# src/builders.py
import json
from typing import Any, Dict, Optional

from .llm_client import LLMClient
from .prompts_en import (
    PLANNER_SYSTEM, PLANNER_USER,
    DEBUGGER_SYSTEM, DEBUGGER_USER,
    COMPLEXITY_SYSTEM, COMPLEXITY_USER,
    CODER_SYSTEM, CODER_USER,
    FIXER_SYSTEM, FIXER_USER,
)
from .extract_cpp import extract_cpp_from_text
from .sandbox import compile_cpp, run_bin
from .schemas import (
    problem_input,
    testgen_message,
    code_message,
    fixer_input, fixer_output, fixer_trace,
)
from .utils import safe_task_id, extract_limits, extract_static_features, baseline_complexity_verdict


def _format_examples(examples: Any) -> str:
    """
    Dataset 'examples' is often list[dict] like:
      [{"input": "...", "output": "..."}, ...]
    Convert to a readable section.
    """
    if not examples:
        return ""
    try:
        if isinstance(examples, list):
            parts = []
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, dict):
                    inp = ex.get("input", "")
                    out = ex.get("output", "")
                    parts.append(f"Example {i}:\nInput:\n{inp}\nOutput:\n{out}")
                else:
                    parts.append(f"Example {i}:\n{str(ex)}")
            return "\n\n".join(parts).strip()
        return str(examples).strip()
    except Exception:
        return str(examples).strip()


def _build_problem_prompt(row: Dict[str, Any]) -> str:
    """
    Build the REAL problem statement from dataset fields.
    """
    title = (row.get("title") or "").strip()
    desc = (row.get("description") or "").strip()
    inp = (row.get("input_format") or "").strip()
    out = (row.get("output_format") or "").strip()
    note = (row.get("note") or "").strip()
    examples = _format_examples(row.get("examples"))

    parts = []
    if title:
        parts.append(f"# {title}")
    if desc:
        parts.append(desc)
    if inp:
        parts.append("# Input\n" + inp)
    if out:
        parts.append("# Output\n" + out)
    if note:
        parts.append("# Note\n" + note)
    if examples:
        parts.append("# Examples\n" + examples)

    return "\n\n".join(parts).strip()


def _try_parse_json_object(s: Any) -> Optional[dict]:
    """
    Robustly parse the first JSON object from a string.

    Handles cases where the model outputs extra text around JSON,
    or code fences, etc. Returns dict or None.
    """
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None

    # 1) direct parse (works when response_format=json_object succeeds)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        # Sometimes backend returns a list with one dict
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # 2) strip markdown fences if present
    if "```" in t:
        parts = t.split("```")
        # pick the largest looking part
        cand = ""
        for p in parts:
            pp = p.strip()
            if len(pp) > len(cand):
                cand = pp
        t2 = cand.strip()
        if t2.lower().startswith("json"):
            t2 = "\n".join(t2.splitlines()[1:]).strip()
        t = t2

    # 3) brace-stack: extract first full {...}
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        c = t[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        return obj[0]
                except Exception:
                    return None
    return None


def _llm_json_call_with_retry(llm: LLMClient, system: str, user: str, *, force_json_object: bool = True) -> Optional[dict]:
    """
    Call LLM expecting a JSON object. Parse robustly.
    If fails, retry once with a strict repair instruction.
    """
    raw = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        force_json_object=force_json_object
    )
    obj = _try_parse_json_object(raw)
    if isinstance(obj, dict):
        return obj

    repair_user = (
        "Your previous output was not valid JSON.\n"
        "Return ONLY a single JSON object that matches the requested schema. No extra text, no markdown.\n\n"
        + user
    )
    raw2 = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": repair_user}],
        force_json_object=force_json_object
    )
    obj2 = _try_parse_json_object(raw2)
    return obj2 if isinstance(obj2, dict) else None


def build_samples_for_row(row, idx: int, cfg):
    """
    MapCoder-Lite style workflow (enhanced for multi-agent SFT):

      Planner (Problem -> Plan)
      Coder-teacher (Plan -> Reference Code)  [clean label]
      Coder-selfplay (Plan -> init_code)      [may be wrong]
      Debugger/TestGen (Problem + init_code -> tests)
        - oracle-label expected_output via reference binary
        - split outputs:
            * debugger_attack_sft: ONLY tests that make init_code fail (else skip)
            * testgen_general_sft: all verified tests (optional but enabled by default)
      Fixer (Problem + init_code + fail_report -> Reference Code) [clean label]
        - records multi-iteration trace + per-step samples
      Coder-iterate (Plan + init_code + fail_report -> Reference Code) [clean label]
      Complexity/Analyst (static + baseline anchor) on both reference and init distributions (configurable).

    Returns: (out_dict, reason)
    """

    schema_version = cfg.get("dataset", {}).get("schema_version", "v3_mapcoderlite")

    # 1) Real prompt from structured fields (NOT row["prompt"])
    prompt = _build_problem_prompt(row)
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "empty_prompt"

    # 2) Reference C++ from generation (used as label and as trusted oracle for expected_output)
    gen = row.get("generation") or ""
    ref_cpp = extract_cpp_from_text(gen)
    if not ref_cpp:
        return None, "no_ref_cpp"

    task_id = safe_task_id(row, idx)
    iteration = 0

    runtime_ms, memory_mb = extract_limits(row)
    constraints = {"runtime_limit_ms": runtime_ms, "memory_limit_mb": memory_mb}

    prob = problem_input(
        task_id=task_id,
        prompt=prompt,
        runtime_limit_ms=runtime_ms,
        memory_limit_mb=memory_mb,
        unit_tests=[]
    )

    llm = LLMClient(**cfg["llm"])

    out = {
        "planner": [],
        "coder": [],
        "coder_iterate": [],
        "debugger_attack": [],
        "testgen_general": [],
        "fixer": [],
        "fixer_steps": [],
        "fixer_traces": [],
        "complexity": []
    }

    # ------------------------------------------------------------
    # Compile reference solution (trusted oracle for labeling tests)
    # ------------------------------------------------------------
    ref_bin, ce = compile_cpp(
        ref_cpp,
        gpp=cfg["sandbox"]["gpp"],
        std=cfg["sandbox"]["std"],
        timeout_sec=cfg["sandbox"]["compile_timeout_sec"],
    )
    if ce:
        return None, "ref_compile_error"

    # ------------------------------------------------------------
    # Planner: ProblemInput -> PlanMessage  (ROBUST JSON)
    # ------------------------------------------------------------
    planner_user = PLANNER_USER.format(
        prompt=prompt,
        constraints_json=json.dumps(constraints, ensure_ascii=False),
        task_id=task_id,
        iteration=iteration
    )
    plan_obj = _llm_json_call_with_retry(
        llm,
        PLANNER_SYSTEM,
        planner_user,
        force_json_object=True
    )
    if not isinstance(plan_obj, dict):
        return None, "planner_json_parse_fail"

    # Normalize envelope
    plan_obj["type"] = "plan"
    plan_obj["task_id"] = task_id
    plan_obj["iteration"] = iteration
    plan_obj.setdefault("constraints", constraints)
    plan_obj.setdefault("problem_statement", plan_obj.get("problem_statement") or prompt)
    plan_obj.setdefault("algorithm", "unknown")
    plan_obj.setdefault("input_bounds", {"n": None, "m": None, "k": None})

    out["planner"].append({
        "schema_version": schema_version,
        "agent": "planner",
        "task": "planner_sft",
        "input": prob,
        "output": plan_obj,
        "meta": {"task_id": task_id}
    })

    # ------------------------------------------------------------
    # Coder (teacher signal): Plan -> Reference Code (clean label)
    # ------------------------------------------------------------
    code_obj = code_message(task_id=task_id, iteration=iteration, code_cpp=ref_cpp)
    out["coder"].append({
        "schema_version": schema_version,
        "agent": "coder",
        "task": "coder_sft",
        "input": plan_obj,
        "output": code_obj,
        "meta": {"task_id": task_id, "code_source": "reference"}
    })

    # ------------------------------------------------------------
    # Coder self-play: sample multiple init attempts
    # ------------------------------------------------------------
    def _strip_fences(code: str) -> str:
        code = (code or "").strip()
        if "```" not in code:
            return code
        parts = code.split("```")
        cand = ""
        for p in reversed(parts):
            if p.strip():
                cand = p
                break
        cand = cand.strip()
        if cand.lower().startswith("cpp"):
            cand = "\n".join(cand.splitlines()[1:]).strip()
        return cand.strip()

    coder_user = CODER_USER.format(
        prompt=prompt,
        constraints_json=json.dumps(constraints, ensure_ascii=False),
        plan_json=json.dumps(plan_obj, ensure_ascii=False),
    )

    num_init = int(cfg.get("coder", {}).get("num_init_attempts", 2))
    num_init = max(1, min(num_init, 6))

    init_attempts = []
    for k in range(num_init):
        init_code_raw = llm.chat(
            [
                {"role": "system", "content": CODER_SYSTEM},
                {"role": "user", "content": coder_user},
            ],
            force_json_object=False
        )
        init_code = _strip_fences(init_code_raw or "")
        init_bin, init_ce = compile_cpp(
            init_code,
            gpp=cfg["sandbox"]["gpp"],
            std=cfg["sandbox"]["std"],
            timeout_sec=cfg["sandbox"]["compile_timeout_sec"],
        )
        if init_ce:
            init_bin = None
        init_attempts.append({
            "k": k,
            "code": init_code,
            "bin": init_bin,
            "ce": init_ce or "",
        })

    # ------------------------------------------------------------
    # For each init attempt: Debugger -> verify via oracle -> find failing tests
    # Pick the attempt with the most failing cases (strongest failure signal)
    # ------------------------------------------------------------
    best = None
    best_reason = None

    max_tests = int(cfg.get("debugger", {}).get("max_tests", 24))
    max_tests = max(8, min(max_tests, 32))

    for att in init_attempts:
        init_code = att["code"]
        init_bin = att["bin"]
        init_ce = att["ce"]

        dbg_user = DEBUGGER_USER.format(
            prompt=prompt,
            code_block="```cpp\n" + (init_code if init_code else ref_cpp) + "\n```",
            task_id=task_id,
            iteration=iteration
        )

        # ROBUST JSON: force json + parse + retry
        dbg_obj = _llm_json_call_with_retry(
            llm,
            DEBUGGER_SYSTEM,
            dbg_user,
            force_json_object=True
        )
        if not isinstance(dbg_obj, dict):
            best_reason = best_reason or "debugger_json_parse_fail"
            continue

        tests = dbg_obj.get("tests", [])
        if not isinstance(tests, list) or not tests:
            best_reason = best_reason or "debugger_no_tests"
            continue

        # Verified tests (oracle expected outputs)
        verified_tests = []
        unit_tests = []

        for t in tests[:max_tests]:
            if not isinstance(t, dict):
                continue
            inp = (t.get("input") or "")
            if not isinstance(inp, str) or not inp.strip():
                continue

            why = (t.get("why") or "")
            category = (t.get("category") or "")
            construction = (t.get("construction") or "")
            target = (t.get("target") or "unknown")

            run_inp = inp if inp.endswith("\n") else inp + "\n"
            rc, stdout, stderr = run_bin(
                ref_bin,
                run_inp,
                timeout_sec=cfg["sandbox"]["run_timeout_sec"],
                max_output_bytes=cfg["sandbox"]["max_output_bytes"],
            )
            if stderr == "TLE" or rc != 0:
                continue

            expected = stdout
            verified_tests.append({
                "category": category,
                "input": inp,
                "expected_output": expected,
                "why": why,
                "construction": construction,
                "target": target
            })
            unit_tests.append({"input": inp, "output": expected})

        if not verified_tests:
            best_reason = best_reason or "debugger_no_verified_tests"
            continue

        # Find failing tests for this init attempt (vs oracle)
        failing = []
        if init_bin is None:
            # compile error case: keep one test as evidence
            failing = [{
                "input": verified_tests[0]["input"],
                "expected_output": verified_tests[0]["expected_output"],
                "actual_output": "",
                "stderr": f"CompileError: {init_ce}",
                "rc": -2,
            }]
        else:
            for t in verified_tests[:max_tests]:
                run_inp = t["input"] if t["input"].endswith("\n") else t["input"] + "\n"
                rc2, out2, err2 = run_bin(
                    init_bin,
                    run_inp,
                    timeout_sec=cfg["sandbox"]["run_timeout_sec"],
                    max_output_bytes=cfg["sandbox"]["max_output_bytes"],
                )
                if err2 == "TLE":
                    failing.append({
                        "input": t["input"],
                        "expected_output": t["expected_output"],
                        "actual_output": out2,
                        "stderr": "TLE",
                        "rc": -1,
                    })
                elif rc2 != 0:
                    failing.append({
                        "input": t["input"],
                        "expected_output": t["expected_output"],
                        "actual_output": out2,
                        "stderr": err2,
                        "rc": rc2,
                    })
                else:
                    if out2 != t["expected_output"]:
                        failing.append({
                            "input": t["input"],
                            "expected_output": t["expected_output"],
                            "actual_output": out2,
                            "stderr": err2,
                            "rc": rc2,
                        })

        # Build fail_report (cap)
        lines = []
        for i, f in enumerate(failing[:8]):
            lines.append(
                f"Case {i+1}:\nINPUT:\n{f['input']}\nEXPECTED:\n{f['expected_output']}\n"
                f"ACTUAL:\n{f['actual_output']}\nSTDERR/RC:\n{f['stderr']} / {f['rc']}"
            )
        fail_report = "\n\n".join(lines).strip() if lines else ""

        # Attack tests are those that actually fail this init attempt
        attack_inputs = set([f["input"] for f in failing if isinstance(f, dict) and "input" in f])
        attack_tests = [t for t in verified_tests if t["input"] in attack_inputs]

        # Score
        score = len(failing)
        if init_bin is None and init_ce:
            score = max_tests + 1  # boost CE attempt so it can be selected if needed

        cand = {
            "attempt": att,
            "verified_tests": verified_tests,
            "unit_tests": unit_tests,
            "failing": failing,
            "fail_report": fail_report,
            "attack_tests": attack_tests,
        }

        if best is None or score > best["score"]:
            best = {"score": score, **cand}
        elif score == best["score"]:
            # tie-break: prefer compiled
            if best["attempt"]["bin"] is None and att["bin"] is not None:
                best = {"score": score, **cand}

    if best is None:
        # planner/coder are still useful
        return out, best_reason or "debugger_no_verified_tests"

    # adopt chosen attempt
    chosen = best
    init_code = chosen["attempt"]["code"]
    init_bin = chosen["attempt"]["bin"]
    init_ce = chosen["attempt"]["ce"]
    verified_tests = chosen["verified_tests"]
    unit_tests = chosen["unit_tests"]
    failing = chosen["failing"]
    fail_report = chosen["fail_report"]
    attack_tests = chosen["attack_tests"]

    # attach unit_tests to problem for downstream agents
    prob["unit_tests"] = unit_tests

    # ------------------------------------------------------------
    # Debugger dataset split
    # ------------------------------------------------------------
    out["testgen_general"].append({
        "schema_version": schema_version,
        "agent": "debugger",
        "task": "testgen_general_sft",
        "input": {
            "problem": prob,
            "code": {"type": "code", "language": "cpp", "code_cpp": init_code},
            "plan": plan_obj,
        },
        "output": testgen_message(task_id=task_id, iteration=iteration, tests=verified_tests),
        "meta": {"task_id": task_id, "code_source": "selfplay_init", "verified_count": len(verified_tests)}
    })

    if attack_tests:
        out["debugger_attack"].append({
            "schema_version": schema_version,
            "agent": "debugger",
            "task": "debugger_attack_sft",
            "input": {
                "problem": prob,
                "code": {"type": "code", "language": "cpp", "code_cpp": init_code},
                "plan": plan_obj,
            },
            "output": testgen_message(task_id=task_id, iteration=iteration, tests=attack_tests),
            "meta": {"task_id": task_id, "code_source": "selfplay_init", "attack_count": len(attack_tests)}
        })

    # ------------------------------------------------------------
    # Produce coder_iterate & fixer only when there is a real failure signal
    # ------------------------------------------------------------
    require_failure = bool(cfg.get("fixer", {}).get("require_failure", True))
    has_failure = bool(failing) and bool(fail_report.strip())

    if (not require_failure) or has_failure:
        # Coder-iterate dataset
        from .schemas import coder_iterate_input
        out["coder_iterate"].append({
            "schema_version": schema_version,
            "agent": "coder",
            "task": "coder_iterate_sft",
            "input": coder_iterate_input(
                task_id=task_id,
                iteration=iteration,
                problem=prob,
                plan=plan_obj,
                code_cpp=init_code,
                fail_report=fail_report,
            ),
            "output": code_message(task_id=task_id, iteration=iteration, code_cpp=ref_cpp),
            "meta": {"task_id": task_id, "code_source": "selfplay_init"}
        })

        # --------------------------
        # Fixer self-play iterations
        # --------------------------
        max_iters = int(cfg.get("fixer", {}).get("max_iters", 2))
        max_iters = max(1, min(max_iters, 5))

        attempts = []
        cur_code = init_code

        def _eval_on_subset(bin_path, failing_cases):
            if bin_path is None:
                return False, "compile_error", len(failing_cases)
            still = []
            for f in failing_cases[:8]:
                run_inp = f["input"] if f["input"].endswith("\n") else f["input"] + "\n"
                rc3, out3, err3 = run_bin(
                    bin_path,
                    run_inp,
                    timeout_sec=cfg["sandbox"]["run_timeout_sec"],
                    max_output_bytes=cfg["sandbox"]["max_output_bytes"],
                )
                if err3 == "TLE":
                    still.append({"stderr": "TLE"})
                elif rc3 != 0:
                    still.append({"stderr": err3})
                elif out3 != f["expected_output"]:
                    still.append({"stderr": "WA"})
            return (len(still) == 0), ("passed" if len(still) == 0 else "still_failed"), len(still)

        attempts.append({
            "iter": 0,
            "code_cpp": init_code,
            "fail_report": fail_report,
            "compile_error": init_ce,
            "failing_count": len(failing),
            "passed_on_subset": False,
        })

        fixer_user = FIXER_USER.format(
            prompt=prompt,
            constraints_json=json.dumps(constraints, ensure_ascii=False),
            plan_json=json.dumps(plan_obj, ensure_ascii=False),
            code_block="```cpp\n" + init_code + "\n```",
            fail_report=fail_report,
        )
        fixed_raw = llm.chat(
            [
                {"role": "system", "content": FIXER_SYSTEM},
                {"role": "user", "content": fixer_user},
            ],
            force_json_object=False
        )
        cur_code = _strip_fences(fixed_raw or "")

        selfplay_fixed = False
        final_status = "still_failed_on_subset"
        final_code_cpp = cur_code

        for it in range(1, max_iters + 1):
            cur_bin, cur_ce = compile_cpp(
                cur_code,
                gpp=cfg["sandbox"]["gpp"],
                std=cfg["sandbox"]["std"],
                timeout_sec=cfg["sandbox"]["compile_timeout_sec"],
            )
            if cur_ce:
                cur_bin = None
                pass_ok = False
                still_cnt = len(failing[:8])
                rep = f"CompileError: {cur_ce}"
            else:
                pass_ok, _status, still_cnt = _eval_on_subset(cur_bin, failing)
                rep = "" if pass_ok else f"Still failing on {still_cnt} cases."

            attempts.append({
                "iter": it,
                "code_cpp": cur_code,
                "fail_report": rep if rep else "(passed_on_subset)",
                "compile_error": cur_ce,
                "failing_count": still_cnt,
                "passed_on_subset": pass_ok,
            })

            # step-wise supervised sample
            out["fixer_steps"].append({
                "schema_version": schema_version,
                "agent": "fixer",
                "task": "fixer_step_sft",
                "input": fixer_input(
                    task_id=task_id,
                    iteration=iteration,
                    problem=prob,
                    plan=plan_obj,
                    code_cpp=cur_code,
                    fail_report=(rep if rep else "(passed_on_subset)"),
                ),
                "output": fixer_output(task_id=task_id, iteration=iteration, fixed_code_cpp=ref_cpp),
                "meta": {"task_id": task_id, "iter": it}
            })

            if pass_ok:
                selfplay_fixed = True
                final_status = "passed_on_fail_subset"
                final_code_cpp = cur_code
                break

            if it >= max_iters:
                final_status = "compile_error" if cur_bin is None else "still_failed_on_subset"
                final_code_cpp = cur_code
                break

            fixer_user2 = FIXER_USER.format(
                prompt=prompt,
                constraints_json=json.dumps(constraints, ensure_ascii=False),
                plan_json=json.dumps(plan_obj, ensure_ascii=False),
                code_block="```cpp\n" + cur_code + "\n```",
                fail_report=(rep if rep else "Still failing."),
            )
            cur_raw2 = llm.chat(
                [
                    {"role": "system", "content": FIXER_SYSTEM},
                    {"role": "user", "content": fixer_user2},
                ],
                force_json_object=False
            )
            cur_code = _strip_fences(cur_raw2 or "")

        keep_only_fixed = bool(cfg.get("fixer", {}).get("keep_only_selfplay_fixed", False))
        if (not keep_only_fixed) or selfplay_fixed:
            out["fixer"].append({
                "schema_version": schema_version,
                "agent": "fixer",
                "task": "fixer_sft",
                "input": fixer_input(
                    task_id=task_id,
                    iteration=iteration,
                    problem=prob,
                    plan=plan_obj,
                    code_cpp=init_code,
                    fail_report=fail_report,
                ),
                "output": fixer_output(task_id=task_id, iteration=iteration, fixed_code_cpp=ref_cpp),
                "meta": {"task_id": task_id, "selfplay_fixed": selfplay_fixed}
            })

            out["fixer_traces"].append({
                "schema_version": schema_version,
                "agent": "fixer",
                "task": "fixer_trace",
                "input": {"task_id": task_id},
                "output": fixer_trace(
                    task_id=task_id,
                    iteration=iteration,
                    attempts=attempts,
                    final_code_cpp=final_code_cpp,
                    selfplay_fixed=selfplay_fixed,
                    final_status=final_status,
                ),
                "meta": {"task_id": task_id, "selfplay_fixed": selfplay_fixed, "final_status": final_status}
            })

    # ------------------------------------------------------------
    # Complexity / Analyst (ROBUST JSON)
    # ------------------------------------------------------------
    code_source_mode = (cfg.get("complexity", {}) or {}).get("code_source_mode", "both_if_failed_else_ref")
    if code_source_mode == "ref_only":
        code_sources = [("reference", ref_cpp)]
    elif code_source_mode == "init_only":
        code_sources = [("selfplay_init", init_code or ref_cpp)]
    elif code_source_mode == "both":
        code_sources = [("reference", ref_cpp), ("selfplay_init", init_code or ref_cpp)]
    else:
        if failing:
            code_sources = [("reference", ref_cpp), ("selfplay_init", init_code or ref_cpp)]
        else:
            code_sources = [("reference", ref_cpp)]

    for cs_name, cs_code in code_sources:
        static_features = extract_static_features(cs_code)
        baseline = baseline_complexity_verdict(plan_obj, static_features, constraints)

        comp_user = COMPLEXITY_USER.format(
            constraints_json=json.dumps(constraints, ensure_ascii=False),
            plan_json=json.dumps(plan_obj, ensure_ascii=False),
            static_features_json=json.dumps(static_features, ensure_ascii=False),
            code_block="```cpp\n" + cs_code + "\n```",
            task_id=task_id,
            iteration=iteration
        ) + "\n\nBaseline (deterministic) verdict anchor:\n" + json.dumps(baseline, ensure_ascii=False)

        verdict_obj = _llm_json_call_with_retry(
            llm,
            COMPLEXITY_SYSTEM,
            comp_user,
            force_json_object=True
        )
        if not isinstance(verdict_obj, dict):
            # don't fail entire row; just warn
            return out, "complexity_json_parse_fail"

        verdict_obj["type"] = "verdict"
        verdict_obj["task_id"] = task_id
        verdict_obj["iteration"] = iteration

        out["complexity"].append({
            "schema_version": schema_version,
            "agent": "complexity",
            "task": "complexity_sft",
            "input": {
                "constraints": constraints,
                "plan": plan_obj,
                "static_features": static_features,
                "baseline": baseline,
                "code": {"type": "code", "language": "cpp", "code_cpp": cs_code}
            },
            "output": verdict_obj,
            "meta": {"task_id": task_id, "code_source": cs_name}
        })

    if not out["debugger_attack"]:
        return out, "debugger_no_failing_tests"

    return out, None