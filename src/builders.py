# src/builders.py
import ast
import json
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, List

from .llm_client import LLMClient
from .prompts_en import (
    PLANNER_SYSTEM, PLANNER_USER,
    DEBUGGER_SYSTEM, DEBUGGER_USER,
    COMPLEXITY_SYSTEM, COMPLEXITY_USER,
    CODER_SYSTEM, CODER_USER,
    BRUTE_CODER_SYSTEM, BRUTE_CODER_USER,
    FIXER_SYSTEM, FIXER_USER,
)
from .extract_cpp import extract_cpp_from_text
from .bug_injector import inject_bugs_cpp
from .sandbox import compile_cpp, compile_cpp_cached, run_bin
from .schemas import (
    problem_input,
    testgen_message,
    code_message,
    fixer_input, fixer_output, fixer_trace,
)
from .utils import safe_task_id, extract_limits, extract_static_features


def _compile(code: str, cfg: Dict[str, Any]):
    """Compile C++ using a persistent cache to avoid recompiling identical code."""
    cache_dir = (cfg.get("sandbox", {}) or {}).get("compile_cache_dir", "./.compile_cache")
    return compile_cpp_cached(
        code,
        gpp=cfg["sandbox"]["gpp"],
        std=cfg["sandbox"]["std"],
        timeout_sec=cfg["sandbox"]["compile_timeout_sec"],
        cache_dir=cache_dir,
    )



def _log_compile_error(stage: str, idx: int, ce: str, cfg: Dict[str, Any], code_snippet: str = ""):
    """Write compile errors to a per-sample log file to debug intermittent failures."""
    try:
        out_dir = cfg.get("output", {}).get("out_dir", "./out")
        log_dir = os.path.join(out_dir, "compile_errors")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{stage}_idx{idx}_pid{os.getpid()}.txt")

        import subprocess
        try:
            gpp_ver = subprocess.run([cfg["sandbox"]["gpp"], "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
        except Exception as e:
            gpp_ver = f"(failed to get g++ version: {e})"

        try:
            ulimit_txt = subprocess.run(["bash", "-lc", "ulimit -a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
        except Exception:
            ulimit_txt = "(ulimit unavailable)"

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"stage={stage}\nidx={idx}\npid={os.getpid()}\n\n")
            f.write("=== g++ --version ===\n")
            f.write(gpp_ver.strip() + "\n\n")
            f.write("=== ulimit -a ===\n")
            f.write(ulimit_txt.strip() + "\n\n")
            f.write("=== compile error ===\n")
            f.write((ce or "").strip() + "\n\n")
            if code_snippet:
                f.write("=== code snippet (first 2000 chars) ===\n")
                f.write(code_snippet[:2000] + "\n")
        # Also print a short pointer for quick debugging
        print(f"[COMPILE_ERROR_LOG] wrote {path}")
    except Exception as e:
        print(f"[COMPILE_ERROR_LOG] failed to write log: {e}")


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



def _cleanup_json_text(t: str) -> str:
    """
    Best-effort cleanup for near-JSON / near-Python-literal:
    - strip ```json ...``` fences
    - normalize smart quotes
    - remove obvious //... and /*...*/ comments (best-effort)
    - extract the first balanced {...} or [...] block
    - remove trailing commas
    - normalize Python literals True/False/None -> JSON true/false/null
    """
    t = (t or "").strip()

    # normalize smart quotes
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")

    # strip ```...``` fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.I)
    if m:
        t = m.group(1).strip()

    # remove obvious comments (best-effort)
    t = re.sub(r"(?m)^\s*//.*$", "", t)
    t = re.sub(r"/\*[\s\S]*?\*/", "", t)

    t = t.strip()

    # extract first balanced block: {...} or [...]
    def extract_balanced(s: str) -> str:
        i1 = s.find("{")
        i2 = s.find("[")
        if i1 == -1 and i2 == -1:
            return s
        if i1 == -1:
            start = i2
            open_c, close_c = "[", "]"
        elif i2 == -1:
            start = i1
            open_c, close_c = "{", "}"
        else:
            if i1 < i2:
                start = i1
                open_c, close_c = "{", "}"
            else:
                start = i2
                open_c, close_c = "[", "]"

        depth = 0
        end = None
        for i in range(start, len(s)):
            c = s[i]
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None:
            return s[start:end].strip()
        return s[start:].strip()

    t = extract_balanced(t)

    # remove trailing commas before } or ]
    t = re.sub(r",\s*([}\]])", r"\1", t)

    # normalize python literals to json literals
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)

    return t.strip()

def _try_parse_json_object(raw: Any) -> Optional[dict]:
    """
    Robust JSON/Python-literal object parser for messy LLM outputs.
    Returns a dict if possible; otherwise None.
    """
    if raw is None or not isinstance(raw, str):
        return None

    t = str(raw).strip()
    if not t:
        return None

    # 0) Sometimes the model returns a JSON string that contains JSON (e.g. "\"{...}\"").
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v[0]
        if isinstance(v, str):
            vv = v.strip()
            if (vv.startswith("{") and vv.endswith("}")) or (vv.startswith("[") and vv.endswith("]")):
                try:
                    v2 = json.loads(vv)
                    if isinstance(v2, dict):
                        return v2
                    if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                        return v2[0]
                except Exception:
                    pass
    except Exception:
        pass

    # 1) direct json parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # 2) cleanup + json parse
    cleaned = _cleanup_json_text(t)
    if not cleaned:
        return None

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # 3) python-literal fallback: handles single quotes, trailing commas, None/True/False, etc.
    try:
        lit = ast.literal_eval(cleaned)
        if isinstance(lit, dict):
            return lit
        if isinstance(lit, list) and lit and isinstance(lit[0], dict):
            return lit[0]
    except Exception:
        pass

    # 4) last resort: if it looks like python dict with single quotes, try a cautious quote swap and json again
    if cleaned.count("'") > cleaned.count('"') and cleaned.startswith(("{", "[")):
        swapped = cleaned.replace('"', '\\"')
        swapped = swapped.replace("'", '"')
        try:
            obj = json.loads(swapped)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0]
        except Exception:
            pass

    return None

def _llm_json_call_with_retry(
    llm: LLMClient,
    system: str,
    user: str,
    *,
    force_json_object: bool = True,
    max_tokens: int | None = None
) -> Optional[dict]:
    """
    Call LLM expecting a JSON object. Parse robustly.
    Retries:
      1) normal (optionally with response_format json_object)
      2) strict repair instruction (temp=0)
      3) fallback: disable json mode + ask to output ONLY minified JSON (temp=0, more tokens)
    """
    raw = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        force_json_object=force_json_object,
        max_tokens=max_tokens
    )
    obj = _try_parse_json_object(raw)
    if isinstance(obj, dict):
        return obj

    repair_user = (
        "Your previous output was not valid JSON.\n"
        "Return ONLY a single JSON object that matches the requested schema. "
        "No extra text, no markdown, no code fences.\n\n"
        + user
    )

    raw2 = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": repair_user}],
        force_json_object=force_json_object,
        temperature=0.0,
        max_tokens=max_tokens
    )
    obj2 = _try_parse_json_object(raw2)
    if isinstance(obj2, dict):
        return obj2

    # final fallback: turn off json mode (some backends behave worse with it) and allow more room
    fallback_user = (
        "Return ONLY valid JSON (one object). No prose.\n"
        "If you included comments or trailing commas before, remove them.\n\n"
        + user
    )

    boosted = None
    if max_tokens is not None:
        boosted = int(max_tokens * 1.5)
    raw3 = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": fallback_user}],
        force_json_object=False,
        temperature=0.0,
        max_tokens=boosted
    )
    obj3 = _try_parse_json_object(raw3)
    return obj3 if isinstance(obj3, dict) else None



def build_samples_for_row(row, idx: int, cfg):
    """
    MapCoder-Lite style workflow (enhanced for multi-agent SFT):

      Planner (Problem -> Plan)
      Coder-teacher (Plan -> Reference Code)  [clean label]
      Coder-selfplay (Plan -> init_code)      [may be wrong]
      Debugger/TestGen (Problem + init_code -> tests)
        - oracle-label expected_output via brute binary
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
        "brute_coder": [],
        "coder_iterate": [],
        "debugger_attack": [],
        "testgen_general": [],
        "fixer": [],
        "fixer_steps": [],
        "fixer_traces": [],
        "complexity": []
    }

    # ------------------------------------------------------------
    # Compile reference solution (teacher label only)
    # ------------------------------------------------------------
    ref_bin, ce = _compile(ref_cpp, cfg)
    if ce:
        _log_compile_error("ref", idx, ce, cfg, code_snippet=ref_cpp)
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
        force_json_object=True,
        max_tokens=int(cfg.get("planner", {}).get("max_tokens", cfg.get("llm", {}).get("max_tokens", 1024)))
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
    # Brute Coder: Plan -> Brute-force Code (small-input correct)
    # Used as the local oracle for differential testing (judge-only setting).
    # ------------------------------------------------------------
    brute_cfg = cfg.get("brute_coder", {}) or {}
    small_constraints = brute_cfg.get("small_constraints") or {
        "notes": "Aim for very small inputs; if variables exist, keep them <= 10-20.",
        "n_max": 12,
        "m_max": 12,
        "k_max": 12,
        "value_abs_max": 20,
        "T_max": 10,
    }
    brute_temperature = brute_cfg.get("temperature", None)
    brute_max_retries = int(brute_cfg.get("max_retries", 2))
    brute_max_retries = max(1, min(brute_max_retries, 4))

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

    brute_cpp = ""
    brute_bin = None
    brute_ce = ""

    brute_user = BRUTE_CODER_USER.format(
        prompt=prompt,
        constraints_json=json.dumps(constraints, ensure_ascii=False),
        plan_json=json.dumps(plan_obj, ensure_ascii=False),
        small_constraints_json=json.dumps(small_constraints, ensure_ascii=False),
    )

    # Try to generate + compile brute solution (retry with compile errors)
    for attempt in range(brute_max_retries):
        brute_raw = llm.chat(
            [
                {"role": "system", "content": BRUTE_CODER_SYSTEM},
                {"role": "user", "content": brute_user},
            ],
            force_json_object=False,
            temperature=brute_temperature,
        )
        brute_cpp = _strip_fences(brute_raw or "")
        brute_bin, brute_ce = _compile(brute_cpp, cfg)
        if not brute_ce:
            break
        brute_user = (
            "Your previous brute C++ code failed to compile. Fix compilation errors and output ONLY corrected C++17 code.\n\n"
            f"Compiler error:\n{brute_ce}\n\n" + brute_user
        )

    if brute_ce or brute_bin is None:
        return out, "brute_compile_error"

    out["brute_coder"].append({
        "schema_version": schema_version,
        "agent": "brute_coder",
        "task": "brute_coder_sft",
        "input": plan_obj,
        "output": code_message(task_id=task_id, iteration=iteration, code_cpp=brute_cpp),
        "meta": {"task_id": task_id, "code_source": "brute_small"}
    })

    # ------------------------------------------------------------
    # Coder self-play: sample multiple init attempts
    # ------------------------------------------------------------
    coder_user = CODER_USER.format(
        prompt=prompt,
        constraints_json=json.dumps(constraints, ensure_ascii=False),
        plan_json=json.dumps(plan_obj, ensure_ascii=False),
    )

    num_init = int(cfg.get("coder", {}).get("num_init_attempts", 2))
    num_init = max(1, min(num_init, 6))

    init_mode = (cfg.get("coder", {}) or {}).get("init_mode", "llm")  # llm | mutate_ref | mixed | tiered
    init_temperature = (cfg.get("coder", {}) or {}).get("init_temperature", None)
    perturb_temperature = (cfg.get("coder", {}) or {}).get("perturb_temperature", init_temperature)
    max_bugs = int((cfg.get("coder", {}) or {}).get("max_bugs_per_init", 2))
    max_bugs = max(1, min(max_bugs, 4))

    def _tiered_mode_list(n: int) -> List[str]:
        coder_cfg = (cfg.get("coder", {}) or {}) if isinstance(cfg, dict) else {}
        w = coder_cfg.get("tier_weights", {}) if isinstance(coder_cfg.get("tier_weights", {}), dict) else {}
        w_mut = float(w.get("mutate_ref", 0.4))
        w_per = float(w.get("llm_perturb", 0.4))
        w_free = float(w.get("llm_free", 0.2))
        total = max(1e-9, w_mut + w_per + w_free)
        w_mut, w_per, w_free = w_mut / total, w_per / total, w_free / total
        c_mut = int(round(n * w_mut))
        c_per = int(round(n * w_per))
        c_free = n - c_mut - c_per
        if c_free < 0:
            c_free = 0
        while c_mut + c_per + c_free < n:
            if w_mut >= w_per and w_mut >= w_free:
                c_mut += 1
            elif w_per >= w_mut and w_per >= w_free:
                c_per += 1
            else:
                c_free += 1
        while c_mut + c_per + c_free > n:
            if c_mut >= c_per and c_mut >= c_free and c_mut > 0:
                c_mut -= 1
            elif c_per >= c_mut and c_per >= c_free and c_per > 0:
                c_per -= 1
            elif c_free > 0:
                c_free -= 1

        modes = ["mutate_ref"] * c_mut + ["llm_perturb"] * c_per + ["llm_free"] * c_free
        import random
        rnd = random.Random(int(cfg.get("hf", {}).get("seed", 0)) + idx * 10007 + n * 31)
        rnd.shuffle(modes)
        if n >= 3:
            if "mutate_ref" not in modes:
                modes[0] = "mutate_ref"
            if ("llm_perturb" not in modes) and ("llm_free" not in modes):
                modes[-1] = "llm_free"
        return modes

    init_modes: List[str]
    if str(init_mode).strip().lower() == "tiered":
        init_modes = _tiered_mode_list(num_init)
    elif str(init_mode).strip().lower() == "mixed":
        init_modes = [("mutate_ref" if (k % 2 == 0) else "llm_free") for k in range(num_init)]
    elif str(init_mode).strip().lower() == "mutate_ref":
        init_modes = ["mutate_ref"] * num_init
    else:
        init_modes = ["llm_free"] * num_init

    init_attempts = []
    for k in range(num_init):
        init_code = ""
        init_ce = ""
        init_bin = None
        bug_meta = []

        mode_k = init_modes[k] if k < len(init_modes) else "llm_free"

        if mode_k == "mutate_ref":
            seed = int(cfg.get("hf", {}).get("seed", 0)) + idx * 1337 + k * 97
            init_code, bug_meta = inject_bugs_cpp(ref_cpp, seed=seed, max_bugs=max_bugs)
        elif mode_k == "llm_perturb":
            perturb_user = (
                "You are given a trusted correct reference solution in C++17. "
                "Rewrite it into a different (but still complete and compilable) C++17 solution. "
                "You MAY introduce subtle logical mistakes (WA), but the code must compile. "
                "Do NOT output any explanations; output ONLY the C++17 code.\n\n"
                "Problem statement:\n" + prompt + "\n\n"
                "Constraints (json):\n" + json.dumps(constraints, ensure_ascii=False) + "\n\n"
                "Planner output (json):\n" + json.dumps(plan_obj, ensure_ascii=False) + "\n\n"
                "Trusted reference code:\n```cpp\n" + ref_cpp + "\n```\n"
            )
            init_code_raw = llm.chat(
                [
                    {"role": "system", "content": "You rewrite C++ solutions."},
                    {"role": "user", "content": perturb_user},
                ],
                force_json_object=False,
                temperature=perturb_temperature,
            )
            init_code = _strip_fences(init_code_raw or "")
        else:
            init_code_raw = llm.chat(
                [
                    {"role": "system", "content": CODER_SYSTEM},
                    {"role": "user", "content": coder_user},
                ],
                force_json_object=False,
                temperature=init_temperature,
            )
            init_code = _strip_fences(init_code_raw or "")

        init_bin, init_ce = _compile(init_code, cfg)

        if init_ce:
            init_bin = None

        init_attempts.append({
            "k": k,
            "code": init_code,
            "bin": init_bin,
            "ce": init_ce or "",
            "bug_meta": bug_meta,
            "init_mode": mode_k,
        })

    # ------------------------------------------------------------
    # Complexity gate + Debugger search for failing init
    # ------------------------------------------------------------
    best = None
    best_reason = None

    max_tests = int(cfg.get("debugger", {}).get("max_tests", 24))
    max_tests = max(4, min(max_tests, 32))

    # Speed knobs to reduce per-problem cost.
    # cap_tests: hard cap of how many candidate tests we actually EXECUTE (even if the LLM returns more).
    # target_mismatches: stop executing tests early once we have enough counterexamples.
    # max_input_ints: skip overly large inputs so brute stays fast.
    cap_tests = int(cfg.get("debugger", {}).get("cap_tests", 12))
    cap_tests = max(2, min(cap_tests, max_tests))
    target_mismatches = int(cfg.get("debugger", {}).get("target_mismatches", 3))
    target_mismatches = max(1, target_mismatches)
    max_input_ints = int(cfg.get("debugger", {}).get("max_input_ints", 220))

    def _complexity_check(code_cpp: str, code_source: str) -> Optional[dict]:
        static_features = extract_static_features(code_cpp)
        baseline = None  # static baseline disabled for dataset generation

        comp_user = COMPLEXITY_USER.format(
            constraints_json=json.dumps(constraints, ensure_ascii=False),
            plan_json=json.dumps(plan_obj, ensure_ascii=False),
            static_features_json=json.dumps(static_features, ensure_ascii=False),
            code_block="```cpp\n" + code_cpp + "\n```",
            task_id=task_id,
            iteration=iteration,
        ) + "\n\nBaseline (deterministic) verdict anchor:\n" + json.dumps(baseline, ensure_ascii=False)

        verdict_obj = _llm_json_call_with_retry(
            llm,
            COMPLEXITY_SYSTEM,
            comp_user,
            force_json_object=True,
            max_tokens=int(cfg.get("complexity", {}).get("max_tokens", cfg.get("llm", {}).get("max_tokens", 1024)))
        )
        if not isinstance(verdict_obj, dict):
            return None

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
                "code": {"type": "code", "language": "cpp", "code_cpp": code_cpp},
            },
            "output": verdict_obj,
            "meta": {"task_id": task_id, "code_source": code_source},
        })
        return verdict_obj

    # Always emit at least one complexity sample so the complexity dataset is populated,
    # but avoid paying an LLM call for every init attempt unless explicitly enabled.
    _complexity_check(ref_cpp, code_source="ref_cpp")

    for att in init_attempts:
        init_code = att["code"]
        init_bin = att["bin"]
        init_ce = att["ce"]

        # -------- Complexity (optional per-attempt) --------
        if bool(cfg.get("complexity", {}).get("per_attempt", False)):
            v = _complexity_check(init_code or ref_cpp, code_source=f"init_attempt_{att['k']}")
            if not isinstance(v, dict):
                best_reason = best_reason or "complexity_json_parse_fail"
                continue
        # NOTE: complexity gating disabled (no static/baseline rejection) for dataset generation.

        # -------- Debugger --------
        if init_ce or init_bin is None:
            best_reason = best_reason or "init_compile_error"
            continue

        dbg_user = DEBUGGER_USER.format(
            prompt=prompt,
            constraints_json=json.dumps(constraints, ensure_ascii=False),
            plan_json=json.dumps(plan_obj, ensure_ascii=False),
            code_block="```cpp\n" + init_code + "\n```",
            task_id=task_id,
            iteration=iteration,
            max_tests=max_tests,
            cap_tests=cap_tests,
        )
        dbg_obj = _llm_json_call_with_retry(
            llm,
            DEBUGGER_SYSTEM,
            dbg_user,
            force_json_object=True,
            max_tokens=int(cfg.get("debugger", {}).get("max_tokens", cfg.get("llm", {}).get("max_tokens", 1024)))
        )
        if not isinstance(dbg_obj, dict):
            best_reason = best_reason or "debugger_json_parse_fail"
            continue

        dbg_obj["type"] = "tests"
        dbg_obj["task_id"] = task_id
        dbg_obj["iteration"] = iteration

        tests = dbg_obj.get("tests") or []
        if not isinstance(tests, list):
            tests = []
        tests = tests[:cap_tests]

        max_output_bytes = int(cfg["sandbox"].get("max_output_bytes", 200000))
        run_timeout_sec = int(cfg["sandbox"].get("run_timeout_sec", 2))
        brute_timeout_sec = int(cfg["sandbox"].get("brute_run_timeout_sec", max(1, run_timeout_sec)))

        mismatches = []
        verified = []

        def _one_test(inp_run: str):
            # oracle: brute output
            rc_b, out_b, err_b = run_bin(
                brute_bin,
                inp_run,
                timeout_sec=brute_timeout_sec,
                max_output_bytes=max_output_bytes,
            )
            if rc_b != 0:
                return None  # brute failed => not trustworthy

            # init output
            rc_i, out_i, err_i = run_bin(
                init_bin,
                inp_run,
                timeout_sec=run_timeout_sec,
                max_output_bytes=max_output_bytes,
            )

            mismatch = None
            if rc_i != 0:
                if rc_i == -1:
                    mismatch = "init_TLE"
                else:
                    mismatch = f"init_RE(rc={rc_i})"
            else:
                if (out_i or "").strip() != (out_b or "").strip():
                    mismatch = "wrong_answer"

            return {
                "input": inp_run.rstrip("\n"),
                "expected_output": out_b,
                "mismatch": mismatch,  # None or reason string
            }

        # Pre-filter candidate inputs to keep brute fast
        candidates = []
        for t in tests:
            if not isinstance(t, dict):
                continue
            inp = t.get("input", "")
            if not isinstance(inp, str) or not inp.strip():
                continue
            inp_run = inp if inp.endswith("\n") else (inp + "\n")
            # Cheap size guard to keep brute fast
            if len(re.findall(r"-?\d+", inp_run)) > max_input_ints:
                continue
            candidates.append(inp_run)

        # Parallel stress-run in small chunks so we can early-stop once we have enough counterexamples.
        max_workers = int(cfg.get("debugger", {}).get("parallel_workers", 4))
        max_workers = max(1, min(max_workers, 32))
        chunk_size = int(cfg.get("debugger", {}).get("parallel_chunk", 8))
        chunk_size = max(1, min(chunk_size, len(candidates) if candidates else 1))

        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            if not chunk:
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_one_test, inp_run) for inp_run in chunk]
                for fut in as_completed(futs):
                    r = fut.result()
                    if not r:
                        continue

                    verified.append({
                        "input": r["input"],
                        "expected_output": r["expected_output"],
                    })

                    if r["mismatch"] is not None:
                        mismatches.append((r["input"], r["expected_output"], r["mismatch"]))

                    if len(mismatches) >= target_mismatches:
                        break

            if len(mismatches) >= target_mismatches:
                break

        if cfg.get("debugger", {}).get("emit_general", True) and verified:
            out["testgen_general"].append({
                "schema_version": schema_version,
                "agent": "testgen",
                "task": "testgen_general_sft",
                "input": {
                    "problem": prob,
                    "plan": plan_obj,
                    "code": {"type": "code", "language": "cpp", "code_cpp": init_code},
                },
                "output": testgen_message(task_id=task_id, iteration=iteration, tests=verified),
                "meta": {"task_id": task_id, "init_k": att["k"], "count": len(verified)},
            })

        if not mismatches:
            best_reason = best_reason or "no_mismatch"
            continue

        fail_report = {
            "mismatch_count": len(mismatches),
            "examples": [
                {"input": inp, "expected_output": exp, "reason": rsn}
                for (inp, exp, rsn) in mismatches[: min(6, len(mismatches))]
            ],
        }

        fail_tests = [{"input": inp, "expected_output": exp} for (inp, exp, _) in mismatches]
        out["debugger_attack"].append({
            "schema_version": schema_version,
            "agent": "debugger",
            "task": "debugger_attack_sft",
            "input": {
                "problem": prob,
                "plan": plan_obj,
                "code": {"type": "code", "language": "cpp", "code_cpp": init_code},
            },
            "output": testgen_message(task_id=task_id, iteration=iteration, tests=fail_tests),
            "meta": {
                "task_id": task_id,
                "init_k": att["k"],
                "mismatch_count": len(mismatches),
                "init_mode": att.get("init_mode"),
                "bug_meta": att.get("bug_meta"),
            },
        })

        if best is None or len(mismatches) > best["mismatch_count"]:
            best = {
                "attempt": att,
                "init_code": init_code,
                "init_bin": init_bin,
                "mismatch_count": len(mismatches),
                "fail_report": fail_report,
                "fail_tests": fail_tests,
            }

    if best is None:
        return out, best_reason or "no_failing_init_found"

    # ------------------------------------------------------------
    # Fixer: Problem + init_code + fail_report -> fixed code
    # ------------------------------------------------------------
    init_code = best["init_code"]
    fail_report = best["fail_report"]

    # schemas.fixer_input expects (problem, plan, code_cpp, fail_report:str)
    fx_in = fixer_input(
        task_id=task_id,
        iteration=iteration,
        problem=prob,
        plan=plan_obj,
        code_cpp=init_code,
        fail_report=json.dumps(fail_report, ensure_ascii=False),
    )

    # In this repo, FIXER outputs ONLY corrected C++ code (see prompts_en.py)
    fixer_user = FIXER_USER.format(
        prompt=prompt,
        constraints_json=json.dumps(constraints, ensure_ascii=False),
        plan_json=json.dumps(plan_obj, ensure_ascii=False),
        code_block="```cpp\n" + init_code + "\n```",
        fail_report=json.dumps(fail_report, ensure_ascii=False),
    )
    fixer_raw = llm.chat(
        [{"role": "system", "content": FIXER_SYSTEM}, {"role": "user", "content": fixer_user}],
        force_json_object=False,
        temperature=(cfg.get("fixer", {}) or {}).get("temperature", None),
    )
    final_code = _strip_fences(fixer_raw or "")
    if not isinstance(final_code, str) or not final_code.strip():
        return out, "fixer_empty"

    out["fixer"].append({
        "schema_version": schema_version,
        "agent": "fixer",
        "task": "fixer_sft",
        "input": fx_in,
        "output": fixer_output(task_id=task_id, iteration=iteration, fixed_code_cpp=final_code),
        "meta": {"task_id": task_id, "init_k": best["attempt"]["k"]}
    })


    # ------------------------------------------------------------
    # Fixer self-play steps + trace (MapCoder-Lite style)
    # ------------------------------------------------------------
    max_iters = int((cfg.get("fixer", {}) or {}).get("max_iters", 0) or 0)
    tests_for_repair = best.get("fail_tests") or []

    def _evaluate_on_tests(code_cpp: str, tests: List[Dict[str, str]]):
        """Compile + run on provided tests. Returns (status, report, compile_err, per_test)."""
        bin_path, ce = _compile(code_cpp, cfg)
        if ce is not None:
            report = {"mismatch_count": len(tests), "examples": [{"input": t["input"], "expected_output": t.get("expected_output",""), "reason": "compile_error"} for t in tests[:6]]}
            return "CE", report, ce, []

        per_test = []
        mismatches = []
        for t in tests:
            inp = t["input"]
            exp = (t.get("expected_output") or "").strip()
            rc, out_s, err_s = run_bin(
                bin_path,
                inp,
                timeout_sec=cfg["sandbox"]["run_timeout_sec"],
                max_output_bytes=cfg["sandbox"]["max_output_bytes"],
            )
            if rc == -1:
                mismatches.append((inp, exp, "TLE"))
            elif rc != 0:
                mismatches.append((inp, exp, f"RE(rc={rc})"))
            else:
                got = (out_s or "").strip()
                if got != exp:
                    mismatches.append((inp, exp, "wrong_answer"))
            per_test.append({"input": inp, "expected_output": exp, "returncode": rc, "stderr": (err_s or "")[:400], "output": (out_s or "")[:800]})

        report = {
            "mismatch_count": len(mismatches),
            "examples": [{"input": i, "expected_output": e, "reason": r} for (i, e, r) in mismatches[:6]],
        }
        status = "OK" if not mismatches else "FAIL"
        return status, report, None, per_test

    if max_iters > 0 and tests_for_repair:
        cur_code = init_code
        cur_report = fail_report
        attempts = []
        selfplay_fixed = False
        final_status = "FAIL"

        for step in range(max_iters):
            # Emit a step-wise SFT sample: (problem + current_code + report) -> next_code
            fx_step_in = fixer_input(
                task_id=task_id,
                iteration=step,
                problem=prob,
                plan=plan_obj,
                code_cpp=cur_code,
                fail_report=json.dumps(cur_report, ensure_ascii=False),
            )

            fixer_user_step = FIXER_USER.format(
                prompt=prompt,
                constraints_json=json.dumps(constraints, ensure_ascii=False),
                plan_json=json.dumps(plan_obj, ensure_ascii=False),
                code_block="```cpp\n" + cur_code + "\n```",
                fail_report=json.dumps(cur_report, ensure_ascii=False),
            )

            fixer_raw_step = llm.chat(
                [{"role": "system", "content": FIXER_SYSTEM}, {"role": "user", "content": fixer_user_step}],
                force_json_object=False,
                temperature=(cfg.get("fixer", {}) or {}).get("temperature", None),
            )
            next_code = _strip_fences(fixer_raw_step or "")
            if not isinstance(next_code, str) or not next_code.strip():
                attempts.append({
                    "step": step,
                    "status": "LLM_EMPTY",
                    "fail_report": cur_report,
                    "code_before": cur_code,
                })
                final_status = "LLM_EMPTY"
                break

            out["fixer_steps"].append({
                "schema_version": schema_version,
                "agent": "fixer",
                "task": "fixer_step_sft",
                "input": fx_step_in,
                "output": fixer_output(task_id=task_id, iteration=step, fixed_code_cpp=next_code),
                "meta": {"task_id": task_id, "init_k": best["attempt"]["k"], "step": step},
            })

            status, new_report, compile_err, per_test = _evaluate_on_tests(next_code, tests_for_repair)

            attempts.append({
                "step": step,
                "status": status,
                "compile_error": compile_err,
                "fail_report_before": cur_report,
                "fail_report_after": new_report,
                "code_before": cur_code,
                "code_after": next_code,
                "per_test": per_test,
            })

            if status == "OK":
                selfplay_fixed = True
                final_status = "OK"
                cur_code = next_code
                cur_report = new_report
                break

            # continue repairing on the same failing set
            cur_code = next_code
            cur_report = new_report

        out["fixer_traces"].append({
            "schema_version": schema_version,
            "agent": "fixer",
            "task": "fixer_trace",
            "input": fx_in,
            "output": fixer_trace(
                task_id=task_id,
                iteration=iteration,
                attempts=attempts,
                final_code_cpp=cur_code,
                selfplay_fixed=selfplay_fixed,
                final_status=final_status,
            ),
            "meta": {"task_id": task_id, "init_k": best["attempt"]["k"], "max_iters": max_iters},
        })

    # ------------------------------------------------------------
    # Coder-iterate: Plan + init_code + fail_report -> fixed code
    # ------------------------------------------------------------
    it_user = (
        "You are improving an initial C++ solution using a failing test report.\n"
        "Return ONLY corrected C++17 code.\n\n"
        "Problem statement:\n" + prompt + "\n\n"
        "Constraints (json):\n" + json.dumps(constraints, ensure_ascii=False) + "\n\n"
        "Planner output (json):\n" + json.dumps(plan_obj, ensure_ascii=False) + "\n\n"
        "Initial code:\n```cpp\n" + init_code + "\n```\n\n"
        "Fail report (json):\n" + json.dumps(fail_report, ensure_ascii=False) + "\n"
    )
    it_raw = llm.chat(
        [{"role": "system", "content": CODER_SYSTEM}, {"role": "user", "content": it_user}],
        force_json_object=False,
        temperature=(cfg.get("coder", {}) or {}).get("iterate_temperature", None),
    )
    it_cpp = (it_raw or "").strip()
    if "```" in it_cpp:
        it_cpp = it_cpp.split("```")[-2].strip() if len(it_cpp.split("```")) >= 3 else it_cpp.replace("```", "").strip()
        if it_cpp.lower().startswith("cpp"):
            it_cpp = "\n".join(it_cpp.splitlines()[1:]).strip()

    out["coder_iterate"].append({
        "schema_version": schema_version,
        "agent": "coder",
        "task": "coder_iterate_sft",
        "input": {
            "plan": plan_obj,
            "init_code": {"type": "code", "language": "cpp", "code_cpp": init_code},
            "fail_report": fail_report,
        },
        "output": code_message(task_id=task_id, iteration=iteration, code_cpp=it_cpp),
        "meta": {"task_id": task_id, "init_k": best["attempt"]["k"]}
    })

    return out, "ok"