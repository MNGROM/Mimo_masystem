import os
import json
import re
import math


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_task_id(row, fallback_idx: int) -> str:
    pid = row.get("problem_id") or row.get("id") or row.get("slug")
    if pid is None:
        return f"ROW_{fallback_idx}"
    s = str(pid)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-:]", "_", s)
    return s[:80]


def extract_limits(row):
    # Best-effort: if dataset provides limits use them; otherwise defaults.
    # Some CF dumps have 'time_limit' (sec) and 'memory_limit' (MB).
    runtime_ms = 2000
    memory_mb = 512

    if row.get("time_limit") is not None:
        try:
            # if it's seconds
            runtime_ms = int(float(row["time_limit"]) * 1000)
        except Exception:
            pass

    if row.get("memory_limit") is not None:
        try:
            memory_mb = int(float(row["memory_limit"]))
        except Exception:
            pass

    return runtime_ms, memory_mb


def extract_static_features(code_cpp: str):
    """Very lightweight static feature extractor for complexity analyst.
    Heuristic only; aims to provide stable signals for SFT conditioning.
    """
    s = code_cpp or ""
    # strip comments (rough)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)

    # count loop keywords
    for_count = len(re.findall(r"\bfor\s*\(", s))
    while_count = len(re.findall(r"\bwhile\s*\(", s))
    do_count = len(re.findall(r"\bdo\b", s))

    # naive nesting estimate: track braces after loop starts
    nesting = 0
    max_nesting = 0
    stack = []
    tokens = re.findall(r"\bfor\b|\bwhile\b|\{|\}", s)
    for tok in tokens:
        if tok in ("for", "while"):
            stack.append("LOOP")
        elif tok == "{":
            if stack and stack[-1] == "LOOP":
                nesting += 1
                max_nesting = max(max_nesting, nesting)
                stack[-1] = "IN_LOOP"
            else:
                stack.append("{")
        elif tok == "}":
            if stack:
                last = stack.pop()
                if last == "IN_LOOP":
                    nesting = max(0, nesting - 1)

    sort_calls = len(re.findall(r"\bsort\s*\(", s))
    map_uses = len(re.findall(r"\bstd::map\b|\bmap<", s))
    umap_uses = len(re.findall(r"\bstd::unordered_map\b|\bunordered_map<", s))
    set_uses = len(re.findall(r"\bstd::set\b|\bset<", s))
    uset_uses = len(re.findall(r"\bstd::unordered_set\b|\bunordered_set<", s))

    recursion = bool(
        re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s*\{[\s\S]{0,2000}?\b\1\s*\(", s)
    )

    return {
        "for_count": for_count,
        "while_count": while_count,
        "do_count": do_count,
        "loop_count": for_count + while_count + do_count,
        "max_loop_nesting": max_nesting,
        "sort_calls": sort_calls,
        "uses_map": map_uses > 0,
        "uses_unordered_map": umap_uses > 0,
        "uses_set": set_uses > 0,
        "uses_unordered_set": uset_uses > 0,
        "has_recursion": recursion,
    }


def _pick_primary_upper(plan_obj: dict) -> tuple[int, dict]:
    """
    Try to infer the dominant input scale from planner variables/input_bounds.
    Returns (upper, debug_info).
    """
    debug = {"picked": None, "candidates": []}
    cand: list[tuple[str, int]] = []

    # 1) variables: [{"name":"n","upper":2e5}, ...]
    vars_list = plan_obj.get("variables") or []
    if isinstance(vars_list, list):
        for v in vars_list:
            if not isinstance(v, dict):
                continue
            name = str(v.get("name") or "").strip()
            u = v.get("upper")
            if isinstance(u, (int, float)) and u > 0:
                cand.append((name, int(u)))

    # 2) input_bounds: {"n":2e5,"m":2e5}
    ib = plan_obj.get("input_bounds") or {}
    if isinstance(ib, dict):
        for k, u in ib.items():
            if isinstance(u, (int, float)) and u > 0:
                cand.append((str(k), int(u)))

    debug["candidates"] = cand[:]

    if not cand:
        return 200000, {"picked": ("fallback", 200000), "candidates": []}

    # Prefer common dominant names, else pick max upper
    prefer = {"n", "N", "m", "M", "q", "Q", "T", "t"}
    preferred = [x for x in cand if x[0] in prefer]
    if preferred:
        name, upper = max(preferred, key=lambda x: x[1])
        debug["picked"] = (name, upper)
        return upper, debug

    name, upper = max(cand, key=lambda x: x[1])
    debug["picked"] = (name, upper)
    return upper, debug


def _nlogn_cost(n: int) -> float:
    if n <= 1:
        return 1.0
    return float(n) * math.log2(float(n))


def baseline_complexity_verdict(static_features: dict, plan_obj: dict, constraints: dict) -> dict:
    """
    Smarter deterministic anchor:
    - Uses runtime_limit_ms to derive an operation budget.
    - Allows higher complexity on smaller input ranges.
    - Still conservative; meant to prevent obvious asymptotic blowups, not micro-opt debates.
    """
    sf = static_features or {}
    max_nest = int(sf.get("max_loop_nesting", 0) or 0)
    sort_calls = int(sf.get("sort_calls", 0) or 0)

    runtime_ms = int((constraints or {}).get("runtime_limit_ms", 2000) or 2000)
    time_sec = max(0.5, runtime_ms / 1000.0)

    # Conservative ops/sec. (CF C++ typical might be higher, but keep anchor stable.)
    ops_per_sec = 8e7
    budget = ops_per_sec * time_sec

    # Infer dominant scale from planner
    n_upper, pick_dbg = _pick_primary_upper(plan_obj or {})

    likely_n2 = max_nest >= 2
    likely_n3 = max_nest >= 3
    has_sort = sort_calls > 0

    # Decide a coarse complexity family
    if likely_n3:
        family = "n3"
        expr = "O(n^3)"
    elif likely_n2:
        family = "n2"
        expr = "O(n^2)"
    else:
        family = "nlogn" if has_sort else "n"
        expr = "O(n log n)" if has_sort else "O(n)"

    # Convert budget to an "allowed n" threshold for that family.
    # c_factor: penalize nested loops / heavier DS a bit.
    c_factor = 1.0
    if sf.get("uses_map") or sf.get("uses_set"):
        c_factor *= 2.0
    if sf.get("has_recursion"):
        c_factor *= 1.3
    if sort_calls > 0:
        c_factor *= 1.2

    allowed_n = None
    if family == "n":
        # Rough per-element budget
        allowed_n = budget / (50.0 * c_factor)
    elif family == "nlogn":
        # Solve n log2 n <= budget / (120*c)
        target = budget / (120.0 * c_factor)
        n = min(max(10.0, float(n_upper)), 1e8)
        for _ in range(6):
            denom = max(1.0, math.log2(max(2.0, n)))
            n = target / denom
        allowed_n = n
    elif family == "n2":
        # n^2 <= budget / (200*c)
        allowed_n = math.sqrt(max(1.0, budget / (200.0 * c_factor)))
    else:  # n3
        # n^3 <= budget / (600*c)
        allowed_n = max(1.0, budget / (600.0 * c_factor)) ** (1.0 / 3.0)

    efficient = n_upper <= int(allowed_n)

    if efficient:
        reason = (
            f"{expr} seems feasible: inferred_upper={n_upper}, allowed≈{int(allowed_n)} "
            f"(time={runtime_ms}ms, c≈{c_factor:.2f})."
        )
    else:
        reason = (
            f"{expr} likely too slow: inferred_upper={n_upper}, allowed≈{int(allowed_n)} "
            f"(time={runtime_ms}ms, c≈{c_factor:.2f})."
        )

    return {
        "estimated_time": expr,
        "assumed_n_upper": n_upper,
        "efficient": bool(efficient),
        "reason": reason,
        "budget_ops": int(budget),
        "allowed_n_rough": int(allowed_n),
        "pick_debug": pick_dbg,
        "c_factor": round(c_factor, 3),
    }