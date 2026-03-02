import os
import json
import re

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
                    nesting = max(0, nesting-1)

    sort_calls = len(re.findall(r"\bsort\s*\(", s))
    map_uses = len(re.findall(r"\bstd::map\b|\bmap<", s))
    umap_uses = len(re.findall(r"\bstd::unordered_map\b|\bunordered_map<", s))
    set_uses = len(re.findall(r"\bstd::set\b|\bset<", s))
    uset_uses = len(re.findall(r"\bstd::unordered_set\b|\bunordered_set<", s))

    recursion = bool(re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s*\{[\s\S]{0,2000}?\b\1\s*\(", s))

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


def baseline_complexity_verdict(static_features: dict, plan_obj: dict, constraints: dict) -> dict:
    """Deterministic, coarse baseline to anchor the Complexity agent.

    This is NOT meant to be perfect; it's meant to be stable and mostly-correct.
    We look at (a) loop nesting, (b) sort calls, and (c) any plan-provided n bound if present.
    """
    sf = static_features or {}
    max_nest = int(sf.get("max_loop_nesting", 0) or 0)
    sort_calls = int(sf.get("sort_calls", 0) or 0)

    # Try to read an n upper bound from planner variables list (preferred) or input_bounds.
    n_upper = None
    try:
        vars_list = plan_obj.get("variables") or []
        if isinstance(vars_list, list):
            for v in vars_list:
                if isinstance(v, dict) and v.get("name") in ("n", "N"):
                    u = v.get("upper")
                    if isinstance(u, (int, float)):
                        n_upper = int(u)
                        break
        if n_upper is None:
            ib = plan_obj.get("input_bounds") or {}
            if isinstance(ib, dict) and isinstance(ib.get("n"), (int, float)):
                n_upper = int(ib["n"])
    except Exception:
        n_upper = None

    # Fallback typical CF bound
    if n_upper is None or n_upper <= 0:
        n_upper = 200000

    # Rule-of-thumb: nested>=2 suggests ~O(n^2) patterns; nested>=3 is definitely too slow.
    likely_n2 = max_nest >= 2
    likely_n3 = max_nest >= 3
    has_sort = sort_calls > 0

    # Very coarse fit decision
    efficient = True
    reason = ""
    if likely_n3:
        efficient = False
        reason = "Detected 3+ nested loops; likely O(n^3) in worst-case."
    elif likely_n2 and n_upper >= 50000:
        efficient = False
        reason = "Detected 2 nested loops; likely O(n^2) and n is large (>=5e4)."
    else:
        # n log n / linear cases are assumed OK for 2s CF by default.
        efficient = True
        reason = "No obvious high-order nesting; likely <= O(n log n)."

    expr = "O(n^3)" if likely_n3 else ("O(n^2)" if likely_n2 else ("O(n log n)" if has_sort else "O(n)"))

    return {
        "estimated_time": expr,
        "assumed_n_upper": n_upper,
        "efficient": efficient,
        "reason": reason,
    }
