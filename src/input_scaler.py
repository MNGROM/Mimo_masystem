import re
from typing import List, Dict, Tuple, Optional

_INT_RE = re.compile(r"-?\d+")

def _proxy_size(inp: str) -> int:
    return len(_INT_RE.findall(inp))

def _try_expand_T_by_case_dup(inp: str, target_T: int) -> Optional[str]:
    """
    If input starts with T and contains T cases, duplicate the first case.
    This is format-preserving and works for many CF problems.
    """
    lines = [ln.rstrip("\n") for ln in inp.strip().splitlines()]
    if not lines:
        return None
    if not re.fullmatch(r"\d+", lines[0].strip()):
        return None
    T0 = int(lines[0].strip())
    if T0 <= 0:
        return None

    # We don't truly parse per-case boundaries (hard).
    # But for many problems, each case consumes a fixed number of lines or is self-describing.
    # A safe trick: if T0==1, we can duplicate "the remaining text" as the case body.
    if T0 != 1:
        return None

    case_body = "\n".join(lines[1:]).strip()
    if not case_body:
        return None

    # Duplicate the exact case body target_T times.
    out = [str(target_T)]
    for _ in range(target_T):
        out.append(case_body)
    return "\n".join(out).strip() + "\n"

def _try_scale_array_input(inp: str, target_n: int) -> Optional[str]:
    """
    Scale common:
      n
      a1 ... an
    """
    lines = [ln.strip() for ln in inp.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    if not re.fullmatch(r"\d+", lines[0]):
        return None
    n0 = int(lines[0])
    arr = _INT_RE.findall(lines[1])
    if n0 <= 0 or len(arr) < n0:
        return None

    base = arr[:n0]
    if not base:
        return None
    reps = (target_n + len(base) - 1) // len(base)
    scaled = (base * reps)[:target_n]
    return f"{target_n}\n" + " ".join(scaled) + "\n"

def generate_profile_inputs_from_tests(
    verified_tests: List[Dict[str, str]],
    ladder: List[int],
    max_base: int = 3
) -> List[Tuple[int, str]]:
    """
    Generate a profiling ladder strictly preserving format by transforming verified tests.
    Returns list of (size_proxy_or_n, input_string).
    Strategy:
      1) Choose up to max_base base tests
      2) Try T-expansion (T=1 case duplication)
      3) Try array scaling
      4) Fallback to raw tests
    """
    tests = []
    for t in verified_tests:
        inp = (t.get("input") or "")
        if inp.strip():
            tests.append(inp if inp.endswith("\n") else inp + "\n")
    if not tests:
        return []

    # choose diverse bases by proxy size
    tests_sorted = sorted(tests, key=_proxy_size)
    bases = [tests_sorted[0]]
    if len(tests_sorted) > 1:
        bases.append(tests_sorted[-1])
    if len(tests_sorted) > 2:
        bases.append(tests_sorted[len(tests_sorted)//2])
    bases = bases[:max_base]

    generated = []

    for base in bases:
        # 1) T expansion
        for T in ladder:
            if T <= 1:
                continue
            out = _try_expand_T_by_case_dup(base, T)
            if out:
                generated.append((T, out))

        # 2) array scaling
        for n in ladder:
            if n <= 1:
                continue
            out = _try_scale_array_input(base, n)
            if out:
                generated.append((n, out))

    # If we generated nothing, fallback to base tests (still legal)
    if not generated:
        uniq = []
        seen = set()
        for b in bases:
            if b not in seen:
                seen.add(b)
                uniq.append((_proxy_size(b), b))
        return uniq

    # sort + dedupe
    generated.sort(key=lambda x: x[0])
    uniq = []
    seen_inp = set()
    for s, inp in generated:
        if inp not in seen_inp:
            seen_inp.add(inp)
            uniq.append((s, inp))
    return uniq