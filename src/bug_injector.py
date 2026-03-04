# src/bug_injector.py
from __future__ import annotations
import re
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

@dataclass
class BugInjection:
    name: str
    applied: bool
    detail: str = ""

def _apply_once(pattern: str, repl: str, s: str, flags: int = 0) -> Tuple[str, bool]:
    m = re.search(pattern, s, flags)
    if not m:
        return s, False
    start, end = m.span()
    return s[:start] + re.sub(pattern, repl, s[start:end], flags=flags) + s[end:], True

def inject_bugs_cpp(code: str, *, seed: int, max_bugs: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
    """Try to inject 1..max_bugs subtle, *compilable* bugs into C++17 competitive programming code.

    Heuristic / regex based:
      - off_by_one_for_le: turn '<= N' into '< N' in a for-loop header
      - sort_descending: make one ascending sort become descending
      - min_to_max: flip one std::min to std::max
      - ll_to_int: change one 'long long' to 'int' (risk overflow)
      - drop_mod_once: remove one '% MOD' (if MOD present)

    Returns (mutated_code, bug_meta).
    """
    rnd = random.Random(seed)
    s = code or ""
    meta: List[Dict[str, Any]] = []

    # candidate transforms (order randomized)
    transforms = [
        "off_by_one_for_le",
        "sort_descending",
        "min_to_max",
        "ll_to_int",
        "drop_mod_once",
    ]
    rnd.shuffle(transforms)

    applied = 0

    for name in transforms:
        if applied >= max_bugs:
            break

        if name == "off_by_one_for_le":
            # for (...) i <= N -> i < N (only once)
            # keep it conservative: only in for headers
            pat = r"for\s*\([^;]*;[^;]*<=([^;\)]*);[^\)]*\)"
            m = re.search(pat, s)
            if not m:
                continue
            header = m.group(0)
            header2, ok = re.subn(r"<=", "<", header, count=1)
            if ok:
                s = s.replace(header, header2, 1)
                meta.append({"bug": name, "detail": "Changed '<=' to '<' in one for-loop condition"})
                applied += 1
                continue

        if name == "sort_descending":
            # sort(v.begin(), v.end()) -> sort(v.rbegin(), v.rend())
            pat = r"\bsort\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\.begin\s*\(\s*\)\s*,\s*\1\.end\s*\(\s*\)\s*\)"
            m = re.search(pat, s)
            if not m:
                continue
            v = m.group(1)
            repl = f"sort({v}.rbegin(), {v}.rend())"
            s = s[:m.start()] + repl + s[m.end():]
            meta.append({"bug": name, "detail": f"Made one sort on '{v}' descending"})
            applied += 1
            continue

        if name == "min_to_max":
            # flip one std::min or min( -> max(
            for pat, repl, desc in [
                (r"\bstd::min\s*\(", "std::max(", "Flipped std::min -> std::max once"),
                (r"\bmin\s*\(", "max(", "Flipped min -> max once"),
            ]:
                s2, ok = _apply_once(pat, repl, s)
                if ok:
                    s = s2
                    meta.append({"bug": name, "detail": desc})
                    applied += 1
                    break
            continue

        if name == "ll_to_int":
            # change one 'long long' token to 'int'
            # avoid changing typedefs like 'using ll = long long;'
            pat = r"\blong\s+long\b"
            it = list(re.finditer(pat, s))
            if not it:
                continue
            # prefer occurrences that are not part of alias / using
            candidates = [m for m in it if "using" not in s[max(0, m.start()-20):m.start()]]
            m = rnd.choice(candidates or it)
            s = s[:m.start()] + "int" + s[m.end():]
            meta.append({"bug": name, "detail": "Changed one 'long long' to 'int' (overflow risk)"})
            applied += 1
            continue

        if name == "drop_mod_once":
            # if MOD constant exists, remove one '% MOD'
            if not re.search(r"\bMOD\b", s):
                continue
            pat = r"%\s*MOD\b"
            m = re.search(pat, s)
            if not m:
                continue
            s = s[:m.start()] + "" + s[m.end():]
            meta.append({"bug": name, "detail": "Removed one '% MOD'"})
            applied += 1
            continue

    # If nothing applied, do a last-resort tiny bug: change one '+=' to '-='
    if applied == 0:
        m = re.search(r"\+=", s)
        if m:
            s = s[:m.start()] + "-=" + s[m.end():]
            meta.append({"bug": "plus_eq_to_minus_eq", "detail": "Changed one '+=' to '-='"})
            applied = 1

    return s, meta
