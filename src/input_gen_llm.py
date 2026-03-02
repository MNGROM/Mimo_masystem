import json
import subprocess
import tempfile
from typing import Optional, Dict, Any

GEN_SYSTEM = """You are an expert competitive-programming input generator author.
You MUST output a SINGLE JSON object and NOTHING ELSE.
No markdown, no code fences.

Goal: infer the EXACT input format from the problem statement and write a Python3 script
that generates ONE VALID test input instance whose size scales with parameter n.

Hard requirements:
- The script MUST be Python3 code.
- It MUST accept n as argv[1] integer.
- It MUST print ONLY the test input to stdout.
- It MUST follow the statement's exact input format (including T testcases if present).
- It MUST generate values within typical constraints; be conservative if unknown.
- Deterministic: use fixed random seed(s).

Output JSON schema:
{
  "type": "input_gen",
  "script_py": "..."
}
"""

GEN_USER_TMPL = """Problem statement:
{problem_prompt}

Write the generator script. Remember:
- strict format adherence
- deterministic
- n controls size

Return JSON only.
"""

def _run_script(script_py: str, n: int, timeout_sec: int = 8) -> Optional[str]:
    """Execute generated script to produce stdin text."""
    try:
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/gen.py"
            with open(path, "w", encoding="utf-8") as f:
                f.write(script_py)
            r = subprocess.run(
                ["python3", path, str(n)],
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            if r.returncode != 0:
                return None
            out = r.stdout
            if out and not out.endswith("\n"):
                out += "\n"
            return out if out.strip() else None
    except Exception:
        return None

def build_generator_script(llm, problem_prompt: str) -> Optional[str]:
    """Ask LLM for a generator script (JSON-wrapped)."""
    user = GEN_USER_TMPL.format(problem_prompt=problem_prompt)
    raw = llm.chat(
        [{"role": "system", "content": GEN_SYSTEM},
         {"role": "user", "content": user}],
        force_json_object=False  # DeepSeek varies; we parse ourselves
    )

    # best-effort parse: find first JSON object
    raw = raw.strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end+1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    script = obj.get("script_py")
    if not isinstance(script, str) or not script.strip():
        return None
    return script

def generate_inputs_by_llm_script(llm, problem_prompt: str, ladder_ns) -> Optional[list[tuple[int, str]]]:
    """
    Use LLM-generated generator to produce a ladder of profiling inputs.
    Returns list[(n, stdin)] or None if failed.
    """
    script = build_generator_script(llm, problem_prompt)
    if not script:
        return None

    results = []
    for n in ladder_ns:
        inp = _run_script(script, n)
        if not inp:
            return None
        results.append((n, inp))
    return results