import re
from typing import Any, List, Dict, Optional

CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
INT_MAIN_RE = re.compile(r"\bint\s+main\s*\(")

def _to_text(x: Any) -> str:
    """Best-effort convert HF cell to plain text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # examples might be list[dict], etc.
    return str(x)

def extract_cpp_from_text(text: str) -> Optional[str]:
    """Extract C++ code from a text blob (generation/editorial/etc.)."""
    if not isinstance(text, str) or not text.strip():
        return None

    m = CPP_FENCE_RE.search(text)
    if m:
        code = m.group(1).strip()
        return code if code else None

    # Fallback: find a chunk containing int main
    if INT_MAIN_RE.search(text):
        chunks = re.split(r"\n\s*\n", text)
        chunks = [c for c in chunks if INT_MAIN_RE.search(c)]
        if chunks:
            return chunks[-1].strip()

    return None