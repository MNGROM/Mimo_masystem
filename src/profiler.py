import time
import resource
from typing import Dict, Any, List, Tuple, Optional

from .sandbox import compile_cpp, run_bin
from .schemas import profile_report
from .input_scaler import generate_profile_inputs_from_tests
from .input_gen_llm import generate_inputs_by_llm_script

def _measure(bin_path: str, inputs: List[Tuple[int, str]], sandbox_cfg: Dict[str, Any]):
    input_sizes: List[int] = []
    runtime_ms: List[float] = []
    peak_memory_mb: List[float] = []

    resource.getrusage(resource.RUSAGE_CHILDREN)

    for size_tag, inp in inputs:
        start = time.perf_counter()
        rc, stdout, stderr = run_bin(
            bin_path,
            inp,
            timeout_sec=sandbox_cfg["run_timeout_sec"],
            max_output_bytes=sandbox_cfg["max_output_bytes"],
        )
        end = time.perf_counter()

        if rc == -1 or stderr == "TLE":
            break

        elapsed = (end - start) * 1000.0
        mem_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        mem_mb = mem_kb / 1024.0

        input_sizes.append(int(size_tag))
        runtime_ms.append(round(elapsed, 3))
        peak_memory_mb.append(round(mem_mb, 3))

    return input_sizes, runtime_ms, peak_memory_mb

def profile_code_strict_format(
    llm,  # LLMClient
    task_id: str,
    iteration: int,
    code_cpp: str,
    constraints: Dict[str, int],
    sandbox_cfg: Dict[str, Any],
    verified_tests: List[Dict[str, str]],
    problem_prompt: str
) -> Dict[str, Any] | None:
    """
    Strict-format profiler:
      1) Prefer LLM-generated input generator to create ladder inputs (format-aware)
      2) Fallback to format-preserving expansions from verified tests
    """
    bin_path, ce = compile_cpp(
        code_cpp,
        gpp=sandbox_cfg["gpp"],
        std=sandbox_cfg["std"],
        timeout_sec=sandbox_cfg["compile_timeout_sec"],
    )
    if ce:
        return None

    # SwiftSolve-style ladder (can tune)
    ladder = [2, 5, 10, 20, 50, 100, 200]

    # 1) LLM generator ladder (best chance to scale for arbitrary formats)
    llm_inputs = generate_inputs_by_llm_script(llm, problem_prompt, ladder)
    if llm_inputs:
        input_sizes, runtime_ms, peak_memory_mb = _measure(bin_path, llm_inputs, sandbox_cfg)
        if input_sizes:
            return profile_report(
                task_id=task_id,
                iteration=iteration,
                input_sizes=input_sizes,
                runtime_ms=runtime_ms,
                peak_memory_mb=peak_memory_mb,
                hotspots={"source": "llm_generator"}
            )

    # 2) Fallback: preserve format by transforming verified tests
    test_inputs = generate_profile_inputs_from_tests(verified_tests, ladder=ladder)
    if test_inputs:
        input_sizes, runtime_ms, peak_memory_mb = _measure(bin_path, test_inputs, sandbox_cfg)
        if input_sizes:
            return profile_report(
                task_id=task_id,
                iteration=iteration,
                input_sizes=input_sizes,
                runtime_ms=runtime_ms,
                peak_memory_mb=peak_memory_mb,
                hotspots={"source": "test_transform"}
            )

    return None