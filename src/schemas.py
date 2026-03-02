from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

def problem_input(task_id: str, prompt: str, runtime_limit_ms: int, memory_limit_mb: int, unit_tests: List[Dict[str, str]]):
    return {
        "type": "problem_input",
        "task_id": task_id,
        "prompt": prompt,
        "constraints": {
            "runtime_limit_ms": int(runtime_limit_ms),
            "memory_limit_mb": int(memory_limit_mb),
        },
        "unit_tests": unit_tests,
    }

def testgen_message(task_id: str, iteration: int, tests: List[Dict[str, str]]):
    return {
        "type": "testgen",
        "task_id": task_id,
        "iteration": int(iteration),
        "tests": tests,
    }

def plan_message(task_id: str, iteration: int, problem_statement: str, algorithm: str, input_bounds: Dict[str, Any], constraints: Dict[str, int]):
    return {
        "type": "plan",
        "task_id": task_id,
        "iteration": int(iteration),
        "problem_statement": problem_statement,
        "algorithm": algorithm,
        "input_bounds": input_bounds,
        "constraints": constraints,
    }

def code_message(task_id: str, iteration: int, code_cpp: str):
    return {
        "type": "code",
        "task_id": task_id,
        "iteration": int(iteration),
        "language": "cpp",
        "code_cpp": code_cpp,
        "compiler_flags": ["-O2", "-std=c++17"],
    }


def fixer_input(task_id: str, iteration: int, problem: Dict[str, Any], plan: Dict[str, Any], code_cpp: str, fail_report: str):
    return {
        "type": "fixer_input",
        "task_id": task_id,
        "iteration": int(iteration),
        "problem": problem,
        "plan": plan,
        "code_cpp": code_cpp,
        "fail_report": fail_report,
    }


def fixer_output(task_id: str, iteration: int, fixed_code_cpp: str):
    return {
        "type": "fixed_code",
        "task_id": task_id,
        "iteration": int(iteration),
        "language": "cpp",
        "code_cpp": fixed_code_cpp,
        "compiler_flags": ["-O2", "-std=c++17"],
    }


def fixer_trace(task_id: str, iteration: int, attempts: List[Dict[str, Any]], final_code_cpp: str,
                selfplay_fixed: bool, final_status: str):
    """Trace of the self-play repair loop (MapCoder-Lite style)."""
    return {
        "type": "fixer_trace",
        "task_id": task_id,
        "iteration": int(iteration),
        "attempts": attempts,
        "final_code_cpp": final_code_cpp,
        "selfplay_fixed": bool(selfplay_fixed),
        "final_status": final_status,
    }


def coder_iterate_input(task_id: str, iteration: int, problem: Dict[str, Any], plan: Dict[str, Any],
                        code_cpp: str, fail_report: str):
    """
    Input schema for an "iterative coder" agent: given a previous attempt + failure report,
    produce an improved solution. We keep it separate from fixer_input so you can train
    Coder and Fixer as different agents if desired.
    """
    return {
        "type": "coder_iterate_input",
        "task_id": task_id,
        "iteration": int(iteration),
        "problem": problem,
        "plan": plan,
        "code_cpp": code_cpp,
        "fail_report": fail_report,
    }

def profile_report(task_id: str, iteration: int,
                   input_sizes, runtime_ms,
                   peak_memory_mb, hotspots=None):
    return {
        "type": "profile_report",
        "task_id": task_id,
        "iteration": int(iteration),
        "input_sizes": input_sizes,
        "runtime_ms": runtime_ms,
        "peak_memory_mb": peak_memory_mb,
        "hotspots": hotspots or {}
    }