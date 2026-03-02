import os
import yaml
import random
from tqdm import tqdm

from src.hf_source import load_hf_rows
from src.builders import build_samples_for_row
from src.utils import ensure_dir, write_jsonl


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["hf"]["seed"])
    ensure_dir(cfg["output"]["out_dir"])

    rows = load_hf_rows(
        dataset_name=cfg["hf"]["dataset"],
        subset=cfg["hf"]["subset"],
        split=cfg["hf"]["split"],
        max_rows=cfg["hf"]["max_rows"],
        seed=cfg["hf"]["seed"],
    )

    planner_samples = []
    coder_samples = []
    coder_iterate_samples = []
    debugger_attack_samples = []
    testgen_general_samples = []
    fixer_samples = []
    fixer_step_samples = []
    fixer_trace_samples = []
    complexity_samples = []

    from collections import Counter
    skip_counter = Counter()
    printed = 0

    for idx, row in enumerate(tqdm(rows, desc="Generating")):
        out, reason = build_samples_for_row(row, idx, cfg)

        if out is None:
            skip_counter[reason] += 1
            if printed < 10:
                print(f"[SKIP] idx={idx} reason={reason}")
                printed += 1
            continue

        planner_samples.extend(out.get("planner", []))
        coder_samples.extend(out.get("coder", []))
        coder_iterate_samples.extend(out.get("coder_iterate", []))
        debugger_attack_samples.extend(out.get("debugger_attack", []))
        testgen_general_samples.extend(out.get("testgen_general", []))
        fixer_samples.extend(out.get("fixer", []))
        fixer_step_samples.extend(out.get("fixer_steps", []))
        fixer_trace_samples.extend(out.get("fixer_traces", []))
        complexity_samples.extend(out.get("complexity", []))

        if reason:
            skip_counter[reason] += 1
            if printed < 10:
                print(f"[WARN] idx={idx} reason={reason}")
                printed += 1

    print("---- SKIP/WARN SUMMARY ----")
    for k, v in skip_counter.most_common():
        print(f"{k}: {v}")

    out_dir = cfg["output"]["out_dir"]

    write_jsonl(os.path.join(out_dir, cfg["output"]["planner_jsonl"]), planner_samples)
    write_jsonl(os.path.join(out_dir, cfg["output"]["coder_jsonl"]), coder_samples)
    write_jsonl(os.path.join(out_dir, cfg["output"]["coder_iterate_jsonl"]), coder_iterate_samples)
    write_jsonl(os.path.join(out_dir, cfg["output"]["debugger_attack_jsonl"]), debugger_attack_samples)
    write_jsonl(os.path.join(out_dir, cfg["output"]["testgen_general_jsonl"]), testgen_general_samples)
    write_jsonl(os.path.join(out_dir, cfg["output"]["fixer_jsonl"]), fixer_samples)

    if "fixer_steps_jsonl" in cfg["output"]:
        write_jsonl(os.path.join(out_dir, cfg["output"]["fixer_steps_jsonl"]), fixer_step_samples)
    if "fixer_traces_jsonl" in cfg["output"]:
        write_jsonl(os.path.join(out_dir, cfg["output"]["fixer_traces_jsonl"]), fixer_trace_samples)

    write_jsonl(os.path.join(out_dir, cfg["output"]["complexity_jsonl"]), complexity_samples)

    print("DONE")
    print("planner:", len(planner_samples))
    print("coder:", len(coder_samples))
    print("coder_iterate:", len(coder_iterate_samples))
    print("debugger_attack:", len(debugger_attack_samples))
    print("testgen_general:", len(testgen_general_samples))
    print("fixer:", len(fixer_samples))
    print("fixer_steps:", len(fixer_step_samples))
    print("fixer_traces:", len(fixer_trace_samples))
    print("complexity:", len(complexity_samples))


if __name__ == "__main__":
    main()
