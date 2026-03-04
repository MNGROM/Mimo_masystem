import os
import yaml
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.hf_source import load_hf_rows
from src.builders import build_samples_for_row
from src.utils import ensure_dir, write_jsonl


def _work(args):
    row, idx, cfg = args
    return idx, build_samples_for_row(row, idx, cfg)


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
    shard_id = int((cfg.get("runtime", {}) or {}).get("shard_id", 0))
    num_shards = int((cfg.get("runtime", {}) or {}).get("num_shards", 1))
    num_shards = max(1, num_shards)
    shard_id = max(0, min(shard_id, num_shards - 1))

    rows = [r for i, r in enumerate(rows) if i % num_shards == shard_id]
    print(f"[SHARD] shard_id={shard_id} num_shards={num_shards} rows={len(rows)}")

    # runtime.workers: number of parallel processes
    workers = int((cfg.get("runtime", {}) or {}).get("workers", os.cpu_count() or 4))
    workers = max(1, workers)

    planner_samples = []
    coder_samples = []
    coder_iterate_samples = []
    debugger_attack_samples = []
    testgen_general_samples = []
    fixer_samples = []
    fixer_step_samples = []
    fixer_trace_samples = []
    complexity_samples = []
    brute_coder_samples = []

    from collections import Counter
    skip_counter = Counter()
    printed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, (row, idx, cfg)) for idx, row in enumerate(rows)]

        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"Generating (parallel={workers})",
        ):
            try:
                idx, (out, reason) = fut.result()
            except Exception as e:
                skip_counter["exception"] += 1
                if printed < 10:
                    print(f"[EXCEPTION] {e}")
                    printed += 1
                continue

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
            brute_coder_samples.extend(out.get("brute_coder", []))

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
    if "brute_coder_jsonl" in cfg["output"]:
        write_jsonl(os.path.join(out_dir, cfg["output"]["brute_coder_jsonl"]), brute_coder_samples)
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
    print("brute_coder:", len(brute_coder_samples))
    print("coder_iterate:", len(coder_iterate_samples))
    print("debugger_attack:", len(debugger_attack_samples))
    print("testgen_general:", len(testgen_general_samples))
    print("fixer:", len(fixer_samples))
    print("fixer_steps:", len(fixer_step_samples))
    print("fixer_traces:", len(fixer_trace_samples))
    print("complexity:", len(complexity_samples))


if __name__ == "__main__":
    main()