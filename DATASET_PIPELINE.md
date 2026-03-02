# Multi-Agent Dataset Generator (OlympicCoder fine-tuning)

This repo generates **multi-agent SFT datasets** for competitive-programming style tasks (e.g., Codeforces),
with a workflow inspired by **MapCoder-Lite**: the model produces an initial attempt, we run it, collect
failures, and then supervise a repair step with a clean teacher solution.

> **Primary goal:** fine-tune a single base model (e.g., **OlympicCoder**) into multiple specialized agents
(Planner / Coder / Debugger / Fixer / Complexity-Analyst) that can be composed into a multi-agent system.

---

## 1. What this generator produces

Running `python run_generate.py` writes JSONL files under `output.out_dir`:

### Core agents
- `planner.jsonl` — **Planner agent**
  - Task: `problem -> plan (strict JSON)`
- `coder.jsonl` — **Coder agent (teacher)**  
  - Task: `plan -> reference C++ code`
- `coder_iterate.jsonl` — **Iterative Coder (optional agent)**
  - Task: `(plan + previous attempt + failure report) -> reference code`
  - Useful if you want Coder to self-improve without delegating to a separate Fixer.
- `debugger_attack.jsonl` — **Debugger (attack-mode)**
  - Task: `(problem + code) -> tests that *actually break that code*`
  - Only emitted when we can verify at least one failing test against the chosen init attempt.
- `testgen_general.jsonl` — **General TestGen**
  - Task: `(problem + code) -> diverse valid tests (oracle-labeled)`
  - Emitted for every problem that yields verified tests; not necessarily adversarial.
- `fixer.jsonl` — **Fixer agent (single-shot)**
  - Task: `(problem + init attempt + fail report) -> reference code`
- `fixer_steps.jsonl` — **Fixer (step-wise)**
  - Task: for each self-play iteration state: `(problem + current code + report) -> reference code`
- `fixer_traces.jsonl` — **Repair traces (analysis / filtering / RL)**
  - Full trajectory of the self-play repair loop.
- `complexity.jsonl` — **Complexity / Analyst agent**
  - Task: `(constraints + plan + static_features + baseline + code) -> verdict`

All lines share a common top-level envelope:
```json
{
  "schema_version": "v3_mapcoderlite_multiagent",
  "agent": "planner|coder|debugger|fixer|complexity",
  "task": "...",
  "input": {...},
  "output": {...},
  "meta": {...}
}
```

---

## 2. End-to-end workflow (per problem)

Implemented in `src/builders.py::build_samples_for_row`.

### Step 0) Load & normalize problem
- Build a full statement from structured fields:
  `title + description + input_format + output_format + note + examples`
- Extract a **trusted reference C++ solution** (`ref_cpp`) from dataset field `generation`.
- Compile `ref_cpp` to a binary `ref_bin` (this is the oracle for expected outputs).

### Step 1) Planner (Problem -> Plan)
- Prompted to output **strict JSON** (variables, bounds, algorithm steps, complexity, edge cases).
- Writes one `planner_sft` sample.

### Step 2) Coder (teacher) (Plan -> Reference Code)
- Writes one `coder_sft` sample with label = `ref_cpp`.

### Step 3) Coder self-play: generate multiple init attempts
- `coder.num_init_attempts` (default 3) calls to the model to generate `init_code`.
- Each attempt is compiled. Compile errors are allowed and treated as a failure signal.

### Step 4) Debugger/TestGen for each init attempt
For each init attempt:
1. Ask the Debugger to produce 12–24 tests **conditioned on that code**.
2. For each proposed test input:
   - Run `ref_bin` to obtain `expected_output`.
   - Discard tests where the oracle fails (TLE/RE/non-zero exit).
3. Run the init attempt on the verified tests and collect **failing cases**:
   - CE (compile error), RE, TLE, or output mismatch (WA).

We then choose the init attempt with the **largest number of failing cases** (ties prefer compiled code),
so downstream Fixer/Debugger data has stronger supervision.

### Step 5) Split Debugger datasets
From the chosen init attempt:
- **General** (`testgen_general_sft`): all verified tests (oracle-labeled).
- **Attack** (`debugger_attack_sft`): only tests that **actually fail** the chosen init attempt.

If no failing tests are found, `debugger_attack.jsonl` will not get a sample for this problem
(and `build_samples_for_row` returns warning reason `debugger_no_failing_tests` for tracking coverage).

### Step 6) Fixer + Iterative Coder datasets (MapCoder-Lite style)
When there is a real failure signal (non-empty `fail_report`):
- Build a concise `fail_report` from up to 8 failing cases, including:
  input / expected / actual / stderr / return code.

Then emit:
- `coder_iterate_sft`:
  `(problem + plan + init_code + fail_report) -> reference code`
- `fixer_sft`:
  `(problem + plan + init_code + fail_report) -> reference code`

#### Self-play repair loop + traces
The generator also runs a self-play loop up to `fixer.max_iters`:
1. Ask Fixer model to repair.
2. Compile the repaired code.
3. Re-run on the **failing subset** (up to 8 cases).
4. If still failing, create a brief updated report and iterate again.

We record:
- `fixer_traces.jsonl`: all iterations, including pass/fail on failing subset.
- `fixer_steps.jsonl`: a supervised sample for each intermediate state
  `(current code + report) -> reference code`.

> Labels remain **clean** (always `ref_cpp`) even if the model’s intermediate repairs are noisy.

### Step 7) Complexity / Analyst (static + baseline anchor)
No profiler is used.
For each code source (configurable):
- `reference` code
- `selfplay_init` code (when failing exists)

We compute:
- `static_features`: heuristic signals (loop nesting, sorts, maps, recursion hints, etc.)
- `baseline`: deterministic “anchor” verdict from `baseline_complexity_verdict(...)`

The Complexity model receives:
- constraints + plan + static_features + code + baseline
and outputs a strict JSON verdict:
- `efficient`, `bottleneck`, `target_agent` (CODER/PLANNER), plus patch or replan advice.

---

## 3. Why this design helps multi-agent fine-tuning

### Planner
- Learns structured reasoning and implementation-oriented planning.

### Coder
- Learns to write correct solutions from plans (teacher forcing with reference code).

### Debugger (attack-mode)
- Learns to generate **counterexamples** conditioned on a specific implementation.
- Because we filter tests to those that actually break `init_code`, supervision is much closer to real multi-agent debugging.

### Fixer / Iterative Coder
- Learns to repair code given concrete evidence:
  failing inputs + expected outputs + actual outputs + stderr/RC.
- Step-wise samples train robustness across intermediate code states.

### Complexity / Analyst
- Learns routing and efficiency judgment using:
  plan complexity + code structure + constraints,
  anchored by a deterministic baseline to reduce label instability.

---

## 4. Configuration knobs

Edit `config.yaml`:

### Self-play diversity
- `coder.num_init_attempts` (default 3)
  - Higher increases probability of finding a “breakable” init attempt, at the cost of more LLM calls.

### Debugger volume
- `debugger.max_tests` (default 24)

### Fixer loop
- `fixer.max_iters` (default 2)
- `fixer.require_failure` (default true)
  - If true: only emit fixer/coder_iterate when a failure report exists.
- `fixer.keep_only_selfplay_fixed` (default false)
  - If true: keep only cases where self-play passes the failing subset.

### Complexity distribution
- `complexity.code_source_mode`:
  - `ref_only`, `init_only`, `both`, `both_if_failed_else_ref` (default)

---

## 5. Recommended training plan for OlympicCoder

A practical SFT curriculum:

1) **Planner SFT**: `planner.jsonl`
2) **Coder SFT (teacher)**: `coder.jsonl`
3) **Fixer SFT**: start with `fixer.jsonl`, then add `fixer_steps.jsonl` for robustness
4) **Debugger SFT**:
   - train `debugger_attack.jsonl` first (counterexample generation),
   - optionally mix in `testgen_general.jsonl` at lower weight.
5) **Complexity/Router SFT**: `complexity.jsonl`

If you want **separate agents**, train separate LoRA adapters per agent using the corresponding JSONL.
If you want a **single multitask model**, keep `agent` and `task` fields and train with a routing header.

---

## 6. Notes on dataset quality & common failure reasons

The generator may return warning reasons (printed in run output), e.g.:
- `debugger_no_failing_tests`: the chosen init attempt did not fail on verified tests (attack data missing)
- `complexity_json_parse_fail`: Complexity model returned invalid JSON (row still mostly usable)

To improve attack coverage:
- Increase `coder.num_init_attempts`
- Increase `debugger.max_tests`
- (Future) add a **two-round debugger**: generate small set → run → generate targeted follow-up.

---

## 7. File map

- `run_generate.py` — main driver; writes JSONL outputs.
- `src/builders.py` — implements per-row generation pipeline.
- `src/prompts_en.py` — system/user prompts for each agent (JSON-safe brace escaping).
- `src/sandbox.py` — compile/run utilities (timeouts, output caps).
- `src/utils.py` — prompt/build helpers + static feature extraction + baseline verdict.
- `src/schemas.py` — message schemas (problem_input, testgen_message, fixer_input, coder_iterate_input, ...).

---
