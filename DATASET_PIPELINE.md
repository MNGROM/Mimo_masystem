# Multi-Agent Dataset Generator (OlympicCoder fine-tuning)

This repo generates **multi-agent SFT datasets** for competitive-programming style tasks (e.g., Codeforces),
with a workflow inspired by **MapCoder-Lite**: the model produces an initial attempt, we run it, collect
failures, and then supervise a repair step with a clean teacher solution.

> **Primary goal:** fine-tune a single base model (e.g., **OlympicCoder**) into multiple specialized agents
(Planner / Coder / Brute Coder / Debugger / Fixer / Complexity-Analyst) that can be composed into a multi-agent system.

---

## 1. What this generator produces

Running `python run_generate.py` writes JSONL files under `output.out_dir`:

### Core agents
- `planner.jsonl` — **Planner agent**
  - Task: `problem -> plan (strict JSON)`
- `coder.jsonl` — **Coder agent (teacher)** - Task: `plan -> reference C++ code`
- `brute_coder.jsonl` — **Brute Coder agent (oracle)**
  - Task: `(plan + small constraints) -> brute-force C++ code`
  - Learns to write guaranteed-correct solutions for small inputs. Used as a local oracle for differential testing.
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
  "agent": "planner|coder|brute_coder|debugger|fixer|complexity",
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

* Build a full statement from structured fields: `title + description + input_format + output_format + note + examples`
* Extract a trusted reference C++ solution (`ref_cpp`) from dataset field `generation`.
* Compile `ref_cpp` to a binary `ref_bin` (acts as teacher label and verification).

### Step 1) Planner (Problem -> Plan)

* Prompted to output strict JSON (variables, bounds, algorithm steps, complexity, edge cases).
* Writes one `planner_sft` sample.

### Step 2) Coder (teacher) (Plan -> Reference Code)

* Writes one `coder_sft` sample with label = `ref_cpp`.

### Step 3) Brute Coder (Oracle Generation)

* Prompted to output a C++ solution that is guaranteed correct for small inputs (ignoring original full constraints).
* Compiles to `brute_bin`, which acts as the local oracle for differential testing.

### Step 4) Coder self-play: generate multiple init attempts

* `coder.num_init_attempts` (default 3) calls to generate `init_code`.
* Supports various `init_modes` (`llm_free`, `mutate_ref`, `llm_perturb`, `mixed`, or `tiered`) to inject bugs or sample diverse implementations.
* Each attempt is compiled. Compile errors are allowed and treated as a failure signal.

### Step 5) Debugger/TestGen for each init attempt

* For each init attempt:
* Ask the Debugger to produce 12–24 tests conditioned on that code.
* Pre-filter candidate inputs (skip overly large inputs) so the brute oracle stays fast.
* For each proposed test input:
* Run the brute-force oracle (`brute_bin`) to obtain `expected_output`.
* Discard tests where the oracle fails.


* Run the init attempt on the verified tests and collect failing cases: CE (compile error), RE, TLE, or output mismatch (WA).


* We then choose the init attempt with the largest number of failing cases (ties prefer compiled code), so downstream Fixer/Debugger data has stronger supervision.

### Step 6) Split Debugger datasets

From the chosen init attempt:

* **General** (`testgen_general_sft`): all verified tests (oracle-labeled).
* **Attack** (`debugger_attack_sft`): only tests that actually fail the chosen init attempt.
* If no failing tests are found, `debugger_attack.jsonl` will not get a sample for this problem.

### Step 7) Fixer + Iterative Coder datasets (MapCoder-Lite style)

When there is a real failure signal (non-empty `fail_report`):

* Build a concise `fail_report` from up to 8 failing cases, including: input / expected / actual / stderr / return code.
* Then emit:
* `coder_iterate_sft`: `(problem + plan + init_code + fail_report) -> reference code`
* `fixer_sft`: `(problem + plan + init_code + fail_report) -> reference code`



#### Self-play repair loop + traces

The generator also runs a self-play loop up to `fixer.max_iters`:

1. Ask Fixer model to repair.
2. Compile the repaired code.
3. Re-run on the failing subset (up to 8 cases).
4. If still failing, create a brief updated report and iterate again.

### Step 8) Complexity / Analyst (static signals)

* For each code source (reference and optionally init attempts):
* We compute `static_features`: heuristic signals (loop nesting, sorts, maps, recursion hints, etc.).
* *(Note: Deterministic baseline is currently disabled for dataset generation).*


* The Complexity model receives: `constraints + plan + static_features + code + baseline`
* And outputs a strict JSON verdict: `efficient`, `bottleneck`, `target_agent` (CODER/PLANNER), plus patch or replan advice.

---

## 3. Why this design helps multi-agent fine-tuning

* **Planner**: Learns structured reasoning and implementation-oriented planning.
* **Coder**: Learns to write correct solutions from plans (teacher forcing with reference code).
* **Brute Coder**: Learns to write simple, brute-force solutions strictly bounded by small constraints, ensuring a localized correct oracle for edge-case differential testing.
* **Debugger (attack-mode)**: Learns to generate counterexamples conditioned on a specific implementation. By using `brute_bin` as an oracle, it enables fully autonomous differential testing pipelines.
* **Fixer / Iterative Coder**: Learns to repair code given concrete evidence: failing inputs + expected outputs + actual outputs + stderr/RC. Step-wise samples train robustness across intermediate code states.
* **Complexity / Analyst**: Learns routing and efficiency judgment using plan complexity + code structure + constraints.

---

## 4. Configuration knobs

Edit `config.yaml`:

### Self-play diversity

* `coder.num_init_attempts` (default 3)
* `coder.init_mode` (`llm`, `mutate_ref`, `mixed`, `tiered`): controls how the initial faulty attempts are generated (via LLM perturbation or syntactic bug injection).

### Debugger volume

* `debugger.max_tests` (default 24)
* `debugger.cap_tests` / `debugger.target_mismatches`: speed knobs to reduce per-problem cost and stop executing tests once enough counterexamples are found.

### Brute Coder

* `brute_coder.small_constraints`: controls the maximum variables/bounds allowed for the differential testing oracle.

### Fixer loop

* `fixer.max_iters` (default 2)
* `fixer.require_failure` (default true)

### Complexity distribution

* `complexity.per_attempt`: (default False) whether to run the complexity agent over every init attempt.

---

## 5. Recommended training plan for OlympicCoder

A practical SFT curriculum:

1. **Planner SFT**: `planner.jsonl`
2. **Coder SFT (teacher)**: `coder.jsonl`
3. **Brute Coder SFT**: `brute_coder.jsonl` (optional, for self-contained testing setups)
4. **Fixer SFT**: start with `fixer.jsonl`, then add `fixer_steps.jsonl` for robustness
5. **Debugger SFT**:
* train `debugger_attack.jsonl` first (counterexample generation),
* optionally mix in `testgen_general.jsonl` at lower weight.


6. **Complexity/Router SFT**: `complexity.jsonl`

---

## 6. Notes on dataset quality & common failure reasons

The generator may return warning reasons (printed in run output), e.g.:

* `brute_compile_error`: the brute coder failed to produce a valid C++ oracle.
* `debugger_no_failing_tests`: the chosen init attempt did not fail on verified tests.
* `complexity_json_parse_fail`: Complexity model returned invalid JSON.

To improve attack coverage:

* Increase `coder.num_init_attempts` or use `mutate_ref` / `tiered` init modes.
* Increase `debugger.max_tests`.

---

## 7. File map

*(Note: File list updated to reflect all relevant files present in the current codebase context).*

* `run_generate.py` — main driver; writes JSONL outputs.
* `config.yaml` — configuration file for tuning generation knobs, API settings, and workflow behavior.
* `requirements.txt` — Python dependencies for running the framework.
* `src/builders.py` — implements per-row generation pipeline (including the new differential testing with `brute_bin`).
* `src/prompts_en.py` — system/user prompts for each agent (JSON-safe brace escaping).
* `src/sandbox.py` — compile/run utilities (timeouts, output caps).
* `src/utils.py` — prompt/build helpers + static feature extraction.
* `src/schemas.py` — message schemas.
* `src/bug_injector.py` — implements syntactic bug injection for generating flawed initial attempts (`mutate_ref` mode).
* `src/extract_cpp.py` — utility to extract pure C++ code blocks from LLM output or dataset fields.
* `src/hf_source.py` — handles loading, filtering, and parsing rows from the HuggingFace dataset.
* `src/llm_client.py` — client wrapper for making LLM API calls with retry logic and format handling.
* `src/profiler.py` — code profiling utilities (used for complexity evaluation or execution bounds).
* `src/input_scaler.py` — tools for scaling down generated inputs to keep brute-force evaluation fast.
* `src/input_gen_llm.py` — auxiliary LLM-based tools for testcase and input generation.

```
