# Feedback for Discovering TSP Heuristics

Hongye XU (72655146) · CS5491 Artificial Intelligence · Individual project

[Milestone report](report/milestone.pdf) · Version: **`milestone-2026-11-19-v2`**

I study whether feedback from an intermediate search state helps a frozen code
language model discover better travelling salesman problem (TSP) heuristics.
The model changes an edge-priority formula inside guided local search (GLS).
Complete-tour quality determines which formulas survive. The selected formula
runs on its own, without a language model or learned critic.

## 1. From an edge action to a rule fragment

The original idea was to learn the value of a single edge-penalty action and use
that prediction to guide program discovery. About 97% of the initial training
labels were zero, and the Transformer gave little validation benefit. That led
to the question studied here: **is one action the right unit of feedback for a
rule that is applied repeatedly?** The initial value-model experiments and
subsequent training trials are preserved in the
[original value-model results](https://github.com/mant000/AI-Project/blob/milestone-2026-11-19/docs/STAGE2_RESULTS.md)
and the [complete research snapshot](https://github.com/mant000/AI-Project/tree/milestone-2026-11-19).
They motivated this experiment; they are not additional repetitions of it.

At the same saved search state, I compare three interventions:

| Intervention | Candidate rule controls | Rest of the search |
|---|---|---|
| Single penalty | One penalty decision | Standard GLS |
| Rule fragment | 25% of remaining work | Standard GLS |
| Persistent rule | All remaining work | No switch |

Every branch has the same remaining total work budget and is scored by actual
continuous Euclidean tour length. Persistent outcomes are a diagnostic target;
they are never shown to the generator.

The bank has **128 instances × three trajectories × four phases = 1,536 states**.
Twelve predefined rules produce 56,832 unique branches. For ten noncontrol
rules, fragment feedback is nonzero in **10.01%** of comparisons, versus
**0.53%** for one penalty. Correlation with persistent outcomes rises from
**0.063 to 0.285**. This supports a longer intervention for describing a rule
change. It does not yet establish better program discovery.

Implementation: [state_feedback.py](scripts/state_feedback.py).
Inputs, snapshots and branch tours: [results/mechanism/bank](results/mechanism/bank).
Statistics: [results/mechanism/analysis](results/mechanism/analysis).

## 2. Does that feedback help discover better rules?

I compare the same frozen **Qwen2.5-Coder-7B-Instruct**, loaded in NF4, under
three feedback conditions. No parameters are trained in this comparison.

| Arm | Information supplied to the generator | Mean development gap (%) ↓ |
|---|---|---:|
| A | Complete-tour outcomes | 2.411957 |
| P | A + measured parent/child single-penalty comparison | 2.464831 |
| F | A + measured parent/child rule-fragment comparison | 2.342285 |
| Standard GLS | Initial rule, `d/(1+p)` | 2.389059 |

Each arm has four paired discovery seeds and 32 outputs per seed. Invalid and
duplicate outputs consume budget: **384 outputs, 361 valid**. Candidate promotion
uses 32 discovery instances. Four elites per lineage are reranked on 32 separate
selection instances. All 12 selected programs freeze before evaluation on
**64 new development instances × five solver seeds**. Bank, discovery, selection
and development instances do not overlap. Final data have not been evaluated.

F−A is **−0.069672 gap percentage points**, paired 95% interval **[−0.173928, 0]**.
Two discovery repetitions improve and two tie. This misses the declared success
criterion: at least 0.05 points of improvement and an interval strictly below
zero. Three F lineages retain standard GLS; one selects
`0.4*d/(1+p) + 0.6*d*mean_d/max_d`. Manual rules and external solvers remain
stronger. The evidence supports a more informative intervention, with an
inconclusive downstream benefit.

Implementation: [discover_rules.py](scripts/discover_rules.py), fixed
[protocol](configs/discovery.json) and [prompts](scripts/prompts.py).
[results/discovery/run](results/discovery/run) holds every prompt, response,
candidate, failure, parent link, program hash, feedback branch and final choice.
[results/discovery/analysis](results/discovery/analysis) holds the report statistics.

An initial complete-function interface often produced bare returns. I changed
the common interface to expression continuation before inspecting development
outcomes. That complete initial batch is in
[results/interface_check](results/interface_check), with the dated
[interface change](docs/interface_change.json). The batches are not pooled.

## 3. Recompute the reported numbers on a CPU

Use **Linux and Python 3.10**. No GPU, Torch, model weights or additional dataset
is needed. The 256 inputs are included. Run from the repository root and use
new output directories.

```bash
git clone --depth 1 --branch milestone-2026-11-19-v2 https://github.com/mant000/AI-Project.git
cd AI-Project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests
python scripts/analyze_results.py --output outputs/analysis
```

This recomputes Figure 1, the A/P/F rows of Table 1, Figure 2 and their intervals
from raw routes. It checks parent promotion, selection, freeze order and feedback
arithmetic. It reads the saved experiment; it does not generate new routes.
Expect `status: passed` in `outputs/analysis/analysis_verification.json`.

| File in `outputs/analysis/` | What to inspect |
|---|---|
| `mechanism_summary.json` | Nonzero rates, correlations and paired intervals |
| `study_summary.json` | Arm means, contrasts, validity and diversity |
| `dev_instance_means.csv` | Per-instance results for each frozen lineage |

To check the initial response interface as well:

```bash
python scripts/analyze_results.py --study results/interface_check/run --output outputs/interface_analysis
```

## 4. Rerun the numerical experiments on a CPU

Reconstruct every bank state and rerun the selected programs, NN and 2-opt:

```bash
python scripts/replay_solvers.py --workers 4 --output outputs/replay
```

Expected: **1,536 exact states, 2,560 exact route replays and six exact feedback
branches**, in `outputs/replay/replay_verification.json`. Identical selected
programs are solved once. This tests the actual solver, rather than saved
summary arithmetic. Use fewer workers if needed.

To repeat the entire 56,832-branch mechanism experiment:

```bash
python scripts/build_state_bank.py --output outputs/mechanism/bank
python scripts/measure_feedback.py --root outputs --phase pilot --output outputs/mechanism/bank
python scripts/measure_feedback.py --root outputs --phase full --output outputs/mechanism/bank
python scripts/analyze_feedback.py --prepare outputs/mechanism/bank --output outputs/mechanism/analysis
```

The full run reuses the nested pilot. Expected: 1,536 measured states, 56,832
branches in total, exact standard/rescaling controls and the rates in Section 1.
The input hash and rebuilt states are checked before measurement.

Reprice the reference/timing routes and assemble the cost ledger:

```bash
python scripts/check_results.py --analysis-proof outputs/analysis/analysis_verification.json \
  --replay-proof outputs/replay/replay_verification.json --output outputs/checks
```

This checks **2,240 reference routes and 576 timing routes**. Actual verification
records are in [results/reproduction](results/reproduction). On the rented
14-vCPU Xeon 6330, the revised commands took 14 s for analysis, 61 s for replay
with four workers, and 259 s for complete mechanism measurement with ten workers. Fewer workers or a slower CPU will
take longer. Elapsed times need not match across machines.

## 5. Regenerate candidates with the frozen 7B model

This optional step repeats stochastic discovery. Use an RTX3090-class 24GB CUDA
GPU, Python 3.10, ten CPU workers and about 25GB for the pinned model/cache.
Install `requirements-gpu.txt` in a **separate environment**; keep the CPU
verification environment unchanged. Reuse compatible packages and a complete
snapshot when available.

```bash
python -m pip install -r requirements-gpu.txt
export TSP_ROOT=/absolute/path/to/run-root
mkdir -p "$TSP_ROOT/artifacts" "$TSP_ROOT/tmp"
# Skip this download if the manifest already points to the verified snapshot.
python scripts/download_model.py --root "$TSP_ROOT"
python scripts/discover_rules.py --root "$TSP_ROOT" \
  --prepare outputs/mechanism/bank --output outputs/discovery/run
python scripts/analyze_results.py --prepare outputs/mechanism/bank \
  --study outputs/discovery/run --output outputs/discovery/analysis
```

Complete Section 4's mechanism run first. Discovery includes selection, freezing
and development evaluation, with the report's four seeds and output budgets.
The recorded run took about 19 minutes, including CPU feedback; its wall-time
cap is two hours. CUDA sampling can vary across systems. Prompts, token IDs,
seeds and executed-source snapshots document the original run.

For the initial interface, use `discover_full_function.py` with another output
directory. Keep its results separate. Pinned model, data and reference-solver
sources are in [docs/data_sources.md](docs/data_sources.md).

## 6. References, timing and report

With the GPU environment, GCC/make and the same `TSP_ROOT`:

```bash
python scripts/setup_references.py --root "$TSP_ROOT"
python scripts/evaluate_references.py --root "$TSP_ROOT" --output outputs/references
python scripts/benchmark_rules.py --root "$TSP_ROOT" --output outputs/timing
```

These verify/download pinned LEHD weights and LKH source, build LKH, repeat the
quality references on the included 64 development inputs, and run a serial
timing sample on 16 prespecified inputs with three repeats. LKH and LEHD use
their own budgets; this is not an equal-compute comparison. Recorded reference
and timing runs took 540 s and 32 s. Final data are not evaluated.

To redraw the figures with an available serif font:

```bash
python scripts/plot_results.py --output outputs/figures
```

To rebuild the course PDF and editable Word file with the required font:

```bash
python -m pip install -r requirements-report.txt
python scripts/plot_results.py --font-directory /path/to/TimesNewRoman --output report/figures
python scripts/build_report.py --font-directory /path/to/TimesNewRoman
```

Text source: [build_report.py](scripts/build_report.py). The PDF, editable Word
file and figures are in [report/](report/). Building the course PDF needs genuine
Times New Roman fonts, which are not redistributed. It has four A4 body pages
plus references, 12pt text and 2.5cm margins.

## Files

| Location | Purpose |
|---|---|
| `src/tsp/` | Data, resumable GLS/2-opt and safe arithmetic rule compiler |
| `scripts/state_feedback.py`, `scripts/feedback_worker.py` | Shared-state interventions |
| `scripts/discover_rules.py`, `scripts/prompts.py`, `scripts/qwen_model.py` | Frozen-model discovery |
| `scripts/analyze_results.py`, `scripts/replay_solvers.py` | CPU analysis and reproduction |
| `configs/` | Fixed settings and author asset hashes |
| `results/mechanism/`, `results/discovery/` | The two research comparisons |
| `results/interface_check/` | Initial response-interface batch |
| `results/reproduction/` | Actual verification outputs and costs |
| `report/` | English PDF, editable source and figures |

Raw records retain the original paths and executed-source hashes: those identify
what ran, rather than the current layout. This checkout contains the code and
evidence for the report. The next experiment, budget and decision points are in
[docs/next_steps.md](docs/next_steps.md).
