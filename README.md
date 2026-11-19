# Feedback for Discovering TSP Heuristics

Hongye XU (72655146) · CS5491 Artificial Intelligence · Individual project

[Milestone report](report/milestone.pdf)

I study how feedback from a search state can help a language model discover
travelling salesman problem (TSP) heuristics. The model proposes an edge-priority
formula inside guided local search (GLS). Actual complete-tour quality selects
which formulas survive. A selected formula runs without a language model.

## 1. Research progress

The proposal asks whether learned action values can guide LLM heuristic discovery,
and whether adapting the generator makes this process more effective. I tested
both parts before examining the time span represented by the feedback.

**Learning action values.** I collected 1,024 training states from 256 TSP50/100
instances and 256 validation states from 64 other instances. Each label measures
one alternative edge penalty followed by conventional GLS. Training zero-label
rates were 97.61% and 96.41%; the two continuation seeds produced identical
outcomes. Across three optimization seeds, the graph Transformer selected actions
with a mean validation benefit of **0.003910 percentage points** over the standard
action. This was a small conditional-action gain, with checkpoints selected on
the same validation set.

**Using the values in discovery.** Four paired discovery seeds generated 24
outputs per arm. A received complete-tour feedback, B added the fixed three-model
Transformer ensemble, and C added measured single-action feedback. On 32 fresh
development instances with three solver seeds, the selected rules averaged
**2.654706%, 2.687714% and 2.687714%** gap for A/B/C. All 96 C prompts gave tied
parent/child values. Some alternative edges in the 32-state bank had nonzero
values, but the proposed rules did not select them. This narrowed the problem
from prediction accuracy to the information the generator actually received.

**Adapting the generator.** I built execution-checked improvements from an
arithmetic rule grammar, then trained a QLoRA adapter for two epochs. From that
shared SFT initialization, I compared terminal-reward and state-augmented GRPO
with three paired optimization seeds and 12 updates per run. All trained
generators were tested with three discovery seeds; all selected programs were
then evaluated on 64 development instances with five solver seeds:

| Generator and feedback | Mean gap (%) ↓ |
|---|---:|
| Original generator, endpoints | 2.333741 |
| SFT, endpoints | 2.202174 |
| SFT, endpoints + predicted state values | 2.209870 |
| Terminal-reward GRPO, endpoints | 2.202334 |
| State-augmented GRPO, endpoints + predicted state values | 2.202174 |

SFT improved the selected rules by **0.131567 gap points** (paired interval
[−0.210681, −0.055730]). The state-GRPO comparison changed the gap by −0.000160
points [−0.008923, +0.008655]. These intervals resample original instances,
conditional on all frozen programs. There was one SFT training run, shared across
the GRPO runs. Generator adaptation helped, while the added state signal still
offered little guidance.

These experiments lead to a more specific question: **does feedback for one
penalty decision describe a rule that will be applied repeatedly?** I compare
single decisions with sustained rule fragments at identical states and budgets,
then test their effect on discovery. The comparisons above and below each retain
their own instance pools, seeds and training settings.

The supporting records are in [value_learning](results/value_learning),
[abc](results/abc) and [adaptation](results/adaptation). Sections 3–4 give CPU
commands to recompute their results and rerun their frozen rules.

## 2. Rule fragments and program discovery

At an identical saved state and remaining work budget, I compare:

| Intervention | Candidate rule controls | Remaining continuation |
|---|---|---|
| Single penalty | One penalty decision | Standard GLS |
| Rule fragment | 25% of remaining work | Standard GLS |
| Persistent rule | All remaining work | No switch |

Tour quality uses continuous Euclidean distances. Persistent outcomes are a
diagnostic target and are never shown to the generator. The bank has **128
instances × three trajectories × four phases = 1,536 states**. Twelve predefined
rules produce 56,832 branches. For ten noncontrol rules, fragment feedback is
nonzero in **10.01%** of comparisons, versus **0.53%** for one penalty; correlation
with persistent outcomes rises from **0.063 to 0.285**.

The downstream comparison uses the same frozen NF4 Qwen2.5-Coder-7B-Instruct:

| Arm | Feedback | Mean development gap (%) ↓ |
|---|---|---:|
| A | Complete-tour outcomes | 2.411957 |
| P | A + measured parent/child single-penalty comparison | 2.464831 |
| F | A + measured parent/child rule-fragment comparison | 2.342285 |
| Standard GLS | Initial rule, `d/(1+p)` | 2.389059 |

Four paired discovery seeds generate 32 outputs per arm: **384 outputs, 361
valid**. Invalid and duplicate outputs consume budget. Candidate promotion uses
32 discovery instances; four elites per lineage are reranked on 32 selection
instances. All 12 selected programs freeze before evaluation on **64 independent
development instances × five solver seeds**. Bank, discovery, selection and
development instances do not overlap.

F−A is **−0.069672 gap percentage points**, with a paired 95% interval of
**[−0.173928, 0]**. Two discovery repetitions improve and two tie. This misses
the declared success criterion of at least 0.05 points of improvement with an
interval strictly below zero. Three F lineages retain standard GLS; one selects
`0.4*d/(1+p) + 0.6*d/(max_d/mean_d)`. The feedback is more informative, while its
downstream benefit remains inconclusive. These are milestone development
results; the final split remains reserved for the frozen final method.

The implementation is [state_feedback.py](scripts/state_feedback.py) and
[discover_rules.py](scripts/discover_rules.py), with fixed
[settings](configs/discovery.json) and [prompts](scripts/prompts.py).
An initial full-function response interface often omitted the function header.
Its complete batch and the subsequent expression-interface amendment are in
[results/interface_check](results/interface_check) and
[docs/interface_change.json](docs/interface_change.json). The batches are separate.

## 3. Environment and result analysis

The reported runs used the following rented machine and software.

| Component | Recorded setup |
|---|---|
| Host | Linux, Xeon Gold 6330 with 14 vCPUs, 60GB RAM |
| GPU | One RTX3090, 24GB; main discovery peak 8.26GiB Torch allocation, 9.38GiB reservation |
| Python / numeric packages | Python 3.10.8; pinned packages in `requirements-cpu.txt` |
| Model runtime | PyTorch 2.1.2+cu121, Transformers 4.45.2, bitsandbytes 0.44.1; `requirements-gpu.txt` |
| Parallelism | Ten CPU workers for branch measurement; four in the replay example below |

Analysis and solver replay need no GPU or model weights. The 256 input instances
are included. The commands below use a Linux shell and Python 3.10:

```bash
git clone --depth 1 --branch milestone-2026-11-19-v4 https://github.com/mant000/AI-Project.git
cd AI-Project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests
python scripts/analyze_progress.py --output outputs/progress
python scripts/analyze_results.py --output outputs/analysis
```

`outputs/progress/progress_summary.json` recomputes the action-label sparsity,
saved model decisions, ABC prompt preferences and development results, and the
SFT/GRPO comparisons. It checks 348 source records and reprices 2,112 unique
frozen-program routes. Expected: `status: passed`.

The second analysis recomputes the mechanism statistics, A/P/F means and paired intervals from
raw routes, and checks parent updates, selection, freezing and feedback arithmetic.
Expected: `status: passed` in `outputs/analysis/analysis_verification.json`.

| Output in `outputs/analysis/` | Contents |
|---|---|
| `mechanism_summary.json` | Nonzero rates, correlations and paired intervals |
| `study_summary.json` | Arm means, contrasts, validity and diversity |
| `dev_instance_means.csv` | Per-instance results for each frozen lineage |

The initial response-interface batch can be analyzed separately:

```bash
python scripts/analyze_results.py --study results/interface_check/run --output outputs/interface_analysis
```

## 4. Rerun the solver and branch measurements

Rerun the 2,112 frozen-program routes from the ABC and adaptation experiments:

```bash
python scripts/analyze_progress.py --replay --workers 4 --output outputs/progress_replay
```

Reconstruct the bank and rerun all unique selected programs, NN and 2-opt:

```bash
python scripts/replay_solvers.py --workers 4 --output outputs/replay
```

Expected: **1,536 exact states, 2,560 exact route replays and six exact feedback
branches**, in `outputs/replay/replay_verification.json`.

To repeat the complete mechanism experiment:

```bash
python scripts/build_state_bank.py --output outputs/mechanism/bank
python scripts/measure_feedback.py --root outputs --phase pilot --workers 10 --output outputs/mechanism/bank
python scripts/measure_feedback.py --root outputs --phase full --workers 10 --output outputs/mechanism/bank
python scripts/analyze_feedback.py --prepare outputs/mechanism/bank --output outputs/mechanism/analysis
```

The full run reuses the nested pilot: 1,536 states and 56,832 branches in total.
The input hash and rebuilt states are checked before measurement. Reference and
timing routes can also be repriced:

```bash
python scripts/check_results.py --analysis-proof outputs/analysis/analysis_verification.json \
  --replay-proof outputs/replay/replay_verification.json --output outputs/checks
```

On the rented Xeon host, analysis took 14 s, replay with four workers took 61 s,
and complete mechanism measurement with ten workers took 259 s. Worker counts
can be changed with `--workers`; use the same count for a pilot and its full run.
Timing varies with hardware. Use new output directories for repeated runs.
[results/reproduction](results/reproduction) contains actual verification records.
The ABC/adaptation analysis with all 2,112 solver replays took 38 s with four workers.

## 5. Run program discovery

The reported generator is
[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242),
revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, frozen with NF4 double
quantization and BF16 computation. No adapter is loaded. I allowed about 25GB
for the complete snapshot and cache. Model weights are downloaded from the
author release and checked against its published hashes.

After the mechanism run in Section 4:

```bash
python -m pip install -r requirements-gpu.txt
export TSP_ROOT="$PWD/outputs/runtime"
mkdir -p "$TSP_ROOT/artifacts" "$TSP_ROOT/tmp"
python scripts/download_model.py --root "$TSP_ROOT"
python scripts/discover_rules.py --root "$TSP_ROOT" --workers 10 \
  --prepare outputs/mechanism/bank --output outputs/discovery/run
python scripts/analyze_results.py --prepare outputs/mechanism/bank \
  --study outputs/discovery/run --output outputs/discovery/analysis
```

The downloader reuses cached files. An existing complete snapshot can instead be
registered with `python scripts/download_model.py --root "$TSP_ROOT" --snapshot
/path/to/snapshot`. It writes `TSP_ROOT/artifacts/model.json`. Discovery includes
candidate generation, selection, freezing and development evaluation. The
recorded 384-output run took about 19 minutes, including CPU feedback. Sampling
can differ across CUDA and package versions; original prompts, replies, token
IDs, seeds and program hashes are retained.

For another model, `download_model.py` accepts `--model REPOSITORY --revision
COMMIT`, or those options together with `--snapshot PATH` for local weights.
Pass its manifest to `discover_rules.py --model-manifest PATH`. The loader uses
Hugging Face `AutoModelForCausalLM` and the tokenizer's chat template, with
`trust_remote_code=False`. `--compute-dtype float16` selects FP16 instead of the
reported BF16; `--config PATH` supplies alternative experiment settings. Each run
saves the effective settings, model identity, manifest hash and executed source.
Different models or settings define new experiments rather than repetitions of
the reported comparison.

## 6. Development baselines

These commands reproduce the manual-rule, LKH and LEHD rows of Table 1, plus the
serial solver timing sample, using the included **development** instances.

```bash
python scripts/setup_references.py --root "$TSP_ROOT"
python scripts/evaluate_references.py --root "$TSP_ROOT" --output outputs/references
python scripts/benchmark_rules.py --root "$TSP_ROOT" --output outputs/timing
```

Setup downloads/verifies the pinned LEHD checkpoint and LKH source, and builds
LKH with GCC/make. LEHD runs on the GPU; LKH and the manual rules run on the CPU.
The serial timing sample uses 16 prespecified inputs and three repeats. The
recorded quality and timing runs took 540 s and 32 s. These solvers use different
budgets, so their quality is context rather than an equal-compute treatment.

Inputs come from the training release of
[Joshi et al.](https://github.com/chaitjo/learning-tsp/tree/bb28f5795924d99f7e7d945695e884f8d1f7df45).
Each released row retains its source location and hashes; the source license is
in `third_party/learning-tsp-LICENSE.txt`. Continuous tour lengths are repriced
locally, including negative gaps to the supplied, noncertified reference tours.
LEHD is pinned to
[CIAM-Group/NCO_code](https://github.com/CIAM-Group/NCO_code/tree/274df3c4975384592b60fe7f79fbb2441ce11c15/single_objective/LEHD);
LKH 3.0.6 to the research-use
[cerebis mirror](https://github.com/cerebis/LKH3/tree/9a51de3161f2746e969f7a2e33220f637da64e54),
with `RUNS=1`, `MAX_TRIALS=1000` and distance scale 1e6. Asset settings and hashes
are in `configs/reference_assets.json` and `configs/reference_asset_hashes.json`.

## Files

| Location | Purpose |
|---|---|
| `src/tsp/` | Data, resumable GLS/2-opt and arithmetic rule compiler |
| `scripts/state_feedback.py`, `scripts/feedback_worker.py` | Shared-state interventions |
| `scripts/discover_rules.py`, `scripts/prompts.py`, `scripts/qwen_model.py` | Frozen-model discovery |
| `scripts/analyze_progress.py` | Value-learning, ABC and adaptation result analysis and frozen-rule replay |
| `scripts/analyze_results.py`, `scripts/replay_solvers.py` | Fragment study analysis and solver replay |
| `configs/` | Experiment settings and author asset hashes |
| `results/mechanism/`, `results/discovery/` | Branch records, generations, candidates, failures, lineage and evaluations |
| `results/value_learning/`, `results/abc/`, `results/adaptation/` | Action labels/predictions, ABC prompts/outcomes and generator-training evaluation |
| `results/research_inputs.jsonl`, `results/research_records.json` | Earlier development inputs and record provenance |
| `results/interface_check/` | Initial response-interface experiment |
| `results/reproduction/` | Verification outputs and costs |
| `report/` | English milestone PDF, editable Word file and figures |

The research interpretation and the final-project plan, including CPU/GPU
allocation and continue/stop criteria, are in Sections 6–7 of the report.
