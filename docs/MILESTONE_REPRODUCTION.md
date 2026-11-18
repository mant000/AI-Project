# Reproduce the milestone

Run commands from the repository root on Linux, Python **3.10**. All commands use
new output directories and leave published evidence untouched. The public input
subset is included; no full dataset, Torch or GPU is needed for tiers 1–2.
Final/test data are never read by these commands. A command failing its assertions
is a failed reproduction; do not substitute the expected summary by hand.

## Tier 1: raw-record audit and report numbers (CPU)

```bash
git clone https://github.com/mant000/AI-Project.git
cd AI-Project
git checkout milestone-2026-11-19
python3.10 -m venv /tmp/tsp-milestone-cpu
source /tmp/tsp-milestone-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/milestone_analyze.py --output /tmp/tsp-audit-expression
python scripts/milestone_analyze.py \
  --study results/milestone-fragment-v1/discovery \
  --output /tmp/tsp-audit-strict
```

The script independently reprices every endpoint and all 56,832 mechanism
branches, verifies source/program hashes, parent updates, top-four selection,
freeze order, paired seeds and all available feedback prompts against saved
rollouts. It computes mechanism metrics and original-instance bootstraps, then
the crossed paired discovery-seed/instance intervals. Expected output:
`study_summary.json`, `mechanism_summary.json`, `dev_instance_means.csv`,
`analysis_verification.json` with `status=passed`. About **10–60 s** per study
on the rented server; actual measured times are in the release verification.
This analyzes records; it does not regenerate routes or LLM outputs.

## Tier 2: reconstruct states and replay selected solvers (CPU)

```bash
python -m unittest discover -s tests -p test_fragment_core.py
python scripts/milestone_replay.py --workers 10 --output /tmp/tsp-solver-replay
```

Expected: all 1,536 state objects match exactly, then all unique selected dev
programs (including standard) plus NN/2-opt replay on the **64 originals × five
solver seeds**, matching tours, lengths and delta counts. Identical programs
across discovery lineages are physically replayed once. Expected output:
`replay_verification.json`, `solver_replay_records.json` and rebuilt snapshots.
Typically tens of seconds to a few minutes, not days. Timing values are not
expected to match across machines. This tier validates deterministic numerical
results; it does not reproduce stochastic GPU generation or retrain old models.

## Tier 3: full new mechanism and GPU discovery

Use an RTX3090-class **24GB CUDA GPU**, 14 CPU threads and roughly 25GB free for
the complete pinned 7B snapshot/cache. Reuse existing compatible packages and
weights when available. For a genuinely new environment, install Python3.10,
`requirements.txt`, `requirements-torch.txt` and `requirements-training.txt`.
The latter supplies PEFT/bitsandbytes required by the shared NF4 loader even
though this experiment trains no parameters. Original versions are recorded in
`environment/remote.freeze.txt` and the new execution artifacts.

```bash
export TSP_ROOT=/absolute/new/run-root
mkdir -p "$TSP_ROOT"/artifacts "$TSP_ROOT"/tmp "$TSP_ROOT"/cache/matplotlib
# Only if the fixed complete base snapshot/manifest does not already exist:
python scripts/download_sources.py qwen --root "$TSP_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$TSP_ROOT/cache/matplotlib"
OUT="$TSP_ROOT/fragment-reproduction"
python scripts/milestone_prepare_portable.py --output "$OUT/prepare"
python scripts/fragment_prepare.py --root "$TSP_ROOT" --phase pilot --output "$OUT/prepare"
python scripts/fragment_prepare.py --root "$TSP_ROOT" --phase full --output "$OUT/prepare"
python scripts/fragment_mechanism_analyze.py --prepare "$OUT/prepare" --output "$OUT/mechanism"
python scripts/fragment_discover_expression.py --root "$TSP_ROOT" \
  --prepare "$OUT/prepare" --output "$OUT/discovery"
python scripts/milestone_analyze.py --prepare "$OUT/prepare" \
  --study "$OUT/discovery" --output "$OUT/analysis"
```

The portable preparation checks the released 256-row input hash and reconstructs
the exact state objects. The original prepare script's full-source allocation
step is bypassed only because this verified bank already exists. Pilot is nested
in full (192 reused states); 56,832 physical branches are not added twice.
Mechanism measurement took about 4.5 min; discovery is bounded at two hours and
uses four seeds × three arms × 32 outputs, not repeated retries until success.
GPU sampling can differ across CUDA/hardware stacks; complete prompts, seed
schedule, rendered input/output tokens and selected-program hashes are public.
The deterministic CPU replay is the exact numerical tier actually verified.

The strict-function precursor can be regenerated separately by replacing the
driver with `fragment_discover.py` and using another output directory. Do not
pool those results with the expression-continuation experiment. The engineering
amendment is `docs/MILESTONE_EXPRESSION_INTERFACE_AMENDMENT.json`; no parameter
training occurs in either experiment. Historical SFT/GRPO commands and their
original protocols are indexed by `docs/HISTORICAL_RESULTS_INDEX.md`.

## Fixed reference solvers and service-time sample

```bash
python scripts/milestone_external_setup.py --root "$TSP_ROOT"
python scripts/milestone_references.py --root "$TSP_ROOT" \
  --output "$OUT/references"
python scripts/milestone_timing.py --root "$TSP_ROOT" --output "$OUT/timing"
```

This downloads only pinned external source/weights if missing, checks original
hashes, and builds LKH with GCC/make; existing verified assets are reused. Source
conditions and quality conventions are in `docs/DATA_AND_MODELS.md`. The fixed
64 dev inputs are used by default; full regeneration can supply `--prepare`.
Parallel quality-run wall times must not be presented as isolated service times.
The timing sample has its own fixed protocol and raw measurement records.

## Figures and editable report

```bash
python scripts/milestone_figures.py --output /tmp/tsp-figures
# Add --font-directory /path/to/TimesNewRoman to reproduce submission styling.
python -m pip install -r requirements-report.txt
python scripts/build_milestone_report.py --font-directory /path/to/TimesNewRoman
```

Figures use verified analysis records only. Default CPU plots use an available
serif font; the submitted plots/PDF use genuine Times New Roman 12pt. Font files
are not redistributed. The report builder produces PDF, editable DOCX, Markdown,
canonical content JSON, five rendered previews and an automated format proof.
The authoritative PDF has four A4 body pages plus one reference page, with 2.5cm
margins. Rendering/editing DOCX in a different office suite can change pagination.

## Report-to-evidence map

| Report item | Raw input | Executable check/output |
|---|---|---|
| Figure 1 and mechanism paragraph | `results/milestone-fragment-v1/prepare/measurements/*.json`, state bank | `milestone_analyze.py` → `mechanism_summary.json`; `fragment_mechanism_analyze.py` adds 12 new branch replays |
| Table 1 A/P/F, Figure 2, intervals | expression `frozen_programs.json`, `dev_evaluation.json`, `evaluations/` | `milestone_analyze.py` → `study_summary.json`, `dev_instance_means.csv` |
| Table 1 manual/LKH/LEHD | expression `references/*.json` | `milestone_references.py`; raw tours repriced by release audit |
| Validity/diversity/feedback counts | both `generations/`, `candidates/`, `feedback/`, `rollouts/` | `milestone_analyze.py` → `lineages` |
| Tokens, CPU/GPU/wall/VRAM | `attempt_costs.jsonl`, all raw generations, `gpu_samples.jsonl`, nested pilot/full costs | release cost ledger; overlapping subprocess deltas are never summed as total CPU |
| Reproduction statement | new-environment/clone metadata and actual solver records | `results/milestone-reproduction-v1/` |
| Format and PDF contents | canonical `report_content.json`, actual figures | `build_milestone_report.py`, `format_verification.json`, rendered previews |

Actual verification scope and exact source commit are recorded alongside the
release proof. This repository does not claim a second physical machine test or
a fresh 7B training run. Private SSH keys and original optimizer backups are not
required for the public CPU tiers.
