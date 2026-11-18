# Learning TSP Heuristic Programs from Search-State Values

CS5491, Group G042 — Hongye XU (72655146).

Can measured feedback about decisions at a shared guided local search state help
an LLM discover better TSP penalty rules? The current milestone studies the
**intervention granularity**: one penalty versus a bounded rule fragment.
The frozen generator proposes arithmetic programs; actual complete-tour quality
always determines parent promotion and program selection. Deployment needs no
LLM or value model.

The November 19 computational release includes the completed new study and all
historical research evidence. The grading release is tagged
`milestone-2026-11-19`; its report and publication verification are linked from
[SUBMISSION.md](SUBMISSION.md). Final test evaluation is not part of this milestone.

## What the new study establishes

The mechanism bank contains **128 originals × three trajectories × four phases
= 1,536 states**, with 12 fixed diagnostic rules and 56,832 unique branches.
On ten noncontrol rules, nonzero feedback increases from **0.53% to 10.01%**;
Spearman correlation to persistent conditional outcomes rises from **0.063 to
0.285**. All original-distance tour costs and standard/rescaling controls pass
independent checks. Persistent outcomes are diagnostic targets, not hints shown
to the generator or evidence of full-run generalization.

The primary utility comparison uses the original pinned NF4
Qwen2.5-Coder-7B-Instruct, four paired discovery seeds, three arms and 32 outputs
per lineage: **384 outputs, 361 valid**. A gets real endpoints; P adds measured
single-penalty feedback; F adds measured fragment feedback. All 12 programs are
frozen before **64 new original dev instances × five solver seeds**.

| Arm | Mean dev gap-to-reference (%) |
|---|---:|
| A: endpoints | 2.411957 |
| P: endpoints + single penalty | 2.464831 |
| F: endpoints + rule fragment | 2.342285 |
| Standard GLS | 2.389059 |

F−A is **−0.069672 gap percentage points**, joint paired 95% interval
**[−0.173928, 0]**: two discovery repetitions improve and two tie. It **does not
pass** the predeclared success gate (at least 0.05pp improvement and an upper
interval below zero). F−P is −0.122546pp [−0.268179, −0.020742]. Three F lineages
retain standard GLS; one finds a softer-penalty mixture. One F−A advantage avoids
an A dev regression. This is a useful preliminary signal, not proof of stable
feedback benefit or a Transformer success.

A preceding complete strict-function batch is preserved separately. Frequent
bare-return responses motivated one common expression-continuation amendment,
recorded before either new dev result was inspected. No favorable batch or seed
is selected, and the two batches are not pooled. The prior stopped 91-output
revision remains incomplete and separate.

## Start with the CPU reproduction

Linux, Python3.10; no Torch, full dataset or GPU needed:

```bash
git clone https://github.com/mant000/AI-Project.git
cd AI-Project
git checkout milestone-2026-11-19
python3.10 -m venv /tmp/tsp-milestone-cpu
source /tmp/tsp-milestone-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/milestone_analyze.py --output /tmp/tsp-audit
python scripts/milestone_replay.py --workers 10 --output /tmp/tsp-replay
```

Analysis independently checks raw routes, feedback arithmetic, lineage, selection
and freeze order, then recomputes every primary statistic. Solver replay
reconstructs all bank states and reruns each unique selected program on all dev
instances/seeds. Use fresh output paths. The exact commands, expected outputs,
measured timings and scope are in
[the reproduction guide](docs/MILESTONE_REPRODUCTION.md).

## Report, evidence and complete GPU workflow

- [English milestone PDF](reports/milestone-2026-11-19/milestone.pdf),
  [editable DOCX](reports/milestone-2026-11-19/milestone.docx),
  [Markdown source](reports/milestone-2026-11-19/milestone.md),
  [figures and provenance](reports/milestone-2026-11-19/figures).
- [Mechanism raw records](results/milestone-fragment-v1/prepare),
  [primary raw generations/evaluations](results/milestone-fragment-expression-v1/discovery),
  [independent analysis](results/milestone-fragment-expression-v1/analysis),
  [fixed strong references](results/milestone-fragment-expression-v1/references),
  [isolated service-time sample](results/milestone-fragment-expression-v1/timing).
- [Frozen primary protocol](configs/milestone_fragment_expression_v1.json),
  [mechanism/strict protocol](configs/milestone_fragment_v1.json),
  [response-interface amendment](docs/MILESTONE_EXPRESSION_INTERFACE_AMENDMENT.json),
  [reference/timing protocol](configs/milestone_reference_v1.json).
- [Data and model entry points](docs/DATA_AND_MODELS.md),
  [28 historical trained inference exports](trained_models/manifest.json),
  [cost and actual verification records](results/milestone-reproduction-v1).
- [Concrete route to final](docs/FINAL_ROADMAP_CN.md), with measured pilots,
  resource caps, prospective freeze and continue/stop conditions.

The full GPU workflow in the reproduction guide rebuilds the bank from the
released **256-row public subset**, measures the same interventions and runs the
frozen model. It reuses a verified snapshot when present; no historical Stage2,
SFT or GRPO rerun is required. Original hardware: rented RTX3090 24GB, 14-vCPU
Xeon6330 quota, 60GB RAM. Core dependencies are pinned separately for CPU,
Torch/CUDA, NF4/PEFT and report generation. Failed outputs, duplicate budgets,
raw prompts/replies/token IDs, program hashes, ancestry and actual costs remain
available; concurrent subprocess CPU deltas are not added as if disjoint.

## Earlier work and provenance

[Historical results index](docs/HISTORICAL_RESULTS_INDEX.md) preserves Stage2
value learning, the original frozen-LLM controls, dense teachers, grammar-derived
SFT, six GRPO runs, crossed controls and solver references. Those results use
separate protocols and data roles; they are not extra repetitions of the new
study. The [dated earlier README](docs/HISTORICAL_README_2026-11-18.md) is retained
for original commands and context, not as the current execution status.

Source data and external solvers are attributed in
[DATA_AND_MODELS](docs/DATA_AND_MODELS.md) and `third_party/`. Full third-party
Qwen/LEHD weights are downloaded from pinned author sources rather than
rehosted. Small own-trained inference weights retain original provenance; full
optimizer/RNG backups are distinct from these exports. Unexecuted local final
scheduling drafts are excluded from this grading release.