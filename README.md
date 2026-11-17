# Learning TSP Heuristic Programs from Search-State Values

Can feedback about decisions at a shared guided local search state help a
frozen code LLM discover better TSP penalty rules?

**Stage 1 is implemented and measured.** This repository provides fixed data
splits, identity/overlap auditing, nearest neighbor, best-improvement 2-opt,
single-edge conventional GLS, complete JSON checkpoints and budget-matched
paired interventions. Transformer training and frozen-7B A/B discovery have
not run. SFT/GRPO are later extensions. Working report text is in
[docs/MILESTONE_DRAFT.md](docs/MILESTONE_DRAFT.md).

## Actual preliminary results

32 project value-validation instances (16 per size), three common solver
seeds, 288 runs, zero errors. Validation is held out from **author train**;
there is no independent author validation file in the received data.

| Size | NN gap-to-reference (%) | 2-opt (%) | GLS (%) |
| --- | ---: | ---: | ---: |
| TSP50 | 22.7997 | 3.2449 | 1.8333 |
| TSP100 | 27.4375 | 4.0042 | 2.8971 |

These baselines do not test the proposed learned-feedback method. Gaps retain
negative values: two GLS runs on one TSP50 instance improve the supplied
reference by 0.0513%. References are not claimed to be certified continuous-
Euclidean optima. Raw logs, summaries and costs are in
[results/stage1](results/stage1); failed attempts are retained in
[results/stage1_attempts](results/stage1_attempts). The predeclared configuration
is [configs/stage1_baseline.json](configs/stage1_baseline.json).

## Environment and data

Measured hardware: RTX 3090 24GiB, Xeon Gold 6330, 14-vCPU cgroup quota,
60GiB RAM; Python 3.10.8, PyTorch 2.1.2+cu121, NumPy 1.26.4,
NVIDIA driver 580.105.08, Ubuntu 22.04. CPU affinity lists more CPUs than
the container quota. Stage 1 uses one CPU process and one BLAS thread, with
zero GPU compute. No new package is needed for stage 1.

On a fresh machine:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch.txt -r requirements.txt
pip check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

On the already-installed AutoDL environment, do not reinstall:

```bash
cd /root/autodl-tmp/tsp-state-value/code
source scripts/env.sh
```

`scripts/env.sh` selects the project venv and puts caches on the data disk;
source it in noninteractive SSH sessions too. Stage 0 package inventories
are in `environment/`. The prior 7B BF16 smoke is infrastructure evidence,
not a program-discovery result. Stage 1 does not load Qwen.

`configs/sources.json` fixes the upstream repositories/revisions and author
URLs for Joshi et al. CP 2021, INViT ICML 2024 and TSPLIB95. Qwen2.5-Coder-7B-
Instruct is fixed at `c03e6d358207e414f1eca0bb1891e29f1db0e242`; its four
author weight hashes were verified. Large data/weights are excluded from Git.

The received Joshi data is a seven-volume 7z repack (5,739,945,309 bytes,
SHA256 `ea6ed955a48028627653c6674fb5038b4073db64e7a2805ff1c88ab6399a64d3`).
Its 11 member hashes/counts are in `configs/joshi_received_files.json`.
This is not a verified hash of the original author gzip. Member integrity,
independent author-prefix agreement and bootstrap reservations passed.
The obsolete interrupted Google download is not needed or restarted.

Provide these verified sources under `$TSP_ROOT`:

```text
data/author_received/data/tsp/tsp{50,100}_train_concorde.txt
data/author_received/data/tsp/tsp{50,100,200}_test_concorde.txt
data/raw/INViT_data.zip
data/raw/ALL_tsp.tar.gz
```

An original author gzip may be extracted under `data/author_received`; member
hashes must match. Generic retrieval tools remain in `scripts/download_sources.py`
and `scripts/download_drive_ranges.py` for machines that reach the author
hosts. Stage 1 launches no download. The repack is not hosted here; author
URLs are the public retrieval route and may be subject to Drive quota.

## Instance isolation and final lock

`configs/data_split_policy.json` fixes original author-train rows per size:
value train 1–2048, discovery 2049–2112, value validation 100001–100256,
program selection 100257–100320, and program development evaluation
100321–100384. These are disjoint project validation pools held out from
author train, never “author val”. TSP50 rows 1–512 stay in value train.
Derived states inherit the original instance's pool.

`splits/manifest.csv` fixes 6,186 selected instances: 4,992 project instances
and 1,194 final identifiers. Fields include source URL/revision/file hash,
original row, project use, coordinate hashes, reference locator and distance
convention. TSPLIB file hashes cover decompressed `.tsp` content; the archive
hash is separate. Auditing checks all 3,840 author test rows, 6,000 INViT
rows in selected distributions and 42 TSPLIB instances against project pools.
Exact and float32-quantized coordinate overlaps are zero; this does not
exclude every possible geometric equivalence.

`configs/final_identity_policy.json` fixes first 256 author-test rows per
size (50/100/200), first 128 INViT rows per distribution
(clustered1/explosion/implosion), and 42 audited unconstrained TSPLIB EUC_2D
instances of size 50–500. Final coordinates were read for identity only;
no final solver run or reference-tour cost was computed. Final performance
is unavailable for method selection. Joshi/INViT use continuous Euclidean
distances; TSPLIB will use rounded EUC_2D at final evaluation.

## Reproduce stage 1

Build the manifest and normalized **project-only** data:

```bash
python scripts/build_manifest.py --root "$TSP_ROOT" \
  --tsplib-archive "$TSP_ROOT/data/raw/ALL_tsp.tar.gz"
```

Expected: 6,186 manifest records, zero overlap, and 4,992 normalized project
instances in `$TSP_ROOT/data/prepared/stage1/project_instances.jsonl`.
Audit time: 20.99 wall seconds / 20.98 CPU seconds on this server. It
validates selected source-file hashes and 512 bootstrap reservations;
it computes no final reference-route length or solver result.

```bash
python -m pytest tests/test_solver.py -q
python scripts/evaluate_baselines.py --root "$TSP_ROOT" \
  --output "$TSP_ROOT/artifacts/stage1-replay"
python scripts/compare_baselines.py results/stage1/baseline_runs.jsonl \
  "$TSP_ROOT/artifacts/stage1-replay/baseline_runs.jsonl" \
  --output "$TSP_ROOT/artifacts/stage1-replay/comparison.json"
```

Expected: ten checks pass; 288 rows with zero errors; identical route, quality,
work and progress fields on replay. `--output` must be new; evaluation
refuses to overwrite logs. Original baseline time was 7.39 seconds including
independent scalar cost checks and JSON logging. Wall times vary and are not
compared for equality. Same-action diagnostic benefits are zero at H=50,000.
The two different-action engineering probes also tie; this is not a label
pilot or evidence of learnability.

An anonymous fresh HTTPS clone of commit `e21a399` replayed all 288 runs
with zero differences in route, quality and work/progress fields. This used
the existing verified environment and inputs, not a newly installed
environment. See `results/stage1/public_replay_verification.json`. Including
failed attempts and replay, measured child commands used 50.48 wall seconds /
50.45 CPU seconds with zero GPU compute; the 20.99-second manifest audit is
included once. Transfers and implementation time are excluded. Full accounting
is in `results/stage1/resource_accounting.json`.

Operational pipeline on the prepared server (before baseline logs exist):

```bash
bash scripts/stage1_preflight.sh full
bash scripts/start_job.sh UNIQUE_SESSION python scripts/measure_command.py \
  --output results/stage1/resources.json -- bash scripts/run_stage1.sh
```

`run_stage1.sh --reuse-manifest` checks existing manifest/normalized-data
SHA256 values instead of rebuilding them. It still refuses to replace
baseline logs. Successful tests/evaluation took 9.51 wall / 9.51 child CPU
seconds, peaking at 139.1MiB child RSS; this includes data loading, tests,
baselines and real-data checkpoint/pair diagnostics. Failed attempts and
their resource costs are preserved; see the draft for accounting.

## Shared solver and paired API

`src/tsp_value/solver.py` defines the interface. A work unit is one nonadjacent
2-opt pair delta evaluation. Each complete sweep has n(n-3)/2 units. Fixed
TSP50/100 caps are 235,000 / 970,000 units (200 sweeps including initial
descent). NN has no delta search; 2-opt stops on convergence. Budget caps
are shared, consumed work differs.

GLS freezes lambda at 0.1 times the first unpenalized local optimum's mean
edge length. It penalizes one maximum distance/(1+penalty) edge, uniformly
resolving exact utility ties. Returned quality uses original distances.

`create_state`, `run`, `save_state`, `load_state` and `clone` preserve full
tour/penalty/progress/RNG state, distance/index caches and the best move in an
unfinished sweep. `run(..., stop_at_decision=True)` stops at a penalty
boundary. `paired_action_set` copies it for all actions using common seeds
and exact additional work, executing each reference once per seed.
`paired_continuation` is the single-candidate wrapper. Benefits are empirical
finite-horizon effects under a fixed continuation, not optimal Q values.

Checks cover valid tours, raw/augmented delta identities, exact small-instance
reference parsing, arbitrary mid-sweep checkpoint replay and seeded ties,
copy isolation, equal-work branches, same-action zero benefit and reference
reuse. Real TSP50/100 training-instance replay records are in
`results/stage1/real_data_correctness.json`.

## Backup and next stage

`scripts/start_job.sh` runs tmux jobs and retains logs/exit codes.
`scripts/backup_remote.sh` verifies the actual file-storage mount, archives
code/results/logs/checkpoints and prepared data, and writes a SHA256 sidecar.
Keep an independently verified local copy. Raw archives/weights are excluded.

Next: training-only label pilot (64 instances, four states, eight candidate
actions, two paired seeds), diagnose H/ties/noise, then freeze labels and
train the Transformer and MLP. Discovery, program selection and program
development evaluation remain untouched by stage 1 baselines. A milestone
submission tag awaits later stages and the formatted report. Only this
implementation directory is public; credentials/private course attachments
remain outside it.
