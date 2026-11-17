# Stage 2 entry: training-only label pilot

Stage 1 is complete: 6,186 manifest identifiers, no project/final coordinate
overlap, ten correctness checks, 288 valid baseline runs and exact anonymous
fresh-source replay. See `results/stage1/` and `docs/MILESTONE_DRAFT.md`.
Final data remains identity-audited only. No labels or model training exist yet.

Use the installed environment (`source scripts/env.sh`). Project-only normalized
instances are `$TSP_ROOT/data/prepared/stage1/project_instances.jsonl`, SHA256
`43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f`.
Their 4,992 records inherit the fixed original author-train row ranges in
`configs/data_split_policy.json`. Validation is project heldout from author
train, never author val. Preserve TSP50 rows 1–512 in value train.

Implement label production using `create_state`, `run(stop_at_decision=True)`,
`save_state` and `paired_action_set`. The existing real-data decision/partial
snapshots are under `$TSP_ROOT/checkpoints/stage1/`. `paired_action_set` executes
one reference branch per state/seed and reuses it for all candidate labels;
the exact branch count is (K+1) times the seed count. Each branch consumes
exactly the requested additional number of delta evaluations.

The planned first pilot uses 64 original value-train instances, balanced by
size, four decision states per instance, eight distinct nonreference current
edges and two common seeds: 4,608 total branches. The original plan suggests
eight CPU workers with one BLAS thread each; measure throughput and RAM before
expansion. Do not count states as independent original problem instances.

Before formal production, diagnose train-only H/state sampling, tie rates,
seed variability and action benefit. The two stage1 different-action probes
tied at H=50,000. They are engineering checks, not a 64-instance pilot, and
do not establish useful labels. Do not silently adopt H=50,000 as final.
The deterministic best-improvement neighborhood may yield limited variation
across common seeds when utility ties are absent; measure this rather than
claiming independent stochastic trajectories.

Freeze pilot-informed H and sampling before touching value validation. Preserve
all failed/zero labels. Then run the planned small train/validation sets and
actual Transformer/MLP training with ranking, regret, realized branch benefit,
loss curves, checkpoint restoration and GPU memory/time measurements.
Discovery, program selection, program development evaluation and every final
source stay outside label training and diagnostic tuning. Do not reinstall,
redownload or reload the 7B model for this CPU label pilot.
