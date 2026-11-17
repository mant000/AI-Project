# Stage 3 entry: frozen 7B discovery A/B

Stage 2 is complete. Read `docs/STAGE2_RESULTS.md`,
`results/stage2/stage2_summary.json`, and `resource_accounting.json` first.
Milestone remains 19 November 2026 23:59 BJT; calculate remaining time again.

The 64-original-instance pilot executed exactly 4,608 branches. The 256 train
+64 heldout validation instances contain 23,040 production branches in total,
including that nested pilot. H is 58,750 deltas for TSP50 and 970,000 for
TSP100. Sampling is fixed at penalty events 0/8/32/96, uniform eight
nonreference current edges, common seeds 71/72. Roughly 97% of labels are
zero and the two continuation seeds are completely redundant in observed
labels/terminal signatures. Do not call them independent random repeats.

Three optimization seeds each trained a real four-layer 128-hidden graph
Transformer and an MLP. Mean validation benefit is +0.003910 pp for the
Transformer and -0.000423 pp for MLP; conventional reference benefit is zero.
Only 39/256 validation states have non-tied outcomes. Some selected-checkpoint
gains come from one or two original instances, and the same validation set
selects checkpoints. Treat feedback as provisional, not validated program
discovery improvement. All losses, predictions, negative outcomes, intervals,
memory figures and checkpoint hashes are retained. Selected-action replay
executed another 1,936 branches with zero label difference. Optimizer/RNG
restoration and graph symmetries passed for all six models.

Reuse `source scripts/env.sh`, the existing Qwen snapshot and prepared project
instances. Do not reinstall or redownload. Models are under
`$TSP_ROOT/checkpoints/stage2/training-v1/{transformer,mlp}-seed{101,202,303}/best.pt`.
Checkpoint payloads contain weights, optimizer, configuration, step and RNG.
Model code is `src/tsp_value/value_model.py`. Preserve the exact feature
construction (no dataset-fitted scaling). For live inference, construct these
features without any future branch labels and verify equivalence on saved
states. Score every current undirected edge, with a common reference edge.
Fix ensemble/seed use before discovery; do not choose seed303 because of its
best validation benefit. Averaging all three predeclared Transformer runs is
an available choice, but its program-discovery benefit has not been tested.

Build the fixed common-state bank from **discovery only**. A and B share the
frozen 7B revision/precision, program interface, initial parent, generation
settings, candidate budget, discovery instances and paired discovery seeds.
A receives real endpoint feedback. B additionally receives parent/child
decisions compared at the same bank states. Generated rules change only
finite current-edge priorities. Keep full prompts/responses, program hashes,
lineage, failures, resource costs and true route outcomes. Use restricted AST,
isolated execution and the existing budget-exact solver.

Select fixed elites on program_select; evaluate once on program_dev_eval
under predeclared solver seeds. Never put value_val examples/errors into LLM
prompts. Keep final author test, INViT and TSPLIB closed. If model feedback
is not useful, retain the negative A/B and label any small direct-branch
feedback comparator explicitly. SFT/GRPO are not part of this milestone step.

Code/results for stage2 are saved locally and on the server; the last public
release is still stage1 until a later publication step. No stage3 job has
started. See the local `milestone/STAGE3_READY_CN.md` and execution status for
the independently verified backup and current commit.
