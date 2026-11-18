# Dense-action teacher pilot: completed negative result

18 November 2026. The bounded pilot in `configs/core_teacher_v2.json` is
complete. It does not establish a better teacher, and will not be enlarged
to search for a favorable architecture or seed.

The same data and loss train MLP, a dense gated message-passing GNN, and the
graph Transformer, each with predeclared seeds101/202/303. GNN and Transformer
have 861,461 /862,081 parameters; MLP has20,353. The GNN replaces attention
with pair-gated mean messages while retaining the same observable features,
global embedding and symmetric edge readout. No city-ID embedding is used.

Train:32 original value_train instances,128 saved states,9,600 current-edge
values. Existing compatible seed71 action costs are reused:1,152 actions,
not relabeled or counted as new. Validation and teacher holdout each use16
previously unused value_val instances /64 states /4,800 actions. Only18,048
missing actions execute new branches, taking140.57 wall /1,082.14 worker CPU
seconds. All new tours recomputed with zero cost error;12 reused actions
were independently replayed exactly. Original-instance and coordinate-role
isolation pass. The continuation remains conventional GLS with the original
size-specific H and common seed71.

Three-seed **prediction ensembles**, evaluated on the teacher holdout only
after all nine checkpoints were selected and frozen:

| Teacher | Selected action benefit (pp) | Top1 regret (pp) | Non-tied pair accuracy |
|---|---:|---:|---:|
| Conventional reference | 0 | 0.184806 | 0.5023 |
| Dense MLP | 0 | 0.184806 | 0.5982 |
| Dense GNN | −0.007764 | 0.192570 | 0.5908 |
| Dense Transformer | −0.001969 | 0.186775 | 0.6168 |
| Stage2 MLP, transfer reference | −0.007254 | 0.192059 | 0.5877 |
| Stage2 Transformer, transfer reference | +0.002354 | 0.182452 | 0.5878 |

There are36 informative states among64, but only16 independent holdout
instances. These remain conditional finite-horizon action effects, not
complete-program improvements. Better average pair ordering does not imply
better top-action selection in these measurements. All models retain most
of the gap to the measured best action.

The dense pilot uses fewer original training instances than stage2 (32 vs256),
so it is not an isolated causal test of action coverage. No result supports
the stronger claim that exhaustive actions are universally harmful or that
GNN/Transformer architectures cannot learn the task. Conversely, no favorable
seed is selected to claim success. The previously frozen stage2 ensemble is
retained as the declared feedback reference; this pilot is not used to select
a new teacher from the holdout.

All nine populated optimizer/RNG restores match the next update exactly
(maximum parameter error0). Permutation and undirected-edge tests pass for
both node sizes. Full training, validation and heldout inference took108.59
wall /108.62 process CPU seconds. Maximum per-model CUDA allocated memory is
138,361,344 bytes. Raw epochs, checkpoints/hashes, all predictions and
freeze-before-holdout evidence are in `results/teacher-v2/` and the recorded
remote `checkpoints/teacher-v2` directory.

Next is the execution-verified, family-isolated corpus and common QLoRA SFT
warm start. New LoRA training packages are pinned additions, with the verified
Torch/Transformers base unchanged. Any synthetic NF4 pilot is infrastructure
evidence only; its adapter must not become the research initialization.
SFT/GRPO effects and state feedback remain to be tested, and final stays closed.
