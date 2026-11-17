# Stage 2 measured results — 18 November 2026

Stage 2 completed the real 64-instance pilot, expanded to 256 train + 64 project-heldout validation instances, trained both models, and replayed selected actions. The Transformer shows a small validation benefit; sparse labels and validation checkpoint selection prevent a strong effectiveness claim. No frozen-generator A/B or final evaluation has run.

At stage2 entry, 18 November 00:07 BJT, about 47h52 remained until 19 November 23:59. The original stage2 target date was already missed; the work used the existing environment and prepared inputs. No package reinstallation, original-data download, or 7B loading occurred.

## Label diagnosis and protocol frozen before validation

Training-only diagnosis v1 used eight original instances with uniform edge candidates, H=50,000 and 50/200/800 sweep equivalents. Diagnosis v2 used 16 nested train instances and high-utility candidates. Diagnosis v3 compared 5/10/20/50/200 sweeps with uniform candidates on the same 16 instances. These are diagnostic work, not extra independent instances. All zero labels are retained. The stage1 engineering probes are excluded from the pilot.

H is locked at 58,750 deltas for TSP50 (50 sweeps) and 970,000 for TSP100 (200 sweeps), selected for per-size informative-state coverage in v3. Longer H often reconverged. Utility-biased candidates reduced signal. This is exploratory train-side protocol selection on a small subset, not proof that these H values are optimal.

Sampling remains four fixed penalty-event boundaries [0,8,32,96] along the conventional GLS trajectory initialized with seed 17, and eight uniformly sampled distinct nonreference current edges with stable instance/state seeds. No future-label filtering or resampling is used. Each branch increments one penalty and uses the identical conventional continuation. State snapshots retain improvement/stagnation information; measured coverage is in `stage2_summary.json`.

The pilot contains 64 instances (32 per size), 256 states, 2,048 averaged candidate labels and exactly **4,608** branches, including 512 reference branches. It took 27.57s, 215.38 worker CPU s, 167.17 branches/s, and 1.176 GiB summed process RSS. TSP50 branch p50/p95: 10.41/14.73ms; TSP100: 80.50/85.41ms. Pilot zero-label rates are 96.875% / 95.996%; informative-state rates are 19.531% / 21.875%.

All paired seed labels and measured terminal signatures coincide (100%); label standard deviation and observed sign disagreement are zero. This means the chosen continuation seeds supplied no independent random repeat, **not** that action values have no modeling or horizon uncertainty. The two repeated labels are averaged once. No confidence interval treats those repeats as independent.

The pilot is copied exactly into the 256-instance training set. New production work is 4,608 pilot +13,824 additional train +4,608 validation = **23,040 branches**, yielding 1,024 train and 256 val states. The train dataset's 18,432 branch count already includes the pilot. Expansion stopped at the planned initial size rather than 512/128 because label coverage remained weak. Train zero labels: 97.607% / 96.411%; val: 97.266% / 98.047%. Train/val have 183/39 informative states. No label-task failed.

## Actual models and validation

Graph Transformer: 4 layers, hidden 128, 8 heads, 862,081 parameters; relative pair attention bias uses normalized distance, current-route adjacency, log penalties and incumbent-route adjacency. Node inputs include centered geometry and incident distance/penalty summaries; global inputs describe size, lambda, best/current length and progress/stagnation. There are no city-ID embeddings. Undirected edge readout uses sums and absolute differences. MLP: 20,353 parameters, edge and pooled features from the same derived tensors, without graph message passing. Input features contain no future branch cost or author reference solution.

Both use AdamW, learning rate 3e-4, batch 16, MSE in percentage-point units plus 0.1 logistic ranking loss on non-tied same-state pairs. Tied labels do not create preference targets. Fixed optimization seeds are 101/202/303; a maximum of 40 epochs, minimum 10, patience 8; minimum validation top1 regret selects the checkpoint. No architecture or loss search followed validation. Each action set includes the eight candidates and the reference; score ties prefer reference.

Instance IDs and unordered coordinate hashes are disjoint; all four states stay with their original instance. Validation is author-train project heldout, not an independent author val/test. Spearman and pair accuracy below use informative/non-tied comparisons; regret and benefit average all states. Benefit in pp is 100 times paired relative length improvement over the reference branch, not gap-to-reference or full-solver improvement.

| Method | Spearman | Non-tied pair accuracy | Top1 regret (pp) ↓ | Selected benefit (pp) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Conventional GLS | 0.1181 | 0.5907 | 0.029643 | +0.000000 |
| MLP (3-seed mean) | 0.1835 | 0.6636 | 0.030065 | -0.000423 |
| Graph Transformer (3-seed mean) | 0.2001 | 0.6790 | 0.025733 | +0.003910 |

Uniform choice among nine actions has expected regret 0.032237 pp and benefit −0.002594 pp. The validation oracle among those nine has benefit 0.029643 pp. High top1-optimal fractions (~95%) are dominated by ties and are not evidence of useful prediction.

| Training run | Epochs / selected epoch | Val benefit (pp) | Original-instance bootstrap 95% interval |
| --- | ---: | ---: | --- |
| mlp-seed101 | 10 / 1 | -0.000677 | [-0.022694609125028364, 0.015365642739925524] |
| mlp-seed202 | 17 / 9 | +0.000000 | [0.0, 0.0] |
| mlp-seed303 | 10 / 1 | -0.000591 | [-0.018357774009928107, 0.013017478023539255] |
| transformer-seed101 | 22 / 14 | +0.003310 | [-0.0050439126789569855, 0.014954742131521925] |
| transformer-seed202 | 15 / 7 | +0.002128 | [0.0, 0.005772748962044716] |
| transformer-seed303 | 28 / 20 | +0.006291 | [0.0, 0.018872091080993414] |

These intervals resample 64 original validation instances with their states grouped; they do not correct the optimism from checkpoint selection on the same validation data. Some gains arise on very few instances (see `selected_action_instance_counts`). Three optimization seeds do not establish significance or program-discovery benefit. TSP100 seed101 selected benefit is negative despite positive combined-scale benefit. Keep these negative and zero outcomes.

Selected actions from all six models and both reference comparators were independently rerun on all 256 validation states: **1,936 extra branches**, maximum absolute difference from saved labels **0**. Repeated actions share a reference rollout. This verifies finite-horizon outcomes; no complete learned-guidance solver is evaluated.

All six runs restore trained weights, populated Adam moments, Python/NumPy/Torch/CUDA/shuffle RNG after epoch1. Their next dropout-enabled update matches uninterrupted execution exactly (parameter and loss error zero). Best checkpoints were restored for the reported validation. Relabeling errors are at most 5.96e-08 pp, and reversing undirected endpoints changes scores by zero. Checkpoint hashes are in individual summaries; local checkpoint copies were checked against them.

## Measured resources

Eight CPU workers, one BLAS thread each, RTX3090 24GB for training only. Summed RSS includes shared pages and is not physical-memory PSS.

| Label scope | New branches | Wall s | Worker CPU s | Summed RSS peak GiB |
| --- | ---: | ---: | ---: | ---: |
| diagnosis-v1 | 2,304 | 31.00 | 180.60 | 1.161 |
| diagnosis-v2 | 3,456 | 45.67 | 350.88 | 1.159 |
| diagnosis-v3 | 5,760 | 13.37 | 101.76 | 1.163 |
| pilot | 4,608 | 27.57 | 215.38 | 1.176 |
| train | 13,824 | 82.54 | 646.85 | 1.357 |
| val | 4,608 | 27.33 | 213.63 | 1.176 |

All label commands sum to 227.48 wall seconds / 1709.09 worker CPU seconds, excluding parent parsing/logging. Whole training command: 235.16 wall seconds / 234.41 process CPU seconds. Individual model wall/epoch/step times are retained. Transformer CUDA peak allocated/reserved: 191.64/216.00 MiB; MLP: 70.46/88.00 MiB. Training-process RSS peak: 1361.00 MiB. CUDA allocator figures exclude driver/context overhead. Training wall includes validation/checkpoint I/O, not GPU-kernel-only time. Replay and plotting command costs are in `resource_accounting.json`. Transfers, implementation and backup are excluded. LLM research calls: zero.

## Artifacts and next step

- `results/stage2/*/states.jsonl.gz`: raw snapshots and both-seed action labels; `branches.jsonl`: individual branch timing, signatures and costs; `tasks.jsonl`: CPU/RSS.
- `results/stage2/training-v1/`: immutable per-epoch logs, per-state predictions, isolation audit and checkpoint hashes.
- `results/stage2/action-replay/`: independent selected-action outcomes.
- `results/stage2/figures/`: four PNG/PDF/SVG figures and two CSV source tables.
- `scripts/run_stage2.sh`: full frozen-protocol replay with new output/checkpoint paths.

Next entry: `docs/STAGE3_ENTRY.md`. Preserve all three trained Transformer checkpoints and predeclare any ensemble/seed use before discovery. State feedback remains provisional; compare frozen-7B endpoint-only versus endpoint+state feedback under matched budgets. If unreliable, explicitly label a small direct-branch-feedback comparator. Do not put value_val contents/errors in LLM prompts, and do not touch final.
