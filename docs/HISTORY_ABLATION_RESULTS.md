# Paired history-feature ablation

This controlled feature study does not support a benefit from retained search history. Both newly retrained three-seed ensembles fail to improve on the conventional reference on a fresh 32-instance/128-state heldout set. The paired full-minus-no-history selected benefit is -0.009375pp; a 5,000-resample original-instance bootstrap 95% interval is [-0.029096, +0.000971]pp. This interval does not support a stable directional effect.

| Ensemble | Selected benefit pp | Top1 regret pp | Non-tied pair accuracy |
|---|---:|---:|---:|
| full | -0.009699 | 0.172306 | 0.5224 |
| no_history | -0.000324 | 0.162931 | 0.6024 |
| R2-dense | +0.008269 | 0.154338 | 0.5545 |
| stage2-frozen | -0.015021 | 0.177628 | 0.6575 |
| reference | +0.000000 | 0.162607 | 0.5027 |

The six new models reuse exactly the saved R2 dense 128 training states and 64 validation states. Architecture, loss, batches, learning rate, checkpoint rule and seeds 101/202/303 are matched. Both arms were retrained on single-thread CPU so different GPU/CPU dropout streams are not confounded with history removal. Paired initial tensor hashes match. All six populated optimizer/RNG checks have zero error, permutation and undirected-edge checks pass, and fresh CPU checkpoint reload predictions match exactly. These are additional controlled models, not a repeat of stage2 label production.

The ablated encoder keeps the current route and fixed GLS lambda but removes penalty memory, best-tour memory and dynamic event/stagnation/work counters. It also replaces best-length normalization with current length. The conventional reference edge remains the same specified action, with all current edges scored and reference last. Thus this is a current-configuration/no-history ablation, not a completely state-independent geometry predictor. A property test verifies invariance to the removed memories and that the original solver state is untouched.

All six checkpoints were frozen before any fresh heldout branches were generated. The new heldout uses value_val offsets 48..63 per size, disjoint from R2 train/validation/heldout. All 9,600 new branch tours and labels were independently recomputed with zero route-cost error; 12 branches were independently replayed exactly. There are 32 independent original instances, not 128 independent graph examples or 9,600 independent problems. No program_dev or final instance supplies these targets.

The pre-existing GPU-trained R2 dense reference changes sign relative to its earlier heldout, and the old stage2 teacher is negative here. These transfer diagnostics illustrate limited reliability; no teacher, seed, threshold, prompt or GRPO reward was replaced using them. CPU/GPU-trained full models are reported as separate references, with no isolated backend-effect claim.

The complete command took 268.03 wall seconds/195.75 parent CPU seconds; fresh label workers used 591.33 CPU seconds. Independent audit/reload/replay added 9.63 wall/9.61 CPU seconds. GPU use was disabled. Raw results/checkpoints retain each run, validation curve, selected epoch, all labels/predictions, costs and source hashes. Run `python scripts/history_ablation.py --root ROOT`, then `python scripts/history_verify.py --root ROOT`; final data remain closed.
