# Controlled QLoRA GRPO results

All six predeclared runs completed from the same independently verified common-SFT epoch2 adapter. Each used 12 fixed parent prompts and four categorical rollouts per group on the same fresh discovery instances (offset40..47 per size). Terminal and state arms used identical prompts, generation settings, work caps and paired sampling seeds. Only the bounded auxiliary reward differed. All three optimization seeds are retained; no model is selected by a favorable route outcome.

| Optimization seed | Reward | Valid outputs | Actual updates | Command wall s | Endpoint worker CPU s |
|---|---|---:|---:|---:|---:|
| 701 | terminal | 47/48 | 12 | 439.95 | 99.50 |
| 701 | state | 47/48 | 12 | 448.19 | 102.56 |
| 702 | terminal | 47/48 | 12 | 442.14 | 101.25 |
| 702 | state | 47/48 | 12 | 445.90 | 99.63 |
| 703 | terminal | 46/48 | 12 | 449.97 | 102.36 |
| 703 | state | 46/48 | 12 | 437.82 | 96.40 |

There were 288 outputs (280 valid), 72 logical optimizer updates and 84 physical updates including two rolled-back optimizer/RNG verification steps per run. All six populated-optimizer next-step checks had zero parameter/loss error, all frozen reference adapters stayed unchanged, and each policy adapter changed from SFT. First-group token IDs and old-policy log probabilities were identical across the paired reward arms for every seed. Every group had nonzero reward variance; no update was silently added for a skipped group.

Despite distinct learned tensors (maximum terminal/state parameter differences 4.61e-5, 3.27e-5 and 2.05e-5), all 140 valid paired generated programs were identical across reward arms. Their paired training-pool route gaps therefore differ by exactly zero. This shows no observed program-output benefit from the auxiliary reward within this small budget; it does not establish that GRPO improves independent route quality. The queued main study uses frozen adapters, new discovery/selection/development instances, and crossed inference-feedback controls to assess that question.

Independent verification recomputed all 8128 unique per-run endpoint routes and all 8,960 valid candidate/state feedback rows, checked prompts, parsing, seed pairing, reward arithmetic/order preservation, token-log-probability lengths and adapter hashes. Failures/duplicates consume the budget. Auxiliary rewards are capped at 0.02 and further reduced by distinct group terminal-reward separation; clipping-induced terminal ties remain a limitation. The one-update-per-group PPO surrogate starts on policy; restore probes do not become hidden training steps.

After all six adapters were frozen, a separate deterministic 12-pair conditional branch diagnostic executed 24 conventional-continuation branches. Eleven pairs tied in true finite-H quality, including one indicated model preference on a true tie; MAE was 0.007925pp. This tiny, deliberately limited diagnostic neither calibrates uncertainty nor tunes rewards/teachers. It is a measured-reference check, not a Transformer success or an optimal-Q estimate.

Across six commands: 2663.98 wall seconds, 1967.26 parent CPU seconds, 601.71 endpoint worker CPU seconds, 160,056 input and 12,488 generated tokens. Endpoint CPU excludes parts of worker startup/decision/monitoring overhead; raw subprocess, decision and GPU power/memory samples are preserved, so these subtotals are not mislabeled as exact total host CPU. CPU-only history ablation overlapped part of later training; no formal wall-time solver advantage is inferred from these command timings.

Artifacts: `results/training/grpo-v1`, `grpo-analysis-v1`; six final adapters are under `checkpoints/grpo-v1/*/final/policy`. Reproduce the saved protocol with `scripts/run_r3.sh`; audit with `python scripts/grpo_verify.py --root ROOT`; recreate tables/figures with `python scripts/grpo_analyze.py`. Existing completed training paths reject retraining. Final data were not opened. The algorithm follows group-relative advantages and clipped token objectives from [DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4.SS1), at a much smaller application-specific scale.
