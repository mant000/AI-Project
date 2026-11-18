# Frozen generator and parameter-learning controls (R4)

All 1,080 predeclared outputs were generated: 1,072 valid and eight invalid. The
same NF4 Qwen2.5-Coder-7B base, conventional initial rule, generation settings,
paired discovery seeds 801/802/803, original instances and endpoint work budgets
were used. Original/A is an additional untrained reference; the factorial
controls use the common frozen SFT initialization. All three GRPO optimization
seeds 701/702/703 are retained. Shared SFT initialization is one training run,
not three independently trained SFT models.

Every context generated 24 candidates. Real discovery route quality maintained
elites; real `program_select` route quality selected each program. All 45
programs were frozen before any main `program_dev_eval` results were computed.
The development set contains 64 original instances, with solver seeds 0/1/2.
Neither development results nor `value_val` labels entered generation prompts.

| Generator / feedback | Mean independent dev gap-to-reference (%) |
|---|---:|
| Original / terminal | 2.371581 |
| SFT / terminal | 2.231818 |
| SFT / terminal + state | 2.247977 |
| Terminal GRPO / terminal | 2.232500 |
| Terminal GRPO / terminal + state | 2.231818 |
| State GRPO / terminal | 2.232500 |
| State GRPO / terminal + state | 2.231818 |
| Conventional GLS | 2.384044 |

SFT/A minus original/A is −0.139762 gap percentage points, with an
original-instance paired bootstrap interval [−0.228031, −0.053491]. The three
paired discovery effects are −0.121594, −0.084205 and −0.213488 pp. This is
development evidence for a program-prior benefit from the grammar-derived SFT
corpus. It neither proves new heuristic novelty nor establishes a state-value
benefit, and requires independent final evaluation.

The primary state-GRPO/B minus terminal-GRPO/A effect is −0.000682 pp,
interval [−0.011282, +0.009745]. Auxiliary-reward-only effects are exactly zero
for these frozen programs; SFT B−A is +0.016159 pp. These results do not support
an added benefit from the state feedback or auxiliary GRPO reward at this
budget. The intervals condition on the frozen programs; all discovery-seed
effects are separately reported and are not treated as independent instances.

The code-only EoH-inspired supplementary control generated 72 outputs, 53
valid. Rank-biased population parents and five exploration/modification
operators were used with the same original frozen base and 24-output budgets.
All three independently frozen programs remained conventional GLS, each with
dev gap 2.384044%. This omits separate thought evolution and is explicitly an
adaptation, not an exact reproduction of EoH.

Independent audit repriced all 7,184 unique main evaluation routes and all
2,720 EoH routes with zero error. It reconstructed raw prompts, tokens, program
hashes, parent updates, population RNG, true-quality selection and the freeze
boundary; recomputed 544 main state-decision sets and 16,128 feedback rows;
rescored the bank with zero error; and passed 12 main plus six EoH solver
replays. Invalid candidates and duplicates consumed the predeclared budget.

Main discovery/selection/dev used 4,694.71 command-wall seconds, 4,218.02 parent
CPU seconds and 465.87 isolated-worker CPU seconds; generation itself used
4,152.19 wall seconds. Tokens: 1,822,259 input / 40,826 output. EoH used 438.97
wall seconds, 243.42 parent CPU seconds and 191.67 endpoint-worker CPU seconds,
with 77,620 input / 3,681 output tokens. These scopes exclude preparation,
previous training and independent audit; summed parent/worker time is not
claimed to be a perfect host-wide accounting total. Peak main CUDA allocated /
reserved memory was 8.28 / 9.38 GiB.

Raw evidence: `results/main/discovery-v1`, `results/main/verification-v1`,
`results/eoh-control-v1`. Tables and figures: `results/main/analysis-v1`.
The separate five-solver-seed development controls and their timing pilot are
still running; final remains unopened at this report's creation.
