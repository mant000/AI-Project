# Preserved historical results (separate from the new milestone comparison)

The milestone release retains the research trajectory and failure records.
Numbers from different instance pools/protocols must not be pooled as repetitions
of the new A/P/F study. The prior publication README is preserved separately;
dated statements there describe their original execution time.

| Work | Results and reproduction entry |
|---|---|
| Baseline engine/data audit | `results/stage1`, historical README, `scripts/replay_stage1.sh` |
| Sparse action-value models | `docs/STAGE2_RESULTS.md`, `results/stage2`, Stage2 scripts/configs |
| First frozen generator loop | `docs/STAGE3_RESULTS.md`, `results/stage3` |
| Four-seed endpoint/predicted/measured comparison | `docs/CORE_R1_RESULTS.md`, `results/core`, `scripts/core_verify.py` |
| Dense teachers and grammar corpus | `docs/CORE_R2_TEACHER_RESULTS.md`, `docs/CORE_R2_CORPUS_AND_TRAINING_ENTRY.md`, `results/teacher-v2`, `results/corpus-v1` |
| Common SFT | `docs/COMMON_SFT_RESULTS.md`, `results/training/sft-v1`, `scripts/run_r3.sh` |
| Six paired GRPO adapters | `docs/GRPO_RESULTS.md`, `results/training/grpo-v1`, `scripts/grpo_verify.py` |
| History feature ablation | `docs/HISTORY_ABLATION_RESULTS.md`, `results/history-ablation-v1` |
| Main crossed controls | `docs/MAIN_CONTROL_RESULTS.md`, `results/main`, `scripts/main_verify.py` |
| Existing solver/EoH references | `docs/SOLVER_CONTROL_RESULTS.md`, `results/solver-controls`, `results/eoh-control-v1` |
| Feedback audit and stopped revision | `docs/FEEDBACK_INFORMATION_AUDIT_CN.md`, `docs/FEEDBACK_REVISION_MECHANISM_RESULTS_CN.md`, `results/feedback-*` |

The interrupted 91-output revision remains incomplete and was not resumed or
pooled. It is not evidence for a completed arm comparison. All 28 historical
inference exports are under `trained_models`, with hashes and model/optimizer
seeds retained. The new policy-fragment study uses the original frozen NF4 base,
not any of these adapters or teachers.

Unexecuted local R5-v2 final scheduling drafts are excluded from the milestone
release. Final benchmark outcomes have not been used to tune this work.
