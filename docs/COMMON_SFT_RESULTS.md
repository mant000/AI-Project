# Common QLoRA SFT results

The executed grammar corpus supports a verified common initialization, not evidence of improved TSP route quality. The base is the original local Qwen2.5-Coder-7B-Instruct revision c03e6d358207e414f1eca0bb1891e29f1db0e242, with NF4 double quantization, BF16 computation and rank-8 q_proj/v_proj LoRA (2,523,136 trainable parameters). The synthetic infrastructure adapter was not used.

307 training examples (8 grammar families) were trained for two epochs using response-only causal cross entropy. Prompt labels were masked; observed training lengths were 591–613 tokens and no response was truncated. The 14-example validation set covers only one family; all splits share underlying discovery TSP instances, so family transfer is not unseen-instance route generalization.

| Checkpoint | Token-weighted validation response NLL |
|---|---:|
| Epoch 0 | 0.534558 |
| Epoch 1 | 0.102282 |
| Epoch 2 | 0.093647 |

Epoch 2 was frozen under the predeclared minimum-NLL rule with epoch0 eligible. Only then were the 40 family-heldout examples opened: response NLL 0.101573. All eight predeclared diagnostic generations parsed successfully under the fixed interface; none was used for checkpoint/prompt selection or claimed as a discovery result.

The optimizer made 154 logical updates and 156 physical updates including two rolled-back restore probes (622 microbatches). The populated 112-state optimizer/RNG next-update check had parameter and loss errors zero. A new-process adapter load matched all 112 saved tensors and the selected validation NLL exactly. The independent verifier also checked full tokenized data, masks, training order and checkpoint selection.

Training/readout diagnostics consumed 1122.17 wall seconds and 1122.70 process CPU seconds; the independent verification added 17.46 wall seconds. Training events used 343,046 prompt and 28,156 response tokens; including restore probes this was 347,524 / 28,514. CUDA peak allocated/reserved memory was 8.636/9.389 GiB. Validation, heldout scoring and eight diagnostic generation costs are included in the measured parent command; raw generation token/time records are retained separately. This is not an end-to-end discovery cost comparison.

Selected adapter SHA256: `05ba33c61d8b2f62a3131b7452d1fe1a88ae182302694d000a8a2497764135b5`. Raw artifacts are in `results/training/sft-v1`, checkpoints in the separately backed-up `checkpoints/sft-v1`. Reproduce with `python scripts/common_sft.py --root ROOT`, then `python scripts/common_sft_verify.py --root ROOT` after a fresh run. Existing completed paths reject retraining. Neither command opens final data. Downstream GRPO and main route comparisons are required before making any quality claim.
