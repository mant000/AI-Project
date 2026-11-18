# Executed corpus and QLoRA readiness

R2 corpus is complete and independently verified. These are **grammar-derived**
program changes, not mislabeled LLM discoveries. Twelve arithmetic families
were fixed before evaluation;143 unique generated programs plus conventional
GLS were actually run on32 new discovery instances with two solver seeds.
All144 programs were valid;9,216 endpoint routes recomputed with zero error.
Evaluation used89.27 wall /666.61 isolated-worker CPU seconds. No LLM calls
were used to enumerate this grammar.

Only strict measured within-family improvements with distinct endpoint-route
vectors become examples. A target need not beat standard GLS; it must beat
its own parent. The prompt contains parent code and its real discovery
metrics, never the target answer. Exact source, family/lineage and target
behavior overlap are controlled across splits:

| Split | Examples | Families | Distinct target programs | Distinct target behaviors |
|---|---:|---:|---:|---:|
| Train | 307 | 8 | 89 | 77 |
| Validation | 14 | 1 | 5 | 5 |
| Heldout | 40 | 2 | 13 | 9 |

All361 improvement pairs passed independent gap/route/order checks. Nineteen
validation targets and five heldout targets were removed because target
behavior had already appeared in an earlier split. Only one validation
family survives, a material limitation for SFT checkpoint selection. Splits
test program-family transfer on shared discovery instances, not unseen TSP
instance generalization. Parent behavior is not asserted to be novel across
splits; target behavior isolation is the recorded guarantee.

Pinned PEFT0.13.2/bitsandbytes0.44.1 were added without changing Torch,
Transformers, Accelerate, NumPy or the existing base stack. Wheels/hashes,
before/after freezes and pip-check evidence are retained. No model or data
was redownloaded.

The **synthetic infrastructure-only** NF4 pilot passed actual LoRA updates,
populated optimizer/RNG restoration and fresh base+adapter reload. It updates
2,523,136 LoRA parameters; both next-step parameter error and reload logit
error are zero. Two logical updates and one verification replay took17.60s;
CUDA peak allocated/reserved was11,130,259,968 /12,306,087,936 bytes. This uses
a short91-token synthetic example and does not guarantee memory for every
research sequence. Its adapter is excluded from research initialization.

Next follow `configs/sft_v1.json`: initialize a new common LoRA from original
Qwen NF4 weights, seed631; response-only loss; two epochs; compare epoch0/1/2
using family-validation NLL, then freeze before inspecting family-heldout
metrics. Check full actual sequence lengths, avoid answer leakage or silent
truncation, and test populated optimizer/RNG restoration on actual corpus
updates. A trained adapter or lower token loss is not a true route-quality
result. The subsequent same-precision frozen/updated × endpoint/state
comparison remains required, with state reward bounded and true route
selection authoritative. Final remains closed.
