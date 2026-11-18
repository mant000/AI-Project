# Data, model entry points and provenance

The milestone uses **256 original TSP instances** (TSP50/100 balanced), released
in `results/milestone-fragment-v1/prepare/benchmark_inputs.jsonl`. Each row retains
the author filename/line, source file and row SHA256, coordinate identity,
reference route/length, original nominal project role, and newly reserved role.
The source is [Joshi et al., learning-tsp](https://github.com/chaitjo/learning-tsp/tree/bb28f5795924d99f7e7d945695e884f8d1f7df45),
whose train archive and fixed revision are in `configs/sources.json`. These rows
were reserved before new outcomes, after checking actual historical use; they
are not a random sample of the author test set. The full original allocation
and exact input hash are retained in `prepare/pool_isolation.json` and
`bank_frozen.json`. Source notices are in `third_party/learning-tsp-LICENSE.txt`.
The original 4,992-row manifest remains in `splits/manifest.csv`.

The public subset is sufficient for every new milestone experiment. Full author
archives are **not needed** for the CPU analysis/solver replay. To reconstruct
the broader historical pipeline, use `scripts/download_sources.py joshi --root
ROOT` and the original preparation scripts documented in the historical README.
The source archive contains large training data; its download is optional.
Continuous Euclidean returned tours are repriced locally; author references are
not certified continuous optima. Negative gaps remain negative.

## Frozen generator

Original base: [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
revision **c03e6d358207e414f1eca0bb1891e29f1db0e242**. No third-party full weights
are copied into this repository. `scripts/download_sources.py qwen --root ROOT`
downloads the pinned safetensors and verifies the author LFS hashes in
`configs/qwen_weight_hashes.json`; it writes `ROOT/artifacts/qwen_download.json`.
An existing complete snapshot is reused by the experiment without a download.
The new comparison uses NF4 double quantization, BF16 linear computation,
the existing `qlora_pilot.load_base` preparation (nonquantized tensors cast to
FP32), evaluation mode, and **zero trainable parameters**. It loads no adapter.

## Released historical trained models

`trained_models/manifest.json` indexes **21 value networks and seven LoRA
adapters**, with exact weight hashes, original checkpoint identity and numerical
reload verification. The small safetensors are included directly in Git.
Value model architecture/feature settings accompany each file. Adapters under
`trained_models/adapters` name the public base model and pinned revision;
`original_adapter_config.json` preserves the original configuration separately.
They require the above base and compatible Torch/Transformers/PEFT dependencies.
These are inference exports, not optimizer/RNG resume checkpoints. Full training
checkpoints remain in closed private backups; no claim is made that the small
export alone resumes training. The milestone fragment study does not use these
models, and releasing them is not evidence that state feedback was effective.

## Strong references

`configs/external_baselines_v1.json` and `results/baselines/prepare-v1` pin the
LEHD author source/checkpoint and an unmodified LKH3.0.6 research source mirror.
`scripts/prepare_baselines.py --root ROOT` documents their original download and
compile path; `scripts/milestone_external_setup.py --root ROOT` offers a portable
setup without overwriting the historical records. Third-party research-use
conditions remain applicable. LKH uses RUNS=1, MAX_TRIALS=1000; continuous
distances are rounded at scale 1e6 for input, then returned tours are priced on
the original coordinates. LEHD uses author greedy or RRC10, never reference
tours as model inputs. Their budgets are algorithm-specific, not equal compute.

Final test data are not included in this milestone subset and were not evaluated
in the new study. Historical data/model paths in raw records are provenance,
not requirements to reproduce the CPU report checks.
