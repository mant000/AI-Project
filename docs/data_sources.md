# Data and external models

## TSP inputs

The 256 Euclidean TSP50/100 instances in
`results/mechanism/bank/benchmark_inputs.jsonl` come from the training archive
released by [Joshi et al.](https://github.com/chaitjo/learning-tsp/tree/bb28f5795924d99f7e7d945695e884f8d1f7df45).
The source revision and archive URL are in `configs/sources.json`; the original
license is in `third_party/learning-tsp-LICENSE.txt`.

Each row retains the source filename, line and hashes, coordinates, reference
tour and new experimental role. The bank, discovery, selection and development
roles contain 128/32/32/64 originals. This subset was reserved before outcomes,
after auditing actual earlier use of the nominal training pool. The original
allocation audit is `results/mechanism/bank/pool_isolation.json`; the frozen
input hash is in `bank_frozen.json` beside it. Full author archives are not
needed for the README commands.

All returned tours are scored on continuous Euclidean distances. Author
reference tours are not certified continuous optima; negative gaps are retained.
The released subset contains no final-test instances.

## Frozen generator

Model: [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
`download_model.py` obtains the pinned safetensors and verifies the author LFS
hashes in `configs/qwen_weight_hashes.json`. It writes the snapshot path and
verification manifest to `TSP_ROOT/artifacts/qwen_download.json`. Existing
verified weights can be reused. Full third-party weights are not rehosted.

`scripts/qwen_model.py` preserves the executed loading convention: NF4 double
quantization, BF16 linear computation and the PEFT k-bit preparation that casts
nonquantized tensors to FP32. Discovery uses evaluation mode, enables the KV
cache, disables all gradients and loads no adapter. PEFT is a loading dependency
here, not evidence of parameter training. Versions are pinned in
`requirements-gpu.txt`; the original machine's package list is
`environment/remote.freeze.txt`.

## Reference solvers

LEHD source and pretrained checkpoint come from
[CIAM-Group/NCO_code](https://github.com/CIAM-Group/NCO_code/tree/274df3c4975384592b60fe7f79fbb2441ce11c15/single_objective/LEHD).
Greedy and ten repair iterations use the author search code; reference tours
are never model inputs.

LKH 3.0.6 comes from the unmodified research source mirror
[cerebis/LKH3](https://github.com/cerebis/LKH3/tree/9a51de3161f2746e969f7a2e33220f637da64e54).
It uses `RUNS=1`, `MAX_TRIALS=1000`. Continuous distances are rounded at scale
1e6 for solver input; returned tours are repriced on the original coordinates.
The mirror was frozen after the official download was inaccessible, and is not
described as the latest LKH version. Research-use source conditions apply.

`configs/reference_assets.json` records revisions and model settings;
`configs/reference_asset_hashes.json` records the downloaded bytes.
`setup_references.py` checks those hashes, reuses available assets and builds LKH
with GCC/make. It writes machine-specific paths under
`TSP_ROOT/operations/reference-assets/`. The wrapper is
`scripts/reference_solvers.py`. These solvers use their own budgets; they are
quality references, not matched-compute treatments.
