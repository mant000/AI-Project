# Historical trained inference exports

See `manifest.json` for the 28 model identities, weight SHA256s and verification.
The six Stage2 models, nine dense-teacher models, six history-ablation models,
common SFT and six GRPO adapters are retained without selecting a favorable seed.

Value networks are defined in `src/tsp_value/value_model.py` and `value_gnn.py`;
the adjacent `config.json` specifies architecture and feature encoding. LoRA
adapters use `Qwen/Qwen2.5-Coder-7B-Instruct` at the pinned revision in
`docs/DATA_AND_MODELS.md`, with rank8/alpha16 on q_proj/v_proj. Load the frozen
NF4 base with `scripts/qlora_pilot.py:load_base`, then use
`peft.PeftModel.from_pretrained(base, ADAPTER_DIRECTORY).eval()`.

Exports omit optimizer and RNG state; original training backups retain them.
They support historical reproducibility. New milestone feedback is directly
measured with the frozen original base and does not call these trained models.
