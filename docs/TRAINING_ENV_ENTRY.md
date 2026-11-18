# Parameter-training environment entry

The existing environment has Torch 2.1.2+cu121, Transformers 4.45.2,
Accelerate 1.0.1 and the full fixed Qwen snapshot. PEFT/bitsandbytes are
not installed at R1 entry. No training or dependency change was performed
while the frozen A/B/C discovery process was running.

Use the smallest compatible additions, `peft==0.13.2` and
`bitsandbytes==0.44.1`, without upgrading the verified stack. Preserve a
before/after package inventory and run `pip check` and a real local
NF4 forward/backward/adapter save/load test before training. No data/model
redownload is necessary. JSONL data loaders and the explicitly implemented
small GRPO loop can use existing PyTorch rather than introducing a new TRL
or datasets stack.

PEFT 0.13.2 declares Torch >=1.13, Accelerate >=0.21 and the existing
Transformers dependency. This satisfies declared lower bounds; actual CUDA
compatibility still requires the measured pilot. See the pinned
[PEFT setup source](https://github.com/huggingface/peft/blob/v0.13.2/setup.py).
The matching [Transformers 4.45.2 guide](https://huggingface.co/docs/transformers/v4.45.2/quantization/bitsandbytes)
documents NF4/BF16 and double quantization. Quantized-base training updates
additional adapter parameters, not the frozen 4-bit base weights.

Pilot defaults to predeclare when the corpus is ready: NF4 with double
quantization and BF16 computation; rank8 LoRA, alpha16, zero dropout,
q_proj/v_proj targets, batch1 with accumulation, gradient checkpointing,
and measured sequence/memory limits. Checkpoints must retain adapter,
optimizer, scheduler, every RNG and dataset order; verify a real subsequent
step after restoration. Quantized and BF16 generator conditions must not
be silently mixed in a causal training comparison.

SFT data must come from executed discovery-only program changes, with
family/lineage separation and explicit source labels. Do not assume the
R1 generated outputs are already diverse training examples. GRPO rewards
remain primarily real endpoint quality; any provisional state term is
bounded, separately reported and compared with a same-initialization
endpoint-only condition. Complete-project goals do not make a negative
auxiliary signal useful by definition.
