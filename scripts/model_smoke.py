"""Load the fixed local 7B snapshot and perform one short greedy generation.

This is infrastructure validation, not a discovery candidate or A/B result.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    root = Path(os.environ['TSP_ROOT'])
    manifest = json.loads((root/'artifacts/qwen_download.json').read_text())
    assert manifest['status'] == 'completed_verified'
    assert manifest.get('author_weight_hashes_verified'), 'Verify author LFS hashes first'
    snapshot = manifest['snapshot_path']
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Reply only with the numeral 1.'}],
                                           tokenize=False, add_generation_prompt=True)
    batch = tokenizer(prompt, return_tensors='pt').to('cuda')
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    result = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
              'model_id': manifest['config']['repository'], 'revision': manifest['config']['revision'],
              'precision': 'bfloat16', 'success': True, 'load_seconds': load_seconds,
              'generation_seconds': time.perf_counter() - start,
              'input_tokens': batch['input_ids'].shape[1],
              'new_tokens': output.shape[1] - batch['input_ids'].shape[1],
              'response': tokenizer.decode(output[0, batch['input_ids'].shape[1]:], skip_special_tokens=True),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'logical_smoke_generation_calls': 1,
              'research_discovery_candidates': 0,
              'scope': 'Single short frozen BF16 inference only; no SFT/GRPO validation'}
    (root/'artifacts/model_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
