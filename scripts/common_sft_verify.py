"""Independent corpus/selection/update audit plus fresh selected-adapter numerical reload."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer
from stage3_common import CODE, read, write, digest, stamp
from common_sft import encode, score
from qlora_pilot import load_base
from tsp_value.programs import canonical_program


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); a = ap.parse_args()
    out = CODE/'results/training/sft-v1'; completed = read(out/'completed.json'); assert completed['status'] == 'completed'
    assert not (out/'verification.json').exists(), 'Already verified'
    start, cpu = time.perf_counter(), time.process_time(); torch.set_num_threads(1)
    protocol = read(out/'protocol_frozen.json'); cfg = protocol['protocol']; corpus = CODE/cfg['corpus']
    for s, expected in protocol['corpus_sha256'].items(): assert digest(corpus/(s+'.jsonl')) == expected
    for source, expected in protocol['source_sha256'].items(): assert digest(CODE/source) == expected
    assert digest(CODE/'configs/sft_v1.json') == protocol['protocol_sha256']
    manifest = read(a.root/'artifacts/qwen_download.json')
    tokenizer = AutoTokenizer.from_pretrained(manifest['snapshot_path'], local_files_only=True)
    data = {s: encode([json.loads(x) for x in (corpus/(s+'.jsonl')).read_text().splitlines()], tokenizer,
                      cfg['max_sequence_tokens']) for s in ('train', 'validation')}
    for s, rows in data.items():
        actual = [json.loads(x) for x in (out/(s+'_tokenized.jsonl')).read_text().splitlines()]
        assert actual == rows
        for r in rows:
            assert all(x == -100 for x in r['labels'][:r['prompt_tokens']])
            assert r['labels'][r['prompt_tokens']:] == r['input_ids'][r['prompt_tokens']:]
    events = [json.loads(x) for x in (out/'update_events.jsonl').read_text().splitlines()]
    training = [e for e in events if e['kind'] == 'training']; expected_groups = []
    for epoch in range(1, cfg['epochs']+1):
        order = list(range(len(data['train']))); random.Random(cfg['seed']+epoch).shuffle(order)
        for i in range(0, len(order), cfg['gradient_accumulation']):
            expected_groups.append([data['train'][j]['example_id'] for j in order[i:i+cfg['gradient_accumulation']]])
    assert [e['example_ids'] for e in training] == expected_groups
    assert [e['logical_step'] for e in training] == list(range(1, len(expected_groups)+1))
    assert completed['logical_updates'] == len(expected_groups) == 154
    restore_check = read(out/'resume_verification.json'); assert restore_check['status'] == 'passed'
    assert restore_check['actual_corpus'] and restore_check['rolled_back_to_logical_update'] == 1
    assert len(events) == 156 and sum(e['microbatches'] for e in events) == 622
    vals = [read(out/f'validation_epoch{e}.json') for e in range(3)]
    for v in vals:
        total = sum(r['nll']*r['response_tokens'] for r in v['detail'])/sum(r['response_tokens'] for r in v['detail'])
        assert abs(total-v['mean_response_nll']) < 1e-12
    chosen = min(range(3), key=lambda e: vals[e]['mean_response_nll'])
    freeze = read(out/'checkpoint_frozen.json'); assert chosen == freeze['selected_epoch'] == completed['selected_epoch']
    assert freeze['at_utc'] < read(out/'heldout_nll.json')['at_utc']
    path = Path(freeze['selected_adapter']); assert digest(path/'adapter_model.safetensors') == freeze['adapter_sha256']
    model = PeftModel.from_pretrained(load_base(manifest['snapshot_path'], False), path, is_trainable=False).eval()
    disk = load_file(str(path/'adapter_model.safetensors')); loaded = get_peft_model_state_dict(model)
    assert disk.keys() == loaded.keys() and len(disk) == 112
    parameter_error = max(float((disk[k]-loaded[k].detach().cpu()).abs().max()) for k in disk); assert parameter_error == 0.
    actual = score(model, data['validation'])
    nll_error = abs(actual['mean_response_nll']-vals[chosen]['mean_response_nll']); assert nll_error <= 1e-5
    diagnoses = []
    for p in sorted(out.glob('heldout_generation_*.json')):
        raw = read(p)
        try: parsed = canonical_program(raw['response']); diagnoses.append(dict(file=p.name, valid=True, **parsed))
        except Exception as exc: diagnoses.append(dict(file=p.name, valid=False, error=str(exc)))
    assert len(diagnoses) == cfg['heldout_generation']['unique_parent_prompts'] == 8
    result = dict(status='passed', at_utc=stamp(), tokenized_corpus_and_response_mask_exact=True,
        logical_updates=154, physical_updates_including_rolled_back_verification=156, physical_microbatches=622,
        epoch0_eligible_selection_correct=True, selected_epoch=chosen, fresh_adapter_tensors=len(disk),
        fresh_adapter_max_parameter_error=parameter_error, fresh_reload_validation_nll_error=nll_error,
        heldout_generation_diagnostics=diagnoses, final_opened=False,
        wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(out/'verification.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
