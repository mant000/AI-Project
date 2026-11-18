"""Synthetic infrastructure-only NF4 LoRA update, optimizer/RNG restore and fresh reload."""
import argparse
import gc
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel
from stage3_common import read,write,stamp,digest
from tsp_value.programs import canonical_program,INITIAL


def load_base(snapshot,checkpointing):
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,
        quantization_config=quant,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    out=root/'code/results/training/qlora-pilot-v1';out.mkdir(parents=True,exist_ok=False);ckdir=root/'checkpoints/qlora-pilot-v1';ckdir.mkdir(exist_ok=False)
    start,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1);random.seed(901);np.random.seed(901);torch.manual_seed(901);torch.cuda.manual_seed_all(901)
    manifest=read(root/'artifacts/qwen_download.json');assert manifest['config']['revision']=='c03e6d358207e414f1eca0bb1891e29f1db0e242'
    assert read(root/'operations/training-environment/completed.json')['base_versions_unchanged']
    write(out/'protocol.json',dict(at_utc=stamp(),scope='synthetic infrastructure only, excluded from SFT/GRPO study and candidate counts',
        model_revision=manifest['config']['revision'],nf4=True,double_quant=True,compute_dtype='bfloat16',rank=8,alpha=16,
        target_modules=['q_proj','v_proj'],dropout=0,optimizer='AdamW 1e-4',source_sha256=digest(Path(__file__))))
    tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True)
    messages=[dict(role='system',content='Return only the requested Python function.'),dict(role='user',content='Return the standard GLS priority rule with signature priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio), with a single return d/(1+p).')]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);response=canonical_program(INITIAL)['source']
    before=tokenizer(prompt,add_special_tokens=False)['input_ids'];after=tokenizer(response,add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
    ids=torch.tensor([before+after],device='cuda');labels=torch.tensor([[-100]*len(before)+after],device='cuda');mask=torch.ones_like(ids)
    torch.cuda.reset_peak_memory_stats();load=time.perf_counter();base=load_base(manifest['snapshot_path'],True)
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    del base;parameters=[p for p in model.parameters() if p.requires_grad];names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all('lora_' in n for n in names);trainable=sum(p.numel() for p in parameters);opt=torch.optim.AdamW(parameters,lr=1e-4)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load
    def step():
        model.train();opt.zero_grad(set_to_none=True);loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        assert torch.isfinite(norm);opt.step();torch.cuda.synchronize();return float(loss.detach()),float(norm)
    first=step();assert len(opt.state)>0
    snapshot=dict(parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},optimizer=opt.state_dict(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(snapshot,ckdir/'after-step1-state.pt');model.save_pretrained(ckdir/'after-step1-adapter',safe_serialization=True)
    expected_loss,expected_norm=step();expected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    restored=torch.load(ckdir/'after-step1-state.pt',map_location='cpu')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in restored['parameters']:p.copy_(restored['parameters'][n].to(p.device))
    opt.load_state_dict(restored['optimizer']);random.setstate(restored['python_rng']);np.random.set_state(restored['numpy_rng']);torch.set_rng_state(restored['torch_rng']);torch.cuda.set_rng_state_all(restored['cuda_rng'])
    resumed_loss,resumed_norm=step();error=max(float((p.detach().cpu()-expected[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad)
    assert error<=1e-6 and abs(expected_loss-resumed_loss)<=1e-5
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
    model.save_pretrained(ckdir/'after-step2-adapter',safe_serialization=True);model.eval()
    with torch.inference_mode():reference=model(input_ids=ids,attention_mask=mask).logits[:,-1,:].float().cpu()
    del model,opt,parameters;gc.collect();torch.cuda.empty_cache()
    fresh=PeftModel.from_pretrained(load_base(manifest['snapshot_path'],False),ckdir/'after-step2-adapter',is_trainable=False).eval()
    with torch.inference_mode():actual=fresh(input_ids=ids,attention_mask=mask).logits[:,-1,:].float().cpu()
    reload_error=float((reference-actual).abs().max());assert reload_error<=1e-4
    fresh.config.use_cache=True
    with torch.inference_mode():generated=fresh.generate(input_ids=ids[:,:len(before)],attention_mask=mask[:,:len(before)],max_new_tokens=8,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    result=dict(status='passed',completed_at_utc=stamp(),scope='synthetic infrastructure only; do not use this adapter as research initialization',
        trainable_parameters=trainable,only_lora_parameters_updated=True,prompt_tokens=len(before),response_tokens=len(after),
        first_update_loss=first[0],first_update_grad_norm=first[1],uninterrupted_next_loss=expected_loss,resumed_next_loss=resumed_loss,
        restore_max_parameter_error=error,fresh_base_adapter_reload_max_logit_error=reload_error,
        populated_optimizer_states=len(snapshot['optimizer']['state']),research_candidate_generations=0,logical_training_updates=2,physical_backward_steps_including_replay=3,
        smoke_generation_tokens=int(generated.shape[1]-len(before)),model_load_wall_seconds=load_seconds,
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        adapter_after_step2_sha256=digest(ckdir/'after-step2-adapter/adapter_model.safetensors'),final_opened=False)
    write(out/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
