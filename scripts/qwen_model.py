"""Frozen NF4 generator loading, with the original numerical preparation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def load_base(snapshot,checkpointing):
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,
        quantization_config=quant,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    return model
