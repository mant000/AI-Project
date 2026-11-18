"""Export every research model, with exact tensor equality and no optimizer pickle requirement."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from safetensors.torch import save_file,load_file
from stage3_common import CODE,read,write,digest,stamp

BASE='Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=CODE/'trained_models');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);begin,cpu=time.perf_counter(),time.process_time();models=[]
    for category in ('stage2/training-v1','teacher-v2','history-ablation-v1'):
        for path in sorted((a.root/'checkpoints'/category).glob('*/best.pt')):
            original=digest(path);ck=torch.load(path,map_location='cpu');tensors={k:v.detach().cpu().contiguous() for k,v in ck['model'].items()}
            folder=a.output/'value'/category/path.parent.name;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'model.safetensors';save_file(tensors,str(target),metadata={'original_checkpoint_sha256':original})
            restored=load_file(str(target));assert restored.keys()==tensors.keys() and all(torch.equal(restored[k],v) for k,v in tensors.items())
            kind='transformer' if category.startswith('history') else path.parent.name.split('-seed')[0]
            config=dict(model_kind=kind,architecture=ck['config']['architecture'],feature_encoding='src/tsp_value/value_model.py:state_features',
                original_checkpoint_relative_path=path.relative_to(a.root).as_posix(),original_checkpoint_sha256=original,
                state_dict_sha256=digest(target),tensor_reload_exact=True,optimizer_in_export=False,
                original_optimizer_and_rng_preserved_in_closed_phase_backups=True)
            write(folder/'config.json',config)
            models.append(dict(name=f'value/{category}/{path.parent.name}',kind=kind,weights=target.relative_to(CODE).as_posix(),
                sha256=digest(target),original_checkpoint_sha256=original,tensor_count=len(tensors),tensor_reload_exact=True))
    adapters=[('sft',a.root/'checkpoints/sft-v1/epoch2')]
    adapters.extend((f'grpo-{arm}-{seed}',a.root/f'checkpoints/grpo-v1/{arm}-seed{seed}/final/policy') for arm in ('terminal','state') for seed in (701,702,703))
    for name,path in adapters:
        folder=a.output/'adapters'/name;folder.mkdir(parents=True,exist_ok=True)
        target=folder/'adapter_model.safetensors';shutil.copyfile(path/'adapter_model.safetensors',target)
        assert digest(target)==digest(path/'adapter_model.safetensors')
        tensors=load_file(str(target));original_cfg=read(path/'adapter_config.json');cfg=dict(original_cfg)
        cfg.update(base_model_name_or_path=BASE,revision=REVISION)
        write(folder/'adapter_config.json',cfg);write(folder/'original_adapter_config.json',original_cfg)
        write(folder/'provenance.json',dict(base=BASE,base_revision=REVISION,
            original_directory_relative_path=path.relative_to(a.root).as_posix(),
            original_config_sha256=digest(path/'adapter_config.json'),original_weights_sha256=digest(target),
            public_config_only_replaces_machine_local_base_path=True,numerical_tensors_unchanged=True,
            selected_by='SFT epoch2 predeclared validation NLL; GRPO last predefined group, all seeds retained'))
        models.append(dict(name=name,kind='lora_adapter',weights=target.relative_to(CODE).as_posix(),sha256=digest(target),
            tensor_count=len(tensors),original_weights_byte_identical=True,base=BASE,base_revision=REVISION))
    assert len(models)==28 and sum(r['kind']=='lora_adapter' for r in models)==7
    files=[p for p in a.output.rglob('*') if p.is_file() and p.name!='manifest.json']
    write(a.output/'manifest.json',dict(status='verified',at_utc=stamp(),models=models,files_sha256={p.relative_to(CODE).as_posix():digest(p) for p in files},
        own_trained_models_only=True,third_party_full_base_or_LEHD_weights_rehosted=False,all_model_and_optimization_seeds_retained=True,
        value_tensor_reload_exact=True,adapter_original_bytes_exact=True,total_bytes=sum(p.stat().st_size for p in files),
        wall_seconds=time.perf_counter()-begin,parent_cpu_seconds=time.process_time()-cpu))
    print('exported and verified',len(models),'models',sum(p.stat().st_size for p in files),'bytes',flush=True)


if __name__=='__main__':main()
