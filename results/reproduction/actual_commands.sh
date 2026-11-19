#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/tsp-state-value
cd "$ROOT/operations/milestone-revision-check"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
PY="$ROOT/envs/milestone-cpu-repro/bin/python"
mkdir -p outputs
exec > >(tee outputs/verification.log) 2>&1
date -Iseconds
git rev-parse HEAD
"$PY" -m pip check
"$PY" -m unittest discover -s tests
"$PY" scripts/analyze_results.py --output outputs/analysis
"$PY" scripts/analyze_results.py --study results/interface_check/run --output outputs/interface_analysis
"$PY" scripts/replay_solvers.py --workers 4 --output outputs/replay
"$PY" scripts/check_results.py --analysis-proof outputs/analysis/analysis_verification.json --replay-proof outputs/replay/replay_verification.json --output outputs/checks
"$PY" scripts/build_state_bank.py --output outputs/mechanism/bank
"$PY" scripts/measure_feedback.py --root outputs --phase pilot --output outputs/mechanism/bank
"$PY" scripts/measure_feedback.py --root outputs --phase full --output outputs/mechanism/bank
"$PY" scripts/analyze_feedback.py --prepare outputs/mechanism/bank --output outputs/mechanism/analysis
"$PY" scripts/plot_results.py --output outputs/figures
"$PY" - <<'PY'
from pathlib import Path
import json,sys
sys.path.insert(0,'scripts')
from analyze_results import read
fresh=read('outputs/analysis/study_summary.json');old=read('results/discovery/analysis/study_summary.json')
for key in ('arm_mean_gap_percent','baseline_mean_gap_percent','contrasts','lineages'):
    assert fresh[key]==old[key],key
assert read('outputs/analysis/mechanism_summary.json')==read('results/discovery/analysis/mechanism_summary.json')
a=read('outputs/mechanism/analysis/summary.json');b=read('results/mechanism/analysis/summary.json')
for key in ['overall','paired_instance_bootstrap','controls','signal_gate_passed']:
    if key in b:assert a[key]==b[key],key
print('REVISED_REPOSITORY_REPRODUCTION_PASSED')
PY
date -Iseconds

# Additional CPU comparison
"$PY" - <<'PY'
from pathlib import Path
import json,subprocess
code=Path.cwd()
def read(p):return json.loads(p.read_text())
def numerical(x):
    if isinstance(x,dict):return {k:numerical(v) for k,v in x.items() if k not in ('wall_seconds','worker_cpu_seconds')}
    if isinstance(x,list):return list(map(numerical,x))
    return x
original=code/'results/mechanism/bank/measurements';rebuilt=code/'outputs/mechanism/bank/measurements'
branches=0
for p in sorted(original.glob('*.json')):
    a,b=read(p),read(rebuilt/p.name)
    assert numerical(a)==numerical(b),p.name
    branches+=a['branches_executed']
proof=dict(status='passed',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),states=len(list(original.glob('*.json'))),
  exact_branches=branches,exact_fields='All numerical fields, routes, actions, program hashes and budgets; only wall/CPU timing fields excluded',
  nested_pilot=read(code/'outputs/mechanism/bank/pilot_completed.json'),additional_full=read(code/'outputs/mechanism/bank/full_completed.json'))
assert branches==56832
(code/'outputs/mechanism_replay_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print('All',branches,'remeasured branches exactly match original numerical outcomes')

PY

# Optional model/reference entry check (existing GPU environment)
"$ROOT/envs/milestone/bin/python" scripts/setup_references.py --root "$ROOT"
"$ROOT/envs/milestone/bin/python" - <<'PY'
import ast,json,os,sys,time
from pathlib import Path
sys.path.insert(0,'scripts')
from experiment_io import read,write,CODE
from reference_solvers import LEHDReference,lkh_solve
from qwen_model import load_base
from transformers import AutoTokenizer
import torch,numpy as np
root=Path('/root/autodl-tmp/tsp-state-value');begin=time.perf_counter()
row=read(CODE/'results/mechanism/bank/program_dev_eval_instances.json')[0]
lkh=lkh_solve(root,row['coords'],0)
print('LKH smoke result keys',list(lkh),flush=True)
model=LEHDReference(root,device='cuda')
lehd=model.solve(row['coords'],0,rrc_budget=0)
for label,result in [('LKH',lkh),('LEHD_RRC0',lehd)]:
    saved=next(r for r in read(CODE/f'results/discovery/references/{label}.json')['results'] if r['instance_id']==row['instance_id'] and r['seed']==0)
    assert saved['best_tour']==result['best_tour'],label
    assert saved['original_length']==result['original_length'],label
print('LKH and LEHD fixture tours match published records exactly',flush=True)
del model
torch.set_default_tensor_type("torch.FloatTensor")
torch.cuda.empty_cache()
manifest=read(root/'artifacts/qwen_download.json')
tok=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True)
qwen=load_base(manifest['snapshot_path'],False).eval().requires_grad_(False)
qwen.config.use_cache=True
assert not any(p.requires_grad for p in qwen.parameters())
batch=tok('def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):\n    return ',return_tensors='pt').to('cuda')
with torch.inference_mode():output=qwen.generate(**batch,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
proof=dict(status='passed',scope='existing author assets and frozen-model loading plus an eight-token synthetic code completion; no effectiveness experiment',
  reference_fixture_routes_exact=True,lkh_fixture_length=lkh['original_length'],lehd_fixture_length=lehd['original_length'],qwen_revision=manifest['config']['revision'],qwen_trainable_parameters=sum(p.numel() for p in qwen.parameters() if p.requires_grad),
  synthetic_response=tok.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True),
  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-begin)
write(CODE/'outputs/optional_entry_check.json',proof);print(json.dumps(proof,indent=2))

PY
