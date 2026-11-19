"""Rebuild the new study's state bank from the small released input subset.

This replaces only the large-source allocation step. The original allocation
audit and input hash remain authoritative. Call measure_feedback.py afterwards
for pilot/full branch measurement, then analyze_feedback.py and the
expression discovery driver. No author test/final data are read.
"""
import argparse
import gzip
import json
from pathlib import Path
from experiment_io import CODE,read,write,digest,stamp
from state_feedback import collect_states
from tsp.data import coordinate_hash


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,default=CODE/'results/mechanism/bank/benchmark_inputs.jsonl')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists(),'choose a new output directory'
    reference=CODE/'results/mechanism/bank';frozen=read(reference/'bank_frozen.json')
    assert digest(a.inputs)==frozen['inputs_sha256'],'only released, previously allocated 256 inputs'
    cfg=read(CODE/'configs/state_feedback.json');rows=list(map(json.loads,a.inputs.read_text().splitlines()))
    assert len(rows)==len({r['instance_id'] for r in rows})==len({coordinate_hash(r['coords']) for r in rows})==256
    a.output.mkdir(parents=True);(a.output/'benchmark_inputs.jsonl').write_bytes(a.inputs.read_bytes())
    pools={role:[r for r in rows if r['project_use']==role] for role in cfg['roles']}
    for role,rr in pools.items():
        lo,hi=cfg['roles'][role];assert len(rr)==2*(hi-lo+1)
        assert all(lo<=r['source_line']<=hi for r in rr)
        write(a.output/f'{role}_instances.json',rr)
    bank=[]
    for item in pools['bank']:bank.extend(collect_states(item,cfg))
    with gzip.open(reference/'states.jsonl.gz','rt') as f:old=list(map(json.loads,f))
    assert bank==old,'state reconstruction differs from released execution'
    with gzip.open(a.output/'states.jsonl.gz','wt') as f:
        for row in bank:f.write(json.dumps(row)+'\n')
    for name in ('feedback_selection.json','selection.json'):(a.output/name).write_bytes((reference/name).read_bytes())
    write(a.output/'pool_isolation.json',dict(status='passed',portable_subset=True,source_allocation_audit_sha256=digest(reference/'pool_isolation.json'),inputs_sha256=digest(a.inputs),original_instances=256,final_opened=False))
    write(a.output/'bank_frozen.json',dict(at_utc=stamp(),states=len(bank),original_instances=128,
          config_sha256=digest(CODE/'configs/state_feedback.json'),bank_sha256=digest(a.output/'states.jsonl.gz'),inputs_sha256=digest(a.inputs),feedback_original_instances=128,final_opened=False))
    write(a.output/'portable_preparation.json',dict(status='passed',all_states_exact=True,states=len(bank),
          original_inputs_sha256=digest(a.inputs),source_sha256=digest(Path(__file__)),
          note='gzip container timestamps may differ; all decompressed state objects match exactly'))
    print('Reconstructed',len(bank),'states exactly from 256 released inputs')


if __name__=='__main__':main()
