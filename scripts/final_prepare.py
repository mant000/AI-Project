"""Open the previously identity-audited final data only after method/protocol freeze."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time
import zipfile
import numpy as np
from stage3_common import CODE,read,write,digest,stamp
from tsp_value.data import parse_author,distances,length,validate_tour,coordinate_hash


def verify_freeze(freeze):
    assert freeze['status']=='frozen' and freeze['method_changes_after_final_prohibited']
    for name,h in freeze['files_sha256'].items():assert digest(CODE/name)==h,name


def parse_invit_reference(raw,n):
    fields=raw.decode().strip().split();assert len(fields)==3
    tour=np.asarray([int(x) for x in fields[0].split(',')],dtype=np.int64)
    if len(tour)==n+1:assert tour[-1]==tour[0];tour=tour[:-1]
    if np.array_equal(np.sort(tour),np.arange(1,n+1)):tour=tour-1
    validate_tour(tour,n)
    return tour,float(fields[1]),float(fields[2])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--freeze',type=Path,default=CODE/'results/final/freeze-v1/protocol_frozen.json');a=ap.parse_args()
    start,cpu=time.perf_counter(),time.process_time();freeze=read(a.freeze);verify_freeze(freeze)
    out=a.root/'data/prepared/final-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        old=read(out/'completed.json');assert old['freeze_sha256']==digest(a.freeze) and old['data_sha256']==digest(out/'instances.jsonl')
        print('reused verified prepared final data',flush=True);return
    manifest=list(csv.DictReader((CODE/'splits/manifest.csv').open()))
    selected=[r for r in manifest if r['project_use'].startswith('final')];assert len(selected)==1194
    rows={r['instance_id']:r for r in selected};source_bytes={};coords={};refs={};reference_meta={};source_checks={}
    expected=read(CODE/'configs/joshi_received_files.json');policy=read(CODE/'configs/final_identity_policy.json')
    for n in (50,100,200):
        member=f'data/tsp/tsp{n}_test_concorde.txt';path=a.root/'data/author_received'/member
        assert digest(path)==expected['files'][member]['sha256'];source_checks[member]=digest(path)
        with path.open('rb') as f:
            for i in range(1,policy['joshi_test_rows_per_size']+1):
                raw=f.readline();key=f'joshi:{Path(member).name}:{i}';xy,tour=parse_author(raw,n)
                assert hashlib.sha256(raw).hexdigest()==rows[key]['source_row_sha256']
                coords[key]=xy;refs[key]=tour
    invit=a.root/'data/raw/INViT_data.zip';assert digest(invit)==policy['invit_archive_sha256'];source_checks['INViT_data.zip']=digest(invit)
    with zipfile.ZipFile(invit) as z:
        for distribution in policy['invit_distributions']:
            member=f'data/data_farm/tsp/tsp100/tsp100_{distribution}.txt';raw_file=z.read(member)
            solution=f'data/solution_farm/tsp100_{distribution}/Gurobi.txt';raw_solutions=z.read(solution)
            source_checks[solution]=hashlib.sha256(raw_solutions).hexdigest()
            raw_rows=raw_file.splitlines(keepends=True);sols=raw_solutions.splitlines(keepends=True)
            assert len(sols)>=policy['invit_rows_per_distribution']
            for i in range(1,policy['invit_rows_per_distribution']+1):
                raw=raw_rows[i-1];key=f'invit:{Path(member).name}:{i}'
                assert hashlib.sha256(raw).hexdigest()==rows[key]['source_row_sha256']
                xy=np.asarray([list(map(float,p.split(','))) for p in raw.decode().split()]);tour,stored,author_time=parse_invit_reference(sols[i-1],100)
                recomputed=length(tour,distances(xy));error=abs(stored-recomputed)
                assert error<=1e-5*max(1.,recomputed)+1e-6,(key,error)
                coords[key]=xy;refs[key]=tour;reference_meta[key]=dict(author_stored_length=stored,
                    author_length_recompute_abs_error=error,author_reference_seconds=author_time,
                    reference_row_sha256=hashlib.sha256(sols[i-1]).hexdigest())
    tsplib=a.root/'data/raw/ALL_tsp.tar.gz';assert digest(tsplib)==policy['tsplib_archive_sha256'];source_checks['ALL_tsp.tar.gz']=digest(tsplib)
    best_path=a.root/'data/raw/tsplib_best_known.json';best=read(best_path)
    assert digest(best_path)==freeze['tsplib_reference_file_sha256'];source_checks['tsplib_best_known.json']=digest(best_path)
    with tarfile.open(tsplib,'r:gz') as tar:
        for row in selected:
            if row['source']!='tsplib':continue
            content=gzip.decompress(tar.extractfile(row['source_file']).read());assert hashlib.sha256(content).hexdigest()==row['source_row_sha256']
            lines=content.decode().split('NODE_COORD_SECTION',1)[1].split('EOF')[0].strip().splitlines()
            xy=np.asarray([[float(x) for x in s.split()[1:3]] for s in lines]);coords[row['instance_id']]=xy
    records=[]
    for row in selected:
        key=row['instance_id'];xy=coords[key];n=int(row['nodes']);assert xy.shape==(n,2) and np.isfinite(xy).all()
        assert coordinate_hash(xy)==row['coordinate_set_sha256'] and coordinate_hash(xy,True)==row['coordinate_float32_sha256']
        mode=row['distance_convention'];tour=refs.get(key)
        reference=float(best[Path(row['source_file']).name.removesuffix('.gz').removesuffix('.tsp')]) if tour is None else length(tour,distances(xy,mode))
        assert reference>0 and np.isfinite(reference)
        group=(f'joshi/{n}' if row['source']=='joshi' else 'invit/'+Path(row['source_file']).stem.removeprefix('tsp100_') if row['source']=='invit' else
            'tsplib/'+('50-100' if n<=100 else '101-200' if n<=200 else '201-500'))
        records.append(dict(instance_id=key,source=row['source'],group=group,nodes=n,project_use=row['project_use'],
            distance_convention=mode,coords=xy.tolist(),reference_tour=None if tour is None else tour.tolist(),reference_length=reference,
            coordinate_set_sha256=row['coordinate_set_sha256'],coordinate_float32_sha256=row['coordinate_float32_sha256'],
            source_row_sha256=row['source_row_sha256'],reference_metadata=reference_meta.get(key,{})))
    path=out/'instances.jsonl'
    with path.open('w') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    assert len(records)==1194 and len({r['coordinate_set_sha256'] for r in records})==1194
    project={r['coordinate_set_sha256'] for r in manifest if not r['project_use'].startswith('final')}
    assert not project&{r['coordinate_set_sha256'] for r in records}
    result=dict(status='completed',at_utc=stamp(),freeze_sha256=digest(a.freeze),data_sha256=digest(path),
        original_instances=len(records),source_checks=source_checks,groups={g:sum(r['group']==g for r in records) for g in sorted({r['group'] for r in records})},
        reference_costs_opened_after_method_freeze=True,wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu)
    write(out/'completed.json',result);write(CODE/'results/final/freeze-v1/data_opened.json',result)
    print('prepared',len(records),'final instances',flush=True)


if __name__=='__main__':main()
