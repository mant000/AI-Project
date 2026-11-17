"""Audit received author files and prepare disjoint project validation views.

The received archive is a repack, not the original author's gzip file. Hashes
identify these exact received files. Author test is only hashed for integrity;
no test coordinates, route costs or solver results are used for development.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import struct
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def file_info(path):
    """Hash and count complete text lines using bounded memory."""
    digest=hashlib.sha256()
    count=0
    last=b''
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):
            digest.update(block)
            count+=block.count(b'\n')
            last=block[-1:]
    assert last==b'\n', f'Incomplete last line: {path}'
    return {'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'lines':count}


def parse_row(raw,nodes):
    """Validate source coordinates and the closed reference permutation."""
    assert raw.endswith(b'\n'), 'Incomplete source row'
    tokens=raw.decode('utf-8').split()
    split=tokens.index('output')
    assert split==2*nodes
    values=[float(x) for x in tokens[:split]]
    assert all(math.isfinite(x) and 0<=x<=1 for x in values)
    tour=[int(x)-1 for x in tokens[split+1:]]
    assert len(tour)==nodes+1 and tour[0]==tour[-1]
    assert sorted(tour[:-1])==list(range(nodes))
    coordinates=sorted(zip(values[::2],values[1::2]))
    canonical=b''.join(struct.pack('<dd',x,y) for x,y in coordinates)
    return hashlib.sha256(raw).hexdigest(),hashlib.sha256(canonical).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True,help='Directory containing data/tsp')
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-prefix',type=Path,required=True)
    parser.add_argument('--write-file-manifest',type=Path)
    parser.add_argument('--extract',action='store_true',help='Verify SHA and safely extract with 7z; extraction checks member CRCs')
    args=parser.parse_args()
    code=Path(__file__).resolve().parents[1]
    policy=json.loads((code/'configs/data_split_policy.json').read_text(encoding='utf-8'))
    manifest_path=args.write_file_manifest or code/'configs/joshi_received_files.json'
    artifact_dir=args.project_root/'artifacts'
    artifact_dir.mkdir(parents=True,exist_ok=True)
    output=artifact_dir/'joshi_received_audit.json'
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_kind':'user-provided repack of author files',
            'original_author_gzip_sha256_verified':False,
            'directory':str(args.directory.resolve()),'archive_path':str(args.archive.resolve()),
            'archive_integrity_verified':False,'research_evaluation_performed':False}

    def save():
        report['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        temporary=output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        temporary.replace(output)

    try:
        save()
        with args.archive.open('rb') as source:
            digest=hashlib.file_digest(source,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
        if digest is None:
            h=hashlib.sha256()
            with args.archive.open('rb') as source:
                for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
            digest=h.hexdigest()
        is_seven_zip=args.archive.suffix=='.7z'
        report.update(archive_bytes=args.archive.stat().st_size,archive_sha256=digest,
                      archive_format='7z_repack' if is_seven_zip else 'author_tar_gzip')
        expected=None
        if not args.write_file_manifest:
            expected=json.loads(manifest_path.read_text(encoding='utf-8'))
            if is_seven_zip:
                assert report['archive_bytes']==expected['received_7z_bytes']
                assert digest==expected['received_7z_sha256'], 'Received archive SHA mismatch'
            else:
                assert report['archive_bytes']==5842160744, 'Unexpected author gzip archive size'
                with gzip.open(args.archive,'rb') as handle:
                    while handle.read(8*1024**2):pass
                report.update(archive_integrity_verified=True,archive_integrity_method='Read complete original gzip stream and verified CRC')
        if args.extract:
            assert is_seven_zip, 'Extract the original tar.gz with tar first'
            assert expected is not None, 'Extraction requires the expected file manifest'
            listing=subprocess.check_output(['7z','l','-slt',str(args.archive)],text=True)
            paths=re.findall(r'^Path = (.+)$',listing,re.MULTILINE)[1:]
            for name in paths:
                path=PurePosixPath(name.replace('\\','/'))
                assert not path.is_absolute() and '..' not in path.parts and path.parts[0]=='data'
                assert not re.match(r'^[A-Za-z]:',name)
            assert 'Symbolic Link =' not in listing and 'Hard Link =' not in listing
            args.directory.mkdir(parents=True,exist_ok=True)
            log=artifact_dir/'received_7z_extraction.log'
            with log.open('w',encoding='utf-8') as handle:
                subprocess.run(['7z','x','-y','-bd',f'-o{args.directory}',str(args.archive)],
                               stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True,timeout=1800)
            report.update(archive_integrity_verified=True,archive_integrity_method='7z extraction checked all member CRCs',
                          extraction_log=str(log.resolve()))
            save()
        files={}
        for path in sorted((args.directory/'data/tsp').rglob('*')):
            if path.is_file():
                name=path.relative_to(args.directory).as_posix()
                files[name]=file_info(path)
                print(f'Hashed {name}: {files[name]["bytes"]} bytes',flush=True)
        assert len(files)==11, 'Unexpected received payload inventory'
        if expected:
            assert files==expected['files'], 'Local and remote received-file hashes/counts differ'
        else:
            manifest_path.write_text(json.dumps({'source_kind':report['source_kind'],
                'upstream_repository':'https://github.com/chaitjo/learning-tsp',
                'upstream_revision':'bb28f5795924d99f7e7d945695e884f8d1f7df45',
                'original_author_gzip_sha256_verified':False,
                'received_7z_bytes':report['archive_bytes'],'received_7z_sha256':digest,'files':files},indent=2)+'\n',encoding='utf-8')
        report['files']=files
        assert not list((args.directory/'data/tsp').glob('*_val*')), 'Review policy if author val files are present'
        prefix_checks=[]
        with tarfile.open(args.archive_prefix,'r|gz') as prefix:
            for member in prefix:
                if member.name in ('data/tsp/tsp50_test_concorde.txt','data/tsp/tsp50_train_concorde.txt'):
                    target=args.directory/member.name
                    assert target.stat().st_size==member.size
                    amount=min(member.size,8*1024**2)
                    data=prefix.extractfile(member).read(amount)
                    with target.open('rb') as source:
                        assert source.read(amount)==data, 'Received file differs from independent author download prefix'
                    prefix_checks.append({'name':member.name,'matching_bytes':amount,'sha256':hashlib.sha256(data).hexdigest()})
                    if member.name.endswith('tsp50_train_concorde.txt'):break
        assert len(prefix_checks)==2
        report['independent_original_prefix_checks']=prefix_checks
        reserved={int(row['source_line']):row for row in csv.DictReader((code/'splits/bootstrap_reserved_rows.csv').open(encoding='utf-8'))}
        seen=set()
        pools={name:0 for name in policy['pools_per_size']}
        views=[]
        prepared=args.project_root/'data/prepared/project_val'
        prepared.mkdir(parents=True,exist_ok=True)
        for nodes in (50,100):
            source=args.directory/f'data/tsp/tsp{nodes}_train_concorde.txt'
            view=prepared/f'tsp{nodes}_val_from_author_train.txt'
            upper=max(b for a,b in policy['pools_per_size'].values())
            assert files[source.relative_to(args.directory).as_posix()]['lines']>=upper
            with source.open('rb') as handle,view.open('wb') as target:
                for line in range(1,upper+1):
                    raw=handle.readline()
                    selected=[name for name,(a,b) in policy['pools_per_size'].items() if a<=line<=b]
                    assert len(selected)<=1, 'Overlapping pool ranges'
                    if not selected:continue
                    pool=selected[0]
                    row_hash,coordinate_hash=parse_row(raw,nodes)
                    assert coordinate_hash not in seen, 'Duplicate coordinate set across project pools'
                    seen.add(coordinate_hash)
                    pools[pool]+=1
                    if nodes==50 and line in reserved:
                        assert row_hash==reserved[line]['source_row_sha256']
                        assert coordinate_hash==reserved[line]['coordinate_set_sha256']
                        assert pool=='value_train'
                    if pool in ('value_val','program_select','program_dev_eval'):target.write(raw)
            views.append({'nodes':nodes,'path':str(view.resolve()),**file_info(view),
                          'original_split':'train','project_use':'validation only; exclude these source rows from training/discovery'})
        assert len(reserved)==512
        report.update(status='completed_verified',project_pools_checked=pools,
            selected_unique_instances=len(seen),bootstrap_512_source_hashes_match=True,
            project_validation_views=views,author_validation_files_present=False,
            validation_policy=policy['validation_policy'],split_policy=policy,
            scope='Stage1 data prerequisites; full instance manifest, final-source overlap check and solver evaluation remain stage1 work')
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','split_policy')},indent=2),flush=True)


if __name__=='__main__':
    main()
