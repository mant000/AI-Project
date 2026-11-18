"""Pinned external solvers. Author LEHD search is called unchanged, without reference tours."""
import contextlib
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import types
import numpy as np
import torch
from stage3_common import CODE, read, digest
from tsp_value.data import distances, length, validate_tour


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


class SilentLogger:
    def info(self,*args,**kwargs):pass


class FixtureClock:
    def get_est_string(self,*args,**kwargs):return 'wrapper','wrapper'


class LEHDReference:
    """Use original _test_one_batch and repair acceptance; replace I/O/logging setup only."""
    def __init__(self, root, device='cpu'):
        self.root=root;self.device=device;cfg=read(CODE/'configs/external_baselines_v1.json')['lehd']
        complete=read(CODE/'results/baselines/prepare-v1/lehd_completed.json')
        for f in complete['files']:assert digest(Path(f['path']))==f['sha256']
        folder=root/'external/lehd';self.cfg=cfg
        if device=='cuda':torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:torch.set_default_tensor_type('torch.FloatTensor')
        model_module=load_module('LEHD.TSP.TSPModel',folder/'TSP/TSPModel.py')
        env_module=load_module('LEHD.TSP.TSPEnv',folder/'TSP/TSPEnv.py')
        # Constructor-only logging utilities are bypassed; the original search methods run unchanged.
        for name in ('LEHD','LEHD.TSP','LEHD.utils'):
            if name not in sys.modules:sys.modules[name]=types.ModuleType(name)
        utility=types.ModuleType('LEHD.utils.utils');utility.__all__=[];sys.modules['LEHD.utils.utils']=utility
        author=load_module('LEHD.TSP.TSPTester',folder/'TSP/TSPTester.py').TSPTester
        class CapturedTester(author):
            def decide_whether_to_repair_solution(self,*args,**kwargs):
                route=super().decide_whether_to_repair_solution(*args,**kwargs)
                self.captured_route=route.detach().clone();return route
        self.tester_class=CapturedTester;self.env_class=env_module.TSPEnv
        self.model=model_module.TSPModel(**cfg['model_parameters']).to(device).eval()
        # Inspected old author pickle also uses scheduler Counter and Python2's set name.
        # Add only these harmless constructors temporarily; arbitrary globals stay disabled.
        allowed=torch._weights_only_unpickler._get_allowed_globals();previous=dict(allowed)
        try:
            allowed['collections.Counter']=Counter
            allowed['__builtin__.set']=set
            checkpoint=torch.load(folder/cfg['checkpoint'],map_location=device,weights_only=True)
        finally:
            allowed.clear();allowed.update(previous)
        self.model.load_state_dict(checkpoint['model_state_dict']);self.model.requires_grad_(False)

    def solve(self, coords, seed, rrc_budget=10, distance_mode='continuous_euclidean', placeholder=None):
        begin,cpu=time.perf_counter(),time.process_time()
        n=len(coords);xy=np.asarray(coords,dtype=np.float64);original=xy.copy()
        normalized=distance_mode=='tsplib_euc_2d'
        if normalized:xy=(xy-xy.min())/max(float(xy.max()-xy.min()),1e-12)
        torch.manual_seed(seed)
        if self.device=='cuda':torch.cuda.manual_seed_all(seed)
        tester=object.__new__(self.tester_class);tester.model=self.model
        tester.env_params=dict(mode='test',data_path='',sub_path=False,RRC_budget=rrc_budget)
        tester.env=self.env_class(**tester.env_params);tester.logger=SilentLogger()
        tester.env.raw_data_nodes=torch.tensor(xy[None],dtype=torch.float32,device=self.device)
        dummy=np.arange(n) if placeholder is None else np.asarray(placeholder);validate_tour(dummy,n)
        tester.env.raw_data_tours=torch.tensor(dummy[None],dtype=torch.long,device=self.device)
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured),torch.inference_mode():
            tester._test_one_batch(0,1,clock=FixtureClock())
        if self.device=='cuda':torch.cuda.synchronize()
        route=(tester.captured_route if rrc_budget else tester.env.selected_node_list)[0].cpu().numpy()
        validate_tour(route,n);cost=length(route,distances(original,distance_mode))
        return dict(best_tour=route.tolist(),original_length=cost,seed=seed,rrc_budget=rrc_budget,
            wall_seconds=time.perf_counter()-begin,process_cpu_seconds=time.process_time()-cpu,device=self.device,
            model_input_reference_tour=False,placeholder_route_only_for_author_logging=True,
            search='author greedy plus original continuous-distance RRC',tsplib_global_minmax_normalization=normalized,
            final_quality_distance_mode=distance_mode,
            author_internal_placeholder_logs_not_quality=captured.getvalue())


def lkh_solve(root,coords,seed,distance_mode='continuous_euclidean',max_trials=1000):
    """Integer input approximation only for continuous data; score returned tour on true distances."""
    begin,cpu=time.perf_counter(),time.process_time()
    meta=read(CODE/'results/baselines/prepare-v1/completed.json');binary=Path(meta['lkh_binary'])
    assert digest(binary)==meta['lkh_binary_sha256'];n=len(coords);d=distances(coords,distance_mode)
    scale=1 if distance_mode=='tsplib_euc_2d' else 1000000
    integer=np.floor(d*scale+.5).astype(np.int64);np.fill_diagonal(integer,0)
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(prefix='tsp-lkh-',dir=root/'tmp') as directory:
        work=Path(directory);problem=work/'problem.tsp';tour=work/'route.tour';params=work/'run.par'
        problem.write_text(f'NAME : course_fixture\nTYPE : TSP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n'+
            '\n'.join(' '.join(map(str,row)) for row in integer)+'\nEOF\n')
        params.write_text(f'PROBLEM_FILE = {problem}\nOUTPUT_TOUR_FILE = {tour}\nRUNS = 1\nMAX_TRIALS = {max_trials}\nSEED = {seed+1}\nTRACE_LEVEL = 1\n')
        proc=subprocess.run([str(binary),str(params)],capture_output=True,text=True,timeout=180)
        assert proc.returncode==0 and tour.exists(),proc.stdout[-2000:]+proc.stderr
        raw=tour.read_text();section=raw.split('TOUR_SECTION',1)[1];route=[]
        for value in section.split():
            if value in ('-1','EOF'):break
            route.append(int(value)-1)
        validate_tour(route,n);cost=length(route,d);integer_cost=int(integer[np.asarray(route),np.roll(route,-1)].sum())
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(best_tour=route,original_length=cost,seed=seed,lkh_seed=seed+1,max_trials=max_trials,
        integer_input_scale=scale,integer_input_route_cost=integer_cost,original_distance_mode=distance_mode,
        input_quantization_route_error_bound=n*.5/scale if scale!=1 else 0.,
        wall_seconds=time.perf_counter()-begin,parent_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
        stdout=proc.stdout,stderr=proc.stderr,source_mirror_commit=meta['lkh_mirror_commit'])
