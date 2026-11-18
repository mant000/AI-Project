"""Fixed all-edge ensemble baseline for online GLS, with explicit feature/inference cost."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from .solver import clone,select_edge
from .value_model import EdgeValue,state_features,collate


def choose_edge(edges, reference, mean, std, threshold=.001):
    """Reference fallback and stable edge order, with no value-based program selection."""
    mean,std=np.asarray(mean),np.asarray(std)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():raise ValueError('Non-finite teacher prediction')
    eligible=np.flatnonzero(mean>np.maximum(threshold,std))
    if not len(eligible):return reference
    maximum=mean[eligible].max();return tuple(edges[int(eligible[np.flatnonzero(mean[eligible]==maximum)[0]])])


class NeuralDirect:
    def __init__(self,root,device='cuda',threshold=.001):
        self.models=[];self.device=device;self.threshold=threshold;self.checkpoints=[]
        for seed in (101,202,303):
            path=Path(root)/f'checkpoints/stage2/training-v1/transformer-seed{seed}/best.pt'
            summary=Path(root)/f'code/results/stage2/training-v1/transformer-seed{seed}_summary.json'
            expected=json.loads(summary.read_text())['checkpoint_sha256'];actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==expected
            ck=torch.load(path,map_location='cpu');model=EdgeValue('transformer',**ck['config']['architecture']).to(device).eval()
            model.load_state_dict(ck['model']);model.requires_grad_(False);self.models.append(model)
            self.checkpoints.append(dict(seed=seed,sha256=actual))
        self.reset_profile()

    def reset_profile(self):
        self.profile=dict(calls=0,changed_actions=0,feature_wall_seconds=0.,inference_wall_seconds=0.,
            total_wall_seconds=0.,host_cpu_seconds=0.,cuda_event_milliseconds=0.)

    def __call__(self,d,p,state):
        begin,cpu=time.perf_counter(),time.process_time();ref=select_edge(clone(state))
        edges=sorted({tuple(sorted((int(a),int(b)))) for a,b in zip(state['tour'],np.roll(state['tour'],-1))}-{ref})+[ref]
        features=state_features(state,edges);batch=collate([features],self.device)
        feature_seconds=time.perf_counter()-begin;infer=time.perf_counter()
        if self.device=='cuda':
            first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);first.record()
        with torch.inference_mode():values=torch.stack([model(batch)[0] for model in self.models])
        cuda_ms=0.
        if self.device=='cuda':last.record()
        predictions=values.cpu().numpy()
        if self.device=='cuda':torch.cuda.synchronize();cuda_ms=float(first.elapsed_time(last))
        inference_seconds=time.perf_counter()-infer
        chosen=choose_edge(edges,ref,predictions.mean(0),predictions.std(0),self.threshold)
        if chosen==ref:
            result=d/(1+p)  # Preserve actual conventional RNG/tie behavior on fallback.
        else:
            result=np.asarray([float(tuple(sorted((int(a),int(b))))==chosen)
                for a,b in zip(state['tour'],np.roll(state['tour'],-1))])
            assert result.sum()==1
        self.profile['calls']+=1;self.profile['changed_actions']+=int(chosen!=ref)
        self.profile['feature_wall_seconds']+=feature_seconds;self.profile['inference_wall_seconds']+=inference_seconds
        self.profile['total_wall_seconds']+=time.perf_counter()-begin;self.profile['host_cpu_seconds']+=time.process_time()-cpu
        self.profile['cuda_event_milliseconds']+=cuda_ms
        return result
