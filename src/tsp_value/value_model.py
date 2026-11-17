"""Permutation-equivariant graph attention and a pooled-feature edge MLP."""
import numpy as np
import torch
from torch import nn

NODE_DIM, PAIR_DIM, GLOBAL_DIM = 6, 4, 6


def state_features(s, edges):
    """Encode only observable state and requested edges; never access labels.

    The final edge must be the common reference, as required by EdgeValue.
    This is the identical, unfitted numerical encoding used in stage 2.
    """
    xy=np.asarray(s['coords'],dtype=np.float32); n=len(xy)
    d=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    tour=np.asarray(s['tour']); best=np.asarray(s['best_tour']); p=np.asarray(s['penalties'],dtype=np.float32)
    route=np.zeros((n,n),dtype=np.float32); incumbent=route.copy()
    route[tour,np.roll(tour,-1)]=1; route=route+route.T
    incumbent[best,np.roll(best,-1)]=1; incumbent=incumbent+incumbent.T
    scale=max(s['best_length']/n,1e-8)
    node=np.column_stack(((xy-xy.mean(0))/max(float(xy.std()),1e-8),
        (route*d).sum(1)/scale, (route*d).max(1)/scale,
        np.log1p((route*p).sum(1)),np.log1p(p.sum(1))))
    pair=np.stack((d/scale,route,np.log1p(p),incumbent),axis=-1)
    glob=np.asarray([n/100,s['lambda_']/scale,s['best_length']/s['current_length'],
        np.log1p(s['penalty_events']),np.log1p(s['stagnation_sweeps']),
        np.log1p(s['delta_evaluations']/max(n*(n-3)/2,1))],dtype=np.float32)
    edges=np.asarray(edges,dtype=np.int64)
    utility=d[edges[:,0],edges[:,1]]/(1+p[edges[:,0],edges[:,1]])
    return dict(node=node.astype(np.float32),pair=pair.astype(np.float32),glob=glob,
                edges=edges,utility=utility.astype(np.float32))


def features(record):
    encoded=state_features(record['state'], record['candidates']+[record['reference_edge']])
    y=np.asarray([x['relative_benefit'] for x in record['records']],dtype=np.float64).reshape(2,-1).mean(0)
    y=np.r_[y,0].astype(np.float32)*100  # percentage points, not fractional gains
    return dict(**encoded,y=y)


def collate(items,device):
    n=max(len(x['node']) for x in items); b=len(items)
    node=np.zeros((b,n,NODE_DIM),np.float32); pair=np.zeros((b,n,n,PAIR_DIM),np.float32)
    mask=np.zeros((b,n),bool)
    for i,x in enumerate(items):
        k=len(x['node']); node[i,:k]=x['node']; pair[i,:k,:k]=x['pair']; mask[i,:k]=True
    result=dict(node=node,pair=pair,mask=mask,glob=np.stack([x['glob'] for x in items]),
                edges=np.stack([x['edges'] for x in items]))
    if all('y' in x for x in items): result['y']=np.stack([x['y'] for x in items])
    elif any('y' in x for x in items): raise ValueError('Mixed labelled/unlabelled batch')
    return {k:torch.as_tensor(v,device=device) for k,v in result.items()}


class GraphBlock(nn.Module):
    def __init__(self,hidden=128,heads=8,dropout=.1):
        super().__init__(); self.heads=heads; self.d=hidden//heads
        self.norm1=nn.LayerNorm(hidden); self.norm2=nn.LayerNorm(hidden)
        self.qkv=nn.Linear(hidden,3*hidden); self.bias=nn.Linear(PAIR_DIM,heads,bias=False)
        self.out=nn.Linear(hidden,hidden)
        self.ff=nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*hidden,hidden))
        self.drop=nn.Dropout(dropout)

    def forward(self,x,pair,mask):
        b,n,h=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,self.d).permute(2,0,3,1,4).unbind(0)
        logits=(q@k.transpose(-1,-2))/(self.d**.5)+self.bias(pair).permute(0,3,1,2)
        logits=logits.masked_fill(~mask[:,None,None,:],-1e9)
        a=self.drop(logits.softmax(-1))
        x=x+self.drop(self.out((a@v).transpose(1,2).reshape(b,n,h)))
        return x+self.drop(self.ff(self.norm2(x)))


class EdgeValue(nn.Module):
    def __init__(self,kind='transformer',hidden=128,layers=4,heads=8,dropout=.1):
        super().__init__(); self.kind=kind
        width=hidden if kind=='transformer' else NODE_DIM
        if kind=='transformer':
            self.embed=nn.Linear(NODE_DIM+GLOBAL_DIM,hidden)
            self.blocks=nn.ModuleList([GraphBlock(hidden,heads,dropout) for _ in range(layers)])
        # Identical symmetric edge/pair/global input, MLP omits graph message passing.
        self.readout=nn.Sequential(nn.Linear(3*width+PAIR_DIM+GLOBAL_DIM,hidden),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,b):
        x=b['node']; glob=b['glob']; mask=b['mask']
        if self.kind=='transformer':
            x=self.embed(torch.cat((x,glob[:,None,:].expand(-1,x.shape[1],-1)),dim=-1))
            for block in self.blocks: x=block(x,b['pair'],mask)
        pool=(x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        edges=b['edges']; idx=torch.arange(len(x),device=x.device)[:,None]
        u=x[idx,edges[:,:,0]]; v=x[idx,edges[:,:,1]]
        edgepair=b['pair'][idx,edges[:,:,0],edges[:,:,1]]
        z=torch.cat((u+v,(u-v).abs(),pool[:,None,:].expand(-1,u.shape[1],-1),
                     edgepair,glob[:,None,:].expand(-1,u.shape[1],-1)),dim=-1)
        score=self.readout(z).squeeze(-1)
        return score-score[:,-1:]


def objective(pred,y):
    mse=((pred-y)**2).mean()
    true_delta=y[:,:,None]-y[:,None,:]
    pred_delta=pred[:,:,None]-pred[:,None,:]
    # Tied labels never create artificial preference supervision.
    valid=true_delta.abs()>1e-7
    rank=torch.nn.functional.softplus(-true_delta.sign()*pred_delta/.1)
    rank=(rank*valid).sum()/valid.sum().clamp_min(1)
    return mse+.1*rank,mse,rank
