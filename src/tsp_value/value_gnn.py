"""Permutation-equivariant dense gated message passing control for edge values."""
import torch
from torch import nn
from .value_model import EdgeValue,PAIR_DIM


class DenseGNNBlock(nn.Module):
    def __init__(self,hidden=128,dropout=.1):
        super().__init__();self.norm1=nn.LayerNorm(hidden);self.norm2=nn.LayerNorm(hidden)
        self.edge_gate=nn.Linear(PAIR_DIM,1);self.message=nn.Linear(hidden,hidden)
        self.update=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.ff=nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*hidden,hidden))
        self.drop=nn.Dropout(dropout)

    def forward(self,x,pair,mask):
        z=self.norm1(x);gate=self.edge_gate(pair).squeeze(-1).sigmoid()
        gate=gate*mask[:,None,:]
        gate=gate.masked_fill(torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None],0)
        message=(gate@self.message(z))/gate.sum(-1,keepdim=True).clamp_min(1e-8)
        x=x+self.drop(self.update(torch.cat((z,message),dim=-1)))
        return x+self.drop(self.ff(self.norm2(x)))


def create_value_model(kind,architecture):
    if kind!='gnn':return EdgeValue(kind,**architecture)
    # Reuse the same global embedding and symmetric edge readout; replace attention only.
    model=EdgeValue('transformer',**architecture)
    model.blocks=nn.ModuleList([DenseGNNBlock(architecture['hidden'],architecture['dropout']) for _ in range(architecture['layers'])])
    model.research_kind='gnn'
    return model
