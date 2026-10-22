"""Causal bimanual position student with training-only geometric heads.

forward() accepts only observed features, a category token, and recurrent state.
Mesh targets never enter the deployed forward path.
"""
from __future__ import annotations
import torch
from torch import nn


class BimanualPriorStudent(nn.Module):
    def __init__(self, feature_dim, num_objects, temporal=True, use_category=True,
                 hidden_dim=64, max_correction=.25):
        super().__init__()
        self.temporal=temporal
        self.use_category=use_category
        self.max_correction=float(max_correction)
        self.embedding=nn.Embedding(num_objects if use_category else 1,12)
        self.encoder=nn.Sequential(nn.Linear(feature_dim+12,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim))
        self.core=nn.GRU(hidden_dim,hidden_dim,batch_first=True) if temporal else nn.Sequential(nn.Linear(hidden_dim,3*hidden_dim),nn.SiLU(),nn.Linear(3*hidden_dim,hidden_dim),nn.SiLU())
        self.delta_head=nn.Linear(hidden_dim,3)
        self.variance_head=nn.Linear(hidden_dim,3)
        self.geometry_head=nn.Linear(hidden_dim,3)
        self.patch_head=nn.Linear(hidden_dim,16)
        self.surface_head=nn.Linear(hidden_dim,12)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(self.variance_head.bias,-4.)

    def forward(self,x,category,state=None):
        if not self.use_category:
            category=torch.zeros_like(category)
        e=self.embedding(category)
        if e.ndim==2:e=e[:,None].expand(-1,x.shape[1],-1)
        z=self.encoder(torch.cat((x,e),dim=-1))
        if self.temporal:z,state=self.core(z,state)
        else:z=self.core(z);state=None
        raw=self.delta_head(z)
        norm=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
        delta=self.max_correction*raw/(1+norm)
        variance=torch.exp(self.variance_head(z).clamp(-9.,0.))
        return {"delta":delta,"variance":variance,"geometry":self.geometry_head(z),
                "patch_logits":self.patch_head(z).reshape(*z.shape[:-1],2,8),
                "surface":self.surface_head(z),"state":state}


def causal_filter(delta_world, contact_prob, variance, state=None, gain=1.,
                  uncertainty=True, carry=False):
    """Apply a bounded correction without looking ahead; optional state carry."""
    if delta_world.ndim!=3:raise ValueError("Expected B,T,3 corrections")
    previous=torch.zeros_like(delta_world[:,0]) if state is None else state
    confidence=((contact_prob[...,:2].amin(-1)-.35)/.55).clamp(0,1)
    precision=.01/(.01+variance.mean(-1)) if uncertainty else torch.ones_like(confidence)
    weight=confidence*precision
    result=[]
    for t in range(delta_world.shape[1]):
        if t==0 and state is None:
            result.append(previous)
            continue
        if carry:
            previous=previous+.25*weight[:,t,None]*(delta_world[:,t]*gain-previous)
        else:
            previous=.75*previous+.25*weight[:,t,None]*delta_world[:,t]*gain
        result.append(previous)
    return torch.stack(result,dim=1),previous


