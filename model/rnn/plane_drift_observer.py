"""Causal drift observer that accumulates bimanual plane measurements.

No test labels or mesh are accepted by forward. The observer tracks a world-space
translation correction; changing normals constrain additional directions.
"""
from __future__ import annotations
import torch
from torch import nn


def plane_accumulate(normals, measurements, weights, decay, state=None, ridge=.5,
                     max_correction=.5, vector_targets=None):
    """Exponentially weighted rank-one normal equations, exact across chunks.

state stores unregularized information A and right-hand side z. Forgetting is
stopped outside contact by passing decay=1. No inverse is formed explicitly.
"""
    b,t,_=normals.shape
    if state is None:
        a0=normals.new_zeros(b,3,3);z0=normals.new_zeros(b,3)
    else:a0,z0=state
    eye=torch.eye(3,device=normals.device,dtype=normals.dtype)
    if vector_targets is None:
        info=weights[...,None,None]*normals[..., :,None]*normals[...,None,:]
        rhs=weights[...,None]*normals*measurements[...,None]
    else:
        info=weights[...,None,None]*eye
        rhs=weights[...,None]*vector_targets
    powers=decay.cumprod(1).clamp_min(1e-12)
    a=powers[...,None,None]*(a0[:,None]+(info/powers[...,None,None]).cumsum(1))
    z=powers[...,None]*(z0[:,None]+(rhs/powers[...,None]).cumsum(1))
    correction=torch.linalg.solve(a+ridge*eye,z.unsqueeze(-1)).squeeze(-1)
    norm=correction.norm(dim=-1,keepdim=True)
    correction=correction*torch.clamp(max_correction/norm.clamp_min(1e-8),max=1.)
    return correction,(a[:,-1],z[:,-1]),a


class PlaneDriftObserver(nn.Module):
    def __init__(self,feature_dim,num_categories,vector_mode=False,hidden_dim=64,
                 memory_frames=90.,ridge=.5):
        super().__init__()
        self.vector_mode=vector_mode;self.memory_frames=float(memory_frames);self.ridge=float(ridge)
        self.embedding=nn.Embedding(num_categories,12)
        self.encoder=nn.Sequential(nn.Linear(feature_dim+12,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim))
        self.rnn=nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        self.measurement_head=nn.Linear(hidden_dim,3 if vector_mode else 1)
        self.trust_head=nn.Linear(hidden_dim,1)
        nn.init.zeros_(self.measurement_head.weight);nn.init.zeros_(self.measurement_head.bias)
        nn.init.zeros_(self.trust_head.weight);nn.init.constant_(self.trust_head.bias,-2.)

    def forward(self,x,category,palms,base,rotation,contact_prob,eligible,valid,
                state=None):
        e=self.embedding(category)[:,None].expand(-1,x.shape[1],-1)
        previous_hidden=None if state is None else state[0]
        previous_information=None if state is None else state[1]
        h,hidden=self.rnn(self.encoder(torch.cat((x,e),-1)),previous_hidden)
        direction=palms[:,:,1]-palms[:,:,0]
        span=direction.norm(dim=-1)
        normal=direction/span[...,None].clamp_min(1e-6)
        midpoint=palms.mean(-2)
        signed=((base-midpoint)*normal).sum(-1)
        confidence=((contact_prob[...,:2].amin(-1)-.35)/.55).clamp(0,1)
        active=confidence*eligible[:,None]*valid*(span>=.12)
        if state is None:
            active=active.clone();active[:,0]=0.
        trust=.02+.98*torch.sigmoid(self.trust_head(h).squeeze(-1))
        weights=active*trust
        decay=torch.exp(-active/self.memory_frames)
        raw=self.measurement_head(h)
        if self.vector_mode:
            local=.5*torch.tanh(raw)
            vector=torch.einsum("btij,btj->bti",rotation,local)
            target=None
            correction,information,a=plane_accumulate(normal,signed*0,weights,decay,previous_information,self.ridge,vector_targets=vector)
            target_value=local
        else:
            # Effective plane offset includes bias of predicted palm anchors.
            target=.3*torch.tanh(raw.squeeze(-1))
            measurement=(target-signed).clamp(-.5,.5)
            correction,information,a=plane_accumulate(normal,measurement,weights,decay,previous_information,self.ridge)
            target_value=target
        return {"correction":correction,"state":(hidden,information),"normal":normal,
                "signed_base":signed,"trust":trust,"target":target_value,
                "information":a,"active":active}
