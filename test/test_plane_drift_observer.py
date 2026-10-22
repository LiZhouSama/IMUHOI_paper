"""Geometric and streaming tests for the accumulated plane observer."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.rnn.plane_drift_observer import plane_accumulate,PlaneDriftObserver


def test_single_plane_does_not_invent_tangent_correction():
    n=torch.tensor([[[1.,0,0]]]).expand(1,60,3)
    y=torch.full((1,60),.2);w=torch.ones(1,60);d=torch.ones(1,60)
    c,_,_=plane_accumulate(n,y,w,d,ridge=.5)
    torch.testing.assert_close(c[0,-1,0],torch.tensor(.2*60/60.5))
    assert torch.count_nonzero(c[...,1:])==0


def test_rotating_normals_recover_three_dimensional_bias():
    n=torch.eye(3).repeat(120,1)[None];truth=torch.tensor([.1,-.2,.3])
    y=(n*truth).sum(-1);w=torch.ones_like(y);d=torch.ones_like(y)
    c,_,a=plane_accumulate(n,y,w,d,ridge=.5)
    torch.testing.assert_close(c[0,-1],truth*(120/120.5),atol=1e-6,rtol=1e-6)
    assert torch.linalg.eigvalsh(a[0,-1]).min()>100


def test_chunk_and_prefix_equivalence_and_no_contact_carry():
    torch.manual_seed(1);n=torch.randn(2,45,3);n=n/n.norm(dim=-1,keepdim=True)
    y=torch.randn(2,45)*.05;w=torch.rand(2,45);d=torch.exp(-w/90)
    full,_,_=plane_accumulate(n,y,w,d)
    first,state,_=plane_accumulate(n[:,:17],y[:,:17],w[:,:17],d[:,:17])
    last,state,_=plane_accumulate(n[:,17:],y[:,17:],w[:,17:],d[:,17:],state)
    torch.testing.assert_close(full,torch.cat((first,last),1),atol=1e-6,rtol=1e-5)
    hold,_,_=plane_accumulate(n[:,:4],y[:,:4],w[:,:4]*0,d[:,:4]*0+1,state)
    torch.testing.assert_close(hold,last[:,-1:,].expand(-1,4,-1),atol=1e-6,rtol=1e-5)


def test_model_streaming_no_future_and_finite_gradient():
    torch.manual_seed(2);m=PlaneDriftObserver(9,2)
    x=torch.randn(2,23,9);cat=torch.tensor([0,1]);h=torch.randn(2,23,2,3)*.2;p=torch.randn(2,23,3)*.1
    R=torch.eye(3).expand(2,23,3,3);cp=torch.ones(2,23,3);elig=torch.ones(2,dtype=torch.bool);valid=torch.ones(2,23,dtype=torch.bool)
    a=m(x,cat,h,p,R,cp,elig,valid)
    b=m(x[:,:11],cat,h[:,:11],p[:,:11],R[:,:11],cp[:,:11],elig,valid[:,:11])
    c=m(x[:,11:],cat,h[:,11:],p[:,11:],R[:,11:],cp[:,11:],elig,valid[:,11:],b["state"])
    torch.testing.assert_close(a["correction"],torch.cat((b["correction"],c["correction"]),1),atol=1e-6,rtol=1e-5)
    assert torch.count_nonzero(a["correction"][:,0])==0
    a["correction"].square().mean().backward()
    assert m.measurement_head.weight.grad is not None and torch.isfinite(m.measurement_head.weight.grad).all()


def test_ineligible_category_is_unchanged():
    m=PlaneDriftObserver(9,2)
    x=torch.randn(1,8,9);h=torch.randn(1,8,2,3);p=torch.randn(1,8,3)
    out=m(x,torch.tensor([0]),h,p,torch.eye(3).expand(1,8,3,3),torch.ones(1,8,3),torch.tensor([False]),torch.ones(1,8,dtype=torch.bool))
    assert torch.count_nonzero(out["correction"])==0


def test_training_wrapper_inference_ignores_all_label_fields():
    from scripts.train_plane_drift_observer import forward
    torch.manual_seed(4);m=PlaneDriftObserver(9,2)
    b={"x":torch.randn(1,12,9),"cat":torch.tensor([0]),"h":torch.randn(1,12,2,3),"p":torch.randn(1,12,3)*.1,
       "R":torch.eye(3).expand(1,12,3,3),"cp":torch.ones(1,12,3),"eligible":torch.tensor([True]),"valid":torch.ones(1,12,dtype=torch.bool)}
    labels={**b,"gt":torch.randn(1,12,3),"gh":torch.randn(1,12,2,3),"contact":torch.zeros(1,12,2,dtype=torch.bool),"mesh":torch.randn(100,3)}
    torch.testing.assert_close(forward(m,b)["correction"],forward(m,labels)["correction"])
