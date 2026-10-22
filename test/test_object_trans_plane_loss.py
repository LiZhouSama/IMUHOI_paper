"""Tests for applying plane loss directly to native OT predictions."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train.rnn.loss.object_trans_plane_loss import bimanual_plane_penalty


def example():
    pos=torch.tensor([[[.15,.2,.3],[.03,.2,.3],[.2,.2,.3]]],requires_grad=True)
    hands=torch.tensor([[[[-.5,0.,0.],[.5,0.,0.]]]*3],requires_grad=True)
    pred={"pred_obj_trans":pos,"ot_hand_positions":hands}
    batch={"obj_name":["box"],"lhand_contact":torch.tensor([[1,1,1]],dtype=torch.bool),"rhand_contact":torch.tensor([[1,1,0]],dtype=torch.bool),"has_object":torch.tensor([True])}
    bands={"box":{"tolerance_m":.05,"eligible":True}}
    return pred,batch,bands


def test_plane_loss_moves_object_only_along_normal():
    p,b,s=example();loss,d=bimanual_plane_penalty(p,b,s);loss.backward()
    assert p["pred_obj_trans"].grad[0,0,0]>0
    assert torch.count_nonzero(p["pred_obj_trans"].grad[...,1:])==0
    assert torch.count_nonzero(p["pred_obj_trans"].grad[:,1:])==0
    assert p["ot_hand_positions"].grad is None


def test_history_mask_and_unknown_object_are_respected():
    p,b,s=example();p["ot_loss_mask"]=torch.tensor([[0,1,1]],dtype=torch.bool)
    loss,_=bimanual_plane_penalty(p,b,s);assert float(loss)==0
    b["obj_name"]=["unknown"];loss,_=bimanual_plane_penalty(p,b,s);assert float(loss)==0


def test_translation_invariance_and_gt_source():
    p,b,s=example();a,_=bimanual_plane_penalty(p,b,s)
    shift=torch.tensor([2.,-3.,4.]);p2={"pred_obj_trans":p["pred_obj_trans"]+shift,"ot_hand_positions":p["ot_hand_positions"]+shift}
    c,_=bimanual_plane_penalty(p2,b,s);torch.testing.assert_close(a,c)
    joints=torch.zeros(1,3,24,3);joints[:,:,22:24]=p["ot_hand_positions"].detach()
    b["position_global"]=joints;v,_=bimanual_plane_penalty(p,b,s,source="gt");torch.testing.assert_close(a,v)


def test_no_contact_loss_remains_differentiable_zero():
    p,b,s=example();b["lhand_contact"].zero_();loss,_=bimanual_plane_penalty(p,b,s)
    loss.backward();assert torch.count_nonzero(p["pred_obj_trans"].grad)==0
