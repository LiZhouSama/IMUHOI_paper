"""Training-only bimanual plane regularizer for the original ObjectTrans loss."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from train.rnn.loss.object_trans_loss import ObjectTransLoss
from utils.human_pose import select_hand_anchor_positions


def bimanual_plane_penalty(pred,batch,bands,source="predicted",beta=.02):
    position=pred["pred_obj_trans"]
    if source=="predicted":
        hands=pred.get("ot_hand_positions")
        if hands is None:raise KeyError("Original OT must return ot_hand_positions")
        hands=hands.detach().to(position)
    elif source=="gt":
        hands=select_hand_anchor_positions(batch["position_global"].to(position)).detach()
    else:raise ValueError(source)
    if hands.shape!=(*position.shape[:2],2,3):raise ValueError((hands.shape,position.shape))
    names=batch["obj_name"]
    if isinstance(names,str):names=[names]
    if len(names)!=len(position):raise ValueError("Object-name batch mismatch")
    tolerance=position.new_tensor([bands.get(n,{"tolerance_m":.1})["tolerance_m"] for n in names])
    eligible=torch.tensor([bool(bands.get(n,{}).get("eligible",False)) for n in names],device=position.device)
    mask=batch["lhand_contact"].to(position.device).bool()&batch["rhand_contact"].to(position.device).bool()
    mask=mask&eligible[:,None]
    has=batch.get("has_object")
    if has is not None:mask=mask&torch.as_tensor(has,device=position.device,dtype=torch.bool).reshape(-1,1)
    for key in ("ot_loss_mask","valid_mask"):
        value=pred.get(key) if key=="ot_loss_mask" else batch.get(key)
        if isinstance(value,torch.Tensor):mask=mask&value.to(position.device).bool()
    # Calculate the geometry in float32 even when the original OT uses AMP.
    direction=(hands[:,:,1]-hands[:,:,0]).float()
    span=direction.norm(dim=-1);normal=direction/span[...,None].clamp_min(1e-6)
    signed=((position.float()-hands.float().mean(-2))*normal).sum(-1)
    mask=mask&(span>=.12)&torch.isfinite(signed)
    excess=(signed.abs()-tolerance.float()[:,None]).clamp_min(0.)
    loss=F.smooth_l1_loss(excess,torch.zeros_like(excess),beta=beta,reduction="none")
    zero=position.sum()*0.
    if mask.any():
        penalty=loss[mask].mean()
        diagnostics={"eligible_frames":mask.sum().detach(),"violation_fraction":(excess[mask]>0).float().mean().detach(),"distance_m":signed[mask].abs().mean().detach()}
    else:
        penalty=zero;diagnostics={"eligible_frames":zero.detach(),"violation_fraction":zero.detach(),"distance_m":zero.detach()}
    return penalty,diagnostics


class PlaneObjectTransLoss(ObjectTransLoss):
    """Adds plane gradients to the native OT forward, not an auxiliary module."""
    TEST_LOSS_KEYS=(*ObjectTransLoss.TEST_LOSS_KEYS,"plane_band","plane_violation_fraction","plane_distance_m")
    def __init__(self,weights,bands,plane_source="predicted",plane_weight=5.):
        super().__init__(weights=weights)
        self.bands=bands;self.plane_source=plane_source;self.plane_weight=float(plane_weight)
        self.calls=0;self.active_calls=0;self.last_diagnostics={}

    def compute_loss(self,pred_dict,batch,device):
        total,losses,weighted=super().compute_loss(pred_dict,batch,device)
        penalty,diagnostics=bimanual_plane_penalty(pred_dict,batch,self.bands,self.plane_source)
        total=total+self.plane_weight*penalty
        losses["plane_band"]=penalty;weighted["plane_band"]=self.plane_weight*penalty
        # Logged diagnostics are not summed into the optimization objective.
        weighted["plane_violation_fraction"]=diagnostics["violation_fraction"]
        weighted["plane_distance_m"]=diagnostics["distance_m"]
        self.calls+=1;self.active_calls+=int(bool(diagnostics["eligible_frames"]>0));self.last_diagnostics=diagnostics
        return total,losses,weighted
