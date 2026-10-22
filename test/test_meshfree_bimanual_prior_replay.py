"""Geometry, causality, and GT-isolation checks for mesh-free replay diagnostics."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diag.meshfree_bimanual_prior_replay import candidates, features, smooth_correction


class ZeroPrior(torch.nn.Module):
    def forward(self, x, k):
        return torch.zeros((len(x),3),dtype=x.dtype,device=x.device)


def sample():
    t=8
    h=np.tile(np.array([[[-.5,0,0],[.5,0,0]]]),(t,1,1))
    p=np.tile(np.array([[.15,.20,.30]]),(t,1))
    rot=np.tile(np.eye(3),(t,1,1))
    x=np.zeros((t,19));x[:,0]=1;x[:,3]=1
    return {"object":"box","p":p,"h":h,"R":rot,"c":p.copy(),"x":x,"cp":np.ones((t,2)),
            "gt":np.zeros((t,3)),"gh":h.copy(),"contact":np.ones((t,2),bool)}


def prior_stats():
    return {"box":{"c_mean":np.zeros(3),"c_cov":np.eye(3)*.01,"signed_mean":0.,
                   "signed_q05":-.1,"signed_q95":.1,"plane_eligible":True}}


def run(r):
    candidates([r],prior_stats(),ZeroPrior(),np.zeros(19),np.ones(19),["box"],torch.device("cpu"))
    return r


def test_projection_sign_and_tangent_preservation():
    r=run(sample())
    projected=r["p"]+r["delta"]["naive_hard_plane"]
    np.testing.assert_allclose(projected[:,0],0,atol=1e-12)
    np.testing.assert_allclose(projected[:,1:],r["p"][:,1:],atol=1e-12)


def test_correction_does_not_read_ground_truth_or_mesh():
    a=sample();b=sample()
    b.pop("gt");b.pop("gh");b.pop("contact")
    run(a);run(b)
    for key in a["delta"]:
        np.testing.assert_array_equal(a["delta"][key],b["delta"][key])


def test_smoothing_is_causal_and_inactive_stays_zero():
    cp=np.ones((20,2));d=np.ones((20,3))*.1
    changed=d.copy();changed[10:]=100
    np.testing.assert_array_equal(smooth_correction(d,cp)[:10],smooth_correction(changed,cp)[:10])
    np.testing.assert_array_equal(smooth_correction(d,np.zeros((20,2))),np.zeros((20,3)))


def test_observation_features_ignore_common_translation():
    t=5;h=sample()["h"][:t]
    imu=np.zeros((t,6,9));obj=np.zeros((t,9))
    imu[:,:,3:9]=np.array([1,0,0,0,1,0]);obj[:,3:9]=np.array([1,0,0,0,1,0])
    x,r=features(h,obj,imu);x2,r2=features(h+np.array([3,4,-2]),obj,imu)
    np.testing.assert_allclose(x,x2,atol=1e-12);np.testing.assert_allclose(r,r2)
