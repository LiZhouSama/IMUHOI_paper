"""Checks for mesh-free student inference, causality, and auxiliary training."""
from __future__ import annotations
import copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.rnn.bimanual_prior_student import BimanualPriorStudent,causal_filter
from scripts.train_bimanual_prior_student import train_loss


def test_temporal_prefix_and_chunk_equivalence():
    torch.manual_seed(1)
    m=BimanualPriorStudent(10,3).eval()
    torch.nn.init.normal_(m.delta_head.weight,std=.1)
    x=torch.randn(2,31,10);k=torch.tensor([0,2])
    whole=m(x,k)
    prefix=m(x[:,:13],k)
    suffix=m(x[:,13:],k,prefix["state"])
    torch.testing.assert_close(whole["delta"][:,:13],prefix["delta"],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(whole["delta"][:,13:],suffix["delta"],atol=1e-6,rtol=1e-5)


def test_no_category_variant_cannot_read_category():
    torch.manual_seed(2)
    m=BimanualPriorStudent(10,3,use_category=False).eval()
    x=torch.randn(2,11,10)
    a=m(x,torch.tensor([0,1]));b=m(x,torch.tensor([2,2]))
    for key in ("delta","variance","geometry"):
        torch.testing.assert_close(a[key],b[key])


def test_training_only_heads_do_not_change_inference_delta():
    m=BimanualPriorStudent(10,3).eval()
    x=torch.randn(1,9,10);k=torch.tensor([1]);a=m(x,k)
    with torch.no_grad():
        m.surface_head.weight.fill_(100);m.patch_head.weight.fill_(-100)
    b=m(x,k)
    torch.testing.assert_close(a["delta"],b["delta"])
    torch.testing.assert_close(a["variance"],b["variance"])


def test_filter_is_causal_and_chunk_state_exact():
    torch.manual_seed(3)
    d=torch.randn(2,20,3)*.1;cp=torch.ones(2,20,3);v=torch.ones(2,20,3)*.01
    for carry in (True,False):
        total,_=causal_filter(d,cp,v,carry=carry)
        assert torch.count_nonzero(total[:,0])==0
        first,state=causal_filter(d[:,:8],cp[:,:8],v[:,:8],carry=carry)
        last,_=causal_filter(d[:,8:],cp[:,8:],v[:,8:],state,carry=carry)
        torch.testing.assert_close(total,torch.cat((first,last),1))
    zeros,_=causal_filter(d,torch.zeros_like(cp),v)
    assert torch.count_nonzero(zeros)==0


def test_mesh_loss_has_finite_gradient_and_can_be_disabled():
    torch.manual_seed(4);n,t=2,16
    m=BimanualPriorStudent(10,3)
    b={"x":torch.randn(n,t,10),"cat":torch.tensor([0,1]),"R":torch.eye(3).expand(n,t,3,3).clone(),
       "cp":torch.ones(n,t,3)*.9,"valid":torch.ones(n,t,dtype=torch.bool),"contact":torch.ones(n,t,2,dtype=torch.bool),
       "p":torch.randn(n,t,3)*.1,"gt":torch.randn(n,t,3)*.1,"h":torch.randn(n,t,2,3)*.1,"gh":torch.randn(n,t,2,3)*.1,
       "residual_target":torch.randn(n,t,3)*.1,"geometry_target":torch.randn(n,t,3)*.1,
       "patch_target":torch.randint(0,8,(n,t,2)),"surface_target":torch.randn(n,t,12),"mesh_confidence":torch.ones(n,t,2)}
    loss,terms=train_loss(m,b,.1);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(m.surface_head.weight.grad).all()
    assert m.surface_head.weight.grad.abs().sum()>0
    m.zero_grad(set_to_none=True)
    small={k:v for k,v in b.items() if k not in ("patch_target","surface_target","mesh_confidence")}
    loss,_=train_loss(m,small,0);loss.backward()
    assert m.surface_head.weight.grad is None


def test_numpy_filter_gain_reuse_is_linear():
    import numpy as np
    from scripts.train_bimanual_prior_student import apply_numpy
    rng=np.random.default_rng(5)
    x={"world":rng.normal(size=(17,3))*.1,"variance":np.ones((17,3))*.01}
    r={"p":np.zeros((17,3)),"cp":np.ones((17,3))*.8}
    for uncertainty in (True,False):
        for carry in (True,False):
            full=apply_numpy(x,r,1,uncertainty,carry)
            half=apply_numpy(x,r,.5,uncertainty,carry)
            np.testing.assert_allclose(half,full*.5)
            np.testing.assert_array_equal(apply_numpy(x,r,0,uncertainty,carry),r["p"])


def test_directional_uncertainty_is_applied_in_object_frame():
    import numpy as np
    from scripts.calibrate_bimanual_student_directional import directional
    rotation=np.array([[[0.,-1,0],[1,0,0],[0,0,1]]])
    records=[{"file":"one","R":rotation}]
    outputs={"one":{"world":np.array([[0.,1,0]]),"variance":np.array([[.09,0,0]])}}
    result=directional(outputs,records)
    np.testing.assert_allclose(result["one"]["world"],[[0.,.1,0]],atol=1e-12)

