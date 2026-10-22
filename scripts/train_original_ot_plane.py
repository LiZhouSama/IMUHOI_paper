"""Fine-tune every parameter of the original OT with optional plane loss.

HP and VC execute on every batch but remain frozen. This script reuses the native
Stage3JointTrainer forward/backward path, including OT recurrent feedback.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader,Sampler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.utils import load_config,load_checkpoint,setup_seed
from dataset.dataset_IMUHOI import IMUDataset
from model import HumanPoseModule,ObjectTransModule
from model.rnn.object_trans_v2 import V2ObjectTransModule
from model.rnn.velocity_contact import build_velocity_contact_module
from train.rnn.train_object_trans import Stage3JointTrainer
from train.rnn.loss.object_trans_plane_loss import PlaneObjectTransLoss
from train.rnn.train_utils import autocast,build_model_input_dict,_restore_rng_state
from train.rnn.loss import resolve_object_trans_loss_weights


class EverySequenceSampler(Sampler):
    def __init__(self,data,k,seed):self.data=data;self.k=int(k);self.seed=seed;self.epoch=0
    def __len__(self):return len(self.data)*self.k
    def __iter__(self):
        ids=np.tile(np.arange(len(self.data)),self.k)
        np.random.default_rng(self.seed+self.epoch).shuffle(ids)
        return iter(ids.tolist())


def json_write(path,value):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def state_digest(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def setup(ds,variant,device,limit=None):
    settings=yaml.safe_load((ROOT/"configs/experiments/ot_plane_full_20261001.yaml").read_text())
    root=ROOT/settings["output_root"];run=root/ds/variant
    source=json.loads((root/"sources.json").read_text())[ds]
    cfg=load_config(str(run/"config.yaml"));cfg.device=device
    cfg.no_trans=bool(getattr(cfg,"no_trans",False));cfg.joint_train=False;cfg.resume_dir=None;cfg.use_multi_gpu=False
    cfg.debug=False;cfg.use_tensorboard=False
    paths={p:json.loads((root/ds/f"{p}_files.json").read_text()) for p in ("train","validation")}
    if limit:
        split=json.loads((root/ds/"split.json").read_text())
        band_preview=json.loads((root/ds/"plane_bands.json").read_text())
        ordinary={r["file"] for rows in (split["train"],split["validation"]) for r in rows if band_preview.get(r["object"],{}).get("eligible",False)}
        paths={p:[v for v in vv if Path(v).name in ordinary][:limit] for p,vv in paths.items()}
    if set(paths["train"])&set(paths["validation"]):raise ValueError("Overlapping train/validation")
    os.environ["IMUHOI_SHARED_MEMORY"]="0"
    datasets={}
    for part in paths:
        datasets[part]=IMUDataset(data_dir=str(ROOT/"process/IMUHOI++cano"/source["directory"]/"train"),sequence_paths=paths[part],window_size=settings["window_frames"],debug=False,
            obj_points_sample_count=256,simulate_imu_noise=True,simulate_imu_noise_seed=42,resolve_bimanual_contact_conflicts=False,require_canonical_skeleton=True,random_window=part=="train")
    sampler=EverySequenceSampler(datasets["train"],settings["windows_per_sequence"],settings["seed"])
    batch=min(limit,settings["batch_size"]) if limit else settings["batch_size"]
    loader={"train":DataLoader(datasets["train"],batch_size=batch,sampler=sampler,num_workers=0,pin_memory=False,drop_last=False),
            "validation":DataLoader(datasets["validation"],batch_size=batch,shuffle=False,num_workers=0,pin_memory=False,drop_last=False)}
    # Match optimizer-update budgets across differently sized corpora.
    scale=settings["reference_steps_per_epoch"]/max(1,len(loader["train"]))
    cfg.epochs=int(math.ceil(settings["epochs"]*scale))
    cfg.early_stopping_min_epochs=int(math.ceil(settings["minimum_epochs"]*scale))
    cfg.early_stopping_patience_epochs=int(math.ceil(settings["patience_epochs"]*scale))
    cfg.eval_interval=max(1,int(round(scale)))
    cfg.milestones=[max(1,int(round(m*scale))) for m in settings["milestones"]]
    if not limit:
        (run/"config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg)),sort_keys=False))
    vc=build_velocity_contact_module(cfg)
    hp=HumanPoseModule(cfg,torch.device(device),no_trans=cfg.no_trans)
    ot=(V2ObjectTransModule if cfg.object_trans_arch=="v2" else ObjectTransModule)(cfg)
    models={"human_pose":hp,"velocity_contact":vc,"object_trans":ot}
    for name,model in models.items():load_checkpoint(model,source["checkpoints"][name]["path"],torch.device(device),strict=True)
    trainer=Stage3JointTrainer(cfg,vc,hp,ot,loader["train"],loader["validation"],joint_train=False,lb_prior_model=None)
    bands=json.loads((root/ds/"plane_bands.json").read_text())
    trainer.ot_loss_fn=PlaneObjectTransLoss(resolve_object_trans_loss_weights(cfg),bands,cfg.plane_source,cfg.plane_weight)
    return trainer,sampler,settings,source,run,models


def smoke(ds,variant,device,smoke_batch_size=4):
    torch.set_num_threads(4);setup_seed(1201);trainer,sampler,settings,source,run,models=setup(ds,variant,device,limit=smoke_batch_size)
    trainer.cfg.ot_history_training_prob=1.0
    batch=next(iter(trainer.train_loader));data=build_model_input_dict(batch,trainer.cfg,trainer.device,add_noise=True)
    hp_before=state_digest(trainer.hp_model);vc_before=state_digest(trainer.vc_model)
    initial={k:v.detach().clone() for k,v in trainer.ot_model.named_parameters()}
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with autocast(enabled=trainer.amp_enabled,dtype=trainer.amp_dtype):
        _,_,_,pred=trainer.forward_all(data,batch,epoch=0,training=True)
        total,raw,weighted=trainer.ot_loss_fn(pred,batch,trainer.device)
    parameter_list=[p for p in trainer.ot_model.parameters() if p.requires_grad]
    plane=weighted["plane_band"]
    base=total-plane
    base_grad=torch.autograd.grad(base,parameter_list,retain_graph=True,allow_unused=True)
    plane_grad=torch.autograd.grad(plane,parameter_list,retain_graph=True,allow_unused=True)
    norm=lambda gs:float(torch.sqrt(sum((g.detach().float().square().sum() for g in gs if g is not None),torch.zeros((),device=trainer.device))))
    trainer.optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(parameter_list,trainer.gradient_clip);trainer.optimizer.step();torch.cuda.synchronize()
    changed=[k for k,v in trainer.ot_model.named_parameters() if not torch.equal(v.detach(),initial[k])]
    report={"dataset":ds,"variant":variant,"forced_own_history":True,"ot_parameters":sum(p.numel() for p in parameter_list),"ot_tensors_with_grad":sum(p.grad is not None for p in parameter_list),
        "ot_changed_parameter_tensors":changed,"hp_unchanged":hp_before==state_digest(trainer.hp_model),"vc_unchanged":vc_before==state_digest(trainer.vc_model),
        "base_gradient_norm":norm(base_grad),"weighted_plane_gradient_norm":norm(plane_grad),"plane_weight":trainer.ot_loss_fn.plane_weight,
        "weighted_plane_loss":float(plane.detach()),"base_loss":float(base.detach()),"gradient_ratio":norm(plane_grad)/max(norm(base_grad),1e-12),
        "plane_diagnostics":{k:float(v) for k,v in trainer.ot_loss_fn.last_diagnostics.items()},"step_seconds":time.perf_counter()-start,"peak_gpu_mib":torch.cuda.max_memory_allocated()/1024**2}
    assert changed and report["hp_unchanged"] and report["vc_unchanged"]
    assert report["ot_parameters"]>100000
    json_write(run/"native_training_smoke.json",report)
    print("SMOKE",json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"));ap.add_argument("--variant",required=True,choices=("baseline","plane_predicted","plane_gt"));ap.add_argument("--smoke",action="store_true");ap.add_argument("--smoke-batch-size",type=int,default=4);ap.add_argument("--resume",action="store_true");a=ap.parse_args()
    if a.smoke:smoke(a.dataset,a.variant,"cuda:0",a.smoke_batch_size);return
    settings=yaml.safe_load((ROOT/"configs/experiments/ot_plane_full_20261001.yaml").read_text());setup_seed(settings["seed"]);torch.set_num_threads(4)
    trainer,sampler,settings,source,run,models=setup(a.dataset,a.variant,"cuda:0")
    initial={k:v.detach().cpu().clone() for k,v in trainer.ot_model.named_parameters()}
    hp_digest=state_digest(trainer.hp_model);vc_digest=state_digest(trainer.vc_model)
    metadata={"dataset":a.dataset,"variant":a.variant,"trainable_parameters":{"object_trans":sum(p.numel() for p in trainer.ot_model.parameters() if p.requires_grad),"human_pose":sum(p.numel() for p in trainer.hp_model.parameters() if p.requires_grad),"velocity_contact":sum(p.numel() for p in trainer.vc_model.parameters() if p.requires_grad)},
        "source_checkpoints":source["checkpoints"],"train_sequences":len(trainer.train_loader.dataset),"validation_sequences":len(trainer.test_loader.dataset),"windows_per_epoch":len(sampler),"steps_per_epoch":len(trainer.train_loader),"window_frames":settings["window_frames"],"native_training":True,"cached_predictions":False,"own_history_probability":settings["history_probability"],"planned_epochs":trainer.cfg.epochs,"minimum_epochs":trainer.cfg.early_stopping_min_epochs,"patience_epochs":trainer.cfg.early_stopping_patience_epochs,"validation_interval":trainer.cfg.eval_interval}
    assert metadata["trainable_parameters"]["object_trans"]>100000 and metadata["trainable_parameters"]["human_pose"]==metadata["trainable_parameters"]["velocity_contact"]==0
    json_write(run/"training_metadata.json",metadata)
    history=[];start_epoch=0;best=float("inf");last_improve=0;start_time=time.time()
    if a.resume and (run/"last.pt").exists():
        checkpoint=torch.load(run/"last.pt",map_location=trainer.device,weights_only=False)
        trainer.ot_model.load_state_dict(checkpoint["model_state_dict"]);trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]);trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.n_iter=int(checkpoint.get("n_iter",0));start_epoch=int(checkpoint["epoch"])+1
        if checkpoint.get("rng_state"):_restore_rng_state(checkpoint["rng_state"])
        history=json.loads((run/"history.json").read_text());history=[r for r in history if r["epoch"]<start_epoch]
        best_ckpt=torch.load(run/"best.pt",map_location="cpu",weights_only=False);best=float(best_ckpt["loss"]);last_improve=int(best_ckpt["epoch"])
    else:
        initial_loss,initial_components=trainer.evaluate(0)
        json_write(run/"initial_validation.json",{"selection_loss":initial_loss,"components":initial_components})
        best=initial_loss;last_improve=-1;trainer.best_loss=initial_loss
        trainer._save_models(-1,initial_loss,"best",{"n_iter":0,"initial_checkpoint_candidate":True})
        print("INITIAL_VALIDATION",initial_loss,flush=True)
    for epoch in range(start_epoch,trainer.cfg.epochs):
        sampler.epoch=epoch;epoch_start=time.time();before=trainer.n_iter
        train_loss,train_components=trainer.train_epoch(epoch)
        evaluated=(epoch%trainer.cfg.eval_interval==0 or epoch+1==trainer.cfg.epochs)
        if evaluated:
            validation_loss,validation_components=trainer.evaluate(epoch)
        elif history:
            validation_loss=history[-1]["validation_loss"];validation_components={}
        else:
            validation_loss,validation_components=trainer.evaluate(epoch);evaluated=True
        trainer.scheduler.step()
        assert trainer.n_iter-before==len(trainer.train_loader)
        record={"epoch":epoch,"evaluated":evaluated,"train_loss":train_loss,"validation_loss":validation_loss,"train_components":train_components,"validation_components":validation_components,
            "optimizer_updates":trainer.n_iter,"updates_this_epoch":trainer.n_iter-before,"windows_this_epoch":len(sampler),"epoch_seconds":time.time()-epoch_start,"elapsed_seconds":time.time()-start_time}
        history.append(record);json_write(run/"history.json",history)
        extra={"n_iter":trainer.n_iter,"experiment":"original_ot_plane_full","plane_source":trainer.ot_loss_fn.plane_source,"plane_weight":trainer.ot_loss_fn.plane_weight,"train_components":train_components,"validation_components":validation_components}
        if evaluated and validation_loss<best:
            best=validation_loss;last_improve=epoch;trainer.best_loss=best;trainer._save_models(epoch,validation_loss,"best",extra)
        if evaluated:trainer._save_models(epoch,validation_loss,"last",extra)
        print("EPOCH",epoch,"updates",trainer.n_iter,"train",round(train_loss,5),"validation",round(validation_loss,5),"plane",train_components.get("ot_plane_band"),"best",last_improve,"seconds",round(record["epoch_seconds"],1),flush=True)
        if evaluated and epoch+1>=trainer.cfg.early_stopping_min_epochs and epoch-last_improve>=trainer.cfg.early_stopping_patience_epochs:break
    trainer._save_models(history[-1]["epoch"],history[-1]["validation_loss"],"final",{"n_iter":trainer.n_iter,"experiment":"original_ot_plane_full"})
    changed={k:float((v.detach().cpu()-initial[k]).norm()) for k,v in trainer.ot_model.named_parameters() if not torch.equal(v.detach().cpu(),initial[k])}
    report={**metadata,"completed_epochs":history[-1]["epoch"]+1,"best_epoch":last_improve,"optimizer_updates":trainer.n_iter,"best_validation_loss":best,"changed_parameter_tensors":changed,
            "hp_unchanged":hp_digest==state_digest(trainer.hp_model),"vc_unchanged":vc_digest==state_digest(trainer.vc_model),"elapsed_seconds":time.time()-start_time}
    assert changed and report["hp_unchanged"] and report["vc_unchanged"]
    json_write(run/"training_complete.json",report);print("TRAINING_COMPLETE",a.dataset,a.variant,trainer.n_iter,flush=True)


if __name__=="__main__":main()







