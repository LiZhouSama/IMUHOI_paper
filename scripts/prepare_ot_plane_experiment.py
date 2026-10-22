"""Prepare matched original-OT fine-tuning configurations and group splits."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.utils import load_config
from utils.ot_experiment import resolve_pipeline_files
OUT=ROOT/"outputs/experiments/ot_plane_full_20261001"
SETTINGS=ROOT/"configs/experiments/ot_plane_full_20261001.yaml"
ORIGINAL={
 "OMOMO":("outputs/IMUHOI++/RNN_cano/relabel_20260918_omomo/experiment_manifest.json","cur_hoi_p800","best","processed_split_data_OMOMO_full","omomo"),
 "BEHAVE":("outputs/IMUHOI++/RNN_cano/relabel_20260922_behave_hpsmooth/experiment_manifest.json","v2_fixed_fusion","final","processed_split_data_BEHAVE","behave"),
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 settings=yaml.safe_load(SETTINGS.read_text());OUT.mkdir(parents=True,exist_ok=True);sources={}
 torch.set_num_threads(2)
 for ds,(manifest_path,variant,tag,directory,key) in ORIGINAL.items():
  manifest=json.loads((ROOT/manifest_path).read_text());cfg_path,modules,pipeline=resolve_pipeline_files(manifest,variant,tag,ROOT)
  cfg=json.loads(json.dumps(load_config(str(cfg_path))))
  split=json.loads((ROOT/f"outputs/experiments/meshfree_temporal_20261001/{ds}_student_split.json").read_text())
  train=[r for r in split if r["partition"]!="validation"];validation=[r for r in split if r["partition"]=="validation"]
  assert not {r["group"] for r in train}&{r["group"] for r in validation}
  sources[ds]={"config":str(cfg_path),"config_sha256":sha(cfg_path),"manifest":manifest_path,"variant":variant,"tag":tag,"dataset_key":key,"directory":directory,"checkpoints":{n:{"path":str(p),"sha256":sha(p)} for n,p in modules.items()}}
  root=OUT/ds;root.mkdir(parents=True,exist_ok=True)
  for part,rr in (("train",train),("validation",validation)):
   (root/f"{part}_files.json").write_text(json.dumps([str(ROOT/"process/IMUHOI++cano"/directory/"train"/r["file"]) for r in rr],indent=2))
  (root/"split.json").write_text(json.dumps({"train":train,"validation":validation,"rule":"Use all original training groups except the fixed validation groups; no test data"},indent=2))
  values={}
  for i,r in enumerate(train):
   data=torch.load(ROOT/"process/IMUHOI++cano"/directory/"train"/r["file"],map_location="cpu",weights_only=False)
   h=torch.as_tensor(data["position_global_full_gt_world"],dtype=torch.float64)[:,[22,23]];o=torch.as_tensor(data["obj_trans"],dtype=torch.float64).reshape(-1,3)
   bi=torch.as_tensor(data["lhand_contact"]).bool()&torch.as_tensor(data["rhand_contact"]).bool();direction=h[:,1]-h[:,0];span=direction.norm(dim=-1)
   signed=((o-h.mean(1))*direction).sum(-1)/span.clamp_min(1e-6)
   v=signed[bi&(span>=.12)].abs().numpy()
   values.setdefault(r["object"],[]).append(v)
  bands={}
  for name,parts in values.items():
   v=np.concatenate(parts)
   med=float(np.median(v));p90=float(np.percentile(v,90))
   bands[name]={"median_m":med,"p90_m":p90,"tolerance_m":max(.02,p90),"eligible":med<=.06 and p90<=.20,"fit_frames":len(v)}
  (root/"plane_bands.json").write_text(json.dumps(bands,indent=2))
  cfg.update(device="cuda:0",gpus=[0],use_multi_gpu=False,debug=False,use_tensorboard=False,num_workers=settings["num_workers"],
      batch_size=settings["batch_size"],epochs=settings["epochs"],lr=settings["learning_rate"],milestones=settings["milestones"],gamma=settings["gamma"],
      train_data_window=settings["window_frames"],test_data_window=settings["validation_window_frames"],windows_per_sequence=settings["windows_per_sequence"],
      input_mix_start_epoch=0,input_mix_end_epoch=0,ot_history_training_window=settings["history_window_frames"],ot_history_training_prob=settings["history_probability"],ot_history_training_start_epoch=0,
      object_trans_init_noise_std=settings["object_init_noise_std_m"],object_trans_init_noise_prob=settings["object_init_noise_probability"],
      resume_dir=None,simulate_imu_noise=True,simulate_imu_noise_seed=42,deterministic_test_windows=True,dataloader_pin_memory=False,
      enable_ot_lb_prior=False,object_trans_arch=pipeline.get("object_trans_arch",manifest.get("object_trans_arch","current")),velocity_contact_arch="v4",eval_oracle_vc=None)
  if ds=="BEHAVE":cfg.update(object_trans_arch="v2",human_pose_smoothing_half_width=7,online_obj_vel_init_diff_frames=3)
  # Source recipe fields remain intact; training owns a new output directory.
  cfg["train"]["window"]=settings["window_frames"]
  for mode in settings["variants"]:
   run=root/mode;run.mkdir(parents=True,exist_ok=True)
   conf={**cfg,"save_dir":str(run),"plane_source":"gt" if mode=="plane_gt" else "predicted","plane_weight":0. if mode=="baseline" else settings["plane_weight"]}
   (run/"config.yaml").write_text(yaml.safe_dump(conf,sort_keys=False,allow_unicode=True))
  print(ds,"native-OT train",len(train),"validation",len(validation),"groups",len({r["group"] for r in train}),len({r["group"] for r in validation}),flush=True)
 (OUT/"sources.json").write_text(json.dumps(sources,indent=2))
 print("Prepared original OT experiment",OUT,flush=True)

if __name__=="__main__":main()


