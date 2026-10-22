"""Summarize actual original-OT training and official full online tests."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/experiments/ot_plane_full_20261001"


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    final_mode="--final" in sys.argv
    selected_mode="--selected" in sys.argv or final_mode
    torch.set_num_threads(2);sources=json.loads((OUT/"sources.json").read_text());all_results={};table=[];all_sequences=[]
    for ds,source in sources.items():
        per={}
        for variant in ("initial","baseline","plane_predicted","plane_gt"):
            run=OUT/ds/variant
            official_path=run/"official_test.json";prediction_path=run/"online_predictions.pt";tag="initial" if variant=="initial" else "window_best"
            if selected_mode and variant!="initial":
                decision_path=run/"full_online_selection.json"
                if not decision_path.exists():continue
                decision=json.loads(decision_path.read_text())
                field="final_test" if final_mode else "selected_test"
                if field not in decision:continue
                official_path=Path(decision[field]["official"]);prediction_path=Path(decision[field]["predictions"]);tag="final" if final_mode else decision["selected"]["tag"]
            if not official_path.exists() or not prediction_path.exists():continue
            official=json.loads(official_path.read_text())["results"][source["dataset_key"]]
            records=torch.load(prediction_path,map_location="cpu",weights_only=False)["records"]
            seq=[];bi_errors=[];bi_hoi=[]
            for r in records:
                n=int(r["publish_len"])
                if not n or not r.get("has_object",True):continue
                p=np.asarray(r["pred_obj_trans_published"],float);gt=np.asarray(r["gt_obj_trans"],float);h=np.asarray(r["pred_palm_glb_pos"],float);gh=np.asarray(r.get("gt_palm_pos_eval",r["gt_palm_pos"]),float)
                contacts=np.stack([np.asarray(r["gt_lhand_contact"],bool),np.asarray(r["gt_rhand_contact"],bool)],-1);both=contacts.all(-1)
                error=np.linalg.norm(p-gt,axis=-1)*100;hoi=np.linalg.norm((p[:,None]-h)-(gt[:,None]-gh),axis=-1)*100
                third=max(1,n//3)
                row={"dataset":ds,"variant":variant,"file":r["seq_file"],"object":r["obj_name"],"frames":n,"seconds":n/30,
                    "object_cm":float(error.mean()),"hoi_cm":float(hoi[contacts].mean()) if contacts.any() else None,"first_third_cm":float(error[:third].mean()),"last_third_cm":float(error[-third:].mean()),
                    "endpoint_drift_cm":float(np.linalg.norm((p[-1]-gt[-1])-(p[0]-gt[0]))*100),"after30_cm":float(error[900:].mean()) if n>900 else None,"after60_cm":float(error[1800:].mean()) if n>1800 else None,
                    "published_acc_mm_frame2":float(np.linalg.norm(np.diff(p,n=2,axis=0),axis=-1).mean()*1000),"palm_cm":float(np.linalg.norm(h-gh,axis=-1).mean()*100)}
                seq.append(row);bi_errors.append(error[both]);bi_hoi.append(hoi[both].mean(-1))
            means=lambda key:float(np.mean([r[key] for r in seq if r[key] is not None])) if any(r[key] is not None for r in seq) else None
            official_obj=official["obj_trans_err_fusion"];official_hoi=official["hoi_err_fusion"]
            # Published-cache replay must match the evaluator; otherwise do not report mixed metrics.
            if abs(means("object_cm")-official_obj)>.01:raise AssertionError((ds,variant,"publication mismatch",means("object_cm"),official_obj))
            if abs(means("hoi_cm")-official_hoi)>.01:raise AssertionError((ds,variant,"HOI mismatch",means("hoi_cm"),official_hoi))
            summary={"dataset":ds,"variant":variant,"checkpoint_selection":"fixed_final_diagnostic" if final_mode else "full_online_validation" if selected_mode else "window_validation","checkpoint_tag":tag,"mpjpe_cm":official.get("mpjpe"),"root_error_cm":official.get("root_trans_err"),"sequences":len(seq),"frames":sum(r["frames"] for r in seq),"official_object_cm":official_obj,"official_hoi_cm":official_hoi,
                     "first_third_cm":means("first_third_cm"),"last_third_cm":means("last_third_cm"),"endpoint_drift_cm":means("endpoint_drift_cm"),"after30_cm":means("after30_cm"),"after60_cm":means("after60_cm"),
                     "published_acc_mm_frame2":means("published_acc_mm_frame2"),"palm_cm":means("palm_cm"),"bimanual_object_cm":float(np.concatenate(bi_errors).mean()),"bimanual_hoi_cm":float(np.concatenate(bi_hoi).mean()),"bimanual_p95_cm":float(np.percentile(np.concatenate(bi_errors),95))}
            if (run/"training_complete.json").exists():
                t=json.loads((run/"training_complete.json").read_text());summary.update(trainable_ot=t["trainable_parameters"]["object_trans"],epochs=t["completed_epochs"],optimizer_updates=t["optimizer_updates"],best_epoch=t["best_epoch"],hp_unchanged=t["hp_unchanged"],vc_unchanged=t["vc_unchanged"])
            per[variant]={"summary":summary,"per_sequence":seq};table.append(summary);all_sequences.extend(seq)
        if "baseline" in per:
            base={r["file"]:r for r in per["baseline"]["per_sequence"]}
            rng=np.random.default_rng(1201)
            for variant in ("plane_predicted","plane_gt"):
                if variant not in per:continue
                differences={}
                for metric in ("object_cm","hoi_cm","last_third_cm","after60_cm"):
                    delta=[r[metric]-base[r["file"]][metric] for r in per[variant]["per_sequence"] if r[metric] is not None and base[r["file"]][metric] is not None]
                    if delta:
                        d=np.asarray(delta);boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(3000)])
                        differences[metric]={"mean_delta_cm":float(d.mean()),"paired_sequence_bootstrap95_cm":np.percentile(boots,[2.5,97.5]).tolist(),"sequences":len(d)}
                per[variant]["vs_matched_baseline"]=differences
        all_results[ds]=per
    suffix="_final" if final_mode else "_selected" if selected_mode else ""
    write_csv(OUT/f"summary{suffix}.csv",table);write_csv(OUT/f"per_sequence_summary{suffix}.csv",all_sequences)
    compact={ds:{v:{k:value for k,value in r.items() if k!="per_sequence"} for v,r in results.items()} for ds,results in all_results.items()}
    (OUT/f"summary{suffix}.json").write_text(json.dumps(compact,indent=2,allow_nan=False))
    print(json.dumps(table,indent=2),flush=True)


if __name__=="__main__":main()


