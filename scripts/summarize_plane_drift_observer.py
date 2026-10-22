"""Summaries of long-sequence drift, fixed-memory and calibration-memory variants."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/"outputs/experiments/plane_drift_20261001"
keys=("obj_cm","hoi_cm","last_third_obj_cm","first_third_obj_cm","late_minus_early_cm","endpoint_drift_cm","after30_obj_cm","after60_obj_cm","bi_obj_cm","bi_obj_p95_cm","acc_mm_frame2","nonbi_obj_cm","ordinary_bi_obj_cm")
rng=np.random.default_rng(1201)
rows=[]
for dataset in ("OMOMO","BEHAVE"):
    for variant in ("vector_state","plane_state","plane_predicted_band","plane_gt_band"):
        for mode,file,seq_file in (("fixed90","result.json","per_sequence.csv"),("calibrated_memory","memory_result.json","memory_per_sequence.csv")):
            paths=sorted((ROOT/dataset/variant).glob("seed*/"+file))
            if not paths:continue
            rr=[json.loads(p.read_text()) for p in paths]
            result={"dataset":dataset,"variant":variant,"mode":mode,"seeds":len(rr)}
            for key in keys:
                vals=[r["result"][key] for r in rr if r["result"][key] is not None]
                base=rr[0]["baseline"][key]
                result[key+"_baseline"]=base
                result[key+"_mean"]=float(np.mean(vals)) if vals else None
                result[key+"_std"]=float(np.std(vals)) if vals else None
                result[key+"_delta"]=float(np.mean(vals)-base) if vals else None
            per=[];base_rows=None
            for p in paths:
                with (p.parent/seq_file).open(encoding="utf-8-sig") as f:seq=list(csv.DictReader(f))
                with (p.parent/"baseline_per_sequence.csv").open(encoding="utf-8-sig") as f:base_rows={r["file"]:r for r in csv.DictReader(f)}
                per.append({r["file"]:r for r in seq})
            files=sorted(per[0]);groups=sorted({r["group"] for r in seq})
            for metric in ("obj_cm","last_third_obj_cm","after60_obj_cm"):
                keep=[f for f in files if per[0][f][metric]!=""]
                if not keep:continue
                d=np.array([np.mean([float(p[f][metric]) for p in per])-float(base_rows[f][metric]) for f in keep])
                unique=sorted({base_rows[f]["group"] for f in keep})
                parts=[d[[i for i,f in enumerate(keep) if base_rows[f]["group"]==g]] for g in unique]
                boots=np.array([np.concatenate([parts[int(i)] for i in rng.integers(0,len(parts),len(parts))]).mean() for _ in range(2000)])
                result[metric+"_group_ci95_lo"],result[metric+"_group_ci95_hi"]=map(float,np.percentile(boots,[2.5,97.5]))
            # Reproduce train-side score only; no test ranking used in selection.
            scores=[]
            for p in paths:
                filename="calibration.json" if mode=="fixed90" else "memory_calibration.json"
                cal=json.loads((p.parent/filename).read_text())["selected"]
                scores.append(cal["selection_score"])
            result["calibration_score_mean"]=float(np.mean(scores));rows.append(result)
fields=list(dict.fromkeys(k for r in rows for k in r))
with (ROOT/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
selection={}
for ds in ("OMOMO","BEHAVE"):
    available=[r for r in rows if r["dataset"]==ds]
    if available:selection[ds]=min(available,key=lambda r:r["calibration_score_mean"])
(ROOT/"summary.json").write_text(json.dumps({"rows":rows,"calibration_selected":selection},indent=2))
print(json.dumps({"rows":rows,"calibration_selected":selection},indent=2))
