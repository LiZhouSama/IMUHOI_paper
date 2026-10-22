"""Aggregate the preregistered student variants and paired sequence outcomes."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT/"outputs/experiments/meshfree_temporal_20261001")
    a=p.parse_args();root=a.root
    results=[]
    for path in sorted(root.glob("*/*/seed*/result.json")):
        r=json.loads(path.read_text());r["path"]=str(path.parent)
        results.append(r)
    variants=sorted({r["variant"] for r in results});summary=[];details=[]
    rng=np.random.default_rng(1001)
    for dataset in ("OMOMO","BEHAVE"):
        for variant in variants:
            runs=[r for r in results if r["dataset"]==dataset and r["variant"]==variant]
            if not runs:continue
            row={"dataset":dataset,"variant":variant,"runs":len(runs),"nonzero_gain_runs":sum(r["selected"]["gain"]>0 for r in runs)}
            for metric in ("object_cm","hoi_cm","bi_object_cm","bi_hoi_cm","bi_object_p95_cm","acc_mm_frame2","nonbi_object_cm"):
                scores=np.array([r["result"][metric] for r in runs]);base=np.array([r["baseline"][metric] for r in runs])
                row[metric+"_baseline"]=float(base.mean());row[metric+"_mean"]=float(scores.mean());row[metric+"_std"]=float(scores.std());row[metric+"_delta"]=float((scores-base).mean())
            ds=[]
            for r in runs:
                path=Path(r["path"]);base={x["file"]:x for x in read_csv(path/"baseline_per_sequence.csv")};seq=read_csv(path/"per_sequence.csv")
                d={x["file"]:float(x["object_cm"])-float(base[x["file"]]["object_cm"]) for x in seq};ds.append(d)
                details.append({"dataset":dataset,"variant":variant,"seed":r["seed"],"epoch":r["best_epoch"],**r["selected"],**r["result"]})
            files=sorted(ds[0]);delta=np.array([np.mean([d[f] for d in ds]) for f in files])
            boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)])
            lo,hi=np.percentile(boots,[2.5,97.5]);row["object_delta_paired_bootstrap_lo"]=float(lo);row["object_delta_paired_bootstrap_hi"]=float(hi)
            row["sequences_improved_pct"]=float((delta<0).mean()*100)
            group_by_file={r["file"]:r["group"] for r in seq}
            group_names=sorted(set(group_by_file.values()))
            group_values=[delta[[i for i,f in enumerate(files) if group_by_file[f]==g]] for g in group_names]
            cluster=np.array([np.concatenate([group_values[int(i)] for i in rng.integers(0,len(group_values),len(group_values))]).mean() for _ in range(3000)])
            clo,chi=np.percentile(cluster,[2.5,97.5])
            row["source_groups"]=len(group_values)
            row["object_delta_group_bootstrap_lo"]=float(clo)
            row["object_delta_group_bootstrap_hi"]=float(chi)
            cal_scores=[json.loads((Path(r["path"])/"calibration.json").read_text())["selected"]["object_cm"] for r in runs]
            row["calibration_object_cm_mean"]=float(np.mean(cal_scores))
            summary.append(row)
    for filename,rows in (("summary.csv",summary),("run_metrics.csv",details)):
        if rows:
            with (root/filename).open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    choices={}
    for dataset in ("OMOMO","BEHAVE"):
        entries=[{"variant":r["variant"],"calibration_object_cm":r["calibration_object_cm_mean"]} for r in summary if r["dataset"]==dataset]
        stat_path=root/dataset/"category_covariance"/"result.json"
        if stat_path.exists():
            stat=json.loads(stat_path.read_text());selected=stat["selected"]
            trial=next(t for t in stat["calibration_trials"] if all(t[k]==v for k,v in selected.items()))
            baseline_cal=next(t for t in stat["calibration_trials"] if t["gain"]==0.)["object_cm"]
            entries.extend([{"variant":"category_covariance","calibration_object_cm":trial["object_cm"]},{"variant":"unchanged","calibration_object_cm":baseline_cal}])
        if entries:choices[dataset]={"ranking":sorted(entries,key=lambda e:e["calibration_object_cm"]),"selected":min(entries,key=lambda e:e["calibration_object_cm"])["variant"]}
    (root/"summary.json").write_text(json.dumps({"runs":len(results),"summary":summary,"training_side_method_selection":choices},indent=2))
    print(json.dumps({"runs":len(results),"summary":summary},indent=2))


if __name__=="__main__":main()


