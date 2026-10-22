"""Summarize directional uncertainty fusion without selecting by test score."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/"outputs/experiments/meshfree_temporal_20261001"
rows=[]
for ds in ("OMOMO","BEHAVE"):
    for variant_dir in sorted((ROOT/ds).glob("*")):
        reports=[json.loads(p.read_text()) for p in sorted(variant_dir.glob("seed*/directional_result.json"))]
        if not reports:continue
        row={"dataset":ds,"variant":variant_dir.name,"runs":len(reports)}
        for metric in ("object_cm","hoi_cm","bi_object_cm","bi_hoi_cm","bi_object_p95_cm","acc_mm_frame2","nonbi_object_cm"):
            values=np.array([r["result"][metric] for r in reports]);base=np.array([r["baseline"][metric] for r in reports])
            row[metric+"_baseline"]=float(base.mean());row[metric+"_mean"]=float(values.mean());row[metric+"_std"]=float(values.std());row[metric+"_delta"]=float((values-base).mean())
        row["calibration_object_cm_mean"]=float(np.mean([json.loads(p.read_text())["selected"]["object_cm"] for p in sorted(variant_dir.glob("seed*/directional_calibration.json"))]))
        rows.append(row)
if rows:
    with (ROOT/"directional_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/"directional_summary.json").write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
