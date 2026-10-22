"""Run the official online evaluator and capture its actual prediction cache.

The wrapper observes the evaluator return value and returns it unchanged. It
adds no inference correction and does not alter the metric implementation.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import eval_batch_online
from scripts.dump_ot_predictions import _record_from


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--capture_predictions",required=True);own,rest=parser.parse_known_args()
    destination=Path(own.capture_predictions);destination.parent.mkdir(parents=True,exist_ok=True)
    original=eval_batch_online._batched_online_predict_dataset
    captured=[]
    def observe(*args,**kwargs):
        pred_cache,sample_cache=original(*args,**kwargs)
        lookahead=int(kwargs.get("online_lookahead",10));records=[]
        for sample in sample_cache:
            pred=pred_cache[sample["_cache_key"]]
            n=max(0,int(sample["human_imu"].shape[0])-lookahead)
            records.append(_record_from(sample,pred,n))
        payload={"schema_version":2,"official_evaluator_args":rest,"online_lookahead":lookahead,"records":records,
                 "source_hashes":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("eval_batch_online.py","eval_IMUHOI.py","model/rnn/imuhoi_model.py","model/rnn/object_trans.py")}}
        torch.save(payload,destination);captured.append(len(records))
        print("CAPTURED_NATIVE_PREDICTIONS",destination,len(records),flush=True)
        return pred_cache,sample_cache
    eval_batch_online._batched_online_predict_dataset=observe
    sys.argv=["eval_batch_online.py"]+rest
    code=eval_batch_online.main()
    if code not in (None,0):return code
    if not captured:raise RuntimeError("Official evaluator did not produce a prediction cache")
    return 0

if __name__=="__main__":raise SystemExit(main())
