"""Read-only schema inspection of the two OMOMO corpora; imports no EgoIMU code."""
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parents[2]
roots = {"ego_split": ROOT.parent / "EgoIMU/process/processed_split_data_OMOMO",
         "ego_unified": ROOT.parent / "EgoIMU/process/processed_data_OMOMO",
         "current": ROOT / "process/IMUHOI++cano/processed_split_data_OMOMO_full"}
for label, root in roots.items():
    print("ROOT", label, str(root), flush=True)
    folders = [root] + [p for p in root.iterdir() if p.is_dir()]
    for folder in folders:
        files = sorted(folder.glob("*.pt"))
        if not files:
            continue
        sample = torch.load(files[0], map_location="cpu", weights_only=False)
        print(json.dumps({"folder": str(folder), "count": len(files), "first": files[0].name,
                          "schema": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                     if isinstance(v, torch.Tensor) or hasattr(v, "shape") else str(v)[:300]
                                     for k, v in sample.items()}}, ensure_ascii=False), flush=True)
