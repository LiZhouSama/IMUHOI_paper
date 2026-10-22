#!/usr/bin/env python3
"""Check whether current VC best/final selection explains velocity drift."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import yaml
    from easydict import EasyDict
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    exp = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
    out = ROOT / "outputs/analysis/egoimu_implementation_bridge_20261017"
    cfg = EasyDict(yaml.safe_load((exp / "recipe.yaml").read_text()))
    manifest = json.loads((exp / "baseline/experiment_manifest.json").read_text())
    vc_folder = Path(manifest["stages"]["velocity_contact"]["checkpoint"]).parent
    models, rows, metadata = {}, {}, {}
    for name in ("best", "final"):
        payload = torch.load(vc_folder / (name + ".pt"), map_location="cuda:0", weights_only=False)
        model = IMUHOIVelocityContactModule(cfg).cuda().eval()
        model.load_state_dict(payload.get("model_state_dict", payload.get("module_state_dict")), strict=True)
        models[name], rows[name] = model, []
        metadata[name] = {"epoch": payload.get("epoch"), "path": str(vc_folder / (name + ".pt"))}
    with torch.inference_mode():
        for entry in json.loads((out / "native_manifest.json").read_text())["entries"]:
            cache = torch.load(out / "native_cache" / entry["file"], weights_only=False)
            data = {k: v.cuda() for k, v in cache["data"].items()}
            target, gtvel = (cache["target"][k][0].cuda() for k in ("obj_trans", "obj_vel"))
            for name, model in models.items():
                vel = model(data)["pred_obj_vel"][0]
                disp = vel.cumsum(0) / 30
                pos = target[0] + torch.cat((disp.new_zeros(1, 3), disp[:-1]), 0)
                error = vel - gtvel
                rows[name].append({"obj_cm": (pos - target).norm(dim=-1).mean().item() * 100,
                                   "velocity_rmse_mps": error.square().mean().sqrt().item(),
                                   "bias_norm_mps": error.mean(0).norm().item()})
    for name in models:
        metadata[name]["n"] = len(rows[name])
        metadata[name]["metrics"] = {key: sum(r[key] for r in rows[name]) / len(rows[name]) for key in rows[name][0]}
    (out / "current_vc_checkpoint_choice.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
