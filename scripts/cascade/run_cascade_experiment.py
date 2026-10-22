#!/usr/bin/env python3
"""Train and evaluate the cascade refine architecture against the current pipeline.

One process per dataset; every GPU stage is resumable and skipped once done.

Stages (the HP and VC-v5ar checkpoints are the ones the baseline pipeline used):
  b0     ObjectTrans v2_armrot_biclose WITHOUT state feedback and WITHOUT the VC+OT
         joint stage (train/cascade/train_ot_b0.py): the baseline's config, schedule,
         early stopping and checkpoint selection, with only those two changes.
  vcr    VC-refine on top of HP / VC / B0-OT (frozen), same data, schedule and
         early-stopping rule as an OT stage (a new module trained from scratch).
  evals  stock eval_batch_online protocol through scripts/cascade/eval_cascade_online.py:
         the baseline (re-evaluated for per-sequence records), B0, the cascade
         (e2 = whole-window OT-refine, main) and ablations.

    python scripts/cascade/run_cascade_experiment.py --dataset behave
    python scripts/cascade/run_cascade_experiment.py --dataset omomo
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/l/anaconda3/envs/SAGE/bin/python"
RNN = "outputs/IMUHOI++/RNN_cano"
DEFAULT_ROOT = f"{RNN}/cascade_refine_20261010"

PRESETS = {
    "behave": {
        "base_config": f"{RNN}/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/config.yaml",
        "hp": f"{RNN}/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt",
        "vc": f"{RNN}/vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930/best.pt",
        "baseline_stage": f"{RNN}/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/"
                          "object_trans_v2_09301506_behave_ot_v2_armrot_biclose_20260930",
    },
    "omomo": {
        "base_config": f"{RNN}/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/config.yaml",
        "hp": f"{RNN}/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt",
        "vc": f"{RNN}/vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930/best.pt",
        "baseline_stage": f"{RNN}/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/"
                          "object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u",
    },
}

# VC-refine recipe (identical for both datasets).
VCR_SETTINGS = {
    "cascade_vcr_hidden_dim": 128,
    "cascade_vcr_num_layers": 2,
    "cascade_vcr_dropout": 0.2,
    "cascade_vcr_contact_weight": 1.0,
    "cascade_vcr_obj_l2_weight": 0.5,
    "cascade_vcr_hoi_weight": 0.5,
    "cascade_vcr_corrupt_prob": 0.5,
    "cascade_vcr_init_noise_prob": 0.8,
    "cascade_vcr_init_noise_min": 0.02,
    "cascade_vcr_init_noise_max": 0.30,
    "cascade_refit_mode": "e2",
    "cascade_hp_history": "stage1",
}
# Free GPU memory (MiB) a stage needs before it may launch (the GPU is shared).
MIN_FREE_MB = {"b0": 5000, "vcr": 6500, "eval": 4500}
SETTLE_SECONDS = 150


def now():
    return datetime.now().isoformat(timespec="seconds")


def free_gpu_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


class Runner:
    def __init__(self, args):
        self.args = args
        self.preset = PRESETS[args.dataset]
        self.root = (PROJECT_ROOT / args.experiment_root).resolve()
        self.ds_root = self.root / args.dataset
        self.ds_root.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.root / ".gpu_launch.lock"
        # A second process for the VC-refine-independent evals keeps its own
        # manifest so the two processes never overwrite each other's record.
        manifest_name = "manifest_independent.json" if args.independent_evals_only else "manifest.json"
        self.manifest_path = self.ds_root / manifest_name
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.is_file() else {
            "dataset": args.dataset, "started_at": now(), "preset": self.preset, "vcr_settings": VCR_SETTINGS,
            "stages": {}, "evals": {},
        }
        self.env = os.environ.copy()
        self.env.update({"PYTHONUNBUFFERED": "1", "IMUHOI_PERSISTENT_WORKERS": "0",
                         "IMUHOI_SKIP_DATALOADER_PRIME": "1", "CUDA_VISIBLE_DEVICES": "0"})

    # ---------------------------------------------------------------- utilities
    def save(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))
        os.replace(tmp, self.manifest_path)

    def log(self, message):
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with (self.ds_root / "orchestrator.log").open("a") as handle:
            handle.write(line + "\n")

    def acquire_launch(self, kind):
        """Serialize launches across the dataset processes and wait for free memory."""
        while True:
            try:
                self.lock_dir.mkdir()
            except FileExistsError:
                time.sleep(20)
                continue
            if free_gpu_mb() >= MIN_FREE_MB[kind]:
                return
            self.lock_dir.rmdir()
            time.sleep(60)

    def run(self, kind, command, log_path):
        self.acquire_launch(kind)
        self.log("RUN " + " ".join(str(part) for part in command))
        try:
            handle = open(log_path, "a")
            process = subprocess.Popen([str(part) for part in command], cwd=PROJECT_ROOT, env=self.env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            # Hold the launch lock until this job's memory shows up in nvidia-smi.
            deadline = time.time() + SETTLE_SECONDS
            while time.time() < deadline and process.poll() is None:
                time.sleep(10)
        finally:
            if self.lock_dir.exists():
                self.lock_dir.rmdir()
        code = process.wait()
        handle.close()
        self.log(f"END rc={code} log={log_path}")
        return code

    def write_config(self, name, overrides):
        cfg = yaml.safe_load((PROJECT_ROOT / self.preset["base_config"]).read_text())
        cfg = copy.deepcopy(cfg)
        cfg.update(overrides)
        path = self.ds_root / name
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return path, cfg

    @staticmethod
    def find_stage(parent: Path, suffix: str, marker: str):
        if not parent.is_dir():
            return None
        for path in sorted(parent.iterdir(), reverse=True):
            if path.is_dir() and path.name.endswith(suffix) and (path / marker).is_file():
                return path
        return None

    def train_stage(self, key, script, cfg_path, cfg, parent, suffix, extra):
        done = self.find_stage(parent, suffix, "final.pt")
        attempt = 0
        while done is None:
            attempt += 1
            if attempt > self.args.max_attempts:
                raise RuntimeError(f"{key} failed {attempt - 1} times")
            command = [PYTHON, script, "--cfg", cfg_path, "--save_dir", parent, "--run_suffix", suffix,
                       "--seed", self.args.seed, "--train_datasets", self.args.dataset, "--num_workers", 2,
                       "--simulate_imu_noise", "--epochs", cfg["epochs"], *extra]
            interrupted = self.find_stage(parent, suffix, "last.pt")
            if interrupted is not None:
                command += ["--resume_dir", interrupted]
            self.manifest["stages"][key] = {"status": "running", "attempt": attempt, "started_at": now()}
            self.save()
            parent.mkdir(parents=True, exist_ok=True)
            self.run("b0" if key == "b0" else "vcr", command, self.ds_root / f"{key}_train.log")
            done = self.find_stage(parent, suffix, "final.pt")
        self.manifest["stages"][key] = {"status": "complete", "stage_dir": str(done), "finished_at": now()}
        self.save()
        return done

    def evaluate(self, name, cfg_path, kind, hp, vc, ot, extra=()):
        out = self.ds_root / "evals"
        out.mkdir(exist_ok=True)
        result = out / f"{name}.json"
        if result.is_file():
            return
        command = [PYTHON, "scripts/cascade/eval_cascade_online.py", "--model_kind", kind,
                   "--extra_json", out / f"{name}_extra.json", "--config", cfg_path, "--dataset", self.args.dataset,
                   "--hp_ckpt", hp, "--vc_ckpt", vc, "--ot_ckpt", ot, "--velocity_contact_arch", "v5",
                   "--object_trans_arch", "v2", "--simulate_imu_noise", "--simulate_imu_noise_seed", self.args.seed,
                   "--seed", self.args.seed, "--num_workers", 2, "--online_batch_size", 48,
                   "--sequence_records_json", out / f"{name}_records.json", "--output_json", result, *extra]
        code = self.run("eval", command, out / f"{name}.log")
        self.manifest["evals"][name] = {"rc": code, "json": str(result), "finished_at": now()}
        self.save()

    # --------------------------------------------------------------------- main
    def independent_evals(self):
        """Baseline and B0 evals: they do not need VC-refine, so they can run during its training."""
        preset = self.preset
        hp, vc = PROJECT_ROOT / preset["hp"], PROJECT_ROOT / preset["vc"]
        b0_stage = self.find_stage(self.ds_root / "b0", f"{self.args.dataset}_cascade_b0_{self.args.run_tag}", "final.pt")
        if b0_stage is None:
            raise FileNotFoundError("B0 has not finished")
        base_stage, base_cfg = PROJECT_ROOT / preset["baseline_stage"], PROJECT_ROOT / preset["base_config"]
        b0_cfg_path = self.ds_root / "b0_config.yaml"
        self.evaluate("b0_best", b0_cfg_path, "b0", hp, vc, b0_stage / "best.pt")
        self.evaluate("b0_final", b0_cfg_path, "b0", hp, vc, b0_stage / "final.pt")
        for ckpt in ("final", "best"):
            self.evaluate(f"baseline_{ckpt}", base_cfg, "stock", hp, base_stage / f"{ckpt}_velocity_contact.pt",
                          base_stage / f"{ckpt}_object_trans.pt")
        self.log("INDEPENDENT EVALS DONE")

    def main(self):
        if self.args.independent_evals_only:
            return self.independent_evals()
        preset, seed = self.preset, self.args.seed
        hp, vc = PROJECT_ROOT / preset["hp"], PROJECT_ROOT / preset["vc"]
        b0_cfg_path, b0_cfg = self.write_config(
            "b0_config.yaml", {"object_trans_state_feedback": False, "save_dir": str(self.ds_root / "b0")}
        )
        vcr_cfg_path, vcr_cfg = self.write_config(
            "vcr_config.yaml",
            {"object_trans_state_feedback": False, "save_dir": str(self.ds_root / "vc_refine"), **VCR_SETTINGS},
        )
        tag = self.args.run_tag
        b0_stage = self.train_stage(
            "b0", "train/cascade/train_ot_b0.py", b0_cfg_path, b0_cfg, self.ds_root / "b0",
            f"{self.args.dataset}_cascade_b0_{tag}",
            ["--velocity_contact_arch", "v5", "--object_trans_arch", "v2", "--object_trans_state_feedback", "none",
             "--hp_ckpt", hp, "--vc_ckpt", vc],
        )
        ot_best = b0_stage / "best.pt"
        vcr_stage = self.train_stage(
            "vcr", "train/cascade/train_vc_refine.py", vcr_cfg_path, vcr_cfg, self.ds_root / "vc_refine",
            f"{self.args.dataset}_cascade_vcr_{tag}",
            ["--hp_ckpt", hp, "--vc_ckpt", vc, "--ot_ckpt", ot_best],
        )
        base_stage = PROJECT_ROOT / preset["baseline_stage"]
        base_cfg = PROJECT_ROOT / preset["base_config"]
        for ckpt in ("final", "best"):
            vcr = vcr_stage / f"{ckpt}.pt"
            self.evaluate(f"cascade_e2_{ckpt}", vcr_cfg_path, "cascade", hp, vc, ot_best,
                          ["--vc_refine_ckpt", vcr, "--cascade_mode", "e2"])
        self.evaluate("b0_best", b0_cfg_path, "b0", hp, vc, ot_best)
        self.evaluate("b0_final", b0_cfg_path, "b0", hp, vc, b0_stage / "final.pt")
        for ckpt in ("final", "best"):
            self.evaluate(f"baseline_{ckpt}", base_cfg, "stock", hp, base_stage / f"{ckpt}_velocity_contact.pt",
                          base_stage / f"{ckpt}_object_trans.pt")
        # Ablations: OT-refine re-fusing only the published frame (e1), and the
        # whole-window re-fusion alone with VC-refine switched off.
        self.evaluate("cascade_e1_final", vcr_cfg_path, "cascade", hp, vc, ot_best,
                      ["--vc_refine_ckpt", vcr_stage / "final.pt", "--cascade_mode", "e1"])
        self.evaluate("cascade_e1_best", vcr_cfg_path, "cascade", hp, vc, ot_best,
                      ["--vc_refine_ckpt", vcr_stage / "best.pt", "--cascade_mode", "e1"])
        self.evaluate("cascade_e2_identity", vcr_cfg_path, "cascade", hp, vc, ot_best,
                      ["--cascade_identity_vc_refine", "--cascade_mode", "e2"])
        if self.args.dataset == "behave":
            # The original proposal also refilled the HumanPose history with refined values.
            self.evaluate("cascade_e2_final_hprefined", vcr_cfg_path, "cascade", hp, vc, ot_best,
                          ["--vc_refine_ckpt", vcr_stage / "final.pt", "--cascade_mode", "e2",
                           "--cascade_hp_history", "refined"])
            self.evaluate("cascade_e1_final_hprefined", vcr_cfg_path, "cascade", hp, vc, ot_best,
                          ["--vc_refine_ckpt", vcr_stage / "final.pt", "--cascade_mode", "e1",
                           "--cascade_hp_history", "refined"])
            # The first e2 implementation started each window from the value published
            # at its first frame (59 interleaved chains); kept to document that failure.
            self.evaluate("cascade_e2p_final", vcr_cfg_path, "cascade", hp, vc, ot_best,
                          ["--vc_refine_ckpt", vcr_stage / "final.pt", "--cascade_mode", "e2p"])
        self.manifest["finished_at"] = now()
        self.save()
        self.log("ALL DONE")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--experiment_root", default=DEFAULT_ROOT)
    parser.add_argument("--run_tag", default="20261010")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--independent_evals_only", action="store_true",
                        help="only the baseline/B0 evals (needs B0 finished); safe beside a running orchestrator")
    return parser.parse_args()


if __name__ == "__main__":
    Runner(parse_args()).main()
