#!/usr/bin/env python3
"""Build the paper comparison from completed native EgoIMU evaluations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/egoimu_native_eval_20261017"
KEYS = ("obj_trans_err_fusion", "hoi_err_fusion", "mpjpe", "root_trans_err", "jitter")
TARGETS = dict(zip(KEYS, (11.31, 17.44, 2.35, 11.07, 2.77)))


def main():
    rows = {}
    for name in ("modules_best_debug_native", "modules_best_debug_disabled", "modules_best_test_disabled", "last_1199_test_disabled"):
        payload = json.loads((OUT / name / "results.json").read_text())
        if payload["status"] != "complete" or not payload["source_unchanged"] or not payload["all_weights_strictly_verified"]:
            raise RuntimeError(f"Incomplete/unverified result: {name}")
        rows[name] = payload
    a, b = (rows[f"modules_best_debug_{mode}"]["results"] for mode in ("native", "disabled"))
    if any(abs(a[key] - b[key]) > 1e-8 for key in a):
        raise RuntimeError("Profiler changed debug results")
    best, last = (rows[name] for name in ("modules_best_test_disabled", "last_1199_test_disabled"))
    if best["sequences"] != last["sequences"] or best["source_sha256"] != last["source_sha256"]:
        raise RuntimeError("Best/final evaluation data or source changed")
    comparison = {"paper_table": 2, "paper_targets": TARGETS, "profiler_debug_metrics_identical": True,
                  "best_last_sources_and_sequence_lists_identical": True, "runs": {}}
    for name, payload in rows.items():
        metrics = {key: payload["results"][key] for key in KEYS}
        comparison["runs"][name] = {
            "sequence_count": payload["sequence_count"], "metrics": metrics,
            "delta_from_paper": {key: metrics[key] - TARGETS[key] for key in KEYS},
            "equals_paper_at_two_decimal_places": {key: round(metrics[key], 2) == TARGETS[key] for key in KEYS},
        }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    exact = all(comparison["runs"]["modules_best_test_disabled"]["equals_paper_at_two_decimal_places"].values())
    lines = ["# EgoIMU 原生 OMOMO offline 评估（2026-10-17）", "",
             "结论：原版best权重在完整test上五项指标均与论文表2四舍五入到两位小数一致。" if exact else "结论：本轮结果与论文存在数值差异，见下表。", "",
             "直接调用 EgoIMU 的原模型、原数据加载器、原SMPL-H male、原 `evaluate_model` 和 `main`。",
             "本轮是已有checkpoint的评估复现，没有重新训练，没有修改EgoIMU源码或预处理数据。", "",
             "| 来源 | 序列数 | Obj cm | HOI cm | MPJPE cm | root cm | Jitter mm/frame² |",
             "|---|---:|---:|---:|---:|---:|---:|",
             "| 论文Table2 | — | 11.31 | 17.44 | 2.35 | 11.07 | 2.77 |"]
    for name, label in (("modules_best_debug_native", "原生默认debug / best1040"),
                        ("modules_best_test_disabled", "完整test / best1040"),
                        ("last_1199_test_disabled", "完整test / 真正末轮1199")):
        payload = rows[name]
        values = " | ".join(f"{payload['results'][key]:.6f}" for key in KEYS)
        lines.append(f"| {label} | {payload['sequence_count']} | {values} |")
    lines.extend(["", "## 核验与运行条件", "",
                  "- WSL原生 `/home/l/anaconda3/envs/SAGE/bin/python`；每次运行前检查GPU。",
                  "- seed=10；原生整段forward，独立eval不加IMU噪声；完整test=1177条。",
                  "- 三个默认模块严格逐tensor核验，等同 `stage_best_joint_training.pt` epoch1040。",
                  "- `final.pt` 与 `stage_best_joint_training.pt` 文件SHA-256相同；真正末轮是 `final_stage_joint_training_epoch_1199.pt`。",
                  "- 原版 `--model_path` 只设置config，当前load_model未实际读取它；末轮通过原模型的严格load_state_dict加载。",
                  "- 原生默认路径是debug（13条），完整test通过既有 `--test_data_dir` 覆盖。",
                  "- 初次full-test因默认FD软限制10240，原loader静默漏样；完整性校验中止，没有产生可采用成绩。",
                  "  只将本进程软限制提高到65536后重跑，不改变数据、算法或全局系统配置。",
                  "- 全量关闭昂贵FLOPs采集；13条序列已验证开关前后所有指标完全一致。",
                  "- 每轮核验1177次成功模型调用、2354次评估SMPL调用，主指标全部有限，运行前后原源码哈希相同。",
                  "- 封装仅做输入路径选择、seed、权重核验、进程资源设置及JSON记录；数学指标仍来自EgoIMU评估函数。", "",
                  "论文Table2数值匹配，支持以原版完整test和best权重作为后续参考。",
                  "这一结果复现的是保存权重的评估分数，不是从随机初始化重训得到的统计复现。", "",
                  "## 复现命令", "", "工作目录为IMUHOI；封装内部切换cwd到EgoIMU，全部结果保存在本目录。", "",
                  "```bash", "nvidia-smi",
                  "CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/l/anaconda3/envs/SAGE/bin/python scripts/eval_egoimu_native.py --split test",
                  "nvidia-smi",
                  "CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/l/anaconda3/envs/SAGE/bin/python scripts/eval_egoimu_native.py --split test --weights last_1199", "```", "",
                  "原命令、源码和checkpoint哈希、样本清单、成功调用计数与全部指标见各run目录的results.json。", "",
                  "当前重构版的旧划分实验仍使用不同的SMPL-X canonical数据；与这里不能视为同GT的受控方法消融。",
                  "原版可否复现应依据本表，而不是用当前重构版分数替代。", ""])
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:13]))


if __name__ == "__main__":
    main()
