"""Produce the reviewable preprocessing audit from saved diagnostic results."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/analysis/omomo_preprocessing_audit_20261016"
OLD = ROOT.parent / "EgoIMU/process/processed_split_data_OMOMO"
NEW = ROOT / "process/IMUHOI++cano/processed_split_data_OMOMO_full"


def main():
    audit = json.loads((OUT / "audit.json").read_text())
    subgroups = json.loads((OUT / "offline_split_subgroups.json").read_text())
    old_info = json.loads((OLD / "split_info.json").read_text())
    new_info = json.loads((NEW / "split_info.json").read_text())
    with (OUT / "per_sequence.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if audit["identity_mismatches"] or audit["frame_count_mismatches"]:
        raise ValueError("Identity/frame mismatch; review pairing before interpreting the audit")
    seeds = []
    universe = {p.name for p in (NEW / "unified").glob("*.pt")}
    for universe_name, names in (("all_5882", universe), ("assigned_5878", universe - set(new_info["unassigned_files"]))):
        for order_name, ordered in (("lexical", sorted(names)), ("numeric", sorted(names, key=lambda n: int(Path(n).stem)))):
            for seed in (0, 1, 10, 42, 2024, 2025, 2026):
                names_copy = list(ordered)
                random.Random(seed).shuffle(names_copy)
                for rounding, count in (("floor", int(len(ordered) * .8)), ("round", int(round(len(ordered) * .8)))):
                    train = set(names_copy[:count]) - set(new_info["unassigned_files"])
                    if train == set(new_info["train_files"]):
                        seeds.append({"universe": universe_name, "order": order_name, "seed": seed, "rounding": rounding})
    omitted = []
    for name in new_info["unassigned_files"]:
        data = torch.load(NEW / "unified" / name, map_location="cpu", weights_only=False)
        omitted.append({"file": name, "seq_name": str(data["seq_name"]), "frames": len(data["obj_trans"]),
                        "old_split": "train" if name in old_info["train_files"] else "test"})
    filter_counts = {}
    for label in ("old", "new"):
        filter_counts[label] = {}
        for split in ("train", "test"):
            group = [r for r in rows if r[f"{label}_split"] == split]
            filter_counts[label][split] = {"files": len(group),
                "long_enough_61": sum(int(r[f"{label}_frames"]) >= 61 for r in group),
                "object_move_at_least_10_frames": sum(float(r[f"obj_contact_{label}_positive"] or 0) >= 10 for r in group
                                                       if r[f"obj_contact_{label}_positive"]),
                "short_files": [r["file"] for r in group if int(r[f"{label}_frames"]) < 61]}
    # Omitted rows have no paired contact counts; read their original labels directly.
    for row in rows:
        if row["new_split"] == "missing":
            data = torch.load(OLD / row["old_split"] / row["file"], map_location="cpu", weights_only=False)
            if int(torch.as_tensor(data["obj_contact"]).sum()) >= 10:
                filter_counts["old"][row["old_split"]]["object_move_at_least_10_frames"] += 1
    details = {"split_order_seed_reconstruction_candidates": seeds, "unassigned_sequences": omitted,
               "loader_filter_counts": filter_counts}
    (OUT / "supplement.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    paired = audit["paired_statistics"]["all"]
    dist = paired["distributions"]
    def fmt(key, stat="mean", digits=4):
        if key in dist:
            value = dist[key][stat]
        else:
            values = [float(r[key]) for r in rows if r.get(key)]
            value = float(np.mean(values)) if stat == "mean" else max(values)
        return f"{value:.{digits}f}"
    lines = ["# OMOMO：EgoIMU 与当前预处理结果审计（2026-10-16）", "",
        "## 结论", "",
        "**数据划分确实发生了大幅变化，不能把上一轮物体误差差距直接归因于模型或噪声。**",
        "但现有证据不足以断言划分就是全部原因：骨架、手物锚点、接触标签和 HOI 计分口径也不同。",
        "本次只读加载两边保存的数据；所有计算、推理和诊断脚本均来自当前仓库，未导入或运行 EgoIMU 代码。", "",
        "## 1. 两份 split 不是同一划分", "",
        "| | EgoIMU 保存的 split | 当前实验使用的 split |", "|---|---:|---:|",
        "| 训练文件 | 4705 | 4705 |", "| 测试文件 | 1177 | 1173 |",
        "| 已分配总文件 | 5882 | 5878 |", "",
        "| 原版归属 | 当前归属 | 序列数 |", "|---|---|---:|"]
    for entry in audit["split_matrix_by_filename"]:
        lines.append(f"| {entry['old']} | {entry['new']} | {entry['n']} |")
    lines += ["", "旧测试集与当前测试集仅重合 **223 条**；当前测试集 **950/1173（80.99%）**来自旧训练集。",
        "另有旧测试集的 **954 条**进入了本轮训练集。逐文件读取 `seq_name` 和帧数后，",
        f"全部 **{paired['paired_sequences']} 对**匹配，身份/帧数不匹配均为 0，因此这不是文件重编号的假象。",
        "两份 split 在各自内部的 `seq_name` train/test 交叉数均为 0；这不等于验证了更高层原始捕捉片段的独立性。",
        "源目录的 manifest 与实际 train/test 文件清单也一致。", "",
        "当前另有 4 条留在 unified、未分配进 train/test：", ""]
    for item in omitted:
        lines.append(f"- `{item['file']}` / `{item['seq_name']}`，{item['frames']} 帧，原版属于 {item['old_split']}。")
    seed_note = json.dumps(seeds, ensure_ascii=False) if seeds else "未找到完全匹配；历史划分的具体生成参数尚不能确定"
    lines += ["", f"对排序方式、常见 seed 和取整方式的重建结果：{seed_note}。",
        "划分源路径只追溯到当前旧目录的 reference manifest；原目录已不在当前工作树，不能仅据生成脚本断定历史操作。", "",
        "## 2. 文件内容的变化（同一序列配对）", "",
        "| 项目 | EgoIMU | 当前 |", "|---|---|---|",
        "| 人体模型 | 预处理代码固定 SMPL-H male；`gender` 字段只是来源标签 | SMPL-X neutral / zero betas |",
        "| 保存关节数 | 22 | 24（body22 + 两个掌根） |",
        "| 姿态保存 | 22 关节 6D | 24 关节轴角，同时缓存全局旋转 |",
        "| 手物几何锚点 | wrist 20/21 | palm 22/23 |",
        "| 地面 | 原预处理坐标 | 每序列 min-foot-Y=0，共同平移人和物体 |",
        "| 手接触清理 | 连续段最短 3 帧 | 填洞最多30帧、最短段30帧，且基于新的手部几何 |",
        "| 物体运动标签 | 原始运动判别 + 最短3帧清理 | 6 Hz低通、±5帧局部窗口、1cm/3°阈值、最短15帧，不填洞 |",
        "| IMU 特征 | loader 按序列位置/旋转合成 | 预先缓存、由 canonical 关节合成 |", "",
        "配对统计（先每序列统计，再取序列平均；标签比率另用帧加权）：", "",
        "| 差异 | 平均 | 说明 |", "|---|---:|---|",
        f"| body22 根相对关节位置差 | {fmt('body22_root_relative_mean_cm')} cm | 去除根平移后仍有骨架差异 |",
        f"| body22 去物体常量平移后位置差 | {fmt('body22_shift_removed_mean_cm')} cm | 包含根/骨架几何变化 |",
        f"| 相同 wrist 的手物向量差 | {fmt('hand_object_vector_wrist_to_wrist_delta_cm')} cm | 骨架变化 |",
        f"| 旧 wrist → 当前 palm 手物向量差 | {fmt('hand_object_vector_old_wrist_to_new_palm_delta_cm')} cm | 实际 OT 监督锚点变化 |",
        f"| 当前 wrist 到 palm 距离 | {fmt('new_wrist_to_palm_cm')} cm | 锚点本身间距 |",
        f"| human IMU 加速度特征 RMSE | {fmt('human_acc_feature_rmse_m_s2')} m/s² | 相同当前合成公式，比较两份骨架输入 |",
        f"| object IMU 加速度特征 RMSE | {fmt('object_acc_feature_rmse_m_s2')} m/s² | 共同常量平移理论上不改加速度；差分有浮点误差 |",
        f"| body22 旋转最大逐元素差 | {fmt('rotation_body22_max_abs', 'max', 8)} | 全部配对的最大值 |",
        f"| 物体旋转最大逐元素差 | {fmt('object_rotation_max_abs', 'max', 8)} | 全部配对的最大值 |",
        f"| 物体平移去每序列常量偏移后最大差 | {fmt('object_shift_removed_max_m', 'max', 8)} m | 检查轨迹形状是否改变 |", "",
        "物体位移常量为每序列独立的竖直平移，不应直接当作模型位置误差；人和物体同时移动，",
        "手物相对误差以及纯平移不变的速度/加速度不因该地面对齐直接变差。", "",
        "| 标签 | 旧正例比例 | 新正例比例 | 不一致帧比例 | 有变化的配对序列 |", "|---|---:|---:|---:|---:|"]
    for key, values in paired["contact_rates"].items():
        lines.append(f"| {key} | {100*values['old_fraction']:.2f}% | {100*values['new_fraction']:.2f}% | {100*values['disagree_fraction']:.2f}% | {values['changed_sequences']} |")
    lines += ["", "这些变化同时影响 VC 的监督、OT 的接触掩码和手物几何监督，以及 HOI 指标统计的帧集合。", "",
        "## 3. 数据加载器的过滤也不同", "",
        "旧 `IMUDataset` 默认要求物体运动标签 >=10帧（包括 eval 未覆盖该参数时）。",
        "当前默认门槛为0、按双手接触计数；本轮完整 test 加载1173条，训练因窗口长度实际加载4699条。",
        "以下是保存文件的可复核过滤计数，不能把文件数直接等同于每个历史训练阶段实际见过的样本数：", "",
        "| 数据 | split | 文件 | 长度>=61帧 | obj_contact>=10帧 |", "|---|---|---:|---:|---:|"]
    for label, splits in filter_counts.items():
        for split, values in splits.items():
            lines.append(f"| {label} | {split} | {values['files']} | {values['long_enough_61']} | {values['object_move_at_least_10_frames']} |")
    lines += ["", "## 4. 当前模型在两组未见测试序列上的 offline 误差", "",
        "使用本轮同一 final 权重重新评估全部当前测试序列，关闭额外噪声，未使用 EgoIMU 模型。", "",
        "| 当前测试子集 | 序列数 | Obj / cm | HOI / cm | 平均时长 / s |", "|---|---:|---:|---:|---:|"]
    for key, label in (("all_current_test", "当前完整测试集"), ("shared_old_and_current_test", "与旧测试重合"), ("old_train_now_current_test", "旧 train → 当前 test")):
        group = subgroups["groups"][key]
        m = group["metrics"]
        lines.append(f"| {label} | {group['n']} | {m['obj_trans_err_fusion_published_cm']['mean']:.4f} | {m['hoi_err_fusion_cm']['mean']:.4f} | {m['duration_s']['mean']:.3f} |")
    lines += ["", f"复核的四项总体误差相对上一轮 JSON 的最大绝对差：{subgroups['previous_run_max_abs_metric_difference']:.9f}。",
        "分组比较只能说明当前模型在这两组的表现差异；不能等价于‘相同模型分别在新旧 split 重训’的因果实验。",
        "**旧 test 中954条已用于本轮训练，不能用本轮权重直接跑旧1177条后称为论文复现。**", "",
        "## 5. 另两处必须核实的历史评估问题", "",
        "1. EgoIMU 当前 `eval_IMUHOI.py:119` 默认指向 `processed_split_data_OMOMO/debug`，目录内只有13条。",
        "   该脚本支持 `--test_data_dir` 覆盖；没有论文运行日志，不能断言论文只测了这13条。debug身份和来源清单见 audit.json。",
        "2. 当前保存代码的 HOI 公式不同：EgoIMU `eval_IMUHOI.py:541-573` 在接触帧算手物相对误差，",
        "   **还在双手无接触帧加入物体相对根的误差**；当前 `utils/hoi_metrics.py:6` 只算 GT 手接触帧。",
        "   同时旧锚点为手腕、当前为掌根。此前18.82 vs论文17.44只是数值并列，尚未对齐统计定义。",
        "   两边 MPJPE evaluator 都取前22个关节；当前存24关节不表示评估平均了24关节。", "",
        "## 6. 建议的最短验证顺序", "",
        "1. 首先核实论文真正使用的 test 路径（完整1177还是其它覆盖路径）与 HOI 计分定义。",
        "2. 增加一个独立 split 对照实验：只把**当前预处理内容**按 EgoIMU 的真实序列身份划分，",
        "   包含 unified 中未分配的4条；噪声、模型、seed、训练预算不变，从 HP 重新训练。",
        "   原目录和现有实验保留。这个对照才可量化 split 的作用。",
        "3. 若仍有显著差距，再单独控制 wrist/palm、接触标签和 SMPL-H/SMPL-X 骨架，避免同时改变所有变量。",
        "   若追求严格历史复现，需要用当前仓库重新实现历史预处理协议；本次没有复制或运行旧代码。",
        "4. 基线与论文对照口径确认后，再在同一划分和噪声下训练 IMUHOI++，同时报告 offline 和 online。", "",
        "本次未启动新训练、未改写任一数据集和既有实验。", "",
        "## 机器可读产物", "",
        "- `audit.json`：全量配对、split、标签和数据分布统计。",
        "- `per_sequence.csv`：每条样本的旧/新归属、身份、几何和标签差异。",
        "- `inventories.json`：两边按 split 的序列身份和运动统计。",
        "- `supplement.json`：排序/seed重建、未分配样本、loader过滤计数。",
        "- `offline_sequence_records.json`、`offline_split_subgroups.json`：本轮模型未见测试集的分组评估。",
        "- `offline_final_recheck.json`：复核的 offline 总体指标。"]
    path = ROOT / "docs/omomo_preprocessing_audit_20261016.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(details, indent=2, ensure_ascii=False), flush=True)
    print("REPORT", path, flush=True)


if __name__ == "__main__":
    main()
