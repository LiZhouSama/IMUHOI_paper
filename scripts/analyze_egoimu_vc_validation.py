#!/usr/bin/env python3
"""Audit completed VC interventions and summarize paired sequence outcomes."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import OUT, EXP, BRIDGE, PYTHON, ARMS, check_gpu, verify_receipt, write_json, now


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def differences(a, b, prefix=""):
    result = {}
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}.{key}" if prefix else key
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            result.update(differences(x, y, path))
        elif x != y:
            result[path] = {"from": x, "to": y}
    return result


def main():
    import numpy as np
    import torch
    import yaml

    protocol = read(OUT / "protocol.json")
    verify_receipt(protocol)
    if read(OUT / "status.json")["status"] != "complete":
        raise RuntimeError("The five-arm queue is not complete")
    rows, paired, configurations, checks = {}, {}, {}, {}
    metric_keys = ("mpjpe", "root_trans_err", "obj_trans_err_fusion", "hoi_err_fusion",
                   "contact_f1_lhand", "contact_f1_rhand", "contact_f1_obj")
    for arm in ARMS:
        directory = OUT / arm
        initial = read(directory / "initialization.json")
        checkpoint = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
        expected = protocol["arms"][arm]["steps_per_epoch"] * 300
        if checkpoint["epoch"] != 299 or checkpoint["n_iter"] != expected:
            raise RuntimeError(f"Fixed-budget mismatch: {arm}")
        optimizer_steps = sorted({int(state["step"]) for state in checkpoint["optimizer_state_dict"]["state"].values() if "step" in state})
        if optimizer_steps != [expected] or not initial["strict_load"] or initial["train_sequences"] != 4115:
            raise RuntimeError(f"Optimizer or initialization mismatch: {arm}")
        if initial["initial_checkpoint_sha256"] != protocol["initial_checkpoint_sha256"]:
            raise RuntimeError(f"Different initial weights: {arm}")
        component = read(directory / "component_final.json")
        sequence_rows = read(directory / "component_sequences.json")
        paired[arm] = {row["file"]: row for row in sequence_rows}
        if component["n"] != 1177 or len(paired[arm]) != 1177:
            raise RuntimeError(f"Incomplete paired component test: {arm}")
        for key, value in component["metrics"].items():
            reconstructed = sum(row[key] for row in sequence_rows) / 1177
            if not math.isclose(reconstructed, value, abs_tol=1e-9):
                raise RuntimeError(f"Aggregate differs from sequence records: {arm}/{key}")
        configurations[arm] = yaml.safe_load((directory / "config.yaml").read_text())
        rows[arm] = {"component": component["metrics"]}
        coverage = {}
        for mode, inference in (("offline", "offline"), ("online", "batch_online")):
            evaluation = read(directory / f"current_pipeline_{mode}_final.json")
            if (evaluation["inference_mode"] != inference or evaluation["hoi_metric_protocol"] != "egoimu"
                    or evaluation["hoi_hand_anchor"] != "wrists_20_21" or evaluation["seed"] != 10):
                raise RuntimeError(f"Evaluation protocol mismatch: {arm}/{mode}")
            log = (directory / f"current_{mode}_eval.log").read_text(errors="replace")
            coverage[mode] = {"progress_reached_1177": "1177/1177" in log, "traceback_count": log.count("Traceback")}
            if not coverage[mode]["progress_reached_1177"] or coverage[mode]["traceback_count"]:
                raise RuntimeError(f"Incomplete or failed pipeline evaluation: {arm}/{mode}")
            metrics = evaluation["results"]["omomo"]
            if not all(math.isfinite(metrics[key]) for key in metric_keys):
                raise RuntimeError(f"Nonfinite scored metric: {arm}/{mode}")
            rows[arm][mode] = {key: metrics[key] for key in metric_keys}
        checks[arm] = {"epoch": checkpoint["epoch"], "optimizer_steps": optimizer_steps,
                       "train_sequences": initial["train_sequences"], "test_sequences": len(paired[arm]),
                       "same_initialization": True, "coverage": coverage}
        del checkpoint
    for mode in ("offline", "online"):
        for key in ("mpjpe", "root_trans_err"):
            if len({row[mode][key] for row in rows.values()}) != 1:
                raise RuntimeError("Frozen HP metric changed")
    names = sorted(paired["w60_v1"])
    if not all(set(value) == set(names) for value in paired.values()):
        raise RuntimeError("Unpaired test identities")

    effects = {}
    contrasts = (("weight_at_w60", "w60_v1", "w60_v10"),
                 ("weight_at_w120", "w120_v1", "w120_v10"),
                 ("window_at_v1", "w60_v1", "w120_v1"),
                 ("window_at_v10", "w60_v10", "w120_v10"),
                 ("combined", "w60_v1", "w120_v10"),
                 ("extra_legacy_bundle", "w120_v10", "legacy_recipe"))
    for label, a, b in contrasts:
        entry = {"from": a, "to": b, "current_pipeline": {}, "paired_native_context": {}}
        for mode in ("offline", "online"):
            entry["current_pipeline"][mode] = {}
            for key in ("obj_trans_err_fusion", "hoi_err_fusion"):
                x, y = rows[a][mode][key], rows[b][mode][key]
                entry["current_pipeline"][mode][key] = {"delta_cm": y - x, "reduction_percent": 100 * (x - y) / x}
        for key in rows[a]["component"]:
            x = np.array([paired[a][name][key] for name in names])
            y = np.array([paired[b][name][key] for name in names])
            delta = y - x
            entry["paired_native_context"][key] = {"mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                "improved_sequence_fraction": float((delta < 0).mean()), "delta_p10": float(np.quantile(delta, .1)),
                "delta_p90": float(np.quantile(delta, .9))}
        effects[label] = entry
    factorial = {}
    for mode in ("offline", "online"):
        factorial[mode] = {}
        for key in ("obj_trans_err_fusion", "hoi_err_fusion"):
            a, b, c, d = [rows[arm][mode][key] for arm in ("w60_v1", "w60_v10", "w120_v1", "w120_v10")]
            factorial[mode][key] = {"mean_weight_effect_cm": ((b-a)+(d-c))/2,
                                    "mean_window_effect_cm": ((c-a)+(d-b))/2,
                                    "difference_in_differences_cm": d-c-b+a}
    config_diffs = {label: differences(configurations[a], configurations[b]) for label, a, b in contrasts}
    allowed = {"weight_at_w60": {"save_dir", "imuhoi_loss_weights.obj_vel", "imuhoi_loss_weights.hand_vel"},
               "weight_at_w120": {"save_dir", "imuhoi_loss_weights.obj_vel", "imuhoi_loss_weights.hand_vel"},
               "window_at_v1": {"save_dir", "train_data_window"}, "window_at_v10": {"save_dir", "train_data_window"}}
    for key, fields in allowed.items():
        if set(config_diffs[key]) != fields:
            raise RuntimeError(f"Unexpected factorial config changes: {key}")
    with (ROOT / "outputs/analysis/omomo_preprocessing_audit_20261016/per_sequence.csv").open() as stream:
        inventory = {r["file"]: r for r in csv.DictReader(stream)}
    object_groups = {}
    for name in names:
        object_groups.setdefault(inventory[name]["old_object"], []).append(name)
    by_object = {obj: {"n": len(files), **{arm: sum(paired[arm][name]["integral_obj_cm"] for name in files) / len(files)
                                          for arm in ARMS}} for obj, files in object_groups.items()}
    object_wins = sum(group["w120_v10"] < group["w60_v1"] for group in by_object.values())
    references = {}
    for method in ("baseline", "proposed"):
        manifest = read(EXP / method / "experiment_manifest.json")
        pipeline = next(iter(manifest["pipelines"].values()))
        directory = Path(pipeline["config"]).parent
        references[method] = {}
        for mode, filename in (("offline", "eval_omomo_offline_final.json"), ("online", "eval_omomo_final.json")):
            result = read(directory / filename)["results"]["omomo"]
            references[method][mode] = {key: result[key] for key in metric_keys}
    references["native_egoimu"] = read(ROOT / "outputs/analysis/egoimu_native_eval_20261017/modules_best_test_disabled/results.json")["results"]
    report = {"created_at": now(), "verification": checks, "factorial_config_differences": config_diffs,
              "rows": rows, "effects": effects, "factorial_mean_effects": factorial, "original_references": references,
              "integral_error_by_object": by_object, "object_groups_improved_combined": object_wins,
              "object_group_count": len(object_groups),
              "limits": ["One training seed; paired file win rates do not measure training-seed uncertainty.",
                         "W120 uses twice as many frames per update as W60; the window effect includes that exposure increase.",
                         "Legacy bundle changes multiple settings and has 12300 vs 6000 updates; not a single-factor test.",
                         "Common 4115 members and paired crop bounds differ from the previous 4701-effective-member baseline.",
                         "Current canonical body/GT differ from paper; native-context comparisons separately fix original HP/OT and GT."]}
    # Native output may contain unscored NaNs; preserve only the four compared values.
    native = references["native_egoimu"]
    report["original_references"]["native_egoimu"] = {key: native[key] for key in ("mpjpe", "root_trans_err", "obj_trans_err_fusion", "hoi_err_fusion")}
    write_json(OUT / "analysis_results.json", report)

    order = ("w60_v1", "w60_v10", "w120_v1", "w120_v10", "legacy_recipe")
    labels = {"w60_v1": "60帧，速度权重1", "w60_v10": "60帧，速度权重10", "w120_v1": "120帧，速度权重1",
              "w120_v10": "120帧，速度权重10", "legacy_recipe": "恢复原版VC配方（组合）"}
    lines = ["# EgoIMU VC重训结果分析（2026-10-18）", "", "五组均完成300轮，主结果使用预先指定的末轮epoch299；所有offline/online及组件评估覆盖1177条完整序列。",
             "共享seed10、4115条训练成员、相同随机初始权重、当前canonical预处理及高斯σ=0.1训练噪声。独立eval不加噪声，HOI使用EgoIMU手腕/非接触root口径。", "",
             "## 结论", "", "已验证的主要原因是VC训练配方：速度监督相对接触监督过弱、60帧训练窗口下的累积漂移更大。",
             "在当前实现、当前预处理、固定HP/OT下，只改速度权重和训练窗口就能显著恢复精度；预处理差异不是当前物体精度差距的必要解释。",
             "这不等同于证明所有训练因素已分离，也不是把canonical结果宣称为严格论文复现。", "",
             "## 当前流水线：仅替换VC，HP/OT固定", "",
             "| VC训练设置 | offline Obj cm | offline HOI cm | online Obj cm | online HOI cm |",
             "|---|---:|---:|---:|---:|"]
    for arm in order:
        row = rows[arm]
        lines.append(f'| {labels[arm]} | {row["offline"]["obj_trans_err_fusion"]:.4f} | {row["offline"]["hoi_err_fusion"]:.4f} | {row["online"]["obj_trans_err_fusion"]:.4f} | {row["online"]["hoi_err_fusion"]:.4f} |')
    lines += ["", "五组人体指标完全一致：offline MPJPE=2.3521、root=9.0861 cm；online MPJPE=2.8022、root=8.8208 cm，确认提升来自VC替换而非HP改变。", "",
              "## 四组对照支持什么", ""]
    for label, title in (("weight_at_w60", "60帧下速度权重1→10"), ("window_at_v1", "权重1下窗口60→120"),
                         ("weight_at_w120", "120帧下速度权重1→10"), ("window_at_v10", "权重10下窗口60→120"),
                         ("combined", "窗口和权重共同恢复")):
        value = effects[label]["current_pipeline"]["offline"]
        lines.append(f'- {title}：Obj下降{-value["obj_trans_err_fusion"]["delta_cm"]:.4f} cm（{value["obj_trans_err_fusion"]["reduction_percent"]:.2f}%），HOI下降{-value["hoi_err_fusion"]["delta_cm"]:.4f} cm。')
    lines += ["", "四组配置差异已逐键核验：除保存目录外，权重对照只改obj_vel/hand_vel两项权重，窗口对照只改训练窗口。",
              "接触预测网络的输入包含未detach的物体/手速度，因此接触损失会反向影响速度头；权重结果与加强速度约束的解释一致，但没有据此声称接触梯度冲突已单独测量。",
              "120帧每次更新也多看一倍帧，当前实验不能把长程依赖收益与更多训练帧的收益分开。", "",
              "## 速度偏置与积分漂移", "", "以下指标在同一套原版完整序列输入上测量；偏置为每条序列速度误差向量均值的模，再按序列平均。", "",
              "| 设置 | 逐坐标速度RMSE m/s | 速度偏置 m/s | 纯积分Obj cm | 末帧Obj cm | 固定原版HP/OT Obj cm |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in order:
        value = rows[arm]["component"]
        lines.append(f'| {labels[arm]} | {value["velocity_rmse_mps"]:.5f} | {value["bias_norm_mps"]:.5f} | {value["integral_obj_cm"]:.4f} | {value["end_obj_cm"]:.4f} | {value["frozen_original_hp_ot_obj_cm"]:.4f} |')
    pair = effects["combined"]["paired_native_context"]
    lines += ["", f'窗口和权重共同恢复后，{pair["integral_obj_cm"]["improved_sequence_fraction"]*100:.1f}%的测试序列纯积分误差下降，'
              f'{pair["bias_norm_mps"]["improved_sequence_fraction"]*100:.1f}%的测试序列速度偏置下降。按物体类别统计，{object_wins}/{len(object_groups)}类纯积分均值改善。',
              "这些是本次训练的配对样本分布，不能代替多随机种子重复实验。", "",
              "## 原版配方组合的额外收益不能单独归因", "",
              "恢复组合在当前HP/OT上比120帧/权重10再好约0.37 cm Obj、0.39 cm HOI；但固定原版HP/OT时，组合Obj=8.8799略差于8.7514，组合HOI=13.7114优于15.7555。",
              "因此，组合不是各项指标均占优，也不能据此认定FP32、窗口内重算或某个学习率设置单独是原因。组合还同时改变batch、采样方式、更新次数（12300对6000）、梯度裁剪和LR。",
              "仅120帧/权重10保留缓存切片、BF16、分组LR和裁剪就已恢复，故这些设置无需全部回退才能解决主要精度损失。", "",
              "## 与论文、此前IMUHOI++的关系", "",
              "原版EgoIMU原生模型/eval已在1177条复现论文：Obj=11.3056、HOI=17.4436 cm。固定原版HP/OT和原版评估GT，仅换此次VC后，组合为8.8799/13.7114，120帧/权重10为8.7514/15.7555；二者均优于原版数值。",
              "当前canonical流水线组合为8.8398/14.5234，优于此前基线16.0125/19.7057；此比较还含训练成员和采样范围差异，不能把所有改善都算到单个参数。",
              "此前IMUHOI++为offline Obj=13.3703、HOI=10.8512 cm；修正后的基线物体误差更低，而++的HOI仍更低。因此不能沿用旧训练配方下的比较宣称++在所有指标更优。", "",
              "## 下一步", "",
              "先将120帧、速度监督权重10设为论文兼容对照的明确配置，保留已有正式结果；在相同有效成员、seed/噪声、HOI口径下重新进行基线OT/联合阶段与IMUHOI++比较，记录真实更新次数和训练帧数。",
              "原版配方组合的0.37 cm额外Obj收益可以随后分解，优先级低于重建公平的baseline/++比较。现有实验足以证明当前实现能达到原版物体精度，无需先整体退回旧预处理或重写VC网络。", "",
              "## 核验与文件", "",
              "analysis_results.json保存逐组指标、逐键配置差异、配对样本胜率和物体类别统计；五组final.pt实际optimizer step均已核验，源码与训练张量receipt未改变。",
              "component_sequences.json为组件级逐序列结果；current_pipeline_*_final.json为完整流水线指标。报告只做统计，不改训练模型或启动新训练。", ""]
    (OUT / "analysis.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for weight, color in ((1, "#5a77a3"), (10, "#179782")):
        chosen = [f"w60_v{weight}", f"w120_v{weight}"]
        axes[0].plot([60, 120], [rows[arm]["offline"]["obj_trans_err_fusion"] for arm in chosen], "o-", color=color, label=f"Velocity weight {weight}")
        axes[1].plot([60, 120], [rows[arm]["component"]["bias_norm_mps"] * 100 for arm in chosen], "o-", color=color)
    axes[0].set(xlabel="Training window (frames)", ylabel="Offline object error (cm)", xticks=[60, 120])
    axes[1].set(xlabel="Training window (frames)", ylabel="Mean velocity bias norm (cm/s)", xticks=[60, 120])
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"factorial_effects.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps({"verification": "passed", "effects": factorial, "paired_combined": pair,
                      "object_groups_improved": [object_wins, len(object_groups)], "report": str(OUT / "analysis.md")}, indent=2))


if __name__ == "__main__":
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use WSL native SAGE")
    check_gpu(minimum=0)
    main()
