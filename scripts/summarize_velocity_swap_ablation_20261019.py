"""Summarize completed velocity interventions without changing experiment state."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, write_json, now, sha
from scripts.run_velocity_swap_ablation_20261019 import OUT, RUN
from scripts.run_omomo_original_recipe_20261018 import read, verify
from scripts.probe_vc_joint_20261019 import clean


def metrics(path):
    value = read(path)
    assert value['hoi_metric_protocol'] == 'egoimu'
    return clean(value['results']['omomo'])


def paired_differences(path, reference):
    records = read(path.with_suffix('.diagnostics.json'))['sequences']
    assert len(records) == len({r['seq_path'] for r in records}) == 1177
    mapping = {r['seq_path']: r for r in reference}
    result = {}
    for new_key, old_key, label in (
        ('obj_trans_err_fusion_published_cm', 'obj_cm', 'fusion_obj'),
        ('hoi_err_fusion_cm', 'hoi_cm', 'hoi'),
    ):
        rows = [{'seq_path': r['seq_path'], 'delta_cm': r[new_key] - mapping[r['seq_path']][old_key]} for r in records]
        deltas = [r['delta_cm'] for r in rows]
        assert all(math.isfinite(d) for d in deltas)
        result[label] = {'mean_delta_cm': statistics.mean(deltas), 'median_delta_cm': statistics.median(deltas),
                         'improved_sequences': sum(d < 0 for d in deltas), 'total': len(deltas),
                         'largest_regressions': sorted(rows, key=lambda r: r['delta_cm'], reverse=True)[:10]}
    return result


def run():
    check_gpu(0)
    verify(read(RUN / 'protocol.json'))
    manifest = read(OUT / 'manifest.json')
    assert all(sha(Path(p)) == digest for p, digest in manifest['fixed_sha256'].items())
    results, differences, velocity, gating = {}, {}, {}, {}
    for mode in ('offline', 'online'):
        results[mode] = {'original_pp': metrics(RUN / f'proposed/eval_{mode}_best.json'),
                         'baseline_reference': metrics(RUN / f'baseline/eval_{mode}_best.json')}
        reference = read(RUN / f'proposed/eval_{mode}_best.coverage.json')['sequences']
        differences[mode], velocity[mode], gating[mode] = {}, {}, {}
        for tag in ('swap', 'swap_fixed_contact_outputs', 'aug_off_120', 'aug_on_120', 'aug_off_300', 'aug_on_300', 'final_off_300', 'final_on_300'):
            path = OUT / f'{tag}_{mode}.json'
            if not path.exists():
                continue
            coverage = read(path.with_suffix('.coverage.json'))
            assert coverage['status'] == 'passed' and coverage['unique_sequences'] == 1177
            results[mode][tag] = metrics(path)
            differences[mode][tag] = paired_differences(path, reference)
            rows = read(path.with_suffix('.diagnostics.json'))['velocity']
            velocity[mode][tag] = {k: statistics.mean(r[k] for r in rows) for k in (
                'velocity_mse_m2ps2', 'velocity_vector_error_mps', 'velocity_bias_norm_mps',
                'integral_obj_cm', 'integral_end_cm', 'integral_first30_cm', 'integral_first60_cm')}
            sequences = read(path.with_suffix('.diagnostics.json'))['sequences']
            gating[mode][tag] = {
                'sequence_mean_weights': {k: statistics.mean(r['gating']['mean_weights'][k] for r in sequences)
                                          for k in ('left_fk', 'right_fk', 'imu', 'static')},
                'frame_mean_weights': {k: sum(r['gating']['mean_weights'][k] * r['num_frames'] for r in sequences)
                                      / sum(r['num_frames'] for r in sequences)
                                      for k in ('left_fk', 'right_fk', 'imu', 'static')},
            }
    contact_check = None
    if 'swap_fixed_contact_outputs' in results['offline']:
        old, new = results['offline']['original_pp'], results['offline']['swap_fixed_contact_outputs']
        contact_check = {k: new[k] - v for k, v in old.items() if k.startswith('contact_')}
        assert all(abs(v) < 1e-6 for v in contact_check.values()), contact_check
    effects = {}
    for mode, arms in results.items():
        if not all(f'aug_{a}_{n}' in arms for a in ('off', 'on') for n in (120, 300)):
            continue
        effects[mode] = {}
        for key in ('obj_trans_err_imu', 'obj_trans_err_fusion', 'hoi_err_fusion'):
            effects[mode][key] = {
                'augmentation_on_minus_off_at_120': arms['aug_on_120'][key] - arms['aug_off_120'][key],
                'augmentation_on_minus_off_at_300': arms['aug_on_300'][key] - arms['aug_off_300'][key],
                'longer_training_300_minus_120_without_aug': arms['aug_off_300'][key] - arms['aug_off_120'][key],
                'longer_training_300_minus_120_with_aug': arms['aug_on_300'][key] - arms['aug_on_120'][key],
            }
    fixed_budget_path = OUT / 'fixed_budget_velocity.json'
    fixed_budget = read(fixed_budget_path) if fixed_budget_path.exists() else None
    final300_crosscheck = {}
    if fixed_budget:
        for arm in ('off', 'on'):
            tag = f'final_{arm}_300'
            if tag in results['offline']:
                delta = results['offline'][tag]['obj_trans_err_imu'] - fixed_budget['means'][f'{arm}_final300']['integral_obj_cm']
                assert abs(delta) < .002, (tag, delta)
                final300_crosscheck[arm] = {'passed': True, 'full_minus_velocity_only_cm': delta, 'tolerance_cm': .002}
    final300_effects = {}
    for mode, arms in results.items():
        if all(f'final_{a}_300' in arms for a in ('off', 'on')):
            final300_effects[mode] = {
                key: arms['final_on_300'][key] - arms['final_off_300'][key]
                for key in ('obj_trans_err_imu', 'obj_trans_err_fusion', 'hoi_err_fusion', 'mpjpe', 'root_trans_err')}
    write_json(OUT / 'comparison.json', {'updated_at': now(), 'full_metrics': results,
               'paired_sequence_changes_vs_original_pp': differences, 'velocity_diagnostics': velocity,
               'factorial_effects_cm_negative_is_better': effects, 'gating': gating,
               'fixed_contact_output_metric_deltas': contact_check,
               'fixed_budget_velocity_means': fixed_budget['means'] if fixed_budget else None,
               'fixed_budget_velocity_effects': fixed_budget['factorial_effects'] if fixed_budget else None,
               'fixed_final300_augmentation_on_minus_off': final300_effects,
               'fixed_final300_velocity_crosscheck': final300_crosscheck})
    labels = {'original_pp': '原 ++ joint-Best', 'baseline_reference': '原 baseline（整体参考）',
              'swap': '替换 baseline 两个速度网络', 'swap_fixed_contact_outputs': '替换速度且锁定原接触输出',
              'aug_off_120': '无增广 / 120 轮内 best', 'aug_on_120': '有增广 / 120 轮内 best',
              'aug_off_300': '无增广 / 300 轮内 best', 'aug_on_300': '有增广 / 300 轮内 best',
              'final_off_300': '无增广 / 固定第300轮', 'final_on_300': '有增广 / 固定第300轮'}
    lines = ['# 速度网络替换与初值增广实验', '', f'更新时间：{now()}', '',
             '本目录为固定组件诊断，不覆盖正式模型结果。仅 OMOMO，恢复的 EgoIMU 旧划分；当前 canonical 预处理，seed10。',
             '完整评估每项 1177 条唯一序列，EgoIMU 手腕 HOI 口径。下表位置指标单位 cm，越低越好。', '',
             '## 完整序列结果', '']
    findings_path = OUT / 'findings.md'
    if findings_path.exists():
        lines[7:7] = [findings_path.read_text(), '']
    for mode, arms in results.items():
        lines += [f'### {mode}', '', '| 条件 | 纯速度积分 Obj | 融合 Obj | HOI | MPJPE | Root |',
                  '|---|---:|---:|---:|---:|---:|']
        for tag, row in arms.items():
            values = [row[k] for k in ('obj_trans_err_imu', 'obj_trans_err_fusion', 'hoi_err_fusion', 'mpjpe', 'root_trans_err')]
            lines.append('| ' + labels[tag] + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        lines.append('')
    if final300_effects:
        lines += ['### 固定第300轮的完整系统增广效应', '',
                  '以下为开启减关闭；所有组件固定方式与上表相同，不使用best选择。', '',
                  '| 模式 | 纯积分Obj变化 | 融合Obj变化 | HOI变化 | MPJPE变化 | Root变化 |',
                  '|---|---:|---:|---:|---:|---:|']
        for mode, row in final300_effects.items():
            lines.append('| ' + mode + ' | ' + ' | '.join(f'{v:+.4f}' for v in row.values()) + ' |')
        lines.append('')
    if 'swap' in results['offline']:
        before, after = results['offline']['original_pp'], results['offline']['swap']
        obj_gain = before['obj_trans_err_fusion'] - after['obj_trans_err_fusion']
        lines += ['## 替换诊断', '',
                  f'Offline 融合 Obj 变化 {after["obj_trans_err_fusion"]-before["obj_trans_err_fusion"]:+.4f} cm；'
                  f'HOI 变化 {after["hoi_err_fusion"]-before["hoi_err_fusion"]:+.4f} cm。',
                  f'预先声明的后续训练门槛为 offline 融合 Obj 至少改善 0.1 cm：{"通过" if obj_gain >= .1 else "未通过"}。',
                  '替换仅涉及 hand_vel_net / obj_vel_net 的 36 个状态张量；其余 116 个张量逐项相等。'
                  '固定 HP 和 OT 权重，但其输出可以随上游输入改变。',
                  '接触网络参数固定并不等于接触预测固定：它读取预测速度。额外 offline 对照保留原 ++ 速度用于接触计算，'
                  '只把输出给下游的两路速度替换，分离接触预测变化的影响。', '']
        weights = gating['offline']['swap']['sequence_mean_weights']
        lines += ['## 为什么积分改善没有直接变成融合收益', '',
                  f'替换后的实际平均融合权重（先序列内平均再跨序列平均）：左手FK {weights["left_fk"]:.2%}、'
                  f'右手FK {weights["right_fk"]:.2%}、速度分支 {weights["imu"]:.2%}、静止分支 {weights["static"]:.2%}。',
                  'OT 的速度候选位置为上一帧融合位置 + 当前预测速度 / 30；静止候选为上一帧融合位置。'
                  '这与从起点累计整段速度的纯积分指标不同。FK 候选和接触门控不断修正位置，'
                  '因此减小长时累计速度偏差不一定明显改变融合结果。',
                  '实现依据：model/rnn/object_trans.py 的 _predict_step、model/rnn/object_trans_v2.py 的固定融合与FK滤波、'
                  'utils/object_fusion.py 的接触权重映射。此机制解释是结合代码和干预结果作出的推断。', '']
        baseline_obj = results['offline']['baseline_reference']['obj_trans_err_fusion']
        recovered = obj_gain / (before['obj_trans_err_fusion'] - baseline_obj)
        lines += [f'原 baseline 融合 Obj={baseline_obj:.4f} cm；替换速度只追回原 ++ 与 baseline 融合差距的 {recovered:.2%}。'
                  '因此当前固定 HP/OT 条件下，速度积分差距不足以解释最终 Obj 差距；'
                  '这不能排除速度训练与 OT 重训之间的相互作用。', '']
        if 'swap' in results['online']:
            original_online, swap_online = results['online']['original_pp'], results['online']['swap']
            lines += [f'Online：融合 Obj 变化 {swap_online["obj_trans_err_fusion"]-original_online["obj_trans_err_fusion"]:+.4f} cm；'
                      f'HOI 变化 {swap_online["hoi_err_fusion"]-original_online["hoi_err_fusion"]:+.4f} cm。', '']
        if contact_check is not None:
            strict = results['offline']['swap_fixed_contact_outputs']
            lines += [f'锁定接触输出的额外对照：所有汇总接触指标与原 ++ 的最大差值为 {max(abs(v) for v in contact_check.values()):.3g}；'
                      f'融合 Obj={strict["obj_trans_err_fusion"]:.4f} cm，相对普通替换仅相差 '
                      f'{strict["obj_trans_err_fusion"]-after["obj_trans_err_fusion"]:+.4f} cm。'
                      '在本干预中，接触预测变化不是融合收益很小的主要原因。', '']
    lines += ['## 300 轮对照的控制条件', '',
              '- 两组从相同 seed10 速度初始化开始；只训练两个速度网络，接触、HP、OT 固定。',
              '- 4115 有效训练序列，K=1，随机 W120，每轮 41 次更新，batch100；300 轮各 12300 次更新。',
              '- AdamW lr0.002、weight_decay0.0001，50/150 轮衰减乘0.2；FP32，无梯度裁剪。',
              '- 物体和手速度 MSE 权重均为10。训练/窗口验证 IMU 高斯噪声σ0.1。',
              '- 增广仅一项开关：50% 窗口的手/物初速度加入每轴σ0.2 m/s 噪声；关闭组概率0。',
              '- 独立进程保持原生 CUDA LSTM；每批输入、目标、CPU/CUDA 随机状态摘要配对核对，隔离初值增广随机流。',
              '- 基种子为10；训练逐轮使用10+epoch，验证使用200010+epoch。两组逐批配对，但不与旧baseline的连续随机数流逐位重放。',
              '- 窗口验证1016条、每10轮；按加权物体+手速度 MSE 选择120/300轮内best。另存 final120/final300。', '',
              '完整 offline 按原评估器逐序列运行；online 使用窗口60、lookahead10、chunk16、batch24，与前述正式对照保持一致。', '',
              '120→300 的配对变化测量延长速度训练的作用，开→关的配对变化测量初值增广的作用。'
              '这里没有重训接触和联合阶段，因此不能把结果等同于整个课程或最终系统重训的收益。'
              'best120/best300 还包含验证选择的影响；这是单种子的机制诊断，不是多种子显著性结论。', '',
              '另一个未在本轮操纵的差异是 baseline 的接触联合监督：velocity_contact_IMUHOI.py '
              '将预测手/物速度直接送入接触网络，没有 detach，接触损失可以反传到速度网络。'
              '本轮两组均只用速度 MSE，符合固定接触的隔离诊断；因此不能把与旧 baseline 的全部残余差异都归因于增广或训练轮数。', '']
    if effects:
        lines += ['## best选择下的两因素对照', '', '以下均为后者减前者（负数表示改善），包含窗口验证选择的影响；训练预算的主要判断应看后面的固定末轮对照。', '']
        for mode, entries in effects.items():
            lines += [f'### {mode}', '', '| 指标 | 增广：120轮 | 增广：300轮 | 300−120：无增广 | 300−120：有增广 |',
                      '|---|---:|---:|---:|---:|']
            for metric, values in entries.items():
                lines.append('| ' + metric + ' | ' + ' | '.join(f'{v:+.4f}' for v in values.values()) + ' |')
            lines.append('')
    else:
        gate_path = OUT / 'gate.json'
        authorization = OUT / 'training_authorization.json'
        if authorization.exists() and read(authorization)['authorized']:
            lines += ['用户已明确要求继续 300 轮对照，目标为区分速度训练时长和初值增广的影响；'
                      '原融合收益门槛未通过这一结论保持不变。', '']
            progress_path = OUT / 'paired_training/progress.json'
            if progress_path.exists():
                progress = read(progress_path)
                counts = {a: (r or {}).get('epochs_completed', 0) for a, r in progress['arms'].items()}
                lines += [f'训练进度：{counts}；输入和随机状态配对检查通过 '
                          f'{progress["matched_epochs_inputs_and_rng"]} 轮。'
                          '完整评估未齐前不对课程与增广的相对贡献下结论。', '']
        elif gate_path.exists() and not read(gate_path)['passed']:
            lines += ['预设融合收益门槛未通过，300 轮训练未自动启动。'
                      '速度积分已明显改善，但是否据此继续区分课程与增广，等待用户确认目标口径。'
                      '当前不对课程与增广的相对贡献下结论。', '']
        else:
            lines += ['300 轮对照尚未全部完成，当前不对课程与增广的相对贡献下结论。', '']
    if fixed_budget:
        means = fixed_budget['means']
        lines += ['## 固定更新次数的速度诊断', '',
                  '此表直接使用第120/300轮结束的参数，不经过验证集best选择。完整1177条序列，GT初速度，关闭评估噪声；'
                  '速度前向独立于HP、接触和OT。MSE单位为 m²/s²，位置单位为 cm。', '',
                  '| 初值增广 | 已训练轮数 | 物体速度MSE | 手速度MSE | 物体速度偏差范数 (m/s) | 纯积分Obj | 积分末帧误差 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for budget in (120, 300):
            for arm in ('off', 'on'):
                row = means[f'{arm}_final{budget}']
                values = [row[k] for k in ('velocity_mse_m2ps2', 'hand_velocity_mse_m2ps2',
                                           'velocity_bias_norm_mps', 'integral_obj_cm', 'integral_end_cm')]
                lines.append(f'| {arm} | {budget} | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
        lines += ['', '固定末轮参数的积分误差差值（负数为改善）：', '']
        for key, value in fixed_budget['factorial_effects']['final']['integral_obj_cm'].items():
            lines.append(f'- {key}: {value:+.4f} cm')
        lines += ['', '固定轮数对照优先用于判断速度训练预算和增广的作用；best checkpoint 的完整融合评估用于观察这些变化能否传递到系统输出。', '']
    lines += ['## 图与数据', '',
              '![固定轮数的积分误差与速度偏差](fixed_budget_effects.png)', '',
              '![配对窗口验证曲线](validation_curves.png)', '',
              '两图另有同名PDF矢量版本。', '',
              '全部指标、逐序列改善数量、速度偏差和积分末端误差见 comparison.json；'
              '覆盖检查见各 *.coverage.json，原始逐序列数据见各 *.diagnostics.json。']
    (OUT / 'analysis.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'available': {m: list(v) for m, v in results.items()}, 'report': str(OUT / 'analysis.md')}))


if __name__ == '__main__':
    run()
