"""Summarize the completed controlled interventions and source audit."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_vc_joint_20261019 import OUT, RUN, ARMS, clean
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, write_json
from scripts.run_omomo_original_recipe_20261018 import verify


def main():
    check_gpu(0)
    verify(json.loads((RUN / 'protocol.json').read_text()))
    head = json.loads((OUT / 'velocity_head_diagnostics.json').read_text())
    audit = json.loads((OUT / 'loss_audit.json').read_text())
    gradient = json.loads((OUT / 'gradient_audit.json').read_text())
    audit.update(effective_vc_test_keys=gradient['vc_test_keys'],
                 effective_proposed_ot_loss_keys=gradient['ot_loss_keys'],
                 configured_but_missing_proposed_ot_loss=['obj_acc_cons'],
                 original_hand_contact_weight_used_by_v5=False,
                 effective_joint_selection='runner OTTrainer.evaluate: six original OT test terms + VC compute_test_loss',
                 caution='common_mismatches only checks scalar equality, not effective loss definitions or active keys')
    write_json(OUT / 'loss_audit.json', audit)
    comparison, pairing = {}, {}
    for mode in ('offline', 'online'):
        comparison[mode] = {}
        for arm in ARMS:
            result = json.loads((OUT / f'{mode}_{arm}.json').read_text())
            assert json.loads((OUT / f'{mode}_{arm}.coverage.json').read_text())['unique_sequences'] == 1177
            comparison[mode][arm] = result['results']['omomo']
        rows = {a: json.loads((OUT / f'{mode}_{a}.diagnostics.json').read_text()) for a in ARMS}
        for arm in ('joint_best', 'joint_final'):
            for section, field in [('sequences', 'obj_trans_err_fusion_published_cm'), ('velocity', 'integral_obj_cm')]:
                if not rows[arm][section]:
                    continue
                before = {r['seq_path']: r for r in rows['prejoint'][section]}
                deltas = [r[field] - before[r['seq_path']][field] for r in rows[arm][section]]
                pairing[f'{mode}_{arm}_{section}'] = {'mean_delta_cm': sum(deltas)/1177,
                    'improved_sequences': sum(d < 0 for d in deltas), 'worsened_sequences': sum(d > 0 for d in deltas)}
    length = {}
    for name, low, high in (('under120', 0, 120), ('120to239', 120, 240), ('240plus', 240, 10000)):
        length[name] = {}
        for arm, rows in head['arms'].items():
            group = [r for r in rows if low <= r['frames'] < high]
            length[name][arm] = {'count': len(group), 'integral_cm': sum(r['integral_obj_cm'] for r in group)/len(group)}
    write_json(OUT / 'comparison.json', clean({'fixed_component_results': comparison, 'paired': pairing,
                                       'velocity_means': head['means'], 'velocity_by_length': length}))
    lines = ['# VC 联合阶段诊断（2026-10-19）', '',
        '结论：纯速度积分的差距在 ++ 独立 VC 阶段已经形成；此次联合训练整体改善积分精度，不能归因为联合阶段引入退化。', '',
        '这是固定组件推理干预，不是新一轮重训。仍使用恢复的 EgoIMU 旧划分和当前 canonical 预处理；全量 1177 条，不使用 K=4 旧实验。',
        '固定 shared/hp/best.pt 和 proposed/joint/best_object_trans.pt。仅替换 VC：独立阶段 best（epoch 180）、联合 best（epoch 100）、联合 final（epoch 299）；epoch 均为零起始。',
        'offline 为主，正式对比统一逐序列；online W60 / lookahead10 / chunk16 / batch24。全部覆盖1177唯一序列，Obj/HOI有限。offline/online Best直接复用完全相同组件和参数的已完成正式评估，命令、checkpoint哈希及冻结源码/数据协议已核验，见两个 best_reuse_receipt.json；联合前和Final各重跑两种模式。固定 HP 权重不意味着人体最终指标不变：固定 OT 的人体修正仍会受 VC 输入影响。', '',
        '| 模式 | VC | 纯速度积分 Obj/cm | 固定 OT 融合 Obj/cm | HOI/cm | MPJPE/cm | Root/cm |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels = {'prejoint':'联合前', 'joint_best':'联合 Best', 'joint_final':'联合 Final'}
    for mode, arms in comparison.items():
        for arm, m in arms.items():
            lines.append(f'| {mode} | {labels[arm]} | {m["obj_trans_err_imu"]:.6f} | {m["obj_trans_err_fusion"]:.6f} | {m["hoi_err_fusion"]:.6f} | {m["mpjpe"]:.6f} | {m["root_trans_err"]:.6f} |')
    lines += ['', 'Final 行仍固定 Best OT，因此不能与此前“全部 Final 组件”的正式结果混用。固定的是联合Best OT，它与联合VC存在共同适配；融合结果衡量替换VC后对这份OT的影响，不等价于整条流水线“不做联合训练”的消融。offline纯速度积分不经过HP/OT，因而更直接回答速度是否在联合阶段退化。online的后续窗口速度初值来自已发布融合轨迹，仍受HP/OT反馈影响，因此online也必须固定HP/OT；不能直接复用全部Final组件的速度积分指标。', '',
        '## 速度误差机制', '',
        '| VC | 逐分量速度 MSE/(m²/s²) | 每序列平均速度偏差模长/(m/s) | 前30帧速度偏差模长/(m/s) | 前30帧积分误差/cm |',
        '|---|---:|---:|---:|---:|']
    for arm, m in head['means'].items():
        lines.append(f'| {labels.get(arm, arm)} | {m["velocity_mse_m2ps2"]:.7f} | {m["velocity_bias_norm_mps"]:.6f} | {m["velocity_first30_bias_norm_mps"]:.6f} | {m["integral_first30_cm"]:.6f} |')
    lines += ['',
        '此表为offline，每序列计算后等权平均；偏差模长不是跨序列全局有符号均值。纯速度快速诊断与正式 baseline / ++ Best 积分指标吻合（容差0.002cm）。',
        '位置由速度累加得到，方向持续一致的小误差会累积。Final 的逐帧 MSE 高于 Best，积分却更好，说明逐帧平方误差与整段轨迹误差不等价。', '',
        '## 权重与实际计算审计', '',
        '- obj_vel=10、hand_vel=10，在独立 VC 速度阶段和联合阶段均实际乘入 MSE；不存在 ++ 意外回落为1的问题。',
        '- EgoIMU 保存配置中的18个共同权重均数值匹配。但数值相同不代表损失定义和启用项完全一致。',
        '- 原版/baseline obj_acc_cons=0.01 实际计算；++ YAML保留0.01，但 ObjectTransLoss 没有该损失键和计算分支，因此该项实际缺失。',
        '- 原版 hand_contact=1 为三通道平均 BCE；++ 不消费这个同名权重，而用 obj_move=1、hand_contact_cond=1、hand_contact_marginal=0.5；条件接触还使用正/负/困难负样本权重1/2/4。',
        '- ++ 接触端点1、端点顺序0.5、端点接触一致性0.5、提前接触0.5；OT额外启用左右骨长时间一致性各1、refine_pose=100、refine_root_trans=10、bimanual_hoi=5、bimanual_prior=1。24条训练成员的运行时检查确认双手、人体修正、骨长一致性等分支产生非零损失；端点顺序项在此次抽取的6个前向批次中均为0，不能将配置非零权重等同于每批均有非零贡献。',
        '- v2骨向量一致性/加速度平滑权重显式为0；gate_weak=0；不要将配置里的所有键都算作启用损失。',
        '- 共同 OT 数值为物体位置1、物体速度一致性1、左右方向/长度各0.1、左右HOI各0.2。++ 的手部几何监督使用掌心，EgoIMU/baseline使用手腕，不能因权重同名就视为相同目标。', '',
        '## 差距在哪里形成', '',
        '1. 两种 VC 的速度子网络结构相同，实验入口还给它们复制了同一份 seed10 随机初始化。额外 v5 接触特征不直接扩展速度子网络。',
        '2. baseline 在独立VC阶段训练速度300轮（12300更新）；++仅前120轮训练速度（4920更新），之后恢复该阶段最优速度参数并冻结，余下180轮训练接触/端点。两者全局LR均在50/150轮下降：++速度阶段没有经历150之后的低LR精修。联合阶段++重新解冻VC、重置优化器和LR，所以不能说++总速度更新次数更少，也不能将后续联合更新视为相同的独立速度训练。',
        '3. ++训练时每窗口以0.5概率给物体/手部速度初值加入每轴sigma0.2m/s噪声；baseline没有。这与offline使用准确GT初值的条件不同。结合早期速度偏差，初始化增广和速度课程是优先排查因素；此次固定组件实验不能把二者的独立贡献拆开。',
        '   独立VC阶段的目标耦合也不同：baseline接触网络直接读取预测速度，接触BCE可反传至速度分支；++前120轮只优化速度MSE，之后训练接触时速度已冻结。++额外OT损失尚未参与，就已有纯速度差距，所以不能将该差距直接归咎于双手/人体修正或缺失加速度项。',
        '4. 接触输入及OT输入中的速度没有detach，联合阶段梯度可回到速度子网络。24训练成员、3个batch的局部梯度诊断可见部分负余弦，但接触/OT梯度模长小于10倍速度MSE梯度；不能凭此宣称额外loss压倒速度监督，更不能称其造成了本次总体积分退化。',
        '5. 实际实验入口覆盖通用训练器：联合Best选择为原版6项OT验证loss + VC test loss；VC test包含10倍obj_vel，但不包含hand_vel，也不直接评估整段速度积分。checkpoint中selection_keys只保存6项OT名，单读通用类或该字段会遗漏VC项。', '',
        '最小后续验证建议：保持已有联合结果，先将baseline训练好的两个速度子网络替换进++，固定接触/HP/OT做诊断；若有效，再用相同300轮速度训练分别开/关初值增广，区分课程与增广。此报告未启动这些重训。', '',
        '## 证据', '',
        '- manifest.json：固定组件、checkpoint哈希与epoch。',
        '- loss_audit.json / checkpoint_metadata.json / gradient_audit.json：配置与实际损失、局部梯度。',
        '- velocity_head_diagnostics.json：baseline及++三阶段的1177条速度诊断。',
        '- offline/online各组 JSON、diagnostics、coverage：全量固定组件结果。',
        '- offline_batch_regression.json：等长batch数值检查未通过严格0.002cm阈值，最大差0.027071cm、平均绝对差0.000357cm；该批量结果未用于主表，保存在 batched_diagnostics。后续恢复逐序列计算。',
        '- comparison.json：全部指标、逐序列改善数及按长度统计。',
        '- 来源：scripts/run_omomo_original_recipe_20261018.py、train/rnn/train_velocity_contact.py、train/rnn/train_object_trans.py、train/rnn/loss/{velocity_contact_loss,object_trans_loss,imuhoi_loss}.py、model/rnn/velocity_contact_v5.py。', '']
    (OUT / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'comparison':comparison, 'paired':pairing, 'length':length}, indent=2))


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    main()
