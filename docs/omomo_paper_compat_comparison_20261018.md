# OMOMO 120帧/速度权重10：基线与IMUHOI++对照

> 2026-10-18更正：用户要求尽可能匹配原版训练配置，本轮的K=4/batch800、旧HP及joint100不符合要求，已退出原版配方公平比较。此文保留记录历史配置；后续以`docs/omomo_original_recipe_fair_comparison_20261018.md`及`original_recipe_k1_seed10_omomo_20261018`实验为准。

用户于2026-10-18授权执行。实验目录为
`outputs/IMUHOI++/paper_compat_w120_v10_seed10_omomo_20261018/`。

## 固定条件

- 使用当前canonical预处理，读取此前已核验的独立数据副本，不修改原数据。
- 新训练阶段共同使用4115条有效训练成员，完整测试1177条；seed=10。
- 训练窗口120帧；训练期间手/物速度损失权重均为10。基线配置使用`imuhoi_loss_weights`，++使用实际生效的`loss_weights`。
- IMU的9维输入独立高斯噪声σ=0.1；独立offline/online评估不额外加噪声。
- batch800，每epoch有放回抽样4N窗口，drop_last后20次更新、16000个窗口、192万帧。
- OT训练300轮、联合阶段100轮，不早停；末轮joint epoch99为主结果，best只作次要输出。
- 保持已有自动联合策略：联合从该次OT窗口验证最优checkpoint启动。
- 评估主指标为完整序列offline，额外输出window60/lookahead10/chunk16的online指标。
- HOI评估使用EgoIMU手腕20/21、非接触root0及逐序列平均口径。

## 共享与复用

两方法都从既有基线的初始HP最佳checkpoint开始，不使用旧联合阶段微调后的HP。
这份HP此前使用4701条有效训练成员、60帧窗口训练；本轮不重新训练HP预训练阶段。
所以“共同4115条”指本轮VC/OT/联合阶段，不表示HP的历史训练也只使用4115条。

基线VC直接复用已完成的`w120_v10/final.pt`（epoch299、6000次更新），不重复训练。
++的VC-v5ar重新训练；其手/物速度子网络严格加载基线实验的同一随机初始权重，36个state tensors逐项一致。
其余结构专有参数按seed10初始化。两种OT各自按独立module seed10初始化，从头训练。

## 保留的方法差异及已知口径修正

1. 基线联合阶段训练HP+OT，冻结VC；++联合阶段训练VC+OT，冻结HP，保留各自方法逻辑。
2. VC-v5继承`SUPPORTS_ENDPOINT_FINETUNE_STAGE=False`，实际预训练为120轮速度、180轮接触/endpoint联合训练，
   **没有endpoint-only尾阶段**。配置中的`vc_joint_stage_epochs=60`不是其实际接触总轮数。
   因此++预训练速度头更新2400次，接触组3600次；基线速度头更新6000次。
   ++随后联合阶段还可更新速度头2000次。统一总阶段预算不等同于每个子网络更新次数完全相同。
3. ++速度独训阶段同时把两个速度损失乘10主要是整体缩放，不能等同于基线中速度损失相对接触BCE的10倍约束。
   ++的OT联合阶段才同时使用速度权重10及其接触辅助损失。
4. ++保留v5ar初始化增广、forearm/object rotation及bimanual closure，这些属于其方法。
5. 基线现有OT trainer传入掌心，而其几何损失和推理使用手腕。本轮在独立适配层把基线OT训练输入改为手腕，
   并检查实际forward输入最大差为0；++保留自己的掌心建模。生产模型/训练/eval源码不改。
   这项修正独立标注，因此新训练与此前“固定旧OT只换VC”的差异不能全部归到OT重训本身。

## 为什么此前Obj数值优于论文

论文原版模型和原生eval已在1177条精确复现Obj11.3056、HOI17.4436 cm。
固定原版HP、OT、完整测试输入及评估GT，仅换入重训VC，Obj/HOI为8.8799/13.7114 cm。
这表明改进并非只能用当前评估口径解释，物体速度预测的积分质量确有提升。

但目前不是原始训练流程的逐项精确重放，不能据此直接宣称严格复现后超越论文：

- 120帧/权重10控制组使用480万个训练窗口；原版最佳VC的11111次更新、batch100对应约111万个窗口。
- 原版配方组合使用123万个窗口，也有提升，说明窗口数量不是完整解释。
- 当前canonical人体监督/接触标签、模块初始化、噪声随机数调用顺序和checkpoint选择均与原始运行有差别。
- 上述剩余因素没有全部逐项分离。此前的2×2实验验证的是速度权重与窗口设置在seed10下的作用。
- 完整canonical流水线的骨架和部分GT不同于原版；必须与固定原版上下文的组件干预分别报告。

## 运行与核验

- 启动：`scripts/run_omomo_paper_compat_20261018.sh`。
- 编排与独立适配层：`scripts/run_omomo_paper_compat_20261018.py`。
- 预检：`scripts/preflight_omomo_paper_compat_20261018.py`，结果为实验目录`preflight.json`。
- 数据字节哈希、生产源码哈希、配置和输入checkpoint哈希保存在`protocol.json`。
- 实时进度：`baseline/ot/progress_ot.json`、`progress_joint.json`，`proposed/vc/progress_vc.json`及`proposed/ot/progress_*.json`。
- `summary.md`每20秒汇总进度，各阶段日志独立保存；所有Python使用WSL原生SAGE，每次执行前检查GPU。
- 启动校验曾因标准化推理接口绕过nn.Module hooks而停止，未生成首轮checkpoint；
  校验移到实际forward入口后从相同初始化重启。修复过程记录在`launcher_repair_receipt.json`，未改训练配置。

本实验比较两套方法在共同数据与全局预算下的结果，并非只改变网络结构的严格消融。
