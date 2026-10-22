# OMOMO：恢复原版训练配方后的公平比较

用户于2026-10-18指出上一轮每序列多采窗口不符合尽可能匹配原版训练配置的要求。本实验修正这一问题，从头训练共同HP及两套VC、OT和联合阶段。正式目录：`outputs/IMUHOI++/original_recipe_k1_seed10_omomo_20261018/`。

## 依据与纠正

依据是EgoIMU论文运行保存的`outputs/IMUHOI/omomo/training_config.yaml`，以及其`train_IMUHOI.py`、`models/do_train_IMUHOI.py`和`dataloader/dataloader_IMUHOI.py`的实际行为；不是现在可能已修改的默认配置。只读取原版配置和实现作为核验依据，本实验执行当前IMUHOI代码及独立适配脚本。

此前`paper_compat_w120_v10_seed10_omomo_20261018`不满足这项要求。其基线OT/联合和++VC已完成，++OT因包装成子类触发v2严格类型检查而退出，没有正式完整评测。旧结果保留为额外采样条件下的探索实验，不作为原版配方公平比较结果。新脚本直接实例化原生v2类，保留其滤波和bimanual closure行为。

| 项目 | 上轮实际配置 | 原版与本轮配置 |
|---|---|---|
| 序列采样 | 有放回抽4N，平均每序列4次，并非保证每条恰好4次 | 无放回打乱N，每序列抽一个随机窗口；drop_last |
| VC batch/epochs | 800/300，复用旧W120权重 | 100/300，从头训练 |
| HP | 复用4701成员、W60训练的旧HP | 60/300，4115成员、W120，从头训练并共享 |
| OT batch/epochs | 800/300 | 60/300 |
| joint batch/epochs | 800/100 | 50/300 |
| 各阶段初始LR | joint默认0.0002 | 全部0.002 |
| VC milestones | 当前分组策略 | 全局50、150 |
| HP/OT milestones | 未逐项恢复 | 100、200、300 |
| joint milestones | 10、20…100 | 80、120、160、200、240、280 |
| 优化器、衰减 | AdamW，当前配置 | AdamW，weight_decay=0.0001，gamma=0.2 |
| 数值与裁剪 | BF16、clip1 | FP32、无梯度裁剪 |
| 基线obj_trans/obj_vel_cons权重 | 0.1/0.1 | 1/1；速度权重10 |
| 新增基线gate/HP contact损失 | gate0.1、HP contact1 | 均0，原版无这两项监督 |
| 窗口验证 | 确定性60帧、1175成员、每5轮 | 随机120帧、原版有效1016成员、每10轮 |
| 特征计算 | 整段缓存后裁剪 | 裁剪后重算导数，匹配原版顺序 |
| OT训练时冻结HP/VC的dropout | 当前trainer设eval关闭 | 原版model.train仍开启；本轮恢复相关Dropout/RNN层训练状态 |
| 选点 | 末轮为主 | 窗口验证best为主，末轮独立报告 |

原版K=1包括训练loader丢弃最后不足一批的序列，并非每条都参与该轮更新。4115个索引无重复，实际预算如下：

| 阶段 | 每轮使用窗口 | 每轮更新 | 300轮总更新 | 300轮总窗口 |
|---|---:|---:|---:|---:|
| VC | 4100 | 41 | 12300 | 1230000 |
| HP/OT | 4080 | 68 | 20400 | 1224000 |
| joint | 4100 | 82 | 24600 | 1230000 |

共同条件：OMOMO旧划分，按原版长度>=121帧和物体运动标签>=10帧筛选训练4115条、窗口验证1016条；所有独立完整序列评估覆盖1177条。训练和窗口验证对人体/物体9维IMU加独立Gaussian sigma0.1，完整eval不再额外加噪声。seed10，12个非persistent workers。独立采样generator逐轮设为10+epoch，使两方法对应阶段使用相同序列顺序与窗口随机流，不受架构消耗随机数的差异影响。

## 仍需明确的边界

- 按用户要求保留当前canonical预处理和当前HP物理实现，人体骨架/接触监督仍不同于原版。这里是尽可能对齐训练配方，不是原始数据与随机数逐位重放。
- 模块按seed10独立构造，与原版组合网络顺序消耗随机数不完全相同。两种VC的相同速度子网络共享同一随机初始化，不加载历史训练权重。HP先训练后共享，原版先VC后HP，两者预训练互不依赖。
- 训练IMU噪声参数和种子相同，但架构及dropout消耗CUDA随机数的顺序不同；不声称两方法每个噪声元素逐位相同。独立loader generator对齐的是序列抽样与裁窗随机流。
- WSL关闭pin_memory；保留原版12个workers。数值计算用FP32及AdamW foreach=False。两方法所有阶段的批次、轮数、LR和里程碑一致。
- ++保留v5ar初始化增广、120轮速度+180轮接触/endpoint联合的课程及专有损失，以及其OT的接触段长度统计先验模拟（min_frames10、noise_std0.02、drop_prob0.3）。全局学习率按原版50/150衰减，不另加分组学习率或额外更新。基线joint微调HP+OT、冻结VC；++joint微调VC+OT、冻结HP。这是方法比较，不是只改变网络结构的消融；不能宣称每个子网络更新数相同。
- 共有损失系数从原版保存配置恢复；++新增接触、端点、bimanual等损失明确保留。去掉GT输入混合warmup、HP输出平滑等原版没有的共同训练增强。
- 基线OT训练手腕，与原版及当前基线loss/eval一致；通过实际forward断言最大差为0。++使用自身掌心表示。
- 原版窗口验证直接使用test划分。本轮遵循该选点协议，同时固定报告末轮，不能把按test窗口选best描述为独立未见测试集选点。基线joint评分恢复原版HP pose/root加六个OT项；++使用相同六个OT项加其VC test loss，公开记录。

## 运行记录

- 编排：`scripts/run_omomo_original_recipe_20261018.py`；启动入口同名`.sh`。
- `protocol.json`冻结配置、当前生产源码、原版核验文件和数据哈希。
- `preflight/`存放每阶段真实batch的前向、反向及验证检查，使用一次性随机模型，权重不用于正式训练。
- 每阶段`loader_receipt.json`记录实际K=1、无放回、4115个互异索引及更新数；`effective_recipe.json`记录实际优化器、loss权重与训练模块。
- `status.json`、`summary.md`及各阶段`progress.json`记录运行进度。训练中逐轮断言累计更新数和学习率，完成时核验epoch299与规定更新数。
- 两方法分别输出`eval_offline_best.json`、`eval_online_best.json`和final对应文件。HOI为EgoIMU口径，offline主比较；online W60/lookahead10/chunk16补充。
