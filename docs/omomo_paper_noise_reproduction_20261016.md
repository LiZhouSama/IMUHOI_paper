# OMOMO 原版噪声协议重训：IMUHOI → IMUHOI++

> 2026-10-16 后续审计修正：已确认 EgoIMU 与本轮 split 大幅不同，测试仅重合223条；
> 同时两份保存代码的 HOI 计分定义不同。本轮10%门槛只作为原始数值筛选记录，
> 不能用 HOI 通过门槛来判定已经复现。下一步应先恢复可比划分和计分口径。
> 详见 [预处理全量审计](omomo_preprocessing_audit_20261016.md)。原实验结果保留。

## 目的与边界

使用当前仓库的 HP、VC-IMUHOI、OT-IMUHOI 三分支版本，从头训练，检查恢复
EgoIMU 的噪声协议后是否接近论文表 2。达到预先登记的数值门槛后，训练
VC-v5ar + OT-v2_armrot_biclose，比较相同数据与噪声下的 offline 和 online eval。
根据用户明确说明，原论文全部指标来自 **offline**；本实验只用 offline 与论文对照，
online 作为同一 checkpoint 的额外测试单独输出。

本实验是“当前重构实现上的噪声协议迁移”，不是严格的论文复现。
保留当前 canonical SMPL-X neutral / zero-betas 数据、重建接触标签、HP 实现、损失、
训练调度；offline 使用完整测试序列推理，online 保留当前滑窗协议。
原版的确切 split、骨架、初始化和指标细节未被恢复。
论文正文没有噪声数值，也没有足够的指标细节，指向的 supplementary 未在此次提供。
不得为了追表中数字反复调整测试集或宣布方法等价。

## 已核实的噪声来源

仅只读检查 EgoIMU 的噪声配置和调用位置；没有运行、导入、复制其代码或加载其权重。

| 证据 | 位置（EgoIMU 内） | 结论 |
|---|---|---|
| 训练默认配置 | `configs/IMUHOI_train.yaml:35-36` | 人体/物体 std 均为 0.1 |
| 保存的 OMOMO 训练配置 | `outputs/IMUHOI/omomo/training_config.yaml:56,106` | 实际保存配置也是 0.1 |
| 输入扰动位置 | `models/do_train_IMUHOI.py:731-745` | 加到全部 9 个特征分量，非轴角噪声 |
| 训练和内部验证调用 | 同文件 `1105,1185` | 两者 `add_noise=True` |
| 独立评估调用 | `eval_IMUHOI.py:313` | `add_noise=False` |

噪声为独立加性 Gaussian：加速度三个分量 std=0.1 m/s²；6D 旋转六个分量 std=0.1
（无量纲），人体和物体相同。不是 0.1 rad，也不是在旋转矩阵上加同等角度噪声。
训练每次取 batch 重新采样；训练内部验证同样加噪声；独立 eval 使用干净的合成 IMU。
两种实现的随机数消耗顺序不同，分布相同不代表逐元素同一份噪声。

在当前仓库直接使用 `utils/utils.py` 的已有特征噪声实现，四项 std 全部设为 0.1。
`simulate_imu_noise=false` 关闭 Noitom 序列扰动。当前默认路径的 `imu_noise_applied`
标志会跳过额外高斯噪声，不能只改四个 std 而保留该开关。
增加 opt-in `validation_add_imu_noise=true`，用于 HP/VC/OT/联合阶段内部验证；
默认 false，已有实验行为保持不变。独立 eval 不带任一加噪 CLI 开关。

## 预先固定的实验

| 项目 | 设定 |
|---|---|
| 数据 | 当前 `process/IMUHOI++cano/processed_split_data_OMOMO_full/{train,test}` |
| 随机种子 | 首轮 42；结果只是单 seed，不作统计显著性结论 |
| 基线 | 新 HP → 新 VC `imuhoi` → 新 OT `imuhoi` → 联合 HP+OT，冻结 VC |
| 改进方法 | 共享本轮**初始** HP → 新 VC `v5`（v5ar 开关）→ 新 OT `v2` + armrot + biclose → 联合 VC+OT，冻结 HP |
| HP 共享 | 只共享本轮从头训练的初始 HP；不使用基线联合训练改过的 HP |
| 预算 | 当前配置：HP/VC/OT 各最多 300 epoch，joint 最多 100；既有 early stopping |
| 数据采样 | batch 800，windows_per_sequence=4；根据实际文件数记录每 epoch 步数 |
| 在线协议 | window 60 / lookahead 10；HP smoothing 7/2；object velocity init diff=3 |
| 论文主评估 | `eval_IMUHOI.py --inference_mode offline`，完整测试序列，预测根平移 |
| 补充在线评估 | `eval_batch_online.py`，与 offline 相同测试集和对应 checkpoint |
| v5ar | heading normalization=false；init noise prob=0.5；obj/hand init vel std=0.2；contact flip=0 |
| v2 | FK filter α=0.35, β=0.05；forearm-object rotation、bimanual closure 开启 |
| 执行 | WSL 原生 SAGE；串行；每个训练/eval 子进程前检查 GPU；workers=0 避免已有 WSL worker 故障 |
| 输出 | offline/online × best/final 均报告；门槛只读 offline final，不在模式或 checkpoint 间择优 |

当前训练器仍以 test-window loss 选择 checkpoint 和早停。这里保留现有流程以减少
额外变量，并明确记录测试集参与模型选择的限制；即使报告 final 也不能消除这点。
后续正式论文比较应另做源序列级 train/validation/test 划分，并对双方重训。
这次比较包含不同的联合训练对象和损失，是完整 pipeline 比较，不是纯 OT 架构消融。

## 目标、自动门槛及未达标处理

用户提供的 `40509.pdf` 第 6 页、表 2（预测根平移；用户确认是 offline 指标）：

| 指标 | 论文 OMOMO | 自动门槛（不差于 110%） |
|---|---:|---:|
| Obj Err / cm | 11.31 | 12.441 |
| HOI Err / cm | 17.44 | 19.184 |
| Pos Err / cm | 2.35 | 2.585 |
| Trans Err / cm | 11.07 | 12.177 |

10% 是本次首次运行前固定的工程筛选门槛，不是论文规定的复现容差。
offline final 四项全部为有限且非负的数值，并满足门槛，才自动继续 IMUHOI++。
online 数字不参与此判断，也不与论文行混用。
比论文更低的误差也会通过数值门槛；仍不得把它称为严格复现。
Pos Err 暂以当前 MPJPE 对应；关节集合、聚合和坐标系等效性尚未得到 supplementary 验证。
论文 Jitter=2.77 写作 mean joint jerk / mm/s³；当前 `jitter` 为二阶差分 / mm/frame²，
`obj_jitter` 更不是该人体指标，全部排除在论文门槛之外，原始 eval JSON 仍保留当前指标。
表 1 属于单独训练的 no-root 模式，不与本次表 2 对照混用。

若未达门槛，队列写 `baseline_complete_gate_not_met` 并停止，保留全部结果。
下一步先定位 HP 的姿态/根平移差异、标签和指标口径、离线与在线误差差异；不自动
改变噪声、降低门槛、缩短训练、转用四分支模型或反复试测试集。

## 运行与产物

```bash
# 在仓库根目录、WSL 中运行。脚本先 nvidia-smi，再使用固定 SAGE Python。
bash scripts/run_omomo_paper_noise.sh --prepare_only
bash scripts/run_omomo_paper_noise.sh
# 读取已有 manifest 汇总；不会训练。
bash scripts/run_omomo_paper_noise.sh --summarize_only
```

实验目录：`outputs/IMUHOI++/paper_noise_omomo_20261016/`。

- `recipe.yaml`：冻结的完整配置；`provenance.json`：源文件哈希、噪声来源、数据清单和元数据。
- `status.json`：顶层状态与每项 offline 门槛结果；`summary.md`、`results.json`：两模型、两种推理模式、best/final 结果。
- `<method>/<variant>/eval_omomo_offline_{best,final}.json`：论文对照的 offline 结果。
- `<method>/<variant>/eval_omomo_{best,final}.json`：online 补充结果（JSON 模式为 `batch_online`）。
- offline 的 `.command.json`：相同 HP/VC/OT checkpoint 路径和明确的 `--inference_mode offline`。
- `baseline/experiment_manifest.json`：HP→VC→OT→joint 的实际进度和权重路径。
- `baseline.log`、各阶段日志：训练和错误证据；`proposed/` 仅在基线达门槛后运行。
- `queue.log`、`queue.pid`：后台启动时记录的输出及进程号。

同一目录再次运行会续接已完成或中断阶段。文件锁防止重复队列；源码、配置或数据清单
发生变化会拒绝沿用原实验，防止混合版本。当前启动器不设置定时通知。

## 首次启动记录（2026-10-16）

噪声/门槛测试及 IMUHOI、v2-armrot-biclose 架构回归测试共 23 项通过。
预检修正了 Python 3.8 下脚本相对 `__file__` 导致的 provenance 路径错误，随后完成审计：
4,705 条训练序列、1,173 条测试序列，源序列名无交叉，每 epoch 23 步。
后台队列已启动（首次 PID 2288238）；HP 阶段实际配置确认四项 std=0.1、
`simulate_imu_noise=false`、`validation_add_imu_noise=true`、`no_trans=false`。
数据加载器实际接纳 4,699 条训练序列（文件清单是 4,705 条；现有窗口/样本过滤发生在
loader 内），测试接纳 1,173 条，实际每 epoch 仍为 23 步。HP 已完成首个训练及验证周期，
并继续训练；当前尚无最终 eval 数值。实际最新进度以 `status.json` 和阶段日志为准。

用户明确论文指标为 offline 后，已修正为 offline 主评估 + online 附加评估。
只停止最外层旧调度器，保留基线 runner 和 HP 训练进程，随后用新调度器接管。
旧 provenance 存入 `provenance.before_offline_correction.json`，训练配置和训练源码不变。
增加模式隔离和两种联合训练 checkpoint 选择测试后，共 27 项相关测试通过。
