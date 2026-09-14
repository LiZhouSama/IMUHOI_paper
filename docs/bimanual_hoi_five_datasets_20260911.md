# 五数据集双手 HOI 误差预算与因果回放

日期：2026-09-11。承接 [物体分支交接文档](hoi_object_branch_handoff.md) 和 [双手 FK 历史分析](bimanual_fk_analysis_export_20260904.md)。本文全部使用 canonical 数据，不沿用旧骨架的 oracle 数字。

## 1. 结论与适用范围

- 只纠正双手相对几何、固定双手中点和物体轨迹时，理想差分修正能降低整集 HOI **0.52-1.26 cm**；这不是任意双手建模方法的性能上限。
- 无真值输入、固定参数的因果软约束，五集实测收益为 **0.09-0.24 cm**。这是自由掌根位置的后处理回放，不是完整 SMPL 精化模块或重训模型的结果。
- 段首硬锁定在 OMOMO、BEHAVE、IMHD、HODome 明显有害；PAHOI 没有明确收益，均值略差。
- 抓握质量门控、左右手可靠性选择、重锚与闭环状态更新仍是不同的问题。本次没有测试这些模块，不能拿差分 oracle 限制它们，也不能宣称它们能回收任意平移 oracle 的收益。

## 2. 基线与评测口径

所有数据集共用 **cur_hoi：HP + VC-v4 + OT-current，启用人体 refine、HOI 对齐损失，static gate 为 none**。模型只在 OMOMO 上训练；其余四集包含跨数据集偏移，不是各自域内重训基线。

- canonical neutral、zero-betas；OMOMO/PAHOI 为 SMPL-X，其他三集为 SMPL-H。
- 在线窗口 60，lookahead 10，seed 42，`simulate_imu_noise=true`；上游并非零延迟。
- 使用完整测试序列；每条序列仅对 GT 接触的手-帧求平均，再对有效序列等权平均。没有 GT 接触的序列不进入 HOI 均值，但保留物体误差统计。
- 使用 published 物体轨迹与 refined 掌根，对照函数为 [compute_hoi_error_at_gt_contact](../utils/hoi_metrics.py)。诊断脚本逐序列检查其结果与该函数一致。
- IMHD 使用 GPU 0、batch 48。HODome 的 GPU batch 48 和 16 均发生 CUDA/cuBLAS 执行错误，后改为 CPU、batch 48；PAHOI 同样使用 CPU、batch 48。全部在 WSL 原生 SAGE 环境执行，没有修改模型逻辑。

| 数据集 | 缓存序列数 | 有效 HOI 序列数 | 发布帧数 | 基线 HOI (cm) |
|---|---:|---:|---:|---:|
| OMOMO | 1173 | 1085 | 196385 | 9.4392 |
| BEHAVE 整段 | 59 | 54 | 86094 | 40.7690 |
| IMHD 整段 | 15 | 15 | 25243 | 34.9671 |
| HODome 整段 | 36 | 36 | 64410 | 41.2805 |
| PAHOI | 113 | 109 | 22599 | 16.2129 |

一致性检查：OMOMO、BEHAVE 缓存复现已有正式指标；IMHD 新缓存与原正式 HOI 相差小于 0.000001 cm。HODome CPU 的 HOI 与原 GPU 结果 41.2822 相差约 0.0018 cm，物体误差相差约 0.0041 cm。PAHOI 是本次新增的同 checkpoint 缓存，没有用不同模型的旧结果作参照。

## 3. 差分修正到底测了什么

定义物体位置误差 $e_o=\hat p-p$、手位置误差 $e_h=\hat h-h$，则每只手的 HOI 向量误差为 $q_h=e_o-e_h$。双手同时接触时：

$$
c=\frac{q_L+q_R}{2},\qquad d=\frac{q_L-q_R}{2},\qquad
E_{\mathrm{both}}=\frac{\|c+d\|+\|c-d\|}{2}\geq\|c\|.
$$

`oracle_diff_100` 使用 GT 双手位置差，对预测双手作等量反向修正，使 $d=0$，但保持物体位置和双手中点不变。只在 GT 双手接触帧修正，单手帧不动。因此它给出该固定共同分量条件下的最小配对 HOI，不是任意人体精化或物体跟踪模型的下界。

$c$ 是共同的**手-物相对误差**，不能直接叫作根节点误差：人和物体若一起平移同一个量，HOI 完全不变。双手相对几何本身无法辨认 $c$。

### 完整结果

所有数值为 cm；“下降”为正表示改善。软约束和硬锁定都不使用 GT 来生成修正。

| 数据集 | 基线 | 理想差分修正后 | 理想下降 | 因果软约束后 | 软约束下降 | 硬锁定后 | 硬锁定恶化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OMOMO | 9.4392 | 8.1770 | 1.2622 | 9.2572 | 0.1819 | 10.5077 | 1.0685 |
| BEHAVE | 40.7690 | 40.2538 | 0.5151 | 40.6805 | 0.0885 | 42.7339 | 1.9649 |
| IMHD | 34.9671 | 34.0931 | 0.8740 | 34.7259 | 0.2412 | 37.8587 | 2.8915 |
| HODome | 41.2805 | 40.5646 | 0.7158 | 41.1700 | 0.1105 | 45.9211 | 4.6406 |
| PAHOI | 16.2129 | 15.1972 | 1.0157 | 16.1106 | 0.1023 | 16.2649 | 0.0521 |

因果软约束：当预测左右接触概率均不低于 0.15 时，在观测物体坐标系中对预测双手向量作 EMA，`alpha=0.1`，再用 `gain=0.5` 将双手差向 EMA 结果拉近。预测双手接触中断后重置状态。参数在五集汇总前固定，没有逐数据集调参。后处理不额外读取未来帧，但输入来自原有 lookahead=10 的上游模型。

硬锁定：使用同一预测接触条件，双手向量冻结为该段第一帧的预测值，`alpha=0`、`gain=1`。

两种回放均固定物体轨迹，所以物体误差不变；软约束的接触帧掌根绝对误差在五集均小幅下降约 0.05-0.11 cm。但修正后没有执行 IK，也没有保证 SMPL 骨长与关节运动学约束，不能据此宣称全身姿态或物理交互已改善。

### 不确定性与部分修正

下表区间为 2000 次配对、序列级 bootstrap 的 95% 区间，seed 42；不是训练后收益的置信区间，也没有按 subject/object 作聚类重采样。

| 数据集 | 理想差分下降区间 | 软约束下降区间 | 消除 25% 差分向量的下降 | 消除 50% 差分向量的下降 |
|---|---|---|---:|---:|
| OMOMO | [1.169, 1.357] | [0.165, 0.199] | 0.492 | 0.892 |
| BEHAVE | [0.409, 0.626] | [0.065, 0.113] | 0.221 | 0.382 |
| IMHD | [0.401, 1.427] | [0.143, 0.351] | 0.372 | 0.648 |
| HODome | [0.548, 0.909] | [0.076, 0.149] | 0.304 | 0.528 |
| PAHOI | [0.710, 1.401] | [0.074, 0.132] | 0.411 | 0.732 |

后两列仍使用 GT 差分方向和 GT 双手接触，是假设敏感性分析，不是可实现模型的预测。IMHD 仅 15 条序列，区间较宽。PAHOI 硬锁定的“收益”区间为 [-0.289, 0.194] cm，跨过零，不能判定有明确改善或恶化。

## 4. 为什么长序列基线很差，差分预算却小

下表的共同/差分模长先在每条序列的 GT 双手帧内平均，再对有双手帧的序列等权平均；它们不能相加，也不能直接从整集基线里相减。“接触帧中双手占比”则是汇总帧数比值。

| 数据集 | 接触帧中双手占比 | 序列等权的双手手-帧占比 | 双手帧共同模长 | 双手帧差分模长 |
|---|---:|---:|---:|---:|
| OMOMO | 54.0% | 55.7% | 7.50 | 5.79 |
| BEHAVE | 50.1% | 61.4% | 38.72 | 6.69 |
| IMHD | 48.9% | 53.9% | 29.30 | 8.31 |
| HODome | 65.5% | 77.2% | 39.43 | 7.23 |
| PAHOI | 50.9% | 51.5% | 12.31 | 6.86 |

BEHAVE、IMHD、HODome 的共同相对误差远大于差分误差；把左右手之间的关系修准，仍会留下共同的手-物偏移。HODome 双手覆盖率最高，但差分修正收益仍小，说明不能仅按双手接触比例预测收益。

### 真值几何也并非段内常量

令 $v(t)=R_{gt}(t)^T(h_L(t)-h_R(t))$。统计 GT 双手接触段内 $\frac12\|v(t)-v(t_0)\|$，以及将段首值换成段内均值后的残差。两者是每只手平均承担一半差分变化的几何尺度，不是完整 HOI 指标；段内均值使用未来帧。

| 数据集 | 段数 | 段长中位数（帧） | 段首冻结残差 | 段内均值残差 |
|---|---:|---:|---:|---:|
| OMOMO | 712 | 104.0 | 5.18 | 2.68 |
| BEHAVE | 196 | 72.5 | 10.46 | 6.37 |
| IMHD | 35 | 112.0 | 13.50 | 10.84 |
| HODome | 130 | 139.5 | 10.82 | 5.96 |
| PAHOI | 67 | 106.0 | 2.19 | 1.17 |

这支持“允许抓握几何变化”的建模，但无法单独区分真实滑移、掌根锚点转动和标注/拟合噪声。PAHOI 的这一残差最小，硬锁定也最接近中性；不能据此把它推广到其他数据集。

## 5. 物体侧 oracle 与尚未测量的门控空间

另一个诊断固定预测手，只在 GT 双手帧令物体位置改变 $-c$，使共同相对误差变为零。此时配对 HOI 为 $\|d\|$，是固定手时任意物体位置可达到的最小配对值。

| 数据集 | 基线 HOI | 任意物体平移 oracle 后 | 下降 |
|---|---:|---:|---:|
| OMOMO | 9.44 | 7.32 | 2.12 |
| BEHAVE | 40.77 | 20.55 | 20.22 |
| IMHD | 34.97 | 24.37 | 10.59 |
| HODome | 41.28 | 15.29 | 25.99 |
| PAHOI | 16.21 | 12.42 | 3.79 |

**这不是融合候选 oracle，更不是双手质量门控的预计收益。** 它使用逐帧 GT 误差，允许任意平移、瞬时跳变，不受四路候选凸包或因果可观测性限制。差分 oracle 与此 oracle 不能当作可加误差预算。

下一项独立实验应先验证：GT 双手帧中，现有左 FK、右 FK、速度、静止候选是否确实存在更低 HOI 的组合；然后使用预测双手几何变化检验这些机会能否被识别，再做保持因果递推的回放。当前只完成了相对几何预算，尚未测量这部分可实现收益。

## 6. 复查与产物

- 诊断实现：[scripts/analyze_bimanual_hoi.py](../scripts/analyze_bimanual_hoi.py)。
- 合成测试：[test/test_bimanual_hoi_analysis.py](../test/test_bimanual_hoi_analysis.py)，覆盖共同/差分误差、单手帧不变、中点保持、因果性、刚体共转、序列等权、JSON 序列化。
- 完整逐序列结果：[all_datasets.json](../outputs/analysis/bimanual_hoi_20260911/all_datasets.json)。
- 冻结 checkpoint 与配置入口：[manifest.json](../outputs/analysis/bimanual_hoi_20260911/manifest.json)。
- 新增缓存与各次运行日志位于 `outputs/analysis/bimanual_hoi_20260911/`；新缓存自带 checkpoint/config/source 的 provenance。旧 OMOMO、BEHAVE 缓存路径在结果 JSON 中。

重复计算不需要重新推理：

```bash
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/l/anaconda3/envs/SAGE/bin/python scripts/analyze_bimanual_hoi.py \
  --dump omomo=outputs/IMUHOI++/RNN_cano/omomo_cano_ot_variants_20260909_c/dumps/cur_hoi_online.pt \
  --dump behave=outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/cur_hoi_behave_online.pt \
  --dump imhd=outputs/analysis/bimanual_hoi_20260911/cur_hoi_imhd_online.pt \
  --dump hodome=outputs/analysis/bimanual_hoi_20260911/cur_hoi_hodome_online.pt \
  --dump pahoi=outputs/analysis/bimanual_hoi_20260911/cur_hoi_pahoi_online.pt \
  --out outputs/analysis/bimanual_hoi_20260911/all_datasets.json
```