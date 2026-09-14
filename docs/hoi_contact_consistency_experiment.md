# IMUHOI++ OT-v3：因果发布残差实验与 GT 几何口径审计

**日期：2026-09-05。状态：训练、缓存验证、完整官方在线评测、回归测试均已完成。**

## 1. 结论先行

在当前正式评测定义下，冻结 HP、VC-v4、OT-v3 及人体 refine，增加一个 **12,867 参数的单向 GRU 发布残差**：

- **正式 HOI：10.8197 → 10.0695 cm，降低 0.7502 cm / 6.93%。**
- **Published object：14.1957 → 13.4507 cm，降低 0.7450 cm / 5.25%。**
- Published jitter：10.2974 → 10.2420 mm/frame²。
- MPJPE、根位置、人体 jitter、接触 F1、raw OT 轨迹等控制量**逐项完全相同**。
- 65.7% 的序列改善；15 类物体的平均正式 HOI 全部改善。

**重要限制：这不是物理交互全面改善的证据。**换用原始预处理 GT 手、仍固定最终预测手与 published 物体时，HOI 从 **9.4654 → 10.1372 cm**，反而变差。正式指标和该几何诊断的 GT 定义不同，必须同时披露。本轮三种兼顾两种 GT 的训练没有通过开发验证门槛，均退回 identity；不能把它们解释为有效改进，也不能据此断言不存在更好的双目标解。

未替用户切换默认模型、默认评测 GT 或可视化行为。当前结果应称为 **“正式 canonical-GT 指标下的输出校正”**，而不是宣称解决了真实接触漂移。

## 2. 基线与正式测试结果

OMOMO 旧 `processed_split_data_OMOMO/test` 的 **1,015 条**；window=60、lookahead=10、batch=48、seed=42、模拟 IMU 噪声；SMPLX_MALE、正式 evaluator 默认零 betas GT FK。最后10帧不评分。

| 方法 | 正式 published HOI (cm) | Published object (cm) | Published jitter (mm/frame²) | 原始预处理 GT HOI (cm) |
| --- | ---: | ---: | ---: | ---: |
| 当前 OT-v3 基线，重新评测 | 10.8197 | 14.1957 | 10.2974 | 9.4654 |
| 共享一维发布校准，lambda=0.35 | 10.6580 | 13.8981 | 10.2552 | 9.7204 |
| **因果 GRU32，gain=0.75** | **10.0695** | **13.4507** | **10.2420** | **10.1372** |

基线与 GRU 都完成了完整官方 `eval_batch_online.py` 推理，分别用时 608.96 / 585.64 秒。标量对照在等价 online cache 上固定参数评分。缓存的 baseline 与正式评测差约 10⁻⁷ cm；GRU cache 分数与重新运行的正式分数也在 10⁻⁵ cm 内。

原 evaluator 的 `obj_jitter=10.2370` 针对 **raw OT**，不是 published output。上表对全部方法重新计算同口径 published jitter，避免错比。

### 配对不确定性与尾部

- 5,000 次序列 bootstrap：HOI 变化 95% CI **[-0.850, -0.654] cm**。
- 按 **102 个 subject-object group** 整组重采样：95% CI **[-1.016, -0.491] cm**。
- 改善超过1 cm：374 条；恶化超过1 cm：110 条。最差单序列变化 +3.554 cm，并非每条都变好。
- HOI p95：21.111 → 20.726 cm。

这些区间只描述当前数据划分与协议，不代表真实 IMU 或其他数据集已经得到验证。

## 3. 为什么不直接沿 handoff 的“手部漂移占70%”路线重训 HP

正式指标为 `mean_GT-contact ||e_object - e_hand||`。共同平移误差可抵消，因此 obj 均值误差阶梯的差值不是 HOI 的可加误差预算，也不能据此推导“OT 只剩0.9 cm空间”。

### 3.1 9.46 与 10.82 的差异主要来自 GT 几何口径

在相同1,015条、相同发布帧与相同 GT contact mask 上的四格对照：

| 物体 / 预测手 / GT手 | HOI (cm) |
| --- | ---: |
| raw / OT输入手 / 预处理GT | 9.4560 |
| raw / OT输入手 / 正式eval GT | 10.8299 |
| raw / refined手 / 正式eval GT | 11.1273 |
| published / refined手 / 正式eval GT | 10.8197 |

预处理 GT palm 与正式 evaluator GT palm 平均相差 **5.1865 cm**。单独换 GT 定义已经使 HOI 增加约1.374 cm；人体 refine 与重发布的净影响约 -0.010 cm。因此不能把 `10.82-9.46` 全归咎于 publisher，也不能拿诊断9.46直接当官方基线。

这两条 GT 路径的区别包括 dataset-selected body model、默认零 betas FK 与预处理关节；本轮测量的是总差异，没有假定某一个因素独占全部5.19 cm。

### 3.2 训练代理目标与最终指标也不一致

旧 `ObjectTransLoss.hoi_error_l/r` 比较单手方向乘骨长与预处理骨向量；正式 eval 比较重新融合后的物体相对最终人体手的位置。新模型直接监督后者，且使用严格相同的 canonical GT FK。

### 3.3 其他审计修正

- 正式1,015条来自旧 test 目录，不是 full/test 1,173条按长度过滤的结果。
- 历史训练是微调 VC+OT、冻结HP；不能说此前所有上游都冻结。本次缓存训练才将选定的全部上游固定。
- HP 对 ignored wrist joints 设置单位局部旋转，并非学习了可观测的腕角。
- oracle 5.03/2.13是特定估计器、输入和拟合方法的实验值，不是所有可学习方法的全局性能下限。
- 原在线协议启动时读取前70帧并发布前60帧，存在启动未来信息例外；后续固定延迟协议不变。新增模块不增加未来信息。

## 4. 方法：只校正最终发布的物体，不污染旧状态

```text
冻结 HP → 冻结 VC-v4 → 冻结 OT-v3 / refine → 原 publisher
                                                ↓
                                  因果 object-only residual → 最终输出
```

`p_final(t) = p_published(t) + gain * c(t) * R_object_IMU(t) * delta_local(t)`

- `c(t)=1-(1-p_left)(1-p_right)`；初始发布帧校正强制为零。
- `delta_local` 为单向 GRU 输出，L2 范数 ≤0.20 m。
- 99维特征只取预测几何、预测接触/融合权重、含噪 IMU 观测；在物体系表达以消除整体平移影响。
- 含左右候选分歧、骨提议/锁存差、前臂相对物体旋转、因果差分、raw-published 差。GT object、GT palm、GT contact **不进特征**。
- hidden=32、单层，跨整个序列保持状态；连续 chunk 与整段输出已测试一致。不每60帧重置。
- 最后一层零初始化；没有加入加速度积分、不可观测腕部真值或新增传感器。
- 最佳 checkpoint：epoch12，开发验证选择 gain=0.75。额外模块12,867参数；CPU单帧连续推理约 **1.09 ms/frame**（4线程，非全系统耗时）。

训练目标：`canonical_HOI + 0.5*object_error + 10*correction_temporal_MSE + 0.1*correction_magnitude_MSE`。AdamW，lr=0.001、weight_decay=0.01、batch=32、gradient clip=1。按序列加权的 HOI 与正式评价一致。

## 5. 自动训练、消融与选择

- Pilot：384训练/96验证；扩大后 **1,152训练/288验证**，其中250条验证序列有GT接触。
- 以原始 `subject_object` 分组，扩大后78训练组/23验证组无交叉；另核对1,440条开发序列与旧1,015条测试的原始 seq_name **零交叉**。
- 归一化只拟合训练数据。全部使用真正 online 上游缓存，不用 offline 推理替代。
- checkpoint / gain只在训练来源的开发验证集选择；要求物体误差不恶化、published jitter增加≤0.2 mm/frame²。
- 11次残差模型训练（3 pilot +5扩大/消融+3双GT）及一次标量拟合；不是11个独立随机种子的重复试验。
- 正式测试前锁定模型与参数，见 `locked_selection.json`。读取测试 baseline、路径/GT口径用于协议核验，但未用测试候选成绩调参。
- 必须披露：历史上游 checkpoint 曾用其原 `test_path` 选择best，因此不能宣称整个系统从未接触测试集。

扩大开发验证集结果；其基线 HOI=8.4596、object=17.9725、published jitter=9.9942、预处理GT HOI=6.7989：

| 变体 | 正式HOI | object | published jitter | 预处理GT HOI | epoch / gain |
| --- | ---: | ---: | ---: | ---: | --- |
| dual_gru32 | 8.4596 | 17.9725 | 9.9942 | 6.7989 | 0 / 0.0 |
| dual_gru32_phys2 | 8.4596 | 17.9725 | 9.9942 | 6.7989 | 0 / 0.0 |
| dual_pointwise | 8.4596 | 17.9725 | 9.9942 | 6.7989 | 0 / 0.0 |
| gru32 | 8.0399 | 17.3803 | 9.9876 | 7.5648 | 12 / 0.75 |
| pointwise | 8.1258 | 17.5352 | 9.9958 | 7.3475 | 15 / 0.75 |
| pointwise_no_extra_imu | 8.2169 | 17.6941 | 9.7992 | 7.3063 | 2 / 0.75 |
| pointwise_no_history | 8.1260 | 17.5354 | 9.9952 | 7.3484 | 15 / 0.75 |
| pointwise_no_publication | 8.2087 | 17.6599 | 9.9716 | 7.0251 | 15 / 0.5 |

一维 lambda 在1152条训练上用41点网格拟合0.35，验证只接受或否决、不重新调参；其验证HOI=8.2968、object=17.7732。

**消融读数：**去掉显式差分对pointwise几乎无影响；去掉raw-published差或额外IMU特征均变差。数据扩大后GRU优于pointwise，说明历史上下文有帮助，但不能把所有收益都归为段内动力学改进。双GT约束的三个变体均选择gain=0，诚实保留失败结果。

## 6. 收益出现在哪些相位

以下各相位对“存在该相位的序列”先独立求均值，不能把行值直接相加作为贡献分解。

| 相位 | 序列数 | 接触手帧 | 基线HOI | 改进HOI | 差值cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 1015 | 204159 | 10.820 | 10.069 | -0.750 |
| single | 905 | 60069 | 10.719 | 10.155 | -0.565 |
| bimanual | 647 | 144090 | 11.276 | 10.505 | -0.770 |
| startup | 993 | 67919 | 9.323 | 8.556 | -0.766 |
| steady | 1010 | 136240 | 11.691 | 10.924 | -0.767 |
| age_0_14 | 1015 | 25830 | 8.885 | 8.106 | -0.779 |
| age_15_59 | 1015 | 75595 | 10.530 | 9.849 | -0.681 |
| age_60_119 | 982 | 72736 | 11.404 | 10.636 | -0.768 |
| age_120_plus | 502 | 29998 | 12.053 | 11.263 | -0.789 |

startup/steady以全序列第60帧划分，age以每只手GT接触段起点计数；只用于评估分组，不是推理输入。

### 每物体

| 物体 | 序列数 | 基线HOI | 改进HOI | 差值cm |
| --- | ---: | ---: | ---: | ---: |
| clothesstand | 80 | 12.258 | 11.406 | -0.852 |
| floorlamp | 60 | 10.248 | 9.398 | -0.850 |
| largebox | 46 | 10.647 | 10.406 | -0.242 |
| largetable | 62 | 11.892 | 10.450 | -1.442 |
| monitor | 66 | 9.336 | 8.952 | -0.383 |
| mop | 99 | 8.163 | 7.578 | -0.585 |
| plasticbox | 62 | 11.920 | 10.782 | -1.138 |
| smallbox | 55 | 11.511 | 10.570 | -0.940 |
| smalltable | 67 | 10.206 | 9.706 | -0.500 |
| suitcase | 48 | 11.808 | 10.642 | -1.166 |
| trashcan | 37 | 9.684 | 9.374 | -0.310 |
| tripod | 95 | 12.286 | 11.342 | -0.944 |
| vacuum | 80 | 9.604 | 8.645 | -0.959 |
| whitechair | 84 | 10.738 | 10.327 | -0.412 |
| woodchair | 74 | 12.386 | 11.921 | -0.465 |

## 7. 与当前冻结人体任务匹配的 oracle 下界

令 `e_h=pred_hand-gt_eval_hand`。单手接触的自由object-only oracle最小HOI为0；双手帧每手平均下界为 `||e_L-e_R||/2`。按正式手帧及序列聚合，本测试集为 **3.4677 cm**。

这给出未考虑因果、初始化、平滑和可观测性的理论最大改善空间约7.352 cm。它不是可部署算法，也不保证网络能达到，但足以说明不能将旧5.03直接当作当前目标的硬地板。

## 8. 工程交付与复现

### 核心代码

- `model/rnn/contact_residual.py`：无标签特征、Pointwise/GRU、固定特征消融。
- `model/rnn/contact_residual_runtime.py`：只接收观测的连续chunk推理；每个新序列reset，不在窗口边界reset。
- `eval_IMUHOI.py`：可选 `config.hoi_contact_residual_checkpoint`，在原publisher之后应用；默认不存在时行为不变。不把校正反馈给HP/VC/OT。
- `scripts/train_hoi_contact_residual.py`：来源检查、分组、训练、验证选择、锁定评估。
- `scripts/calibrate_hoi_publisher.py`：一维发布校准与oracle诊断。
- `scripts/dump_ot_predictions.py` / `utils/ot_dump_selection.py`：train/test与序列清单控制，保存canonical GT与published轨迹以及完整来源。
- `scripts/analyze_hoi_contact_residual.py`：锁定后相位、GT几何、cluster bootstrap、运行耗时诊断。

### 已保存产物

所有新产物位于 `outputs/IMUHOI++/RNN2/hoi_contact_consistency_20260905/`：

- 推荐用于**当前正式指标研究**的权重：`expanded/gru32/best.pt`。
- 完整官方结果：`official_residual_eval.json`；同一次运行同时记录uncorrected控制指标。
- 直接使用的配置：`official_residual_config.yaml`；启动脚本 `official_residual_eval.sh`。
- 固定标量对照：`expanded/publisher_calibration.json`。
- 实验汇总/来源：`experiment_summary.json`、`final_source_sha256.json`、各cache provenance、分组与原始身份审计JSON。
- 所有训练曲线、未胜出checkpoint和失败双GT报告保留，不覆盖旧实验。

在WSL、仓库根目录执行已保存脚本即可复现官方评估：

```bash
bash outputs/IMUHOI++/RNN2/hoi_contact_consistency_20260905/official_residual_eval.sh
```

该脚本先执行nvidia-smi，再用 `/home/l/anaconda3/envs/SAGE/bin/python`。也可在自己的配置中设置 `hoi_contact_residual_checkpoint` 指向上述权重；普通 `model.inference` / 可视化不会被静默改写，应在原publisher后显式使用runtime。

### 验证状态

**211项相关测试通过**；只有两条已有TensorBoard/distutils弃用警告。包括因果性、60帧跨chunk一致、GT污染隔离、位置平移不变、初值与gain=0恒等、真实训练/保存/加载/评估、不同长度padding、来源路径隔离、canonical GT FK、旧v3/refine/online指标回归。另有真实SMPLX GT FK与预测palm一致性检查；完整新旧官方流水线的控制指标完全相同。

## 9. 下一步的优先级

1. **先固定论文的GT几何合同。**训练骨向量、初始化、人体现有body model和最终HOI必须描述同一套手锚点；不要通过切换GT定义把诊断变化算作模型收益。本轮没有擅自改该合同。
2. 将已验证的published监督用于更一致的OT训练，而不是只保留后处理；但需要新的干净开发协议，不能继续用本次测试成绩调参。
3. 若目标是同时改善真实形状下的接触，下一步应研究接触条件化的局部手臂/腕部精化，并保留传感器旋转、骨长和全身精度约束；不要整体平移人体。当前object-only双GT尝试未过门槛，是这一步的动机而非效果证明。
4. 真实IMU、其他数据集、多随机种子尚未验证。不要将本轮OMOMO结果直接外推。
