# Canonical OMOMO（VC-v4 上 OT-current vs OT-v3）：mesh 测试期重放、差距诊断、HOI 改进方向

**日期：2026-09-08。** 基线来自 `outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/`（共享 HP，VC-v4；OT current / v3 各自 `--joint_train`；
`eval_batch_online.py`，1173 条，window 60 / lookahead 10 / seed 42 / 模拟 IMU 噪声；两条 OT 配置里 `object_trans_static_gate: none`）。
本文全部重放数字来自两份在线 dump（`.../dumps/ot_{current,v3}_online.pt`，`scripts/dump_ot_predictions.py`），
重放 publisher 复现正式 JSON 到 3e-5 cm（published obj 13.9447 / 13.8943，HOI 9.8644 / 9.9945）。误差单位 cm。

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| Q1 测试期给 mesh，"接触时把手吸附到表面"能否提升指标 | **onset 吸附：不能**（HOI +1.9）。**逐帧把 refined 手拉到发布物体表面 3 cm 内：HOI −0.58**（current 9.86→9.29，v3 9.99→9.41），但接触帧掌根绝对误差 **+0.45**（11.69→12.16），是指标收益不是物理收益。骨向量软投影 HOI −0.28/−0.30、obj ±0.1；地面非穿透（人-物同步抬升）obj **−0.2~−0.3**、HOI 不变（canonical 的 HP 竖直误差已比旧版小，收益比旧缓存的 −1.2 缩水） |
| Q3 为什么 VC-v4 上 v3 拉不开 current | 两个相反的效应抵消：v3 的软锁存让段内漂移少 **1.5 cm**（61–120 帧 growth 12.1 vs 13.7），但 v3 在**接触起点处已落后 1.6 cm**（onset 帧 6.02 vs 4.41）——固定的 VC 驱动融合权重在无接触间隙/假接触帧被拖走（gap 内漂移 7.5 vs 3.9 cm/100 帧；FP 帧误差 19.2 vs 17.1），且 onset 处的骨向量提议更差（7.89 vs 7.42），jitter 更高（8.79 vs 8.43）。current 的**学习门控**已吃掉旧版 v3b 靠静止硬门控拿到的那部分收益。逐序列两者高度互补（62% 序列相差 >1 cm，择优 12.78 vs 13.9） |
| Q2 不给 mesh、目标压低 HOI | 训练验证过的结论（1173 条官方评测）：**对 current 加度量对齐损失 `hoi_fused`（融合输出相对输入手 vs 真值相对向量，obj_trans 5→1，`static_gate` 保持基线的 `none`）HOI 9.86→9.44、obj 13.94→13.72、jitter 8.43→7.38，三项同时最优（cur_hoi）；加静止门控的版本 9.59**，机制是学习门控在接触帧把 FK 权重从 0.88 降到 0.75、融合输出更贴近真值相对向量；同一损失在 v3（固定权重）上无效（10.03）；关闭状态反馈全面变差（10.69）；旧版有效的因果发布残差在 canonical 上 HOI 变化 0.000（GT 掌根合同一致后系统性偏移消失）；静止硬门控对 v3 obj −0.18 但对 current 有害。剩余的 HOI（骨误差里 5.6 cm 沿手误差方向）由手部漂移决定，OT 侧杠杆基本用尽，下一步只能是在线协议对齐训练或臂链精化 |

---

## 1. 基线（正式评测，`comparison_summary.json`）

| | obj raw | obj published | HOI | obj jitter | MPJPE | root | F1 L/R/obj |
|---|---|---|---|---|---|---|---|
| OT current (VC-v4) | 14.35 | 13.94 | **9.86** | 8.43 | 2.99 | 6.06 | .942/.953/.971 |
| OT v3 (VC-v4) | 14.37 | **13.89** | 9.99 | 8.79 | 2.98 | 6.05 | .943/.953/.970 |

两条 pipeline 的 refined 手在接触帧误差相同（11.69 vs 11.70），接触概率平均绝对差 0.0075——差别几乎全在 OT。

---

## 2. Q1：测试期给 mesh 的重放（publisher 级，refined 手；`scripts/mesh_probe/cano_replay.py`）

| 变体 | current obj / HOI | v3 obj / HOI | 副作用 |
|---|---|---|---|
| 基线 | 13.94 / 9.86 | 13.89 / 9.99 | |
| 骨向量尖端硬投影到最近表面点 | 14.30 / 9.89 | 14.09 / 10.14 | 变差 |
| 骨向量软投影（>3 cm 才拉到 3 cm） | 14.09 / **9.58** | 13.82 / **9.70** | obj ±0.1 |
| onset 时把手吸附到表面（整段常量偏移，骨向量反补偿） | 13.94 / **11.78** | 13.89 / **11.77** | 预测 onset 处掌根误差 8.5→11.1（GT 对上的子集 8.0→8.4，仍变差） |
| **逐帧把 refined 手拉到发布物体表面 3 cm 内**（预测接触帧） | 13.94 / **9.29** | 13.89 / **9.41** | 接触帧掌根误差 11.69→**12.16**（变差） |
| 地面非穿透，训练集标定全局地面 −3.3 cm，人-物同步抬升 | **13.78** / 9.86 | **13.67** / 9.99 | 触发 30% 帧；掌根 11.69→11.75 |
| 同上，2 cm 余量 | 13.71 / 9.86 | 13.61 / 9.99 | 触发 22%；掌根 11.67 |
| 同上，按物体地面偏置 | 13.71 / 9.86 | 13.61 / 9.99 | |
| 地面：逐序列 GT 地面（oracle 上界） | 13.43 / 9.86 | 13.31 / 9.99 | |

读法：
- **手吸附只在"逐帧、相对发布物体"这一形式下有收益，且收益全部来自 HOI 度量的相对性**：它把手往物体拉，
  相对向量的法向分量被修掉，但手的绝对位置变差 0.45 cm。要不要用取决于论文怎么定义"改善"；作为 HOI 指标手段可用，作为物理改进不成立。
- onset 吸附在 canonical 版本上同样为负（与昨天旧缓存一致）：预测 onset 只有 80% 对得上真值，且手误差主要沿表面切向。
- canonical HP 的竖直误差（|e_y| 5.6 vs |e_x|,|e_z| 7.4/7.5）已明显小于旧版（7.15），预测穿地（>2 cm）帧 22% 对 GT 自身 17%，
  所以地面非穿透的余地缩到 −0.2~−0.3。
- 真值骨向量 oracle（`gtbone_*`）：onset 冻结的真值骨 → obj 不变（13.94/13.82）、HOI 8.87；逐帧真值骨 → obj 11.8/11.6、HOI 4.5/4.2。
  说明 obj 误差几乎与骨向量无关（手误差主导），而 HOI 对骨向量极敏感。

---

## 3. Q3：v3 为什么没拉开 current（`analyze_ot_dumps.py --compare`，raw 轨迹 + stage-1 手）

### 3.1 误差随 onset 后帧数（err | 相对 onset 的增长）

| 帧数 | current | v3 | 差（v3−current） |
|---|---|---|---|
| onset 帧 | 4.41 | **6.02** | +1.61 |
| 1–15 | 8.75 / +4.33 | 9.34 / +3.33 | +0.59 / −1.00 |
| 16–30 | 11.74 / +7.32 | 11.82 / +5.80 | +0.08 / −1.52 |
| 31–60 | 13.94 / +9.58 | 14.12 / +8.16 | +0.18 / −1.42 |
| 61–120 | 17.38 / +13.68 | 17.50 / +12.13 | +0.12 / −1.55 |
| 121+ | 18.74 / +15.51 | 18.86 / +13.82 | +0.12 / −1.69 |

**v3 的段内增长确实少 1.0–1.7 cm——软锁存在起作用；但 onset 帧就落后 1.6 cm，两者抵消。**

### 3.2 onset 落后从哪来：无接触阶段

| | current | v3 |
|---|---|---|
| 首个 onset 处误差 | 3.80 | 5.47 |
| pre-first-contact 误差 | 3.65 | 4.00 |
| 接触间隙 (gap) 内误差增量 | 1.49（3.9/100 帧） | **2.84（7.5/100 帧）** |
| FP 帧（预测接触、GT 无）误差 / 每帧增量 / 预测位移 | 17.05 / +0.34 / 1.36 cm | **19.19 / +0.51 / 1.76 cm** |
| FN 帧 w_static | 0.41 | 0.30 |
| GT 静止帧 w_static / w_imu | 0.74 / 0.21 | 0.86 / 0.09 |
| obj jitter | 8.43 | 8.79 |

v3 沿用 v2 的**固定 VC 驱动融合权重**：p_L=0.3 就给 FK 0.3 的权重，假接触/闪烁直接把物体拖走；current 的学习门控在这些帧上更保守。
旧版 v3b 之所以赢 v2 0.5 cm，有 0.27 来自静止硬门控（`object_trans_static_gate: both`），而本次两条 OT 都是 `none`，
current 的学习门控部分替代了它。事后静止否决（obj-IMU 规则）在两者上都能拿 −0.45（raw 14.35→13.90 / 14.37→13.93）。

### 3.3 骨向量（GT 接触帧，published 手）

| | current | v3 |
|---|---|---|
| onset 处骨误差 | **7.42** | 7.89 |
| 段内增长 1–15 / 16–30 / 31–60 / 61+ | 0.70 / 1.80 / 2.70 / 3.83 | **0.41 / 1.35 / 2.34 / 3.48** |
| 全段骨误差 | 10.02 | 10.19 |
| 补偿系数 α（骨误差·手误差/‖手误差‖²） | −0.22 | −0.23 |

v3 的 onset 提议差 0.47（onset 帧 α=1 直接用未平滑的提议），段内锁存少漂 0.35，总账仍输 0.17。

### 3.4 交叉重放（同一套 v3 refined 手；publisher 级）

| 骨向量 × 融合权重 | obj | HOI |
|---|---|---|
| current 骨 × current 权重 | 13.92 | **9.85** |
| v3 骨 × v3 权重 | 13.89 | 9.99 |
| v3 骨 × current 权重 | 14.00 | 10.02 |
| **current 骨 × v3 权重** | **13.79** | **9.85** |

**current 的骨向量更好（HOI），v3 的固定权重对 obj 略好（−0.13）**；v3 的骨向量（软锁存+Rrel）在 canonical 版本上没有 HOI 优势。

### 3.5 按物体 / 按序列

| 物体 | current obj/HOI | v3 obj/HOI | 谁赢 |
|---|---|---|---|
| clothesstand | 13.46 / 9.87 | 14.60 / 11.13 | current |
| woodchair | 15.48 / 11.82 | 14.81 / 12.49 | 分裂 |
| whitechair | 15.21 / 10.69 | 15.42 / 11.22 | current |
| trashcan | 14.96 / 10.06 | 14.02 / 9.47 | v3 |
| largetable / smalltable | 14.59 / 14.61 | 14.07 / 14.06 | v3（平移类） |
| floorlamp | 13.89 / 9.77 | 13.50 / 9.25 | v3 |
| vacuum / mop | 12.29 / 11.90 | 12.72 / 12.06 | current |

逐序列：v3 更好 52%，|Δ|>1 cm 的序列 62%（v3 差 >1 cm 29%、好 >1 cm 33%），**逐序列择优 12.78**（vs 13.9）。
两条模型失败的序列不同——和旧版的观察一致（v2/v3 择优 12.74）。

### 3.6 改进（按证据）

1. **v3 + 静止硬门控 both**（训练中，§5）：直接修 §3.2 的 gap/FP 拖动；预期 obj −0.3~−0.5。
2. **v3 的 onset 提议用短 EMA 而不是裸提议**（onset 处 α=1 → 改为对 onset 前几帧提议的平均，或 onset 后 3–5 帧内 α 从 1 线性降到 gate）——针对 §3.3 的 +0.47。
3. **融合权重：固定权重 + 静止门控 + 学习残差**（不是纯学习门控）。交叉重放显示固定权重对 obj 更好，学习门控对 FP 更保守，二者可合并为"固定权重上乘一个学习的收缩因子"。
4. **分歧门控/集成**：两条轨迹择优 12.78，平均约 −0.3；把 current 与 v3 的候选差作为置信度输入 publisher（residual GRU 的特征里已有 candidate_disagreement）。
5. HOI 对齐训练（§4）。

---

## 4. Q2：不给 mesh，怎么压 HOI

事实：
- HOI 度量 = 发布物体相对 refined 手的向量误差 = FK 帧上的骨向量（世界系）误差。obj 误差由手误差主导（真值骨向量 oracle 不改变 obj），HOI 由骨向量主导。
- 现在的骨误差 10.0 cm 里，**沿手误差方向 5.6 cm、正交 7.3 cm，α=−0.22**：head 为了 `obj_trans` 损失（权重 5）学会部分补偿手误差；
  这对 obj 有利（不补偿 obj 会 +1.0），但对 HOI 是纯损失。真值骨（onset 冻结）HOI 8.87 是"不补偿 + 段内刚性"的下界。
- 现有 `hoi_error_l/r` 只监督 FK 分支的方向×长度，不监督融合输出；融合在接触帧仍有 6–12% 权重给 IMU/静止分支（w_fk 0.88/0.94）。

方向与验证结果（§5 有完整表）：
1. **HOI 对齐损失**（已实现 `hoi_fused_l/r`：`‖(fused − hand_in) − (obj_gt − palm_gt)‖` on GT 接触帧；`train/rnn/loss/object_trans_loss.py`，默认权重 0）。
   **在 current 上有效：`static_gate: none`（基线设置）时 HOI 9.86→9.44、obj 13.94→13.72、jitter 8.43→7.38（cur_hoi）；加静止门控 9.59**；在 v3 上无效（10.03）。机制不是"去补偿"（α 反而更负），而是学习门控在接触帧降低 FK 权重（0.88→0.75）。
2. 静止硬门控 both：v3 obj −0.18 / HOI −0.06；current 变差（+0.12 / +0.21）。
3. 因果发布残差 GRU：canonical 上 **HOI 变化 0.000**（§5.3），不再有效。
4. 关闭状态反馈：全面变差（§5.4），反馈通路必须保留。
5. refine 头用度量对齐目标（相对向量）：未做；它等价于 Q1 的"逐帧把手拉向物体"的无 mesh 版本，会换来掌根绝对误差上升，需同时报告 MPJPE。
6. 长视野 / 在线协议对齐训练（52% 接触帧在 onset 60 帧之后，段内 growth 是主项）：未做，是 OT 侧剩下的最大杠杆；HP 侧的臂链精化（物体 IMU 作第七传感器）是碰到 5.6 cm 沿手误差分量的唯一路径。

---

## 5. 训练实验（`scripts/run_omomo_cano_ot_variants.py`，上游 HP/VC-v4 与基线相同，`--joint_train`，300+100 epoch，batch 200 × 1 窗口/序列 = 基线协议）

> **静止门控（`object_trans_static_gate`，`model/rnn/object_trans.py`）**：判定物体静止时把四路融合权重强制为 `[0,0,0,1]`（保持上一帧位置）。
> `imu` = 物体 IMU 因果规则（近 5 帧转动 <0.6° 且 |acc| <0.35）；`pmove` = VC 的 `p_move` <0.5；`both` = 任一成立；`none` = 关闭。
> 20260907 的两条基线都是 `none`。本节变体名里带 `gate` 的设为 `both`，`cur_hoi` 保持 `none`，因此它与基线 current 的唯一差别是新增损失。

| 变体 | 改动 | obj published | obj raw | HOI | jitter | MPJPE / root | F1 L/R/obj |
|---|---|---|---|---|---|---|---|
| 基线 ot_current | 无 | 13.94 | 14.35 | **9.86** | 8.43 | 2.99 / 6.06 | .942/.953/.971 |
| 基线 ot_v3 | 无 | 13.89 | 14.37 | 9.99 | 8.79 | 2.98 / 6.05 | .943/.953/.970 |
| **v3_gate** | static gate both | **13.72** | **14.20** | 9.93 | 8.90 | 2.99 / 6.06 | .943/.953/.970 |
| v3_gate_hoi | + hoi_fused 10/10，obj_trans 5→1 | 13.79 | 14.24 | 10.03 | 8.94 | 2.99 / 6.08 | .941/.952/.970 |
| cur_gate | current + static gate both | 14.06 | 14.53 | 10.08 | 8.45 | 2.99 / 6.08 | .942/.953/.971 |
| **cur_gate_hoi** | current + gate + hoi_fused 10/10，obj_trans 5→1 | 13.83 | 14.34 | **9.59** | **7.42** | 2.99 / 6.07 | .942/.953/.971 |
| v3_gate_hoi_nofb | v3 + gate + hoi_fused + 无状态反馈 | 14.40 | 14.85 | 10.69 | 9.22 | 2.99 / 6.06 | .941/.953/.970 |
| **cur_hoi** | current + hoi_fused 10/10，obj_trans 5→1，`static_gate: none`（与基线相同，唯一改动是损失） | **13.72** | 14.24 | **9.44** | **7.38** | 3.00 / 6.04 | .942/.953/.971 |

第一批读数（官方 `eval_batch_online.py`，1173 条，同协议）：
- **静止硬门控对 v3 有效但小**：obj −0.18（13.89→13.72）、raw −0.17、HOI −0.06；比重放的事后静止否决（raw −0.45）少，说明训练时带门控后 head 把一部分收益吃回去了；jitter 略升（8.79→8.90）。
- **静止硬门控对 current 有害**：obj +0.12、HOI +0.21。current 的学习门控已经在静止帧给了 0.74 的 w_static，再叠加硬门控会和它学到的 IMU 分支权重（0.21）冲突。
- **HOI 对齐损失（v3_gate_hoi）没有降低 HOI**：10.03 vs v3_gate 9.93、基线 v3 9.99。§4 的"去补偿"假设在当前训练协议下不成立，原因见 §5.1（dump 诊断）。
- 三个变体都没有在 HOI 上超过基线 current（9.86）。

第二批读数：
- **HOI 对齐损失在 current（学习门控）上有效**：cur_gate_hoi HOI **9.59**（比基线 current −0.27、比 cur_gate −0.49），jitter **7.42**（−1.0），obj 13.83（比基线 −0.11）；同一损失在 v3（固定 VC 权重）上无效。差别在于损失作用的对象：v3 的融合权重是固定的，`hoi_fused` 只能推骨向量（§5.1 已证明推不动）；current 的四路门控可学，损失把"接触帧上融合输出必须和手保持真值相对向量"直接教给了门控，同时把静止硬门控带来的伤害（cur_gate +0.21）抵消掉。
- **关掉状态反馈明显变差**（nofb：obj +0.7、HOI +0.7、jitter +0.3）：反馈通路 `R_objᵀ(p_prev − h)` 不是单纯的"补偿手误差"，它把物体的位置信息带进骨向量提议，去掉后先验骨向量更差。补偿是有益的副作用，不应该去掉。

### 5.1 第一批变体的 dump 诊断（`scripts/mesh_probe/variant_stats.py`，published 物体 + refined 手）

| 量 | base current | base v3 | v3_gate | v3_gate_hoi | cur_gate |
|---|---|---|---|---|---|
| obj / HOI | 13.94 / 9.86 | 13.89 / 9.99 | **13.72** / 9.93 | 13.79 / 10.03 | 14.06 / 10.08 |
| GT 静止帧：w_static / 每帧位移 cm / 误差 | 0.74 / 0.260 / 8.83 | 0.86 / 0.283 / 8.41 | 0.96 / **0.122** / **8.10** | 0.96 / 0.128 / 8.18 | 0.96 / 0.100 / 8.47 |
| 接触间隙内漂移（cm/100 帧） | 3.9 | 7.5 | **1.3** | 2.2 | – |
| 首个 onset 帧误差 | **3.64** | 5.28 | 4.87 | 4.96 | 3.74 |
| 段内误差增长 Δ | 13.69 | **11.87** | 12.46 | 12.50 | 13.70 |
| onset 骨误差 / 60 帧后增长 | 7.36 / 3.38 | 7.84 / 2.83 | 7.77 / 2.92 | 7.87 / 2.86 | 7.64 / 3.20 |
| 补偿系数 α / 沿手误差分量 / 正交分量 | −0.225 / 5.60 / 7.34 | −0.227 / 5.68 / 7.50 | −0.215 / 5.66 / 7.43 | **−0.216 / 5.73 / 7.43** | −0.227 / 5.72 / 7.49 |
| FP 帧误差 | 13.47 | 14.49 | 13.62 | 13.55 | 13.12 |

读法：
1. **静止门控做到了它该做的**：v3 的静止帧位移 0.283→0.122 cm/帧、间隙漂移 7.5→1.3 cm/100 帧、静止帧误差 8.41→8.10。但段内增长反而 +0.6（11.87→12.46），首个 onset 帧误差只从 5.28 降到 4.87，仍比 current 的 3.64 高 1.2——**v3 的 onset 落后不是静止漂移造成的，而是 onset 帧上固定权重立刻把 p_contact 的权重交给一个 7.8 cm 误差的骨向量提议**；current 的学习门控在 onset 更"懒"（TP 帧 w_static 0.04 vs 0.01，FN 帧 0.41 vs 0.30），代价是段内增长更大（13.7 vs 12.5）。两者各占一半好处。
2. **HOI 对齐损失没有改变补偿**：α 在 −0.215~−0.227 之间纹丝不动，沿手误差分量 5.6–5.7 不变。离线训练窗口（GT 初始化）里补偿本就很小，α 是在线滑窗从含手误差的已发布历史初始化后的结构效应，损失函数改不了。
3. **cur_gate 变差的位置**：largebox +1.2、floorlamp +0.4、tripod +0.4——IMU 规则把"纯平移、不转动"的慢速移动判成静止并钉住，而 current 的学习门控原本用 IMU 积分分支（静止帧 w_imu 0.21）跟随这类运动；v3 的固定权重本来就没有这个通路，所以门控对它只赚不赔。
4. 按物体 HOI，v3_gate 在 floorlamp/mop/trashcan/vacuum/smalltable 上优于 current，在 clothesstand/largebox/tripod/whitechair 上劣于 current——细长可转动物体是 v3 家族的弱项，与 onset 提议误差一致。

### 5.2 针对 onset 落后的两个免训练重放（v3_gate dump）

| 变体 | obj | HOI | jitter |
|---|---|---|---|
| v3_gate 原样 | 13.72 | 9.93 | 8.95 |
| onset 处骨向量改为 dead-reckoning 继承（`replay_bone_variants.py inherit`） | 15.60 | – | – |
| 继承 + 半径 3 / 5 / 10 cm 内允许 head 修正 | 14.87 / 14.63 / 14.48 | – | – |
| **onset 后 K 帧内 FK 权重线性爬升、其余给静止分支**（`onset_ramp_replay.py`），K=5 / 10 / 15 / 30 | 13.69 / 13.66 / **13.64** / 13.64 | 9.93 / 9.93 / 9.93 / 10.01 | 8.88 / 8.69 / **8.54** / 8.11 |

同样的 K=15 爬升在 base v3 上 13.89→13.81、在 current 上 13.94→13.92。结论：onset 时"继承已知位置"（改骨向量）在 canonical 上依旧为负（与旧版一致）；
"软交接"（改权重）是免费的小收益：obj −0.08、jitter −0.4、HOI 不变，可以直接进 publisher，但不足以填平与 current 的 onset 差距（首个 onset 帧 4.87 vs 3.64）。

### 5.3 因果发布残差（GRU32）在 canonical 上重训：基本无效

按 2026-09-05 的流程在 canonical 训练集上重做：分组开发集 1152/288 条（78/23 个 subject-object 组，零交叉），
v3_gate 上游的真实在线 dump（`hoi_residual_cano_20260908/train_dev_online.pt`），GRU32、gain 在验证集上选择。

| | 开发验证 HOI | obj | jitter |
|---|---|---|---|
| 基线（gain 0） | 6.285 | 11.33 | 7.65 |
| 选中（epoch 1，gain 0.75） | 6.265 | 11.30 | 7.62 |
| epoch 25（过拟合） | 6.66 | 11.30 | 8.04 |

旧版同一模块在开发验证上是 8.46→8.04、正式测试 10.82→10.07。**canonical 的 GT 掌根与 refined 掌根合同一致后，旧版残差利用的那部分系统性几何偏移不存在了**
（旧报告已警告"正式指标下的输出校正"不等于交互改善），剩下的 HOI 是不可从特征预测的手部漂移。

正式测试（1173 条，同协议，`hoi_residual_cano_20260908/official_residual_eval.json`）：

| | obj published | HOI | published jitter |
|---|---|---|---|
| v3_gate 未校正 | 13.717 | 9.933 | 8.949 |
| v3_gate + 残差 GRU32（gain 0.75） | 13.673 | **9.933** | 8.904 |

HOI 变化 0.000，obj −0.04。以 current 为上游的第二条残差链已取消（同样的开发验证读数不会不同）。

### 5.4 第二批 dump 诊断：HOI 对齐损失在 current 上改了什么

| 量 | base current | cur_gate | **cur_gate_hoi** | v3_gate_hoi | nofb |
|---|---|---|---|---|---|
| obj / HOI / jitter | 13.94 / 9.86 / 8.48 | 14.06 / 10.08 / 8.51 | 13.83 / **9.59** / **7.48** | 13.79 / 10.03 / 9.00 | 14.40 / 10.69 / 9.27 |
| 接触帧 FK 总权重 w_fk | 0.880 | 0.871 | **0.752** | 0.930 | 0.930 |
| 接触帧物体误差 | 14.02 | 14.13 | **13.31** | 14.13 | 14.86 |
| 首个 onset 帧误差 | 3.64 | 3.74 | **2.62** | 4.96 | 6.03 |
| 段内误差增长 Δ | 13.69 | 13.70 | 14.29 | 12.50 | 11.63 |
| 骨误差 onset / 全段 / 60 帧后增长 | 7.36 / 10.02 / 3.38 | 7.64 / 10.23 / 3.20 | 7.54 / **9.74** / **2.85** | 7.87 / 10.18 / 2.86 | 8.76 / 10.88 / 2.50 |
| 补偿 α / 沿手误差 / 正交 | −0.225 / 5.60 / 7.34 | −0.227 / 5.72 / 7.49 | **−0.265** / 5.61 / **6.95** | −0.216 / 5.73 / 7.43 | −0.203 / 6.00 / 8.04 |
| FP 帧误差 / 预测位移 | 13.47 / 1.36 | 13.12 / – | **12.54 / 0.94** | 13.55 / – | 14.37 / – |
| TN（无接触）帧误差 | 15.07 | – | **16.90** | – | – |
| "never"（无接触标签）序列误差 | 24.3 | – | **29.7** | – | – |

读法：
1. `hoi_fused` 在 current 上改变的是**学习门控**：接触帧 FK 权重 0.88→0.75，物体更多地由 IMU/静止分支平滑地带着走，接触帧物体误差 −0.7、onset 帧误差 −1.0、jitter −1.0，骨向量 60 帧后的增长 −0.5，正交分量 −0.4；
   α 反而更负（−0.265）——所以 HOI 改善不是"去补偿"，而是融合输出在接触帧上更贴近真值相对向量。逐序列 57% 改善，接触段越多的序列改善越大（corr(Δ, n_seg) = −0.33）。
2. 代价在无接触阶段：TN 帧误差 +1.8、"never" 序列 +5.4、段内增长 +0.6——门控更依赖 IMU/静止分支后，释放后与无标签序列的漂移更大。obj 净 −0.11。
3. **第三批 cur_hoi（`static_gate: none`，门控设置与基线一致，唯一改动是 hoi_fused 损失）三项指标都是最好**：obj 13.72（与 v3_gate 持平）、HOI 9.44（比基线 current −0.42、比 cur_gate_hoi −0.15）、jitter 7.38。静止硬门控对 current 只有害处（与学习门控的 IMU 分支冲突），HOI 对齐损失单独用即可。
4. **nofb 变差的机制**：onset 骨误差 7.9→8.8、全段 10.2→10.9，静止/onset 误差全面上升；反馈 `R_objᵀ(p_prev − h)` 是骨向量提议最重要的输入之一，不是可以拿掉的"补偿通道"。

协议注记：`configs/IMUHOI_train_rnn.yaml` 在 2026-09-08 18:38 被改为 `batch_size: 800, windows_per_sequence: 4`（每 epoch 4 倍窗口），基线 OT 的保存配置是 batch 200 / 1 窗口；首次启动的变体沿用了 800×4，每 epoch 慢 2.6 倍且与基线 epoch 预算不等价，已按基线协议重启。
工程注记：本机 WSL 上全尺寸训练在 DataLoader pin-memory 线程 SIGSEGV（`-X faulthandler` 定位到 `Tensor.pin_memory`），
`dataloader_pin_memory: false` 后正常；orchestrator 已默认关闭。

---


## 5.5 已落地的默认配置

`configs/IMUHOI_train_rnn.yaml` 的默认 OT 改为 cur_hoi：`object_trans_arch: current`，并新增按架构生效的权重块

```yaml
object_trans_arch_loss_weights:
  current: {obj_trans: 1.0, hoi_fused_l: 10.0, hoi_fused_r: 10.0}
```

`resolve_object_trans_loss_weights`（`train/rnn/loss/object_trans_loss.py`）只把当前选中架构的块叠在共享 `loss_weights` 之上，
所以 v2/v3/legacy 的有效权重与新增此机制前逐位一致（`obj_trans` 5、`hoi_fused` 权重 0）。共享 `loss_weights` 块本身未改动。
`object_trans_static_gate` 保持 `none`（对 `current` 正确；训练 v3 时应设 `both`）。
`velocity_contact_arch` 未改动，但上表全部在 VC-v4 上测得，复现时需 `--velocity_contact_arch v4`。
测试见 `test/test_rnn_ot_arch_loss_weights.py`。

## 6. 复现

```bash
# dumps（两条 pipeline，在线协议）
python scripts/dump_ot_predictions.py --manifest outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/experiment_manifest.json --variant ot_v3 --mode online --out .../dumps/ot_v3_online.pt --online_batch_size 48
# 对比分解
python scripts/analyze_ot_dumps.py current=.../ot_current_online.pt v3=.../ot_v3_online.pt --compare current v3
# mesh 重放 + 交叉重放（需 outputs/analysis/mesh_probe_20260907/floor_calibration_train.json）
python scripts/mesh_probe/calibrate_floor_train.py && python scripts/mesh_probe/cano_replay.py
# 训练变体
python scripts/run_omomo_cano_ot_variants.py --experiment_root outputs/IMUHOI++/RNN_cano/omomo_cano_ot_variants_20260908 --variants v3_gate v3_gate_hoi --max_parallel 2
```
