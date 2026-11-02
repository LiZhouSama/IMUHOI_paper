# OT 模块消融实验方案（服务器端执行）

编写：2026-11-02（本地 WSL 机器上的 agent，交给服务器 `llz@202.120.59.70:/disk2/mmzhou/IMUHOI` 上的 agent 执行）。
项目根目录在服务器上就是 `/disk2/mmzhou/IMUHOI`，下文所有路径都相对它。代码注释用英文，汇报用简体中文。

## 0. 目标

IMUHOI++ 的物体位移模块（OT，`model/rnn/object_trans_v2.py`，发表配置 `v2_armrot_biclose`）相对 IMU-HOI 的三分支 OT
（`model/rnn/object_trans_IMUHOI.py`）有一组改动。本实验在 OMOMO、BEHAVE、IMHD、PAHOI、HODome 五个数据集上做三组消融，
每一行都重训 OT 和联合阶段、重新在线评测：

| 组 | 行名 | 去掉的东西 | 回答的问题 |
|---|---|---|---|
| 参考 | `full` | 无（发表配方在服务器上重跑） | 本机上的基准 |
| A 闭环抓取估计 | `no_closedloop` | 状态反馈、双手闭合候选（biclose）、前臂-物体相对旋转 B_s、FK 候选 α–β 跟踪滤波、双手共享的抓取预测器（改成每只手一个开环 LSTM） | 闭环、双手一致的抓取偏移估计值多少 |
| B 门控融合策略 | `learned_gate` | 闭式接触融合权重（改成 IMU-HOI 式学习门控）、静止候选、IMU 静止门控 | 闭式融合 + 静止处理值多少 |
| C GT 接触 | `gtc`、`imuhoi3b`、`imuhoi3b_gtc` | 接触概率在训练和测试中都换成 GT，新模型（我们）和旧模型（三分支 IMU-HOI）各一次 | OT 在完美接触下的上限，以及新旧 OT 对接触质量的依赖 |

指标列：物体误差、HOI 误差、**双手交互帧上的物体误差和 HOI 误差（新增）**、单手接触帧上的两项误差、物体抖动。每行相对参考行给出按序列配对 bootstrap 的 95% 置信区间。

不在本实验范围内的 OT 改动（各行都保持发表配置）：接触段长度统计 Λ（`object_trans_lb_stats`）、手掌锚点、偏移向量 HOI
损失和长度一致性损失、GT→预测的上游输入调度、联合阶段微调 VC+OT（HP 冻结）、refine 头、在线协议（窗口 60、前瞻 10、HP 平滑
7、物体速度 3 帧差分初始化）。

## 1. 第 0 步：开始前必须完成的检查（任何一项不通过都先停下来汇报）

### 1.1 文件完整性

本地用 `scripts/upload/make_manifest.py` 生成了 `scripts/upload/upload_manifest.tsv`：上传清单里每个文件的大小，以及检查点、
配置和代码的 sha256。在服务器项目根目录运行：

```bash
python scripts/upload/verify_upload.py            # 全部校验（数据只比大小，其余比 sha256）
python scripts/upload/verify_upload.py --no-hash  # 只比大小，先快速看一遍
```

要求输出 `ALL OK`。四组含义：

| kind | 清单 | 内容 |
|---|---|---|
| `data` | `scripts/upload/filelist_data.txt` | 五个数据集的 train/test、完整序列 test，SMPL-H/SMPL-X 身体模型 |
| `ckpt` | `scripts/upload/filelist_ckpt.txt` | IMUHOI++ 发表版：HP、VC 阶段 best.pt、OT+联合 pipeline 目录、配方 |
| `ckpt_imuhoi3b` | `scripts/upload/filelist_ckpt_imuhoi3b.txt` | 三分支 IMU-HOI：VC-IMUHOI 阶段 best.pt、三分支 OT pipeline 目录 |
| `code` | `scripts/upload/filelist_code.txt` | 代码（包括尚未提交 git 的 VC 消融改动和本方案） |

有缺失或不一致时，把报告里的路径列出来交给用户重传，不要自己猜着补。`code` 组如果因为服务器上的代码比清单更新
（例如已经开始改代码）而不一致，要确认不一致的只是你自己改过的文件。

### 1.2 环境

本地参考环境：Python 3.8.20、torch 1.13.1+cu117（cuDNN 8500）、pytorch3d 0.7.3、numpy 1.23.5、RTX 4080 SUPER。
服务器环境不同也可以，但要在汇报里写明版本。

把下面 4 处写死的本地路径改成服务器路径：

- `scripts/run_behave_full_pipeline_trust.py` 中的 `SAGE_PYTHON`（编排器用它启动所有子进程）；
- `scripts/run_remaining_datasets_comparison.py` 中的 `PYTHON`；
- `outputs/IMUHOI++/RNN_cano/vc_ablation_omomo_20261030/run_ablation.sh` 中的 `cd`、`PY`，以及同目录 `worker.sh` 中的
  `ROOT`（只在继续 VC 消融时用到，本实验用不上）。

配置 YAML 里的 `save_dir` 是本地绝对路径，但运行时会被 `--save_dir` / `--experiment_root` 覆盖，不用改。

### 1.3 单元测试（修改代码前后各跑一次，都要全部通过）

```bash
python -m pytest -q test/test_rnn_object_trans_v2.py test/test_rnn_ot_v2_armrot_biclose.py \
  test/test_rnn_ot_known_prefix_fast_path.py test/test_rnn_ot_bimanual_closure.py \
  test/test_rnn_ot_forearm_obj_rotation.py test/test_eval_hoi_metric.py test/test_rnn_online_inference.py \
  test/test_rnn_vc_ot_compat.py test/test_rnn_ot_arch_loss_weights.py test/test_rnn_vc_v5.py test/test_rnn_imuhoi_arch.py
```

本地结果：97 passed。

### 1.4 端到端复现发表版评测（检验数据 + 检查点 + 代码 + 环境）

在服务器上，用发表版检查点对 10 个 pipeline（5 个数据集 × 新/旧模型）跑在线评测，与随检查点一起上传的发表版 JSON 比较。
评测命令与编排器 `evaluate()` 相同（见 4.3），只是 `--output_json` 写到新文件。允许的差异（不同 GPU 的浮点差别）：

- 物体误差、HOI 误差、根节点误差：≤ 0.05 cm；

超出这个范围，说明数据、检查点或环境有问题，先停下来汇报。对照的 JSON：

| 数据集 | 新模型（我们） | 旧模型（三分支） |
|---|---|---|
| OMOMO | `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json` | `outputs/IMUHOI++/RNN_cano/imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch/eval_omomo_best.json` |
| BEHAVE | `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json` | `outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch/eval_behave_best.json` |
| IMHD / PAHOI / HODome | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/{ds}/proposed/v2_armrot_biclose/eval_{ds}_best.json` | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/{ds}/imuhoi3b/vc_imuhoi_ot_3branch/eval_{ds}_best.json` |

对应的论文数值（物体 / HOI，cm）：

| 模型 | OMOMO | BEHAVE | IMHD | PAHOI | HODome |
|---|---|---|---|---|---|
| 新模型 | 13.34 / 10.48 | 22.93 / 17.19 | 32.69 / 17.23 | 9.95 / 8.06 | 26.66 / 17.99 |
| 旧模型 | 17.84 / 15.04 | 95.45 / 79.90 | 103.24 / 97.49 | 12.72 / 11.02 | 81.24 / 58.91 |

`MPJRE` 不参与比较：10-25 之后的评测代码修正过它的计算。

## 2. 实验矩阵

每个数据集 6 个训练行，外加 2 个只评测、不重训的行：

| 行 | 编排器变体 | 训练什么 | 评测 |
|---|---|---|---|
| `full` | `v2_armrot_biclose`（已有） | OT + 联合，HP 与 VC 阶段取发表版 | 正常 |
| `no_closedloop` | `v2abl_no_closedloop`（新增） | 同上 | 正常 |
| `learned_gate` | `v2abl_learned_gate`（新增） | 同上 | 正常 |
| `gtc` | `v2_armrot_biclose_gtc`（新增） | 同上，OT 训练时接触 = GT | `--oracle_vc_contact all` |
| `imuhoi3b` | `vc_imuhoi_ot_3branch`（已有） | 三分支 OT + 联合（联合阶段微调 HP+OT），HP 与 VC-IMUHOI 取发表版 | 正常 |
| `imuhoi3b_gtc` | `vc_imuhoi_ot_3branch_gtc`（新增） | 同上，OT 训练时接触 = GT | `--oracle_vc_contact all` |
| `full_oracle_eval` | — | 不训练，用 `full` 的检查点 | `--oracle_vc_contact all` |
| `imuhoi3b_oracle_eval` | — | 不训练，用 `imuhoi3b` 的检查点 | `--oracle_vc_contact all` |

为什么参考行也要在服务器上重训：训练在同一台机器、同一份代码上是逐位确定的（本地用 OMOMO 验证过：重跑与发表版检查点
完全相同），但换了 GPU 或驱动就不再逐位相同，差别相当于换了一个种子。所以消融行必须与服务器上重训的 `full` / `imuhoi3b`
比较，不能直接和发表版数值相减。发表版数值只用于 1.4 的复现检查，以及在汇报里注明“服务器重训的参考行与发表版相差多少”。

GT 接触这一组的三层对比是：真实接触（`full`）→ 只在测试时换成 GT（`full_oracle_eval`，相当于本地 A3 实验 `experiments/05`）
→ 训练和测试都用 GT（`gtc`）；旧模型同理。

## 3. 需要实现的代码改动

总原则：

- 所有新开关**默认关闭**，关闭时与现有代码逐位一致。用发表版检查点加载新、旧代码，在随机输入上比较前向输出必须完全相等
  （做法参考 `scripts/vc_probe/` 下的 VC 消融：把 HEAD 版本复制成临时模块做对照）。
- 每个开关都要写单元测试（`test/` 下，风格参照 `test/test_rnn_ot_bimanual_closure.py`），并在检查点里登记标记 buffer，
  让配置不匹配的加载直接报错，不能静默错用。只在开关打开时才注册新 buffer，这样旧检查点照常加载
  （参照 `model/rnn/velocity_contact_v5.py` 的 `_v5_ablation_code`）。

### 3.1 每只手一个开环抓取头（A 组）

开关：`object_trans_per_hand_heads: true`（只支持 `object_trans_arch: v2`；要求 `object_trans_state_feedback: false`、
`object_trans_bimanual_closure: false`、`object_trans_forearm_obj_rotation: false`，否则直接报错）。

- 现状：`ObjectTransModule.__init__`（`model/rnn/object_trans.py`）建一个双手共享的 `self.object_trans_head`
  （`RNNWithInit`，2 层 LSTM，隐层 `object_trans_hidden_dim` = 192）。v2 在 `V2ObjectTransModule.__init__` 里把它重建成
  `n_output=8` 的版本（左 FK 方向 3 + 长度 1、右 FK 方向 3 + 长度 1）。
- 改成两个 `RNNWithInit`（左、右），各输出 4 维，隐层、层数、dropout 与共享头相同。长度输出的 bias 用同样的
  `object_trans_length_init` 初始化（对应输出下标 3）。两个头的输出拼成 8 维，下游解码和融合不变。
- 每只手的输入：从共享输入里切出
  - 物体部分：`obj_imu`、`obj_rot_delta`、`obj_vel`；
  - 接触：本手的 p_s 和物体运动 p_m；
  - 本手：手掌位置、前臂 IMU；
  - 本手的 Λ 分位数。

  不给另一只手的任何信息。共享输入的拼接顺序见 `_build_prediction_inputs`：
  `[obj_imu(9), obj_rot_delta(3), obj_vel(3), contact(3), lhand_pos(3), lhand_imu(9), rhand_pos(3), rhand_imu(9), (forearm), lb(6)]`。
  Λ 的 6 维是 `[左, 右] × [q10, q50, q90]`，用前要核对排列。
- 每只手的初始化向量：本手在窗口起点的方向（3）和长度（1）、本手的初始接触状态、本手的 Λ 分位数。共享头的 init 布局见
  `_grasp_prediction_init` / `prediction_init`：`[l_dir, l_len, r_dir, r_len, contact_init(3), lb(6)]`。
- 实现建议：写一个与 `RNNWithInit` 接口一致的包装模块（`initial_state`、`step`、`forward_with_state`，以及被外部读取的属性
  `n_hidden`、`n_rnn_layer`、`init_net`、`linear2`），把状态做成两个头状态的组合。这样扫描、在线推理和 known-prefix 快速路径
  的代码都不用动。
- 验收：
  - 单元测试覆盖形状、梯度，以及“改动另一只手的输入不影响本手输出”；
  - 在线推理的快速前缀路径和逐帧路径结果一致（参照 `test/test_rnn_ot_known_prefix_fast_path.py`）。

### 3.2 v2 的学习门控融合与静止候选开关（B 组）

开关：`object_trans_v2_fusion: closed_form | learned`（默认 `closed_form`）、`object_trans_static_candidate: true | false`
（默认 `true`）。静止门控用现有开关 `object_trans_static_gate: none`。

- 现状：`V2ObjectTransModule._decode_prediction_output` 用 `utils/object_fusion.py::build_contact_fusion_weights` 由接触概率
  闭式算出四路权重 `[左 FK, 右 FK, 惯性, 静止]`。FK α–β 滤波只存在于 v2（`_predict_step`），所以**不能**改用
  `object_trans_arch: current` 来做这一行，否则会把属于 A 组的滤波也一起关掉。
- `learned`：
  - 头的输出增加 G 个门控 logits（G = 3，无静止候选；G = 4，有静止候选）。
  - 权重 `w = softmax((z + β·log π) / τ)`，β = `gating_prior_beta`（5.0），τ = `gating_temperature`（5.0），与 IMU-HOI 相同。
  - 先验 π：G = 3 时用 IMU-HOI 的 `IMUHOIObjectTransModule._build_gating_prior`，即 `(p_L, p_R, 1 − max(p_L, p_R))`；
    G = 4 时用 `build_contact_fusion_weights`。
  - 没有静止候选时，在静止那一列补 0，交给现有的四路融合代码。门控 logits 随输出返回，用于诊断。
- 保持不变：FK 滤波、状态反馈、前臂旋转、biclose。biclose 的第五路权重由 `_closure_weights5` 用它自己的 pair logit 计算，
  不受影响。
- 损失：学习门控需要从融合位置拿到梯度。保留 v2 配方中融合位置的 `obj_trans`（λ = 5），再加上 `hoi_fused_l` / `hoi_fused_r`
  （λ = 10，与之前的学习门控 OT `current` 架构相同，见 `object_trans_arch_loss_weights.current`）。要核实这两项在 v2 输出上
  确实被计算；如果没有，需要补上。
- 验收：
  - 单元测试覆盖权重之和为 1、无静止候选时静止权重恒为 0、先验的形式；
  - 快速前缀路径与逐帧路径一致；
  - 开关关闭时与现有 v2 逐位一致。

### 3.3 GT 接触训练（C 组，新旧模型都要用）

开关：`ot_train_oracle_contact: true`。

- 位置：`train/rnn/train_object_trans.py::Stage3JointTrainer.forward_all`。这里先得到传给 OT 的 `contact_prob`：
  - 我们的模型：GT 与 VC 预测按 `prediction_mix_probability` 调度混合；
  - 三分支：`self.imuhoi_arch` 分支直接用 VC 预测。

  开关打开时，在这之后统一换成 `build_gt_velocity_contact_outputs(batch, ...)['pred_hand_contact_prob']`
  （GT 的左手、右手、物体运动三路）。训练和验证都要换（只要拿得到 `batch`），OT 阶段和联合阶段都要换。
- 联合阶段：VC（我们的模型）或 HP（三分支）照常微调，VC 的损失照常计算，只是 OT 看不到 VC 的接触输出。
- 要核实：所有 OT 架构（v2 和 imuhoi）都只通过这个 `contact_prob` 读接触；如果还有别处读 `vc_out['pred_hand_contact_prob']`
  （例如 Λ 统计、初始接触状态），也要一并处理，并在汇报里写清楚。
- 评测：用现成的 `eval_batch_online.py --oracle_vc_contact all`，即在 OT 读取前把 VC 的三路接触换成 GT
  （`IMUHOIModel._apply_vc_oracle`）。

### 3.4 双手交互帧指标（评测）

位置：`eval_IMUHOI.py`（`eval_batch_online.py` 调用它的指标计算），和现有物体误差、`hoi_err_fusion` 写在同一处。

- 每个序列在已发布帧上取 GT 掩码：双手帧 `both = gt_l & gt_r`，单手帧 `single = gt_l ^ gt_r`。
- 对每种掩码计算两项：
  - 物体误差：`‖pred_obj − gt_obj‖`（用与 `obj_trans_err_fusion_refine` 相同的预测）在掩码帧上取平均，×100 转成 cm；
  - HOI 误差：`compute_hoi_error_at_gt_contact(pred_obj, gt_obj, pred_hand, gt_hand, mask, mask)`，口径与 `hoi_err_fusion`
    一致，只是把接触帧限定在掩码帧上。
- 新增结果键：`obj_trans_err_bimanual`、`hoi_err_bimanual`、`obj_trans_err_single_hand`、`hoi_err_single_hand`，以及每个序列
  的帧数 `n_bimanual_frames`、`n_single_hand_frames`。
- 与其它指标一样先按序列平均，再在有这类帧的序列之间平均；没有这类帧的序列记为 NaN。同时写进 `--sequence_records_json`
  的逐序列记录。
- 验收：在发表版检查点上，所有已有指标与 1.4 的结果逐项相同，只是多出这几个新键。

### 3.5 编排器（`scripts/run_behave_full_pipeline_trust.py`）

- 变体的 `overrides` 在命令行 `--set` **之后**生效（`variant()` 里先 `--set` 再 `deep_update(config, spec["overrides"])`）。
  所以**不能**用 `--set` 去关 `v2_armrot_biclose` 变体自带的开关，必须新增变体：

```python
"v2abl_no_closedloop": {
    "label": "ablation A: open-loop per-hand grasp heads (no feedback, biclose, forearm-object rotation, FK filter)",
    "ot_arch": "v2",
    "overrides": {
        "loss_weights": {"bimanual_hoi": 0.0, "bimanual_prior": 0.0},
        "object_trans_forearm_obj_rotation": False,
        "object_trans_bimanual_closure": False,
        "object_trans_state_feedback": False,
        "v2_fk_filter_alpha": 1.0,
        "v2_fk_filter_beta": 0.0,
        "object_trans_per_hand_heads": True,
    },
},
"v2abl_learned_gate": {
    "label": "ablation B: IMU-HOI learned 3-way gate, no static candidate, no IMU static gate",
    "ot_arch": "v2",
    "overrides": {
        "loss_weights": {"bimanual_hoi": 5.0, "bimanual_prior": 1.0, "hoi_fused_l": 10.0, "hoi_fused_r": 10.0},
        "object_trans_forearm_obj_rotation": True,
        "object_trans_bimanual_closure": True,
        "object_trans_v2_fusion": "learned",
        "object_trans_static_candidate": False,
        "object_trans_static_gate": "none",
    },
},
"v2_armrot_biclose_gtc": {
    "label": "IMUHOI++ OT trained and evaluated with GT contact",
    "ot_arch": "v2",
    "overrides": {
        "loss_weights": {"bimanual_hoi": 5.0, "bimanual_prior": 1.0},
        "object_trans_forearm_obj_rotation": True,
        "object_trans_bimanual_closure": True,
        "ot_train_oracle_contact": True,
    },
},
"vc_imuhoi_ot_3branch_gtc": {
    "label": "IMU-HOI three-branch OT trained and evaluated with GT contact",
    "ot_arch": "imuhoi",
    "overrides": {"ot_train_oracle_contact": True},
},
```

- `evaluate()`：
  - 变体配置里 `ot_train_oracle_contact` 为真时，评测命令追加 `--oracle_vc_contact all`；
  - 始终追加 `--sequence_records_json <输出 JSON 同名的 _records.json>`，供配对 bootstrap 使用。
- 新增变体后，先用 `--dry_run` 打印配置，核对上面这些键最终确实生效（特别是 `v2_fk_filter_alpha` 为 1.0、
  `object_trans_static_gate` 为 none）。

## 4. 运行

### 4.1 每个数据集的冻结输入

| 数据集 | HP（新旧模型共用） | VC 阶段（新模型） | VC-IMUHOI 阶段（旧模型） |
|---|---|---|---|
| OMOMO | `omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt` | `vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930/best.pt` | `imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/best.pt` |
| BEHAVE | `behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt` | `vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930/best.pt` | `imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/best.pt` |
| IMHD | `remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp/human_pose_10100114_imhd_hp_remaining3_20261010/best.pt` | `remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_v5/velocity_contact_v5_10100229_imhd_vc_v5_remaining3_20261010/best.pt` | `remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10100654_imhd_vc_imuhoi_remaining3_20261010/best.pt` |
| PAHOI | `remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp/human_pose_10100950_pahoi_hp_remaining3_20261010/best.pt` | `remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_v5/velocity_contact_v5_10101041_pahoi_vc_v5_remaining3_20261010/best.pt` | `remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101407_pahoi_vc_imuhoi_remaining3_20261010/best.pt` |
| HODome | `remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp/human_pose_10101550_hodome_hp_remaining3_20261010/best.pt` | `remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_v5/velocity_contact_v5_10101621_hodome_vc_v5_remaining3_20261010/best.pt` | `remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101856_hodome_vc_imuhoi_remaining3_20261010/best.pt` |

（表中路径都在 `outputs/IMUHOI++/RNN_cano/` 下。）

### 4.2 命令模板（输出根目录 `OUT=outputs/IMUHOI++/RNN_cano/ot_ablation_20261102`）

OMOMO、BEHAVE，新模型的 4 行（与发表版 `vc_v5_20260930/protocol_1002/run_ot_omomo_v5ar_unified.sh`、`vc_v5_20260930/run_ot.sh`
一致）：

```bash
python scripts/run_behave_full_pipeline_trust.py --dataset $DS --experiment_root $OUT/$DS/$ROW --run_tag abl_$ROW \
  --vc_arch v5 --skip_hp $HP --skip_vc $VC --max_parallel 1 --max_attempts 3 --resume --variants $VARIANT \
  --set v2_fk_filter_alpha=0.35 --set v2_fk_filter_beta=0.05 --set v5_heading_normalize_imu=false \
  --set v5_init_noise_prob=0.5 --set v5_obj_vel_init_noise_std=0.2 --set v5_hand_vel_init_noise_std=0.2 \
  --set v5_contact_init_flip_prob=0.0 --set human_pose_smoothing_half_width=7 --set online_obj_vel_init_diff_frames=3
```

OMOMO、BEHAVE，旧模型的 2 行（与 `outputs/IMUHOI++/RNN_cano/chain_imuhoi3b_20261012_omomo_unified.sh` 一致）：

```bash
python scripts/run_behave_full_pipeline_trust.py --dataset $DS --experiment_root $OUT/$DS/$ROW --run_tag abl_$ROW \
  --vc_arch imuhoi --skip_hp $HP --skip_vc $VC_IMUHOI --max_parallel 1 --max_attempts 3 --resume --variants $VARIANT \
  --set human_pose_smoothing_half_width=7 --set online_obj_vel_init_diff_frames=3 --check_gpu_before_stage
```

IMHD、PAHOI、HODome：发表版由 `scripts/run_remaining_datasets_comparison.py` 用配方
`outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/recipe.yaml` 一次训出 HP、VC、OT。这里复用 HP 和 VC 阶段，
只训 OT + 联合：

```bash
RECIPE=outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/recipe.yaml
# new model rows: --vc_arch v5 --skip_vc $VC; old model rows: --vc_arch imuhoi --skip_vc $VC_IMUHOI
python scripts/run_behave_full_pipeline_trust.py --cfg $RECIPE --dataset $DS --experiment_root $OUT/$DS/$ROW \
  --run_tag abl_$ROW --vc_arch v5 --variants $VARIANT --seed 42 --gpu 0 --num_workers 2 --eval_batch_size 48 \
  --max_parallel 1 --schedule_scale auto --resume --check_gpu_before_stage --skip_hp $HP --skip_vc $VC
```

各数据集的训练轮数、学习率里程碑和早停窗口由编排器按每轮步数自动缩放，与发表版相同，不要手动改。

### 4.3 只评测、不重训的两行（以及 1.4 的复现评测）

```bash
# new model: PIPE=$OUT/$DS/full/v2_armrot_biclose, archs v5 / v2
# old model: PIPE=$OUT/$DS/imuhoi3b/vc_imuhoi_ot_3branch, archs imuhoi / imuhoi, and --hp_ckpt $STAGE/best_human_pose.pt
STAGE=$(ls -d $PIPE/object_trans_* | tail -1)
python eval_batch_online.py --config $PIPE/config.yaml --dataset $DS --hp_ckpt $HP \
  --vc_ckpt $STAGE/best_velocity_contact.pt --ot_ckpt $STAGE/best_object_trans.pt \
  --velocity_contact_arch v5 --object_trans_arch v2 --simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 \
  --num_workers 2 --online_batch_size 48 --per_joint_rot_error --oracle_vc_contact all \
  --output_json $OUT/$DS/full_oracle_eval/eval_${DS}_best.json \
  --sequence_records_json $OUT/$DS/full_oracle_eval/eval_${DS}_best_records.json
```

三分支模型的联合阶段会微调 HP，所以它的 `--hp_ckpt` 要用 stage 目录里的 `best_human_pose.pt`；编排器的 `ckpt_set` 也是这样
选的。做 1.4 的复现检查时，去掉 `--oracle_vc_contact`，`PIPE` 指向发表版目录。

## 5. 汇总与报告

- 写 `scripts/ot_ablation/summarize_ot_ablation.py`（可以参照 `scripts/vc_probe/summarize_vc_ablation.py` 的统计部分和
  `analyze_contact_events.boot_weights`）：读取每一行的 `eval_{ds}_best.json` 和 `_records.json`，输出
  `$OUT/{ds}/summary.md`、`metrics.json`。
- 主表列：物体误差、HOI 误差、双手帧物体误差、双手帧 HOI 误差、单手帧 HOI 误差、物体抖动。
- 每个消融行给出相对参考行（新模型行对 `full`，旧模型行对 `imuhoi3b`）的差值，以及按序列配对 bootstrap 的 95% CI
  （1000 次，所有行共用同一组重采样）。双手帧指标只在有双手帧的序列上做 bootstrap。
- 每个数据集报告双手帧的总帧数和涉及的序列数。IMHD 只有 15 个序列，置信区间会很宽，要标出来。
- 写一页中文 README，放在 `experiments/07_ot_module_ablation/`，格式参照 `experiments/06_vc_module_ablation/README.md`：
  问题、设置、结果表、机制解释、局限、复现命令。再给一个论文格式的总表（Markdown 和 LaTeX 各一份，参照
  `scripts/vc_probe/make_vc_ablation_paper_table.py`）。
- 汇报里必须写清楚：
  - 服务器重训的 `full` / `imuhoi3b` 与发表版的差距；
  - 每个开关实际生效的配置（从 pipeline 的 `config.yaml` 摘出来）；
  - 所有失败、重试和续训的情况。

## 6. 执行顺序与资源

1. 第 0 步全部通过（1.1 → 1.4）。
2. 实现 3.1–3.5 并通过验收；先在 OMOMO 上用 2 个 epoch 的冒烟配置（`--set epochs=2 --set joint_train_epochs=1
   --set eval_interval=1 --set joint_train_eval_interval=1`，写到临时目录）把 4 个新变体各跑通一次训练和评测。
3. 正式运行，优先顺序：
   - OMOMO、BEHAVE 的 `full`、`no_closedloop`、`learned_gate`、`gtc`、`imuhoi3b`、`imuhoi3b_gtc`，以及两个只评测的行；
   - 然后是 IMHD、PAHOI、HODome 的同样各行。
4. 本地的单行耗时参考（RTX 4080 SUPER，独占 GPU，OT + 联合 + 两次评测）：OMOMO 约 2.8 h、BEHAVE 约 3.2 h、IMHD 约 2.4 h、
   PAHOI 约 2.3 h、HODome 约 1.8 h。30 个训练行合计约 75 GPU 小时（三分支行的耗时可能不同）。

## 7. 注意事项

- **一块 GPU 同时只跑一个训练作业。** 本地两次出过事故：
  1. 显存接近上限时，WSL/WDDM 不报 OOM，而是换页，作业停滞；
  2. 两个 OT 训练并发时出现过不确定的运行时数据损坏：loss 变成 1.4e-45、CUBLAS 执行失败、续训时 AdamW 除零。

  服务器是原生 Linux，第 1 种不会以同样方式出现，但第 2 种原因不明。多卡时每卡一个作业；同一卡上不要并发。
- **训练崩溃后先查原因，不要直接用 `--resume` 续训**：崩溃前保存的检查点可能已经坏了。可以用确定性来判断：同一命令单独重跑，
  比较崩溃前各轮的 loss；如果逐位相同，说明是运行时损坏而不是代码问题。
- **BEHAVE 的在线评测很吃显存**（本地约 5 GB，序列长达 1399 帧），而它是在编排器内部启动的，绕过任何外部启动闸门。
- **不要改动或覆盖上传来的发表版目录**（`vc_v5_20260930*`、`imuhoi3b_*`、`remaining3_*`）。所有新输出写到 `$OUT` 下。
- **`pgrep -f <pattern>` 会匹配到自己所在 shell 的命令行**（模式也出现在那条命令里），在循环里用它做等待条件会永远等下去。
  要把模式锚定在解释器路径上，或者直接用 PID。
- **模型选择**：`best.pt` 按测试窗口损失选择，早停也看测试集。这与发表版一致，但属于共同局限，要在 README 的局限一节写明。
- **IMU 噪声**：训练和评测都用数据集级的 Noitom 噪声模型（`process/imu_noise.py`），每个序列一个固定的随机实现，种子为 42。
  `--simulate_imu_noise_seed 42` 不要改。

## 8. 交付物

- 改动的代码和新测试（单元测试全部通过；开关关闭时逐位一致的核对记录）。
- `$OUT/{ds}/{row}/`：每一行的 pipeline 目录、评测 JSON 和逐序列记录。
- `$OUT/{ds}/summary.md`、`metrics.json`，以及 `experiments/07_ot_module_ablation/`（README、各数据集汇总、论文格式表）。
- 一段中文汇报：
  - 第 0 步检查结果（完整性、环境版本、复现差距）；
  - 每组消融的主要数字和置信区间；
  - 双手交互帧上的发现；
  - 与发表版的差距；
  - 所有异常情况。
