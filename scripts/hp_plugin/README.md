# HP 插件实验（2026-11-05）

## 目的

验证 IMUHOI++ 的交互部分（VC + OT + 联合阶段）可以作为插件，接到其他 IMU 人体姿态估计方法上使用。

做法：把我们的 HP 换成 Comparisons/ 下四个方法的人体姿态，在其冻结的输出上重训 VC 和 OT。四个方法是 DIP、TIP、TransPose、GlobalPose，都是论文版复现（protocol v3），用的检查点与已发表对比表相同（`selected.pt`）。5 个数据集 × 4 个方法，共 20 个任务。

每个数据集报告三行，测试集都是同一个全序列测试集：

- **X 单独**：方法自身的物体头；
- **X + 我们的 VC/OT**；
- **IMUHOI++**：发表版，即我们自己的 HP。

## 设计

### 缓存

比较方法按各自论文的在线协议，对每个序列跑一次推理：

- 用的是 `Comparisons/eval_comparison.py --save_hp_cache`，与已发表对比表是同一套代码、同一检查点；
- IMU 噪声与主流程一致（seed 42）。

每帧的 SMPL 局部旋转和根位移写入缓存（`utils/external_hp_cache.py`）。缓存覆盖三处数据：

- 训练窗口（`train_path`）；
- 训练时的验证窗口（`test_path`，用于选 best）；
- 在线评测（`full_sequence_test_path`）。

### 外部 HP 模块

`model/rnn/external_human_pose.py` 中的 `ExternalHumanPoseModule` 按我们 HP 的输出格式回放缓存：

- FK 用同一套零 betas SMPL-24 骨架，手腕、手掌、`p_pred`、全身旋转与我们的 HP 一一对应；
- 模块没有参数，始终冻结；
- VC、OT、联合阶段和在线评测的代码都不变。

接入方式：

- 配置里设 `external_hp_cache_dir`，`build_model_input_dict` 会把每个窗口对应的缓存帧附加为 `ext_hp_*`；
- HP 的“检查点”就是缓存的 `manifest.json`；
- 配置与检查点不一致时直接报错，不会悄悄退回 GT 或随机 HP。

### 根轨迹

把方法自己的逐帧根位移从窗口初值积分：训练窗口从 GT 起点开始，在线时从已发布轨迹开始。

- 在线时与方法自己的轨迹逐帧一致。冒烟测试中 6 个序列的阶段一根误差，与 TIP 自己的根误差相同，精确到小数点后 4 位。
- 不加我们 HP 的地面修正，也不做输出平滑；配置里的 `human_pose_smoothing_half_width` 只作用于我们自己的 HP。

### 延迟

- **DIP、TransPose** 的在线协议用 5 帧未来。在窗口末尾，最后 5 帧的姿态还没有产生，所以这 5 帧保持最后一帧可得姿态，根按最后速度外推。训练和在线用同一规则，保证整个系统仍只看 10 帧前瞻。
- **TIP、GlobalPose** 是因果的，不需要这样处理。

### OT refine 头

保留。它只修正 10 个推断关节和根，其余关节沿用外部方法的旋转。

### 与已发表配方的关系

VC/OT 配方与已发表的 IMUHOI++ 相同，只换了 HP：

- **OMOMO / BEHAVE**：用 `vc_v5_20260930` 的 v5ar VC 配方（300 / 986 epoch），以及 unified OT 配方。这与 VC 模块消融 `run_ablation.sh` 中 `full` 行的命令相同。
- **IMHD / PAHOI / HODome**：用 `remaining3_v5ar_vs_imuhoi3b_20261010/recipe.yaml`，由编排器一次训练 VC、OT 和联合阶段。

### 需要在结果表里注明的两点

- **DIP 没有平移估计**：与已发表对比表一致，用 GT 根位移（`build_hp_cache.py --dip_root gt`）。这一行的根是 oracle。如果想用 DIP 的固定首帧根，在 `run_plugin.sh` 的缓存命令里加 `--dip_root fixed`，并换一个缓存目录。
- **TIP 推理时历史 dropout 0.8 常开**（官方实现的行为），预测依赖随机数状态和序列处理顺序。缓存把每个序列的一次实现固定下来，之后训练和评测都用这同一份。服务器上的 TIP 缓存因此不会与已发表的 TIP 评测逐位相同，只在整体上一致。另外三个方法是确定性的：本地重建的 6 个 OMOMO 序列与已发表结果逐位相同。

## 本地已做的验证（2026-11-05）

- **单元测试** `test/test_external_human_pose.py`：
  - 用 OMOMO（SMPL-X）和 BEHAVE（SMPL-H）的 GT 姿态回放，关节、手掌都与语料 GT 一致，误差 < 1 mm；
  - 窗口切片和补齐；
  - 在线发布帧与窗口根一致；
  - 延迟保持；
  - refine 推导；
  - 配置/检查点不一致时报错。
- **冒烟测试**（24 个训练、6 个测试 OMOMO 序列）：
  - 四个方法的缓存都能生成；
  - VC（含 HP 参与的联合阶段）→ 编排器 OT + 联合 → 在线评测（含逐序列记录）全链路通过；
  - OMOMO/BEHAVE 路径和 recipe 路径各测一遍。

## 服务器上运行

1. `git pull`。其中包括噪声 key 修复（f1bb5b3），它决定服务器上的噪声与发表版一致，必须有。
2. 上传比较方法的检查点和已发表评测（本地执行，约 1.2 GB）：

   ```bash
   bash scripts/upload/upload_scp.sh llz@202.120.59.70 /disk2/mmzhou/IMUHOI scripts/upload/filelist_ckpt_comparisons.txt
   ```

   在服务器项目根目录校验：`sha256sum -c --quiet scripts/upload/comparisons_sha256.txt && echo SHA_OK`。
3. 跑单元测试，还要通过 OT 消融方案 1.4 的噪声 key 检查：

   ```bash
   python -m pytest -q test/test_external_human_pose.py test/test_sequence_noise_key.py
   ```

4. 启动。每张卡同一时间只跑一个任务，`GPUS` 填可用的卡：

   ```bash
   IMUHOI_PYTHON=$(which python) GPUS="0 1 2 3" nohup bash scripts/hp_plugin/queue.sh > hp_plugin_queue.out 2>&1 &
   ```

   队列会按 `scripts/hp_plugin/jobs.txt` 的顺序分配任务，最耗时的排在前面（BEHAVE 的 VC 要训 986 epoch，GlobalPose 缓存要跑 CPU 物理优化）。中断后再运行同一命令即可续跑：已完成的任务会跳过，其余任务从断点继续。单个任务也可以直接运行：

   ```bash
   bash scripts/hp_plugin/run_plugin.sh <dataset> <method> <gpu>
   ```

5. 查看进度：
   - 队列：`outputs/IMUHOI++/RNN_cano/hp_plugin_20261105/queue/queue.log`；
   - 各任务：`<ds>/<method>/job.log`，以及 `cache.log`、`vc.log`、`pipeline.launcher.log`、`pipeline/orchestrator.log`。
6. 汇总：`python scripts/hp_plugin/summarize_hp_plugin.py`，输出写到 `hp_plugin_20261105/summary.md` 和 `metrics.json`。

## 必须检查的项（任何一项不对，先停下来汇报）

- **基线复现**：每个缓存 `manifest.json` 里的 `baseline_reproduction`，即重建的全测试集评测与已发表评测的逐序列比较。
  - DIP / TransPose / GlobalPose：逐序列中位差应在浮点噪声量级（≤ 0.01 cm）。
  - TIP：只看整体差。
- **外部 HP 确实生效**：每个训练和评测日志开头都应有 `External HumanPose: <method> from ... (latency N frames ...)`。DIP 和 TransPose 是 5 帧，TIP 和 GlobalPose 是 0 帧。
- **噪声 key**：日志里应有 `IMU noise keys re-rooted to the reference checkout`，含义见 OT 消融方案 1.4。

## 资源（估计）

- **每个任务的内容**：
  - 缓存：在 train / 验证 / 测试三处数据上各跑一遍比较方法。本地 OMOMO 上，DIP、TransPose、GlobalPose 约 6–7 序列/秒，TIP 是自回归的，约 1 序列/秒。GlobalPose 物理优化默认用 12 个 CPU 进程，可用 `PHYSICS_WORKERS` 调整。
  - VC 训练、OT + 联合训练、两次在线评测。本地 4080 上 OT + 联合 + 评测单行约 2–3 h，VC 另算。
- **内存**：每个任务都预加载整套训练和验证数据，外加缓存（缓存约为数据体积的 1/4），BEHAVE 和 HODome 最大（训练集 5–6 GB）。按机器内存决定并行的卡数。

## 输出（`outputs/IMUHOI++/RNN_cano/hp_plugin_20261105/`）

- `cache/<ds>/<method>/`：
  - `manifest.json`；
  - `seq/`：逐序列缓存；
  - `eval/`：三处数据上比较方法自己的评测。全测试集那份就是表中的“X 单独”。
- `<ds>/<method>/`：
  - `job.log`；
  - `vc/`（仅 OMOMO / BEHAVE）；
  - `pipeline/v2_armrot_biclose/eval_<ds>_{best,final}.json` 及对应的 `_records.json`。
- `queue/`、`summary.md`、`metrics.json`。

## 口径说明

“X 单独”来自比较方法自己的评测，统计全部帧；“X + VC/OT”来自我们的在线评测，不统计最后 10 帧前瞻帧。已发表对比表也是这样比的。对漂移敏感的指标（根误差、物体误差）因此有约 1–3% 的口径差，不影响结论的方向。`summarize_hp_plugin.py` 里的配对 bootstrap 基于逐序列均值。
