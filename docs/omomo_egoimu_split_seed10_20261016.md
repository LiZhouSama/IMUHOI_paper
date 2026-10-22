# OMOMO：恢复 EgoIMU 划分、seed=10、旧 HOI 口径

## 本轮设定

按用户要求恢复 EgoIMU 保存的实际 train/test 身份，从 HP 开始重新训练。
只读使用旧 split 清单、样本身份、保存的训练种子和 HOI 定义；不运行或导入 EgoIMU 代码，
不加载其 checkpoint，不采用其预处理张量。

| 项目 | 设置 |
|---|---|
| 训练种子 | **10**，取自 `EgoIMU/outputs/IMUHOI/omomo/training_config.yaml:109` |
| 划分种子 | **42**，取自旧 `split_info.json`；直接恢复清单，不再次随机划分 |
| 训练 / 测试 | **4705 / 1177**，按序列身份核实，恢复 unified 中4条原训练样本 |
| 数据内容 | 当前 canonical SMPL-X neutral / zero-betas、24关节、当前接触标签、当前缓存 IMU |
| 基线模型 | HP → VC `imuhoi` → OT `imuhoi` 三分支 → 联合 HP+OT，冻结VC |
| 初始化 | 所有初始模块从头训练，无上一轮 HP/VC/OT 权重转入 |
| 噪声 | 上一轮 Gaussian 协议：人体/物体全部9维 std=0.1；Noitom序列噪声关闭 |
| 内部验证 | 同上一轮加噪声；独立 offline / online eval 干净输入 |
| 训练预算 | 同上一轮冻结 recipe：HP/VC/OT 最多300 epoch，joint最多100；原有早停 |
| 批次与采样 | batch800，每序列4窗口，workers0；每epoch仍23优化步 |
| 在线设置 | window60 / lookahead10；HP smoothing7/2；物体速度初始化差分3帧 |
| 主评估 | 完整1177条测试序列的 offline，报告best和final |
| 附加评估 | 同组权重的online，报告best和final |

完整训练配置来自**上一轮冻结的 `outputs/IMUHOI++/paper_noise_omomo_20261016/recipe.yaml`**，
不从可能正在编辑的默认配置重新生成。改动仅有 OMOMO 路径、训练种子42→10、
已关闭序列噪声的种子42→10，以及只在评估器读取的 `eval_hoi_protocol: egoimu`。
模型结构、训练损失、标签和训练窗口均沿用上一轮。

这里同时改变划分和训练种子，不能解释成只改变 split 的因果消融。按用户要求优先对齐
历史 seed；若后续要独立估计 split 的影响，需再做相同 seed 的配对实验。
仍沿用当前骨架和标签，属于历史划分/种子/HOI口径下的当前重构实现，不是完整历史环境复现。
现有 test-window 选 checkpoint 和早停也继续保留。

## 数据恢复与校验

独立实验目录下的 `data/train`、`data/test` 是当前 unified 张量的独立文件副本，
不是旧数据、硬链接或软链接。每个文件执行以下校验：

1. 旧实际样本与当前样本的 `seq_name`、帧数相同。
2. 当前文件具有 neutral / zero-betas canonical 元数据。
3. 副本与当前 unified 文件 SHA-256 一致。
4. 清单完整覆盖5882条，train/test在文件ID和序列身份上均无交叉。

保留原数据和旧实验。恢复的4条为 `2329.pt`、`2331.pt`、`2332.pt`、`2333.pt`，
均按旧清单归入训练集，不改变其当前预处理内容。

## EgoIMU HOI 定义

以当前模型预测及当前预处理 GT 的**手腕20/21、根0**为几何锚点：

- 左手GT接触帧，统计物体相对左手腕的位移误差。
- 右手GT接触帧，统计物体相对右手腕的位移误差；双手接触帧贡献两条观测。
- 双手均无GT接触的帧，统计物体相对根的位移误差，贡献一条观测。
- 每序列将上述观测合并后取均值、换算成cm；再对序列均值等权平均。

在当前 `utils/hoi_metrics.py` 中独立实现数学定义；两个评估入口共用同一评估器。
主键 `hoi_err_fusion` 采用此定义，并以 `hoi_err_fusion_egoimu` 同值留档。
原接触帧/掌根口径额外保存在 `hoi_err_fusion_contact_palm`，避免与上一轮主指标混淆。
JSON 标明 `hoi_metric_protocol=egoimu`、`hoi_hand_anchor=wrists_20_21`、
`hoi_noncontact_reference=root_0`。未配置该选项的其它实验仍采用原默认口径。

## 队列与后续比较

实验根目录：`outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016/`。

```bash
# WSL，脚本先检查GPU，再使用固定SAGE Python。
bash scripts/run_omomo_egoimu_split.sh
bash scripts/run_omomo_egoimu_split.sh --summarize_only
```

`status.json`、`queue.log` 和各阶段日志记录实际进度。
`data/inventory.json` 保存每个样本身份、帧数和内容哈希；`recipe.yaml` 冻结训练配置；
`provenance.json` 冻结配置来源、参考文件与源码哈希。
同一目录可续接中断阶段；文件锁防止重复训练；配置或源码变化时拒绝混用。

延续前一轮安排，offline final 的 Obj/HOI/MPJPE/root 四项均不高于论文表2参考值的110%
才自动继续 IMUHOI++。这是预先固定的数值筛选门槛，不是论文规定的严格复现标准。
IMUHOI++ 使用该轮初始 HP、相同 split/seed/噪声/HOI口径，重训 v5ar VC、
v2-armrot-biclose OT 及联合 VC+OT；未通过则记录结果后停止。

## 启动前验证

35项相关测试通过，包含旧HOI的双手权重、无接触根项、共同平移不变性、offline/online
共同评估器、seed读取、身份校验、文件独立复制和训练配方的有限改动。

## 实际启动记录

后台队列已启动（首次PID2500730），HP已完成首个训练及验证周期。
全部5882条恢复样本通过身份、帧数和副本哈希验证；另外核对了上一轮实际使用的5878份
预处理文件，均与本轮对应内容逐字节一致。验证结果保存于实验根目录 `launch_verification.json`。

实际HP配置确认 seed=10、`no_trans=false`、四项std=0.1、序列噪声关闭、内部验证加噪声，
三个预训练模块路径均为空。训练窗口加载4701条（旧train中4条短于61帧），
仍为23步/epoch；完整独立eval使用1177条测试序列。最新阶段以 `status.json` 和日志为准。

## 2026-10-17 显式继续 IMUHOI++

基线已完成全部offline/online、best/final评估，offline final Obj16.01246、
HOI19.70566、MPJPE2.35213、root9.08615cm，未通过原数值门槛。
用户随后明确要求“分析一下差异出处，顺便跑一下同条件的IMUHOI++”。
据此通过 `scripts/continue_omomo_egoimu_proposed.sh` 启动++，无需更改原门槛。
启动核验冻结源码、5882份数据哈希、seed与HOI口径，复用本轮初始HP，
重训v5ar VC、v2-armrot-biclose OT及原定联合阶段，保留全部正式评估。

本次发现基线OT训练/推理锚点不一致、历史损失权重及优化配置差异；
分析及独立诊断见 `docs/egoimu_reproduction_difference_20261017.md`。
运行中不修改冻结模型和训练源码，不将诊断结果替换为正式基线结果。
