# Experiments

论文补充实验的整理目录。每个子文件夹包含：
- 一份 `README.md`：问题、设置、主要发现、局限，以及重新生成的命令；
- 生成出来的汇总表格和图的副本。

原始导出和 checkpoint 仍放在 `outputs/IMUHOI++/RNN_cano/` 下，具体路径见各子文件夹的 README。

| 文件夹 | 问题 | 日期 |
|---|---|---|
| [`01_all5_imuhoipp_vs_imuhoi3b`](01_all5_imuhoipp_vs_imuhoi3b/README.md) | 统一协议下，IMUHOI++ 与原版三分支 IMUHOI 在五个数据集上的对比 | 2026-10-12 |
| [`02_vc_interaction_mode_probe`](02_vc_interaction_mode_probe/README.md) | VC 内部是否编码交互方式（搬起 vs 推/拖）；补充：姿态先验能否提前判断接触开始、是否有“正在/即将交互”的姿态原型 | 2026-10-27/28 |
| [`03_vc_contact_event_timing`](03_vc_contact_event_timing/README.md) | 姿态先验是否让接触起止时刻更准（事件对齐的接触响应曲线、事件级 F1、时刻误差，含 7/3 与 0/0 协议敏感性） | 2026-10-28 |
| [`04_vc_body_part_knockout_attribution`](04_vc_body_part_knockout_attribution/README.md) | VC 是否真的在用姿态先验、用哪个身体部位和哪段时间（身体部位剔除 + 积分梯度） | 2026-10-28 |
| [`05_contact_error_sensitivity`](05_contact_error_sensitivity/README.md) | 接触的时刻偏差和抖动对物体/HOI 误差的影响，真实 VC 离完美接触还有多远（回答 CVPR 审稿人 7TmV） | 2026-10-28 |
| [`06_vc_module_ablation`](06_vc_module_ablation/README.md) | VC 模块消融，每行重训重测：HP 先验（整包）、条件接触概率、身体部位；五个数据集（另外三个的身体部位行进行中） | 2026-10-30 至 11-02 |

共同约定：
- 统一在线协议：窗口 60 / 前瞻 10，HP 平滑半宽 7，物体速度用 3 帧差分初始化（简称 7/3）。03 另外报告了 0/0 协议下的结果，因为 7/3 会影响 IMUHOI 的接触估计。
- 训练 seed 42，IMU 噪声 seed 42，只用测试集。
- 单个 seed、best checkpoint 按测试窗口损失选择，这是所有实验共同的局限。
