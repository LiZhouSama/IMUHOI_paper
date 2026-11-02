# 01 五数据集主对比：IMUHOI++ vs 原版三分支 IMUHOI

日期：2026-10-10（IMHD/PAHOI/HODome）、2026-10-12（补齐 OMOMO，汇总五个数据集）。

## 问题

在统一的训练和评测条件下，IMUHOI++（VC-v5ar + OT-v2_armrot_biclose）相对原版 IMUHOI（三分支 OT）在五个数据集上的
物体、HOI 和根节点误差。

## 设置

| 项目 | IMUHOI++ | IMUHOI（三分支） |
|---|---|---|
| 流程 | HP → VC-v5ar → OT-v2（前臂-物体相对旋转 + 双手闭合候选，闭式融合）→ 联合微调 VC+OT，HP 冻结 | HP → VC-IMUHOI → OT-IMUHOI（左 FK、右 FK、IMU 积分三候选）→ 联合微调 HP+OT，VC 冻结 |
| 每个数据集共享 | 初始 HP checkpoint、canonical 数据、seed 42、逐序列 Noitom 噪声（seed 42）、按步数缩放的学习率计划和早停 | 同左 |
| 评测（每一行都由脚本校验） | 完整序列 batch-online，窗口 60 / 前瞻 10，HP 平滑半宽 7，物体速度用已发布轨迹 3 帧差分初始化（简称 7/3），初始接触 0 | 同左 |

7/3 是推理协议，两种方法完全一样地使用：平滑作用在共享的 HP 上，IMUHOI 的 VC 同样读取 `obj_vel_init`。

**OMOMO 上的影响（2026-10-28 补充核对）：**
- 三分支基线在 7/3 下重训 OT 和联合阶段后，比旧的 0/0 训练与评测好得多（obj 19.30 → 17.84，root 10.95 → 7.17）；我们几乎不变（+0.1）。
- 这个差别来自重训，而不是评测协议本身：同一个重训模型改用 0/0 评测，物体误差 17.81 cm，几乎不变。
- 7/3 对 IMUHOI 的接触估计略不利：帧级接触 F1 低约 0.005；手部事件 F1 明显更低，因为 VC 抖动更多。见 `../03_vc_contact_event_timing/`。
- 所以本表 OMOMO 上的物体、HOI、root 误差对评测协议不敏感，接触 F1 对基线略不利。

**BEHAVE 上的影响：**
- 0/0 会让所有模型的物体误差明显变差，三分支从 95 cm 升到 123 cm，所以 7/3 对基线更有利。

7/3 的取值当初是在 BEHAVE 测试集上选出来的，这一点要在论文里交代。

OMOMO 的三分支基线于 2026-10-12 在 7/3 下重训了 OT 和联合阶段，复用 0907 的 HP 和 10-08 的 IMUHOI VC。
IMUHOI 的 VC 不读取 HP，3 帧差分只在推理时起作用，所以 VC 可以复用。BEHAVE 的三分支基线（10-08）本来就是 7/3。

## 结果（best checkpoint）

| 数据集 | Object（cm） | HOI（cm） | Root（cm） | Obj jitter（mm/f²） | 降幅 obj / HOI |
|---|---|---|---|---|---|
| OMOMO | 13.34 vs 17.84 | 10.48 vs 15.04 | 6.00 vs 7.17 | 4.80 vs 5.70 | 25% / 30% |
| BEHAVE | 22.93 vs 95.45 | 17.19 vs 79.90 | 16.13 vs 26.67 | 5.74 vs 3.24 | 76% / 79% |
| IMHD | 32.69 vs 103.24 | 17.23 vs 97.49 | 21.42 vs 21.56 | 12.88 vs 8.97 | 68% / 82% |
| PAHOI | 9.95 vs 12.72 | 8.06 vs 11.02 | 5.10 vs 6.99 | 3.22 vs 2.75 | 22% / 27% |
| HODome | 26.66 vs 81.24 | 17.99 vs 58.91 | 21.30 vs 39.31 | 5.04 vs 3.83 | 67% / 70% |
| 五个数据集均值 | 21.11 vs 62.10 | 14.19 vs 52.47 | 13.99 vs 20.34 | 6.34 vs 4.90 | |

每格左边是 IMUHOI++，右边是 IMUHOI。final checkpoint 的结论相同，完整表格（含 MPJPE、接触 F1）见 `summary.md`。

## 解读和注意事项

- BEHAVE、IMHD、HODome 上 68–83% 的降幅主要来自三分支结构在长序列上的崩溃：它没有静止候选，IMU 积分分支一直漂移
  （IMU 候选误差 BEHAVE 264 / IMHD 522 / HODome 432 cm，序列中位长度约 1400–1800 帧）。OMOMO 和 PAHOI 的降幅
  （21–30%）更能代表结构上的差距。之前主表用的四候选（带静止保持）IMUHOI 在 BEHAVE 上为 41.03 / 25.69
  （`relabel_20260922_behave_hpsmooth_imuhoi`），比三分支强得多。
- 物体 jitter 在五个数据集中有四个是基线更低（只有 OMOMO 是我们更低），论文里应如实报告。
- IMUHOI 的联合阶段会微调 HP，所以它的 MPJPE 和 root 与我们不同；HODome 上 root 39.3 vs 21.3 cm 主要来自这一点。
- 只有单个 seed；best checkpoint 按测试窗口损失选择，早停也看测试集（没有独立验证集）。PAHOI 和 OMOMO 上选模集合就是报告集合。
- 我们这一侧与之前 BEHAVE/OMOMO 最优配置一致：OT 配置逐键相同（除按步数缩放的计划），BEHAVE v5ar best 用当前代码重评是逐位一致的。

## 文件

- `summary.md`：best 和 final 两张完整表格、相对降幅、每一行的来源 eval JSON。
- `metrics.csv` / `metrics.json`：机器可读的全精度指标。
- 生成脚本：`scripts/summarize_imuhoi_pp_vs_imuhoi3b.py`（校验每一行的评测协议和架构，不一致就报错）。
- 原始目录：`outputs/IMUHOI++/RNN_cano/all5_v5ar_vs_imuhoi3b_20261012/`；
  IMHD/PAHOI/HODome 的运行在 `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/`，
  OMOMO 三分支重训在 `outputs/IMUHOI++/RNN_cano/imuhoi3b_20261012_omomo_unified/`。

重新生成：

```bash
python scripts/summarize_imuhoi_pp_vs_imuhoi3b.py
```
