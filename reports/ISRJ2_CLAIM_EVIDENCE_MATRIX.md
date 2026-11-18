# ISRJ 第二篇研究逐项复核

2026 年 11 月 18 日。远端基准为 `Hxin0203/ISRJ_2` 的 `codex/1103-experiment`，提交 `c29ea9c98f0c1f10ed23bb8f7b60248e90f14ca1`。本次核对远端文件清单、本地巡检原文与通用方法论文，并复算了一个统计反例。没有读取旧 Gate 数组。以下“未核实”仅限本次取得的证据。

## 十项判断

| 方向 | 证据与可写入论文的结论 |
|---|---|
| ① 复数相位等变性 | 已有 16 个场景的小样本诊断，足以把“相位不一致解释两模型差距”降为待证假设。样本量和对照范围不足以排除其他条件下的影响。[相位诊断](../evidence/outputs/phase_scale_diagnostic_20261117_v1/SUMMARY.json) |
| ② 固定去斜 | 已有历史消融。历史无去斜组微批为 32，新控制组为 8，因而仍缺完全配对的固定骨干四格。这个协议差异的实际影响未知。[协议核对](ISRJ2_PAIRED_EVIDENCE_CHECK.md) |
| ③ 尾部损失 | 两套骨干的原损失和尾部损失均已有种子 17、29、41 的 DEV 汇总。虚警改善的跨骨干差分均值为 −0.217 个百分点，依次留出一个种子后范围为 −0.391 至 +0.065。可报告两组各自的变化，跨骨干额外收益的方向对种子敏感。[四格汇总](../evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json) · [复算](ISRJ2_PAIRED_EVIDENCE_CHECK.md) |
| ④ U-Net 与新训练机制 | U-Net 加尾部损失已有 DEV 评价。提案中的四状态交互项另有项目内同型先例 D1-Pair，其审计记载正式种子数为 0/3。需区分所约束的输出量，不能用代数相似性宣布性能或首创。[既有审计](ISRJ2_PERFORMANCE_AUDIT_RSDR_20261117.md) |
| ⑤ 教师与蒸馏 | 同一条件下已有教师与基线的汇总比较。主网格上两个方向都存在仅一个模型报警的情形；教师与学生有分歧，仍不足以说明学生接受监督后会改善。未核实用户所列三种子蒸馏和独立等虚警评价闭环。[教师配对证据](ISRJ2_PAIRED_EVIDENCE_CHECK.md) |
| ⑥ 多脉冲 | 评价时使用多个脉冲与模型前向联合使用多个脉冲，是两个不同的实验条件。已审材料不足以支持联合输入的模型增益，也不足以把可能的捷径写成观察结果。[既有复核](ISRJ2_PAIRED_EVIDENCE_CHECK.md) |
| ⑦ 长程关系 | 提案给出了重复间隔，并对网络影响范围作了估计。有效感受野指输入各位置对输出的实际影响，仅凭下采样次数无法确定。即使某个统计量与错误相关，也不能由此确定模型是否已利用该信息，更不能解释两模型差距。[Luo 等的原论文](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html) |
| ⑧ 检测感知与场景级风险 | 重建误差与下游指标可能给出不同评价。提案中固定阈值的两个边际项与只依赖配对分数差的项也有区别。给两个分数同加一个常数，分数差不变，与固定阈值的相对位置却变了。单独报告分数差无法确定固定阈值的表现。[Balestriero 与 LeCun](https://proceedings.mlr.press/v235/balestriero24b.html) |
| ⑨ Hankel 与结构化分离 | 理想单指数序列的 Hankel 阵秩为一，是可直接证明的特例。这一特例不能推出一般混合观测具有唯一分解。近碰撞提示精度与稳定性问题，仍允许观测映射为单射。[已有反例与证明](ISRJ2_IDENTIFICATION_CLAIMS_REVIEW.md) |
| ⑩ 外部数据与独立确认 | 本表涉及新候选的结果最高为 DEV 筛查。旧模型的独立评价不能代替新候选的外部数据与独立确认；当前材料不足以给候选 A、B 或组合排序。[既有评价范围](ISRJ2_IDENTIFICATION_CLAIMS_REVIEW.md) |

## P0 的两个状态冲突

U-Net 尾部实验目录仍保留早期 STATUS 文件；后续跨骨干汇总列出全部三种子。归一化的后续巡检也记录三种子完成，DEV 门槛失败。[四格汇总](../evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json) · [归一化巡检](../evidence/outputs/ISRJ2_AUTOMATION_INSPECTION_20261117T2224Z.md)

本地巡检 `ISRJ2_AUTOMATION_INSPECTION_20261118T0018Z.md` 在 2026 年 11 月 18 日 00:18 UTC 记录频率处理三种子各完成 256 轮，DEV 未过门槛。核对的文件 SHA-256 为 `c963094516a60bea3c00244c1ae5eb0e20fe8f2329e8c2d50802e7b11e3b125f`。在本报告的远端基准提交中，相应实验目录仅列出 `QUEUE_BINDING.json`、`STATUS.json`。本项依据是本地巡检，未从该远端目录独立取得最终评价摘要。两项具体配置的负结果只约束这两项配置。

## 同一个“尾部”可以给出相反排名

以下是对提案中两种统计口径的算术核验。给 10 个场景等权，每个场景含 10 个误差，均以越小越好。甲在每个场景有一个误差 2、其余为 0。乙在一个场景的全部误差为 1、其余九个场景全部为 0。

| 聚合规则 | 甲 | 乙 | 误差较小 |
|---|---:|---:|---|
| 各场景取最大误差，再平均 | 2 | 0.1 | 乙 |
| 先求各场景平均误差，再取最差 10% 场景的均值 | 0.2 | 1 | 甲 |

这两行使用完全相同的数据。对这十个等权场景，第二行恰为场景平均损失的经验上尾 CVaR（置信水平 0.9）。本次用精确分数核验四个数值。该例说明为何论文需要写明尾部取在哪一层，以及“一个样本”指位置、帧还是独立场景。它没有估计任何 ISRJ 模型性能。[风险敏感微调论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/845fe27a3e5854999f972bbe729e5237-Abstract-Conference.html)

## 跨领域引文能支持到哪一步

| 引文 | 原研究支持的范围 |
|---|---|
| Luo 等，有效感受野 | 分析卷积网络中实际影响与理论覆盖的差别。原文没有测量本项目的网络。[论文](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html) |
| Duan 等，CANConv | 在遥感影像全色锐化任务上研究非局部自相似，属于非局部建模先例。跨领域结果无法确定本项目是否存在同一瓶颈。[作者原文](https://arxiv.org/abs/2404.07543) |
| Balestriero 与 LeCun | 研究重建学习得到的特征对下游感知任务的信息价值，提供区分优化指标与任务指标的理由。[论文](https://proceedings.mlr.press/v235/balestriero24b.html) |
| Xia 等，STk | 研究通用 Top-k 模块与损失。引用时应写明取值、排序索引还是子集选择；例如唯一最大值附近，最大值函数本身可微。[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e0ebde5c2152b03420d9258f7139a58-Abstract-Conference.html) |
| Kim 等，风险敏感微调 | 研究强调高损失样本的数据生成和微调。可作为场景级风险思想的相关工作，已有帧内局部尾部试验仍要按原定义报告。[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/845fe27a3e5854999f972bbe729e5237-Abstract-Conference.html) |
| Vasu 等，MobileOne | 作者报告特定模型、设备上的推理延迟。参数量不足以推断本项目 CPU、GPU 或端到端时延，后者仍需对应设备实测。[作者原文](https://machinelearning.apple.com/research/mobileone) |

## 尚未验证的核心问题

四项 P1 诊断在本轮均无新增 ISRJ 场景级结果。近碰撞的代数反例没有给出项目数据中的发生率，长程相关与失败共现也不能单独证明模型遗漏了信息。四状态等式只约束两个增量是否相等，即使两个增量同时为零也可满足；若要写“保持了目标”，需要来自独立保真指标的证据。候选 A、候选 B 和 A+B 本轮均无新增训练收据，因而无法判断哪一条路线兼具创新性与性能优势。[代数检验](ISRJ2_IDENTIFICATION_CLAIMS_REVIEW.md) · [原研究笔记的 18 篇文献分类](ISRJ2_LIGHTWEIGHT_LITERATURE_20261117.md)
