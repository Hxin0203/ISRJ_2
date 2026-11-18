# ISRJ 第二篇阶段实验总报告

截至 2026 年 11 月 18 日。覆盖早期预算预研、11 月中旬基线比较，以及 11 月 17–18 日完成的五项开发集筛查。证据固定在实验分支提交 `b4de8c3ed62fcf064a8b9c5baecb88f737922941`。本次从既有评价 JSON 重算均值，未新增训练或访问旧 Gate 数组。

## 一、主要结论

最近五项实验全部完成三种子 DEV 评价。两套骨干加入尾部损失后，三个种子的受扰无目标帧虚警率都下降；两套骨干的平均目标增量损伤也都略有上升。归一化、频率处理、扩张卷积三个具体配置均未过预设门槛。普通 U-Net 在此次四组比较中仍是有竞争力的基线。新候选尚缺独立确认，教师和长程结构仍缺直接模型结果。[四组原始汇总](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json) · [最终收据核验记录](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_PROGRESS_RECEIPTS_20261118.md)

## 二、最近五项实验逐项复盘

每项均与对应原损失或原配置基线作同种子比较。种子为 17、29、41，DEV 数据摘要哈希相同。下表先在各个种子内求候选减基线，再对三个种子等权平均。虚警指受扰无目标帧虚警率；Pd 为检测率。两项变化用**百分点**，损伤沿用原指标单位，SJRIF 沿用 dB。正的损伤变化表示保真变差。

| 改动 | Δ虚警 ↓ | ΔPd ↑ | Δ损伤 ↓ | ΔSJRIF ↑ | DEV | 原始证据 |
|---|---:|---:|---:|---:|---|---|
| U-Net 尾部损失 | −0.477 | −0.174 | +0.000680 | +0.065 | 通过 | [评价](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_mf_tail_screen_20261117_v2/DEV_SCREEN_SUMMARY.json) · [完成收据](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_mf_tail_screen_20261117_v2/PIPELINE_STATUS.json) |
| CorrectionNet 尾部损失 | −0.694 | +0.260 | +0.001056 | +0.092 | 通过 | [评价](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/correction_mf_tail_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/correction_mf_tail_20261117_v1/PIPELINE_STATUS.json) |
| U-Net 噪声归一化 | +2.561 | −0.521 | +0.011419 | −0.261 | 未通过 | [评价](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_noise_norm_screen_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_noise_norm_screen_20261117_v1/PIPELINE_STATUS.json) |
| U-Net 频率处理 | +4.470 | −12.587 | +0.087674 | −1.449 | 未通过 | [评价](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_frequency_screen_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_frequency_screen_20261117_v1/PIPELINE_STATUS.json) |
| U-Net 扩张卷积 | +0.347 | +0.304 | +0.002354 | −0.028 | 未通过 | [评价](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_dilation_screen_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/unet_dilation_screen_20261117_v1/PIPELINE_STATUS.json) |

筛查规则同时考虑三项条件：种子级受扰帧虚警变化的中位数须改善，每个种子的 Pd 降幅不超过 2 个百分点，平均损伤增量不超过 0.005。因此“通过”允许部分指标轻微恶化。上述结果来自同一开发集，三个种子衡量训练波动；共同使用的 256 个 DEV 父场景不能按三个种子扩算成 768 个独立场景。[四组原始汇总](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json)

- **U-Net 尾部损失。** 三个种子的虚警均下降，平均 Pd 轻微回落。训练改动达到了事先规定的开发筛查目标；尚无该候选在新独立场景上的等虚警结论。
- **CorrectionNet 尾部损失。** 三个种子的虚警均下降，平均 Pd 微升，但三个种子的目标损伤也都增加。该配置改善了自身的虚警指标，仍有保真代价。
- **噪声归一化。** 三个种子的虚警均升高，平均损伤增量超过 0.005 的容许界限。
- **频率处理。** 三个种子的虚警、Pd、损伤、SJRIF 均向不利方向变化。此前巡检中的失败结论已由最终评价摘要和完成收据证实。
- **扩张卷积。** 此项在较早巡检中仍为等待状态，最终收据时间为 11 月 18 日 08:57 UTC。平均 Pd 微升，但三个种子的虚警均升高，筛查失败。这个改动没有直接测试提案中的长程重复证据分支。

## 三、四组绝对值：尾部损失追回了多少差距

同一批 256 个 DEV 父场景，取三个种子的等权均值。这里的百分数为绝对率。[四组原始汇总](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json)

| 骨干与损失 | 受扰无目标帧虚警率 ↓ | Pd ↑ | 目标增量损伤 ↓ | SJRIF ↑ |
|---|---:|---:|---:|---:|
| U-Net 原损失 | 11.111% | 86.806% | 0.048145 | 10.605 dB |
| U-Net 尾部损失 | 10.634% | 86.632% | 0.048825 | 10.670 dB |
| CorrectionNet 原损失 | 15.234% | 86.198% | 0.062494 | 10.312 dB |
| CorrectionNet 尾部损失 | 14.540% | 86.458% | 0.063550 | 10.405 dB |

按未取整值计算，原损失下两骨干虚警相差 4.123 个百分点，加尾部损失后仍相差 3.906 个百分点，缩小约 **5.3%**。逐种子比较两骨干的虚警变化之差，依次为 0.000、−0.781、+0.130 个百分点。种子 29 对平均差贡献最大，去掉它后差分变号。相比之下，加尾部后 Pd 的绝对差只剩约 0.174 个百分点。剩余差距主要体现在虚警与目标损伤。[逐种子和历史协议复核](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_PAIRED_EVIDENCE_CHECK.md)

## 四、此前几轮实验提供了什么背景

| 已做工作 | 实际观察 | 结论与证据 |
|---|---|---|
| 早期损伤预算预研 | 单目标候选在原开发集过线，冻结后两个种子在新同分布样本上的平均损伤为 0.05375、0.05351，均超出 0.05。另立的双目标研究有两个种子通过预设开发核验 | 保留冻结失败记录。双目标预研属于另一研究设置，其开发结果不能覆盖此次候选。[历史对照与计时汇总](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_RESEARCH_REVIEW_20261117.md) |
| 等参数骨干与效率 | CorrectionNet 与普通 U-Net 都有 59,714 个实际参与前向的参数。在 RTX 5060 Ti、FP32、batch=1 时，两者前向延迟中位数分别为 3.394、2.203 ms | 计时脚本版本未登记，原始收据排除 RMS/去斜，与现存脚本入口口径有冲突。保留暂定读数，完整前向速度排序待核验。[主稿计时说明](https://github.com/Hxin0203/ISRJ_2/blob/5723605ccbde2f439a5f1e985e5ca050bf6ce67b/manuscript/ISRJ_2_论文修订稿_20261116.md#L1135) · [历史对照与计时汇总](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_RESEARCH_REVIEW_20261117.md) |
| 历史独立校准比较 | CorrectionNet / U-Net 的 Pd 为 47.11% / 51.04%，实测受扰帧虚警为 0.857% / 0.819%，损伤为 0.06094 / 0.04626 | 这是旧模型的独立校准工作点。不能与上表约 86% 的 DEV Pd 直接相减，更不能给新候选沿用独立确认结论。[历史对照与计时汇总](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_RESEARCH_REVIEW_20261117.md) |
| 固定去斜消融 | 较早原损失 DEV 中，移除去斜后虚警约从 15.3% 降至 10.9%，Pd 从 86.0% 升至 87.3%。旧独立校准中损伤从 0.06094 降至 0.04353，实测虚警则从 0.857% 升至 0.895% | 数据支持去斜设置与表现有关，方向依赖指标和工作点。历史无去斜组微批 32，本轮为 8。[历史消融诊断](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/RESIDUAL_FAILURE_DIAGNOSTIC_REPORT_20261117.md) · [逐种子和历史协议复核](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_PAIRED_EVIDENCE_CHECK.md) |
| 相位与尺度诊断 | 已在 16 个场景上比较完整模型、无去斜模型和 U-Net 的变换响应，观察到多个模型存在不一致 | 该样本不足以把相位不一致归为 CorrectionNet 性能差距的主因。[相位与尺度诊断原始摘要](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/evidence/outputs/phase_scale_diagnostic_20261117_v1/SUMMARY.json) |
| 教师 RSDR-Net 与 CorrectionNet 的历史配对证据 | 已有主网格含 168 个条件，每条件 256 个父场景。用各条件公布的边际计数取界，仅 CorrectionNet 报警占全部评价帧的 13.423%–20.424%，仅教师报警占 0.074%–7.075% | 这是由计数约束得到的范围，并非置信区间，也没有定位具体帧。两模型确有分歧；这里没有接受蒸馏的学生。[逐种子和历史协议复核](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_PAIRED_EVIDENCE_CHECK.md) |

## 五、对照原计划逐项核销

| 原计划 | 当前完成度 | 哪部分仍缺直接证据 |
|---|---|---|
| P0 尾部三种子训练、统一 DEV 评价 | 两种骨干及各自原损失对照已归档 | 新候选的独立等虚警确认 |
| P0 噪声归一化、频率筛查，及后续扩张卷积 | 三项均完成，负结果已入档 | 本次各配置已结案，不能外推到所有输入表示或普通模块 |
| P0 固定骨干“去斜 × 损失” | 部分完成。保留去斜的两格有本轮结果，移除去斜有历史原损失组 | 同协议下方两格和交互效应；已完成的“骨干 × 损失”四格回答另一个问题 |
| P1-A 可辨识性碰撞 | 完成一般数学反例 | 项目场景中的碰撞率；数学上接近的观测可对应不同真值，同时映射仍可为单射 |
| P1-B 长程相似性 | 有文献与结构假设 | 未取得项目级相似性审计收据；扩张卷积负结果不能替代 |
| P1-C Hankel 奇异谱 | 有理想单指数的秩一论证 | 未取得实际场景谱分布及其稳定性的评价摘要 |
| P1-D 失败关联 | 有已有配置的失败记录和部分诊断 | 提案所列综合场景关联审计尚缺完整收据；相关性本身也无法确定原因 |
| 候选 A：四状态交互一致性 | 公式已核验，找到既有 D1-Pair 同型先例；历史工程审计记录该候选正式训练为 0/3 | 新候选的正式训练、保真及独立结果。两种背景下的增量可同时为零，满足交互等式仍可完全丢失目标增量 |
| 候选 B：长程重复分支；A+B 组合 | 至本报告证据版本，均未见正式模型结果 | 单项和组合的直接对照 |
| 选择性蒸馏 | 完成教师与基线的历史比较 | 正式 ISRJ 学生、同预算无教师对照、普通蒸馏对照、独立评价 |
| 检测感知、场景级 CVaR、结构化分离与深度展开 | 完成部分数学和文献比对 | 各自的模型结果；本轮局部尾部约束没有测量最差场景均值 |
| 多脉冲与仿真捷径 | 已区分跨脉冲评价与联合输入 | 三类跨脉冲条件下的联合模型比较、外部数据验证 |
| 文献、新颖性、部署与泛化 | 已完成 18 篇分类、已有模型的前向计时，以及部分近邻比对 | 新候选的完整近邻审查、CPU/GPU 端到端延迟、峰值内存、外部数据和独立确认 |

计划核对依据：[18 篇文献分类表](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_LIGHTWEIGHT_LITERATURE_20261117.md) · [代数检验与可辨识性反例](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_IDENTIFICATION_CLAIMS_REVIEW.md) · [十项主张与文献复核](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_CLAIM_EVIDENCE_MATRIX.md) · [项目内 D1-Pair 审计](https://github.com/Hxin0203/ISRJ_2/blob/b4de8c3ed62fcf064a8b9c5baecb88f737922941/reports/ISRJ2_PERFORMANCE_AUDIT_RSDR_20261117.md)。

## 六、本阶段可以写入论文的结论

在已测试的五项改动中，局部尾部损失在两套骨干上都带来了可复核的 DEV 虚警下降，并伴有轻微平均损伤上升。其收益只追回了原有跨骨干虚警差距的一小部分。三项未过门槛的具体配置给出了同样重要的负结果，其中扩张卷积还说明平均 Pd 微升可以与所有种子虚警恶化同时出现。

教师方向目前回答了“两个已有模型是否会犯不同的错”，尚未回答“学生是否学到了教师的优势”。四状态和近碰撞的工作澄清了假设的数学含义，均未产生新的雷达模型性能结果。依据这批证据，目前可以报告完整的开发筛查及其取舍，还无法把上述任何新候选认定为已获独立验证的最佳模型。
