# 两份研究笔记的完成度与最终 DEV 收据

本次从训练服务器只读核对五组 DEV 评价摘要和完成收据。每组的完成收据均指向对应评价摘要的实际 SHA-256，配置哈希一致，评价种子均为 17、29、41。五组记录相同的 DEV 数据哈希。核对前的 GitHub 提交为 `0707e31a255fd296f9dc06923e5f15760085e732`；其中 CorrectionNet 尾部组两份收据已与服务器逐字节一致，本次补档其余八份。所有结论均限于归档实验。

| 实验 | DEV 筛查 | 服务器原始收据 |
|---|---|---|
| U-Net 频率处理 | 未通过 | [评价摘要](../evidence/outputs/unet_frequency_screen_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](../evidence/outputs/unet_frequency_screen_20261117_v1/PIPELINE_STATUS.json) |
| U-Net 噪声归一化 | 未通过 | [评价摘要](../evidence/outputs/unet_noise_norm_screen_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](../evidence/outputs/unet_noise_norm_screen_20261117_v1/PIPELINE_STATUS.json) |
| U-Net 扩张卷积 | 未通过 | [评价摘要](../evidence/outputs/unet_dilation_screen_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](../evidence/outputs/unet_dilation_screen_20261117_v1/PIPELINE_STATUS.json) |
| U-Net 尾部损失 | 通过 | [评价摘要](../evidence/outputs/unet_mf_tail_screen_20261117_v2/DEV_SCREEN_SUMMARY.json) · [完成收据](../evidence/outputs/unet_mf_tail_screen_20261117_v2/PIPELINE_STATUS.json) |
| CorrectionNet 尾部损失 | 通过 | [评价摘要](../evidence/outputs/correction_mf_tail_20261117_v1/DEV_SCREEN_SUMMARY.json) · [完成收据](../evidence/outputs/correction_mf_tail_20261117_v1/PIPELINE_STATUS.json) |

扩张卷积的完成时间为 2026-11-18 08:57 UTC，晚于先前的等待记录。该配置在三个种子上均未改善受扰无目标帧虚警率。扩张卷积和提案中的非局部重复分支使用的信息不同，这次负结果不足以判定后者。

## 笔记其余事项

| 事项 | 已完成 | 尚缺 |
|---|---|---|
| 18 篇轻量网络文献 | [分类与逐篇判断](ISRJ2_LIGHTWEIGHT_LITERATURE_20261117.md) | 候选方法的新颖性仍需同任务近邻比较 |
| 相位与尺度 | 小样本变换诊断 | 普遍性与性能差距的因果解释 |
| 去斜与尾部 | 历史去斜消融、两种骨干各自的尾部对照 | 固定 CorrectionNet 的同协议去斜 × 损失四格 |
| 四状态干预 | 约束的代数核验、与较早 D1-Pair 的比对 | 候选 A 的正式训练与独立评价 |
| 可辨识性、长程相关、Hankel 谱、失败关联 | 近碰撞反例、已有小样本诊断及历史汇总的审阅 | 未取得提案指定四项 P1 场景审计的完整收据；候选 B 与 A+B 也缺直接结果 |
| 教师蒸馏 | 教师与基线的配对证据复核 | 正式 ISRJ 学生蒸馏、同预算对照和独立等虚警评价 |
| 多脉冲 | 已区分多脉冲评价与联合输入 | 三种跨脉冲条件下的联合输入对照 |
| 检测感知、场景级风险、深度展开 | 已区分局部尾部与场景尾部，整理相关数学主张 | 未核实新候选的正式模型结果 |
| 轻量化与外部确认 | 文献评价口径和已有 DEV 范围已归档 | 新候选在实际设备的端到端延迟、峰值内存、分布外和独立确认结果 |

[逐项证据与引用](ISRJ2_CLAIM_EVIDENCE_MATRIX.md) · [既有配对比较](ISRJ2_PAIRED_EVIDENCE_CHECK.md)。本次新增工作是找回、验证和归档最终收据，没有启动训练或访问旧 Gate 数组。
