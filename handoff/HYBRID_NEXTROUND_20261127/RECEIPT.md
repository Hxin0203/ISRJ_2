# 本轮执行回执：ReCoVP-Hybrid 零训练工作包

日期：2026-11-27 UTC。任务书 SHA256：3d56839b53a3a66a752bc5a11b76d63ffe1021ecf9b5a77537f539110df269c7。本轮不含训练、不恢复暂停种子、不访问独立确认集。

## 结论

- WP11b：门对比度及 32–160 轮轨迹已有输出；末轮 alpha 比与 seed 67 数值见阶段报告。E0 与 SNR、噪声底、rho 的相关性仍待统一口径复核。
- WP12b：未通过审计。输出协议/脚本 SHA 与当前登记文件不同；pooled residual 有 18/310 个评测帧及 16/155 个校准帧违反 oracle 下界检查，逐脉冲比值出现极端值。现有分类图和比例不得作为正式结论。原结果保留。
- WP8c：seed 17 重跑摘要数值字段差异 0，临时块清理成功；独立包装器形式要求未满足，已注明。
- WP2r：固定条件功能结果已完成，不能解释为 held-out 性能结论。
- WP4：一轮限定搜索未找到 Base/P0 检查点。
- WP5b：指定注册网格目录缺失，依协议停止。
- WP6：已登记协议，现有 E1 汇总文件不足以产生所需完整波形、RD 和阶段差值图。
- WP7：已登记协议；完整冻结模型入口/身份未核实前未执行正式计时。

WP12b 协议 SHA256：2ddddefef75d5f4fa72cdfa98ba1a66dcf057e01c67f7fc40bd1448e9a443b91；当前脚本 SHA256：1744afce4fbef7fb79b65acec63ce5fdfd0edc99a0c6a3a4e1f75794e53dc5dc。WP6 协议 SHA256：df691978999d472cd4f06f7f136b211297540869ba6b51c6a049a465cb1cee23。WP7 协议 SHA256：38ad69ee74b8b427316bd769f1245812137226bfe49faec9831a0df44f6734b2。阶段报告 SHA256：674ed2af1f97d6615dcea51906e744e69d435f7559178275ba43f3737bcb78ac。

GPU 快照（索引、UUID、型号、利用率、显存）：

`
0, GPU-f12e73b1-3bed-5a1e-bcb0-f9726dd1621d, NVIDIA GeForce RTX 5060 Ti, 1 %, 904 MiB, 16311 MiB 1, GPU-bb60dd3a-738e-2398-908f-7f77ac05549b, NVIDIA GeForce RTX 5060 Ti, 0 %, 0 MiB, 16311 MiB
`

服务器详细结果与原始数据仍留在 Server B；本交接不复制训练数据、权重或完整日志。

