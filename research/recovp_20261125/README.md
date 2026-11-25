# ISRJ_2：ReCoVP 研究快照

UTC快照：2026-11-25T02:10:31.495371+00:00。这是开发/机制证据，不是论文创新或独立确认结论。

## 已核验的正式 DEV 结果

同一256个DEV parent；以下轻量模型为seed17，RSDR保留原冻结seed40312102/epoch32及不同训练预算。三类型在parent内平均，不能扩为768独立样本。

| 模型 | 帧虚警↓ | Pd↑ | JCR↑ | absCPL↓ | 损伤↓ |
|---|---:|---:|---:|---:|---:|
| ReCoVP-Base | 6.901% | 84.896% | 12.256 dB | 0.176 dB | 0.071620 |
| ReCoVP-NoEquiv | 6.641% | 83.854% | 12.097 dB | 0.202 dB | 0.069523 |
| P0-NoDechirp-Tail | 10.286% | 87.630% | 16.355 dB | 0.556 dB | 0.044794 |
| UNet-Tail | 10.807% | 86.458% | 15.755 dB | 0.685 dB | 0.050780 |
| RSDR-Net | 5.078% | 86.979% | 17.760 dB | 1.054 dB | 0.094703 |

ReCoVP-NoEquiv按预设虚警优先规则获选，但相对Base的虚警差仅-0.260 pp，配对95%区间[-1.172,+0.651] pp跨零，不是显著全面优胜。

相对第一篇RSDR：NoEquiv的Pd低3.125 pp（配对区间[-5.339,-0.781] pp）、JCR低5.663 dB；absCPL低0.852 dB，目标增益保持更好。相对P0：虚警低3.646 pp，但Pd也低3.776 pp、旧损伤更高。不能只用虚警或参数量宣称全面跑赢。

## 当前执行状态

详细完成种子和当前轮次见 [PROGRESS.json](PROGRESS.json)。本快照的正式DEV仅包含已评分种子，不把完成训练但未评分的seed29/41冒充三种子均值。579工况、2811方法工况计划已冻结，未执行部分明确待完成。

新增精确同参U-Net宽度(21,40,74)，实参数58943、CPU全参数梯度检查通过；正式训练与调度尚未开始，不能把代码实现写成已获得消融结果。既有UNet-Tail为59714参数，比ReCoVP多1.31%，其真实对照已保留。

## 阅读入口

- [历史路线审计与阴性结果](HISTORY.md)
- [正式训练复现协议](reports/recovp_formal_dev/REPRODUCE.md)
- [第一篇对齐扫描协议](reports/recovp_isrj1_suite/REPRODUCE.md)
- [正式DEV分层结果与配对区间](results/formal_dev/)
- [机制审计汇总](results/audits/)
- [CPU预检NO_GO原报告](reports/recovp_gate_summary/REPORT.md)：这是后续GPU人工放行之前的历史结果，不是当前训练状态。

## 复现与边界

源码基线5723605，工作树锚点fcf9245。将本目录tools/与所需evidence布局叠加到独立的基线工作树根目录后运行，不要在旧ISRJ目录写入。现有脚本中的历史SHA绑定保持原样；reports/为只读发布副本，原始本地evidence路径由快照收据追溯。

私有TRAIN/DEV数据、历史模型权重需在本地按原收据配置；本仓库不提供这些原始文件。所有主统计采用原生snr_db/jsr_db有效支撑功率定义，宽度扫描明确单列JSR_E。历史审计只导出全局/类型汇总，不把旧强度分档重新命名为JSR。

数据集、逐parent数组、权重、传输包、原始训练日志及任何CONFIRM/历史TEST数组均未上传。小JSON是明确标注的聚合压缩导出，完整原始本地JSON的SHA256在清单中保留。所有负结果及已完成seed均保留。SoftShare/CVaR尚未正式训练；CONFIRM继续锁定。
