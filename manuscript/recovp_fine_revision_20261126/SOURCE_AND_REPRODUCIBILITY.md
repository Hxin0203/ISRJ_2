## 附录 A 结果来源与复现说明

本文的实验数值来自作者工作区中已存在的训练与 DEV 记录。表3 使用 `reports/RECoVP_HYBRID_SEED17_EXPERIMENT_REPORT_20261126.md` 与 `evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json`；表4 和门控、延迟摘要同时核对 `evidence/outputs/recovp_hybrid_seed17/full_review/REPORT.md`。Base 消融及等虚警结果使用 `evidence/outputs/recovp_research_stage_review_20261125_r2/STATISTICAL_REVIEW.md`。GitHub 已归档版与本地后续复核版的范围分别记录于本次审阅文件。

方法公式对应 `tools/vp_1103_recovp_hybrid.py`、`tools/vp_1103_recovp_hybrid_objective.py`、`tools/unet_tail_loss/objective.py` 与 `paper2/network.py`；指标与检测对应 `paper2/evaluation_1103_v4.py`、`paper2/radar_eval.py`。数据配置对应 `paper2/data_1103.py`。checkpoint SHA-256 为 `e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`，DEV SHA-256 为 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`。正文与来源清单的 SHA 见伴随审查记录。

核心训练与 DEV 报告的已核验归档版本为提交 `35d42c5800498e55bae0c3858b22fb3e7a3c1328`，见 https://github.com/Hxin0203/ISRJ_2/tree/35d42c5800498e55bae0c3858b22fb3e7a3c1328 。本文新增修订与运行说明见伴随 `REPRODUCIBILITY.md` 及同步回执，后者记录本次最终远端版本。现有公开摘要不足以逐数重算检查点表格；按代码重新生成场景和训练则是另一种复现目标。训练数据、原始数组和检查点不随本文公开。


