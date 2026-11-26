# 审阅输入与边界

审阅对象为同目录 `REVIEW_INPUT_FROZEN.md`，本轮结束前不修改冻结副本。中文算法类雷达信号处理研究论文草稿，未指定期刊，不按 Nature 录用标准作门槛。

三份审阅分别检查数学与源码、实验证据、正式论文结构与语言。每份只读取本文和以下原始资料，不读取其他审阅意见。主要关切需给稳定 ID、Major/Minor、Blocking Yes/No、正文位置、证据文件位置及可执行的修正验收条件。区分已证明、有限样本支持和未完成。无真实审稿人身份，不作编辑录用决定。

共同来源

- `tools/vp_1103_recovp_hybrid.py`
- `tools/vp_1103_recovp_hybrid_objective.py`
- `tools/unet_tail_loss/objective.py`
- `paper2/network.py`, `paper2/data_1103.py`, `paper2/evaluation_1103_v4.py`, `paper2/radar_eval.py`
- `reports/RECoVP_HYBRID_SEED17_EXPERIMENT_REPORT_20261126.md`
- `evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json`
- `evidence/outputs/recovp_hybrid_seed17/full_review/REPORT.md`
- `evidence/outputs/recovp_research_stage_review_20261125_r2/STATISTICAL_REVIEW.md`
- `doi_registry.json`
- 作者提供论文的本地抽取文本位于 `G:/Files/Markdown/ISRJ_2/草稿/草稿1125/审阅与修订_20261126/`，文件为 `MREDNet_local_reading.txt`, `GMM_local_reading.txt`, `Meng_local_reading.txt`。仅在本地使用，不复制全文至公开仓库。

只读审阅，不启动训练或推理，不访问 CONFIRM/TEST，不改源码、原始输入或其他报告。保存本人报告到本目录指定文件，完成后冻结。主要评价原创性、科学问题的重要性、技术可靠性与可读性，对跨学科广泛性按专业期刊边界说明。摘要不增加详细结果数值。
