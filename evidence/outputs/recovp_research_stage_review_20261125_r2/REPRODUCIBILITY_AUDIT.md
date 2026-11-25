# 研究归档的文档一致性复核

本轮只读复核既有研究收据和归档文字，不运行模型、训练、推理、性能测试、场景扫描或新的统计重采样。日期以 UTC 记录；2026-11-25 UTC 对应本地 2026-11-24。

## 输入与版本

来源工作树为 `.cindy-worktrees/thoughtful-tesla`，其已记录提交为 `fcf9245c0bd9d2a85d3261138908b9852df2a750`。原归档目录为 `evidence/outputs/recovp_research_closure_20261125_063119/`。原归档及其全部历史实验收据保持不变。

来源归档记录了 224 个输入和 547 项检查，失败为 0。这些数字属于原归档。本轮单独执行 456 项文档/源码字节一致性及收据引用检查，失败为 0；两轮数量不合并，也不解释为实验复现次数。

本轮重新哈希 204 个文本、源码及暂停标记文件，并在检查末尾再次比对。另有 20 项数据、指标数组和权重文件仅保留原归档的身份记录，本轮未打开、解码或重新哈希。不能把这 20 项说成本轮直接验证的二进制资产。逐项范围见 `DOCUMENT_RECHECK.json`，历史完整输入清单见 `INPUT_MANIFEST.json`。

## 八份指定文件的引用关系

| 原始文件 | 本轮检查 |
|---|---|
| STATUS.json | 当前字节与原归档记录相同；研究对象与进度字段分开解释 |
| SUMMARY.json | SHA 与 STATUS、VERIFICATION 引用一致 |
| CORE_SUMMARY.json | SHA 与 STATUS 引用一致 |
| VERIFICATION.json | 当前字节身份未变；其中数值 PASS 仅作为历史收据 |
| QUEUE_CLOSED.json | 正式评分总表 SHA、已完成种子及 0 扫描记录一致 |
| QUEUE_OVERRIDE.json | 当前字节身份未变；保留研究主干覆盖与历史队列状态 |
| REPORT.md | SHA 与 STATUS、VERIFICATION 引用一致 |
| REPRODUCE.md | 当前字节身份未变；本轮没有执行其中的评测命令 |

原 STATUS 与 VERIFICATION 所记 DEV 和检查点身份一致。DEV 档案 SHA 为 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`；Base/17 检查点 SHA 为 `c67dab24e4f4f2d9b3d51fdda369000f964e0ffc830db8e1bcfdead8148f6748`。这里验证的是收据间一致性，未重新读取两个二进制文件。

七条单种子汇总行的模型身份与全局指标均逐项对应原 SCORE.json 的 `identity.method` 和 `metrics.global`。八条既有配对区间的跨零/触零判断与报告一致；未重算区间、重校准门限或重新筛选结果。

## STATUS.model 的含义

原 `STATUS.model = RSDR-Net` 保持原样。源码 `tools/vp_1103_recovp_dev_gap_audit.py:181–182` 在依次评测三个参照模型时更新此进度字段，最后一个为 RSDR-Net。因此它表示最后评测的参照模型，并不把研究主干改为 RSDR-Net。

新状态单独列出 `research_subject = ReCoVP-Base`、`research_subject_seed = 17` 和 `last_evaluated_control_model = RSDR-Net`。历史 WINNER 与后续人工路由分别保留。

## 统计与叙述边界

- 所有汇总仍为固定 256 DEV parent 的探索性结果，独立统计单位为 parent。
- 单种子与多种子均值、正式盲输出与 Oracle 诊断、原名义门限与同 DEV 等虚警分别记录。
- Base 与 P0 的两项既有等虚警配对区间均包含零，不支持显著优越、等效或非劣结论。
- RSDR 的比较槽位为 17，原生训练种子为 40312102、epoch 32，训练协议不同。
- 固定算子条件性质不推广为完整自适应模型保证；固定顺序的有符号能量分摊不写成独立因果贡献。
- 数据窗 1800 点与参考模板 600 点分开；四状态为 Y00=E、Y10=S+E、Y01=J+E、Y11=S+J+E。

暂停标记存在且未修改。本轮没有查询或控制服务器进程，因此不把标记和历史队列收据表述为服务器实时进程审计。

## 本轮复核的复现方式

仅使用 Python 标准库运行 `tools/review_recovp_archive_documents_20261125.py`，提供 `--source-root`、`--prior` 和一个尚不存在的 `--out` 目录。该脚本拒绝覆盖输出目录，不导入模型，不执行历史 REPRODUCE 中的命令。首次本轮校验把归档子对象与 SCORE 父对象比较，产生了 14 项模式误报；修正对象层级后，本目录的第二次校验为 456 项全部通过。第一次结果保留在同级无 `_r2` 的目录，不用于科学判断。

本轮报告正文保留既有归档文字，新增来源与复核范围说明。离线文字审计、备份指纹和输出清单分别记录在 `TEXT_AUDIT.json` 与 `STATUS.json`。哈希一致只支持身份与未修改判断，不证明算法正确、性能目标达成或独立泛化。
