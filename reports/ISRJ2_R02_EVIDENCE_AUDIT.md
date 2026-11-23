# R02 来源与引用审计（R4-A）

2026-11-23 UTC；B-R02-LITERATURE。状态：PARTIAL。已完成可核验的引用登记与本地证据可达性检查；未完成教师来源闭环，不登记整个任务COMPLETED。

## 实际读取的本地来源

ISRJ2_RESEARCH_QUEUE.json、ISRJ2_RESEARCH_VALIDATION_MATRIX.md、ISRJ2_R3_EXECUTION_SUMMARY.md、ISRJ2_IDEA_BACKLOG.md、IDEA_REVIEW_05.md；原始《gptpro创新性研究.md》的蒸馏与参考文献段落。R3追加检索与原始引用分开：Hinton/Cho/Qian来自R3检索，DCKD/VRM是原始文件明确引用。

## 教师来源与可比性

|记录|可以核实|不能据此认定|
|---|---|---|
|IDEA_REVIEW_05.md|报告称历史教师与基线有报警分歧，并声明无正式学生比较|不能确认教师检查点、训练数据隔离、共同场景或全面更强|
|卡片链接ISRJ2_PAIRED_EVIDENCE_CHECK.md|当前工作区该文件不存在；读取失败|不能将卡片转述升级为已读取原始配对收据|
|R3审计摘要|冻结25组完成与教师蒸馏假设分开记录|不能把这些组数当成蒸馏学生试验完成|
|原始资料的历史比较|资料本身警示不同测试集排名不能直接比较|不能用不同TEST网格数字建立配对优势或显著性|

教师准确版本、训练/评价数据角色及重叠检查在本次可读来源中尚未闭合。没有读取权重、数组或旧Gate；“未核实”不等于已经发现泄漏。仅有边际计数不能复原逐场景联合事件；范围界也不是置信区间。

## 文献登记

|文献|已核实题录/来源|本轮阅读级别|限制|
|---|---|---|---|
|Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network|2015 arXiv:1503.02531，DOI 10.48550/arXiv.1503.02531；页面另注明NIPS 2014 workshop|官方元数据及9页PDF可读|不能混写预印本年份与workshop年份；未验证本项目迁移收益|
|Cho, Hariharan, On the Efficacy of Knowledge Distillation|ICCV 2019；CVF页记4794–4802|官方搜索索引摘要；直接页403|DOI候选10.1109/ICCV.2019.00489未完成出版方交叉核验；页码差异待解，不擅自合并|
|Zhou等，Dynamic Contrastive Knowledge Distillation for Efficient Image Restoration|原始资料登记AAAI2025，39(10):10861–10869，10.1609/aaai.v39i10.33180|本轮出版方访问被拒|仅保留待核验引用，不能继承原文件“已读PDF”为本轮全文阅读|
|Zhang, Xie, Cai, Ma, VRM: Knowledge Distillation via Virtual Relation Matching|CVF ICCV2025，2707–2717；IEEE DOI 10.1109/ICCV51701.2025.00260|官方题录、摘要及PDF首段可读取|不是逐式全文核验；2026 Xplore收录日期不改变2025会议年份|

来源：[Hinton元数据](https://arxiv.org/abs/1503.02531)、[PDF](https://arxiv.org/pdf/1503.02531)、[Cho官方页](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html)、[DCKD出版方](https://ojs.aaai.org/index.php/AAAI/article/view/33180)、[VRM官方页](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VRM_Knowledge_Distillation_via_Virtual_Relation_Matching_ICCV_2025_paper.html)、[VRM DOI](https://doi.org/10.1109/ICCV51701.2025.00260)。

已核实的文献存在性只支持一般学术背景。没有证据支持“采用蒸馏就必然使本项目学生受益”或“知识选择概念首次提出”。本报告不推荐模型、损失或专项验证路线。

## 验收与缺口

完成：本地引用缺件核查、四条引用分级、年份/页码/阅读级别风险登记。缺口：教师版本与原配对收据、DCKD出版方全文、Cho页码及DOI主源交叉核对、VRM全文引用逐段匹配。因此保留PARTIAL，队列以BLOCKED记录缺失证据，允许选择下一独立任务。

