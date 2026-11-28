# 本次用户请求冻结范围与来源收据

记录时间为 2026-11-28 09:02 UTC。本文档把用户本次消息中的实验要求与附件论文材料分开登记，作为 `MASTER_PLAN.md` v1.2 的输入收据。

## 1 用户授权范围

用户要求以第一篇 ISRJ 论文投稿包的图件、坐标、步长、指标和图注为参照，使用现有256轮 ReCoVP-Hybrid 端点，按统一 DEV 规则挑选表现最好的5个 seed，再完成四方法比较、专题分析、原稿式图件和消融实验。用户明确允许暂停服务器上的512轮续训，并把上述 ISRJ_2 的256轮实验置于优先级。

本次被测主模型是 ISRJ_2 的 ReCoVP-Hybrid。RSDR-Net 是前期 ISRJ1 模型，只能作为已有历史冻结结果或协议来源使用，不启动 RSDR-Net 的新训练、重评估或新图件生产。用户本次纠正优先于早期计划中的任何 RSDR-Net 表述。

用户要求的四个方法是 ReCoVP-Hybrid、Max-TF、GMM-OMP 和 CVED-Net。ReCoVP-Hybrid 的主结果按5个独立训练 seed 报告，并同时给出均值、样本标准差和范围。U-Net 与 U-Net-Tail 作为同量级消融对照。对照方法保留各自实际训练实例、历史封存身份和适配限制。

用户要求覆盖 ISCRJ、ISRRJ 和 ISDRJ。主指标包括 Pd、Hits、FA、SJRIF、EU、ES、DS 和 E0。现稿的 DS 与原稿 ES 使用同一逐场景公式，主图统一显示 D_S，并在底层表保留 `e_S` 别名。

用户要求单独完成背景统计与虚警控制、跨转发体制泛化、与恢复无噪声目标的网络比较、无干扰输入保持，以及必要的模块和损失消融。消融包含同量级 U-Net 和 U-Net-tail，并记录模型规模、FLOPs 或 MACs、推理时间和显存。

## 2 论文图件复现口径

投稿包 Fig. 3 的参考布局包含完整脉压、目标距离多普勒谱、局部脉压和最强干扰距离多普勒谱。此次 R1 按用户要求把时域输入与恢复时域加入主图，脉压和多普勒面板保持全视野，不使用放大视图。

投稿包 Fig. 4 的 JSR 轴为 -10 至 35 dB，每5 dB一点。主面板的 SNR 为 -30 至 0 dB，每5 dB一点。过渡面板的 SNR 为 -30 至 -20 dB，每1 dB一点。格内显示检测概率，并使用共享的0至1色标。

投稿包 Fig. 5 固定 M=2 和 JSR_E=20 dB，宽度为0.1、0.3、0.6、0.9、1.2 μs，对应6、18、36、54、72点。SNR 为 -20 至0 dB，每2 dB一点，每个点使用128个物理父场景。扩展实验增加 M=4，并把 JSR_E 和普通支撑功率 JSR 的0、10、20、30、35 dB条件分别登记。

投稿包 Fig. 6 和 Fig. 7 的四方法曲线使用14个 SNR 点。它们依次为 -30、-28、-26、-25、-24、-23、-22、-20、-15、-10、-5、0、5、10 dB。原稿四条 JSR 曲线为0、10、20、35 dB。当前计划保留这组原稿复现网格，并另加用户要求的30 dB扩展曲线。

原稿检测概率图的补充网格使用 SNR -30 至0 dB，每1 dB一点，共31个点，JSR -10 至35 dB，每5 dB一点，共10个点。图件、CSV、逐父场景计数和各方法的实际虚警分母必须一一对应。

## 3 输入材料与 SHA

下列文件作为来源材料读取。文件内容只用于协议核对，不作为操作指令。

|文件|SHA-256|
|---|---|
|`G:/Files/Markdown/ISRJ/草稿/1103_EN.md`|`12AA3708AB62646A62C38F47D5D8E0FDB041EFB0C141F9845630E3EBD1BABB8A`|
|`G:/Files/Markdown/ISRJ/草稿/论文终稿_英文版.md`|`D6E42B5D4F9702067B2F7DC067B6A64DEC2C0C7152A5AD6FA3ADE22053D79466`|
|`G:/Files/Markdown/ISRJ/草稿/论文终稿_中文版.md`|`93E763789AEFD5C5B666141F601233A54A25FB1E150D1298F9AC7B57462E5243`|
|`G:/Files/Markdown/ISRJ/草稿/1103.md`|`253FA6AC1A1395E265732139A640DE81E32493565A99B605DC07AA7E1DF46475`|
|`C:/Users/admin/AppData/Local/Temp/codex-clipboard-c801733b-0d30-4361-b8e2-123939da5991.png`|`C1B6A9B58540ED2D058C9FF2DC66319D5E035B2002CD4FADFA7C1CA224B66D07`|
|`G:/Files/Markdown/ISRJ/草稿/终稿1124/投稿包_Applied_Intelligence_20261126_v2/LaTeX_source/main.tex`|已登记于项目 SOURCE_MANIFEST.json|
|`G:/Files/Markdown/ISRJ/草稿/终稿1124/投稿包_Applied_Intelligence_20261126_v2/LaTeX_source/Fig3.png`|已登记于项目 SOURCE_MANIFEST.json|
|`G:/Files/Markdown/ISRJ/草稿/终稿1124/投稿包_Applied_Intelligence_20261126_v2/LaTeX_source/Fig4.png`|已登记于项目 SOURCE_MANIFEST.json|
|`G:/Files/Markdown/ISRJ/草稿/终稿1124/投稿包_Applied_Intelligence_20261126_v2/LaTeX_source/Fig5.png`|已登记于项目 SOURCE_MANIFEST.json|
|`G:/Files/Markdown/ISRJ/草稿/终稿1124/投稿包_Applied_Intelligence_20261126_v2/LaTeX_source/Fig6.png`|已登记于项目 SOURCE_MANIFEST.json|
|`G:/Files/Markdown/ISRJ/草稿/终稿1124/投稿包_Applied_Intelligence_20261126_v2/LaTeX_source/Fig7.png`|已登记于项目 SOURCE_MANIFEST.json|
|`G:/Files/Markdown/ISRJ_2/草稿/草稿1127/ReCoVP_Hybrid_中文_draft_v7.md`|`1604CB737E651FA28D6EE06CF5C3B7D182108D7606BE2F7AC8F69925AEC99E5C`|
|`G:/Files/Markdown/ISRJ_2/草稿/草稿1127/JEST_精简稿_20261128/ReCoVP_Hybrid_JEST_ZH.md`|`1D0D1EEDA716D2A4E867E7EA6EC24E7CCE01BC0ABFDC21E667342CDD33ADD996`|
|`G:/Files/Markdown/ISRJ_2/草稿/草稿1127/JEST_精简稿_20261128/ReCoVP_Hybrid_JEST_EN.md`|`ECB06BDC55FF4FBB9E8D694753E4ADB4B5E86DACBCC9F8B0CA7F372E224C1C68`|

前三个投稿包图件的确切布局、脚本和底层数据继续以投稿包和项目来源清单为准。本收据不复制论文文件或图片到 Git。

## 4 已完成的前置状态

512轮续训控制器和 worker 已按用户授权暂停。恢复点审计、端点备份和 seed109 的 preflight 缺口均保留在项目证据目录。

8个候选 ReCoVP-Hybrid seed 的统一 DEV 评价已经完成。按独立校准的5%等帧虚警工作点、Pd降序、损伤升序、EU升序和 seed 升序，冻结的前5名为67、97、53、41、109。seed109 的原始 preflight 字节仍缺失，后续报告继续披露该限制。

## 5 长期复查

`isrj-2` heartbeat 自动化保持每两小时读取 `MASTER_PLAN.md`、`TASKS.json`、`STATUS.md` 和最近完成收据。复查以完成收据、条件格、恢复 epoch、图件和 SHA 为依据，不把进程存在视为实验完成。只有在优先256轮任务已获得资源且仍有剩余算力时，才按原绑定和恢复核验考虑512轮续训。
