# 复现范围、模型身份与环境缺口

日期为 2026-11-26 UTC。本文件是审阅后的运行说明，不授权启动训练或访问保留数据。

## 可核验版本

已通过 GitHub 插件核验远端分支 `codex/hybrid-seed17-pipeline-20261125` 的提交 `35d42c5800498e55bae0c3858b22fb3e7a3c1328`。它包含先前核心报告与 seed17 复核交接。本次论文修订和 Notion 回执的最终版本另记 `SYNC_RECEIPT.json`，不能将本地尚未推送的内容说成已在该旧提交内。

## 数据和计算角色

- TRAIN512、DEV256 来自 `paper2/data_1103.py`，生成种子分别为 2026111301、2026111302。四状态与三干扰变体在基础场景内共享目标和噪声，统计以基础场景为单位。
- Hybrid seed17 固定 epoch256，不挑最佳 DEV 轮次。48 更新/轮，共 12,288 更新。有效批量32，微批量8，四次累积。
- Hybrid checkpoint SHA 为 `e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`，DEV SHA 为 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`。核心 JSON SHA 为 `877cfe7f6baf4fb3a59bff58da4a344e5437e43f000b3b3a9808a47acbda7df1`。
- Bootstrap 为256个基础场景的4000次重采样，边际指标种子170000+指标索引、配对差种子340000+配对指标索引（按归档顺序从零开始），均值百分位区间，三变体先在基础场景内平均。它不估计训练种子总体，也不消除DEV选型偏差。
- 当前 Hybrid 未使用独立确认集或历史 TEST。旧 CorrectionNet/U-Net 的 TEST 或校准结果属于各自冻结方案，不是 Hybrid 的独立证据。

## 表3身份矩阵

|展示名|结构与表示|训练目标与预算|来源与已知限制|
|---|---|---|---|
|ReCoVP-Base|观测驱动局部窗字典与正则系数解，无精修分支|原生 seed17/epoch256；不是同配置 Hybrid 消融|`recovp_research_stage_review_20261125_r2/RESEARCH_STAGE_SUMMARY.md`；当前包未封存全部 Base 配置闭包，不能宣称损失/更新预算完全相同|
|P0-NoDechirp-Tail|原 CorrectionNet 的恢复骨干和局部残差结构，取消固定去斜；RMS和修正输出保留|seed17/epoch256；波形+0.1脉压+目标损伤+0.05尾峰|`tools/unet_ablation/models.py:StructuralAblation` 与旧评分归档；112,756注册参数中59,714有效|
|UNet-Tail|普通实值I/Q U-Net，宽度18/42/74，无去斜或局部残差块；保留RMS|seed17/epoch256；与P0相同损失族|同文件 `PlainUNet`；59,714参数均有效。完整训练身份应以各自配置SHA为准|
|ReCoVP-Hybrid|K16字典、13/26/44特征网络、9/13/17/21复值精修，逐脉冲恢复|seed17/epoch256，12,288更新；最终损失+0.25阶段辅助项|`recovp_hybrid_seed17/CONFIG.json`、`PREFLIGHT.json` 及核心JSON的源哈希|
|RSDR-Net|原生可靠度引导复数恢复模型；不同预算参考|原生seed40312102/epoch32，不能把比较槽位17当原生种子|`RESEARCH_STAGE_SUMMARY.md`；3,017,877有效实参数，不是匹配训练控制|

表3是同DEV的描述性系统比较。尚缺各比较模型逐parent向量、统一Equal-PFA及完整配置/依赖闭包，不能把矩阵说明升级为公平性已经全部验收。

## 实际可公开的复核命令

在仓库根目录运行以下离线命令，仅读取现有文稿和轻量收据。

```powershell
python tools/audit_manuscript_20261126.py
```

已有模型评测入口为 `tools.vp_1103_recovp_hybrid_evaluate --batch-parents 2`。它必须在服务器B `D:/code/Xin/ISRJ_222`，由控制器设置合法物理GPU映射并校验设备UUID、源码、数据和checkpoint身份。这里提供定位而不重新派发。训练入口为 `tools/vp_1103_recovp_hybrid_train.py`，不绕过其主机与目录保护。完整生成、训练及环境安装步骤需结合原运行配置，不能从这一个评测入口推导为第三方完整复现成功。

## 环境与数值边界

已知设备为 NVIDIA GeForce RTX 5060 Ti，卷积主要float32/complex64，系数求解complex128。当前轻量包没有完整 Python/PyTorch/CUDA/驱动版本冻结清单，故将其列为未闭合项，而不把本机现环境冒充原训练环境。

复缩放结构性质依赖理想算术及一致峰选择。随机因子幅度[0.2,5.0]、相位[-π,π]，种子17256；同一基础场景变体内各状态/脉冲共用因子。归档误差使用未缩放原输出的范数（下限1e-24）作分母，而非缩放后的期望输出范数；三变体、四状态和八脉冲在每个parent内展平。DEV误差均值与P95超过既定1e-5严格界限，应保留失败。原型的数值输入条件为非零脉冲RMS在[1e-12,1e12]，但当前DEV没有逐输入RMS范围导出，不能宣称已覆盖全域。

CUDA event计时为5次预热、40次测量，覆盖两阶段模型及门控等中间计算，不包含完整接收链。跨模型同机计时和CPU计时尚缺。

## 公开限制

不上传训练集、检查点、权重、原始数组、缓存、凭证或完整日志。公开配置、源码和摘要可以辅助重新生成仿真、重新训练；没有原检查点和逐场景输出时，不能逐数重算已有表格。仓库代码可读、来源SHA一致、原始实验逐数复现，是三个不同层级。
