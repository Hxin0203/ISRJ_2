# 8-seed、256 轮收敛结果

训练图按轮展示 B4 总损失及分量；实线为从原始 128 轮检查点续训的 seed 17/29/41，虚线为五个从头训练的新 seed。灰带是八个训练 seed 间的描述性均值 ± 样本标准差。

DEV 图只在 32、64、96、128、160、192、224、256 轮检查点上评估。阴影同样是 seed 间标准差，不是置信区间；逐 seed 的 parent bootstrap 区间保存在 CSV。DEV 估计以 256 个物理 parent 为单位，不把脉冲或 jammer 变体当作独立样本。

文件包括两张 PNG/PDF 图、逐 seed CSV、跨 seed 汇总 JSON 和 SHA-256 manifest。训练权重、训练数组与每 parent 的完整 DEV 数组留在本地 outputs/。未读取 TEST 或旧项目 held-out Gate 数组。
