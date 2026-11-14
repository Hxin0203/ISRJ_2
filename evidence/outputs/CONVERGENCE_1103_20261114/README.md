# 1103 训练收敛与 DEV 检查点轨迹

训练图使用三种 seed 的完整逐更新历史，并按 epoch 求均值后绘制训练目标及分量。DEV 图只使用 32、64、96、128 轮的四个冻结检查点；阴影是三个 seed 间的描述性标准差，不是统计置信区间。DEV 图不代表逐 epoch 验证。

数据文件：`TRAINING_LOSS_BY_EPOCH.csv`、`DEV_METRICS_BY_CHECKPOINT.csv`、`DEV_MEAN_AND_SD.json`。图文件同时提供 PNG 与 PDF。原始训练历史和权重仍留在 outputs/；未读取独立 TEST 或旧项目 held-out Gate 数组。

输入与代码 SHA-256 记录于 `MANIFEST.json`。
