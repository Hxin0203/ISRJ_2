# ReCoVP-Hybrid 结构图的波形与字典素材

三张主图取自同一仿真物理样本（父样本 ID 为 `2026112901000000`，脉冲索引 0）。蓝色波形展示已知目标分量，棕色波形展示接收观测，热力图展示模型读取该观测时生成的字典。可在 Visio 中直接插入 SVG，保留纵横比。每张图均有同名 PNG。

| 建议用途 | 素材 | 显示量 |
|---|---|---|
| 标注目标参考回波 | [纯净 LFM](01_lfm_clean_detail.svg) | 无噪声目标的同相分量，中心 192 个采样点 |
| 放入图 (a) 的 Observed I/Q | [ISRRJ 受扰观测](02_isrrj_observation_detail.svg) | 目标、干扰与噪声之和的同相分量，标称 SNR +10 dB、JSR +30 dB |
| 放入图 (a) 的 Local dictionary | [局部字典幅值](03_dictionary_magnitude_detail_db.svg) | 与受扰波形相同的 512 点，16 行依次对应原子 1 至 16 |

纯净图截取采样区间 [1294, 1486)，受扰图和主字典图均截取 [1282, 1794)。索引从 0 起算。两条波形各自按窗口峰值调整显示范围，保留原始采样；请用它们比较形态。另有[无干扰、SNR +10 dB 的观测](01_lfm_unjammed_10dB_detail.svg)、[全时窗纯净 LFM](01_lfm_clean_full.svg) 和[全时窗受扰观测](02_isrrj_observation_full.svg)。纯净目标分量与含噪无干扰观测应分别标注。

热力图从 seed 17、第 256 轮冻结模型的第一阶段直接提取复矩阵 B，维度为 1800 个快时间采样 × 16 个原子。主图使用全矩阵统一参考峰值的幅度分贝值，20 log₁₀(|B| / max|B|)，低于 −48 dB 的值显示为深蓝色，0 dB 显示为黄色。保留模型给出的原子次序。另附[全矩阵线性色标版](03_dictionary_magnitude.svg)和[线性色标局部图](03_dictionary_magnitude_detail.svg)，可核对原始幅值对比。全矩阵峰值约 0.473595。

## 数据与复现

已登记的复数组件位于 `data/hybrid_condition_master_20261127/components/test_0_0.npz`。图示选取首个父样本的 repeat 类别和首个受扰脉冲。SNR、JSR 沿用仿真器的有效区间功率口径。哈希、输入索引、检查点身份与字典数值核验见 [dictionary_provenance.json](dictionary_provenance.json)。

在工作区根目录依次运行：

```powershell
python manuscript/jest_signal_insets/plot_waveforms.py
python manuscript/jest_signal_insets/plot_dictionary.py
python manuscript/jest_signal_insets/make_preview.py 03_dictionary_magnitude_detail_db.png
```

本地 `outputs/jest_signal_insets/observations.npz` 保存绘图所用的完整复波形；`outputs/jest_signal_insets/dictionary.npz` 的 `B` 保存完整复字典。图示对应一个仿真脉冲，模型性能由稿件中的独立统计结果描述。
