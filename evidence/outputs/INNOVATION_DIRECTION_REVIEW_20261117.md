# ISRJ_2 创新方向复核：失效区间诊断、文献依据与对已有建议的判断

日期：2026-11-17。性质：研究决策备忘，不是新方法，也不是新的确认实验。本文没有启动训练，没有读取权重或原始数组；新增数值全部来自对已入库 TEST 汇总表的离线重读，脚本为 `tools/jnr_regime_diagnostic.py`，输出在 `evidence/outputs/JNR_REGIME_DIAGNOSTIC_20261117/`。

## 0. 结论

当前稿件把受扰背景虚警写成一个平均数（主网格 20.42%）。按干噪比重读后，这个平均数拆成三个机制不同的区间，其中两个区间里 CorrectionNet 的失败有明确的结构原因。这比“给 U-Net 再加一个模块”更适合作为第二篇的出发点：先说清普通小网络在哪里、为什么失败，再用一个最小机制去解决它。

建议的主线是“让校正器具备与后级检测器相当的相参处理增益，并用目标缺席状态下的脉压峰值而不是残差能量来训练”。另一份建议中的等变网络和教师蒸馏，在现有数据下都接不上这三个失效区间，建议降为辅助检查。多脉冲方向在当前生成器下属于仿真捷径，暂不启动。

能否超过等参数 U-Net 无法事先保证。可以保证的是比较方式：按区间、按折中前沿、按计算预算比较，使结论不依赖弱基线或工作点差异。

## 1. 新增证据：按干噪比重读冻结 TEST

目标缺席的受扰状态 \(J+E\) 中没有目标，输入只通过 \(\mathrm{JNR}=\mathrm{SNR}+\mathrm{JSR}\)（dB，沿用主生成器的有效支撑功率口径）依赖强度参数；不同 SNR 只改变整体幅度。CorrectionNet 带 RMS 归一化，对同一 JNR 的各条件给出完全相同的虚警（稠密网格内最大差值为 0），所以它的 \(J+E\) 行为是 JNR 的一元函数。

**表 1. 稠密网格、重复转发、每条件 512 个 parent 的 \(J+E\) 帧虚警率（%）。CorrectionNet 为 seed 29 第 128 轮，RSDR-Net 为 F-R7。RSDR-Net 一栏给出同 JNR 各条件的均值及随输入幅度变化的范围。**

| JNR/dB | 不处理 | CorrectionNet | RSDR-Net 均值 [最小, 最大] | \(S+E\) 参考 | CorrectionNet 残差/干扰能量，dB |
|---:|---:|---:|---|---:|---:|
| −10 | 2.3 | 2.7 | 2.3 [2.3, 2.3] | 1.0 | 0.0 |
| −5 | 11.9 | 12.3 | 10.2 [8.8, 11.7] | 1.0 | 0.0 |
| 0 | 40.2 | 37.7 | 14.3 [4.9, 31.6] | 1.0 | −0.5 |
| 5 | 62.7 | 48.8 | 8.4 [2.3, 25.2] | 1.0 | −3.7 |
| 10 | 70.1 | 45.9 | 5.6 [1.6, 16.4] | 1.0 | −8.3 |
| 15 | 70.3 | 37.1 | 3.5 [1.2, 7.2] | 1.0 | −13.2 |
| 20 | 70.5 | 24.0 | 2.9 [1.6, 5.7] | 1.0 | −18.8 |
| 25 | 70.9 | 24.4 | 3.3 [2.3, 5.1] | 1.0 | −23.8 |
| 30 | 70.9 | 37.1 | 5.2 [4.9, 5.5] | 1.0 | −27.9 |
| 35 | 70.1 | 52.9 | 10.7 [10.7, 10.7] | 1.0 | −30.5 |

三类干扰合并的主网格给出同样的形状，并延伸到更高干噪比：JNR 为 40 和 45 dB 时，CorrectionNet 的虚警为 45.8% 和 58.6%，残差能量相对干扰约 −32 至 −33 dB。

### 1.1 三个区间

**区间 A，干扰低于或接近噪声（JNR 约 ≤ 0 dB）。** 校正器基本不动作：残差能量与输入干扰能量相差不到 0.5 dB，虚警与不处理一致。但这些逐点不可见的干扰经过脉压后已经造成 12%–40% 的帧虚警。这个区间并不冷门：SNR 为 −20 dB、JSR 为 20 dB 时目标检测概率约 96%，而 \(J+E\) 虚警为 37.7%。

**区间 B，中等干噪比（约 5–25 dB）。** 残差停在噪声能量的 0.20–0.23 倍，不随干扰增强而下降；虚警 24%–49%。

**区间 C，高干噪比（约 ≥ 30 dB）。** 相对抵消深度饱和在 −30 dB 上下，绝对残差随干扰增长。稠密网格中虚警回升到 37%–53%，主网格 40–45 dB 处为 46%–59%。

### 1.2 两个附带发现

**残差能量不能预测虚警。** JNR 为 30 dB 时，RSDR-Net 的背景残差能量比（0.52）是 CorrectionNet（0.25）的两倍，虚警却是 5.2% 对 37.1%。决定虚警的是残差中能被匹配滤波器相参积累的部分，而不是残差总能量。现有端点损失和脉压损失都是能量型的，这一点与消融结果一致：去掉脉压损失后检测率只从 47.11% 变为 46.46%。

**RSDR-Net 的背景抑制依赖输入绝对幅度。** 同为 JNR 0 dB、形状完全相同的 \(J+E\) 输入，只因整体幅度不同，RSDR-Net 的虚警在 4.9% 到 31.6% 之间变化。它没有幅度归一化，而训练数据中目标幅度限制在 \([0.15,2]\)、噪声幅度由 SNR 决定，绝对幅度因此成为可用线索。这属于生成器先验，不是物理先验。本文的 No RMS 消融（检测率 49.57%，高于完整模型的 47.11%）与这一解释相容。统一比较表中“RSDR-Net 背景虚警 7.08%”应附带这一说明。

### 1.3 区间 A 不是信息不足

三条旁证表明弱干扰可以被处理：本地复现的 Max-TF 在主网格 JNR −5 到 0 dB 处虚警为 8.2%–17.8%，低于 CorrectionNet 的 13.8%–39.1%（它依靠时频相参分析，代价是很大的目标失真）；RSDR-Net 在某一输入幅度下达到 4.9%；带真实支撑的 BCS 始终约 0.9%。逐点干噪比为 0 dB 的干扰，在一个 6–72 点的切片内相参积累后有约 8–19 dB 的增益，多个切片合计更高。信息在，现有小网络取不到。

### 1.4 限制

每个模型只有一个冻结种子和检查点；全部为单目标仿真；TEST 已被分析过，这次重读只作诊断，不能当作新的确认；等参数 U-Net 和四项消融没有逐 JNR 的表，三个区间是否同样出现在 U-Net 上尚未检验；生成器对各受扰脉冲独立抽取干扰参数，这会影响距离–多普勒图上的虚警水平。

## 2. 机制假设（均待证伪）

**H1 处理增益失配。** 后级检测器拥有约 27.8 dB 的脉压增益，校正器的卷积路径感受野约 55 点，且没有频率选择能力。校正器看不见、检测器看得见的干扰就成为虚警。同时，平方误差下的最优解是条件均值 \(\mathbb E[Q\mid Y]\)（修订稿第 2.8 节），干扰不确定时它向零收缩；剩下的部分仍是相参的线性调频切片。区间 A 的“不动作”与这一收缩一致。

**H2 单次回归的相对精度下限。** 区间 C 中相对残差停在 −30 dB 左右，绝对残差因此随干扰增强而增长：JNR 为 35 dB 时已达噪声能量的 0.42 倍，并集中在受扰片段内。单次前向的相对精度不够，无法把强干扰的残差压到噪声以下。

**H3 去斜使信号频带随时间旋转。** 去斜参考相位为 \(\pi\mu t^2\)，覆盖整个 1800 点接收窗。原始坐标中目标与干扰始终位于 \(|f|\le B/2=15\) MHz，是采样带宽的一半，且不随时间变化。去斜后，时刻 \(t\) 的有效频带中心在 \(-\mu t\)（对 \(f_s=60\) MHz 取模），每 \(f_s/\mu=20\) μs 即 1200 点绕采样频带一周。平移等变的小卷积核能够利用前一种不变的频带结构，无法利用后一种随位置变化的结构；步长为 2 的下采样在全频带信号上还会产生混叠。这与 No dechirp 优于完整模型的方向一致，也是对另一份建议中对易关系 \(D_\mu T_\tau=e^{j\pi\mu\tau^2}M_{-\mu\tau}T_\tau D_\mu\) 的具体化。推论是：去斜只有与具备频率分辨力的算子（长核、短时傅里叶或 FFT）搭配才有益，与 3–5 点卷积搭配反而不利。可用 {原始, 去斜} × {短核, 长核或谱分支} 的交叉实验检验。

**H4 RMS 归一化隐去了噪声基准。** 归一化后网络无法知道残差相对噪声底的高低。可检验的替代是以带外频段（\(|f|>B/2\)）稳健估计的噪声标准差作归一化：仍然严格尺度等变，同时让干噪比成为可观测量，又不引入 RSDR-Net 那种对绝对幅度的依赖。

## 3. 其他工作怎样找到创新点

以下条目均已核对题名、出处和年份；除特别说明外，阅读层级为摘要与元数据。它们的原始性能不能换算成本项目的预期收益。

| 做法 | 代表工作 | 对本项目的含义 |
|---|---|---|
| 做减法，找出不必要的部件 | NAFNet（ECCV 2022） | U 型外形不是问题；删掉去斜和残差块后更好，本身就是可报告的结果 |
| 训练期复杂、推理期简单 | RepVGG（CVPR 2021）；ECBSR（ACM MM 2021）把固定的边缘算子分支在推理时并入单个卷积 | 物理先验可以只出现在训练期分支里，推理时不增加开销 |
| 用领域先验设计表示，缩小问题 | SincNet（SLT 2018）只学滤波器带边；DeepFilterNet（ICASSP 2022）；GTCRN（ICASSP 2024，23.7 K 参数、39.6 MMACs/s）；TIGER（ICLR 2025）按先验分带；Conv-TasNet（TASLP 2019） | 小模型的提升主要来自在合适的表示里工作，并在降采样后的帧率上计算 |
| 只学习模型不确定的部分 | KalmanNet（IEEE TSP 2022）；ALISTA（ICLR 2019）；NA-ALFISTA（IEEE TRS 2024，用单个 GRU 单元预测展开 FISTA 的参数，用于车载雷达干扰抑制）；DeepSIC（IEEE TWC） | 已知的相参处理交给固定算子，学习能力留给切片结构 |
| 低成本的全局算子 | Fast Fourier Convolution（NeurIPS 2020）；LaMa（WACV 2022） | FFT 是没有参数的全局相参积累器 |
| 约束输出形式 | Deep Filtering（IEEE SPL）；掩模波束形成（ICASSP 2016） | 输出对观测线性时，网络不易凭空生成成分 |
| 让训练目标对齐任务 | 可微 Neyman–Pearson 损失（IEEE TGRS 2023；IEEE TAES 2025）；Tilted ERM（arXiv 2007.01162）；感知–失真折中（CVPR 2018） | 平方误差最优不等于检测最优；虚警可以进入训练目标 |
| 一个模型覆盖整条折中前沿 | YOTO（ICLR 2020）；Pareto HyperNetworks（ICLR 2021） | 抑制与保真的折中可以做成可调旋钮 |
| 以实测开销而非参数量论轻量 | FasterNet（CVPR 2023）；MobileOne（CVPR 2023） | 见下 |

**关于“轻量”。** 当前模型参数为 59,714，但 Conv1d 运算量约 92–95 M MAC/脉冲，折合每个采样点约 5.1–5.3 万次；若按 16 kHz 采样率折算，GTCRN 约为每点 2.5 千次（跨领域，仅作量级参照）。收据中的单脉冲延迟为 2.2–3.4 ms（计时范围仍待核实），而脉冲重复间隔为 0.03 ms。参数少不等于轻量。运算量集中在全分辨率层；先用带步长的分析层降到帧率再处理，是语音领域小模型的通行做法，也恰好提供 H1 所需的相参积累。

## 4. 对另一份建议的判断

| 建议 | 判断 | 依据 |
|---|---|---|
| 相位等变或复数等变网络 | 降为低成本检查，不作主线 | 三个失效区间都不由相位处理引起；主网格 168 个条件上，目标增量的整体相位偏差平均 0.08°、最大 0.46°，整体复增益项只占目标增量误差的约五分之一，其余是形状误差；随机初相的训练数据已提供近似等变；若检查发现有影响，用增广或规范化（arXiv 2211.06489）即可精确获得，不需要新骨干；复值网络在 ISRJ 文献中已常见 |
| 去斜与时移的对易关系 | 保留并具体化为 H3 | 能解释已有的负结果，并给出可检验的交叉实验 |
| 以 RSDR-Net 为教师的选择性蒸馏 | 至多作辅助 | 教师在 JNR ≤ −5 dB 同样不动作；它在 0–15 dB 的优势随输入幅度变化，尺度等变的学生无法也不应模仿；它在高干噪比下残差能量更大，输出级模仿会教错对象；容量相差约 50 倍；教师未发表，审稿人无法核验。若做，应限定在目标缺席状态的脉压域、JNR ≥ 15 dB |
| 多脉冲联合结构 | 现在不做 | `paper2/scenes_1103.py` 中各受扰脉冲用独立随机流抽取干扰参数，受扰脉冲是随机子集；真实转发干扰脉间相参。跨脉冲收益在当前生成器下是仿真捷径。先为生成器增加脉间相参模式，这一步对虚警评价本身也有必要 |
| 背景残留尾部损失 | 升为主线的一半 | 表 1 直接支持“能量不等于虚警”。但只改损失不够：区间 A 中网络缺的是相参积累能力 |
| 骨干×训练四格对照、预先定义胜负、按 parent 重采样 | 同意，并补充三项 | 逐 JNR 曲线；扫描目标增量权重得到的折中前沿；等参数、等运算量、等延迟三种预算 |

另一份建议没有覆盖的问题：失效的区间结构；训练集只有 512 个 parent；运算量口径下的“轻量”并不成立；教师的幅度依赖；2026 年四篇直接近邻（CV-FISTA-Net、MCA-ADMM-Net、ISRJ-FNet、MoRPCA）的摘要均报告了实测数据。

## 5. 建议路线

**第 0 步，只做推理。** 用已有检查点在服务器上对完整模型、等参数 U-Net、No dechirp、No RMS、No target-increment 和 RSDR-Net 输出逐 JNR 的虚警、残差/干扰能量、校准阈值下的检测率和目标增量误差；RSDR-Net 另做输入幅度扫描；去斜模型按目标时延分组。要回答的问题是 U-Net 是否同样存在三个区间。

**第 1 步，在普通 U-Net 上做互相独立的单因素实验，各三个种子。**

1. 目标：在 \(J+E\) 状态加入脉压域或距离–多普勒域的峰值型损失。记 \(r=C_\theta(J+E)-J\)，对 \(|\mathcal H r|^2\)（以噪声脉压功率归一）在各单元上取软最大或倾斜平均，而不是求和。检验 H1 的目标部分。
2. 相参积累：在同等参数和运算量下加入一个全局谱分支（去斜坐标中 FFT → 逐点卷积 → IFFT，物理上即网络内的拉伸处理），或带步长的长核分析层。与原始或去斜坐标交叉，同时检验 H1 的结构部分和 H3。
3. 递推：权值共享的两次校正 \(T_\theta\circ T_\theta\)，每次重新归一化，不增加参数。检验 H2。
4. 噪声基准归一化替换 RMS。检验 H4。
5. 对照：在线生成训练场景，或把 parent 数从 512 扩到 2048 和 8192，区分“归纳偏置在小数据下有益”和“容量不足”。

预先写定通过条件，例如 JNR 在 −5 到 10 dB 的虚警下降幅度、目标增量误差的允许上升量，再看结果。

**第 2 步，只合并通过的因素**，并以原理命名方法，而不是罗列模块。在几个规模点上与 U-Net 比较折中前沿。

**第 3 步，新 parent 上的确认与外部有效性：** 脉间相参干扰、多目标与近邻目标、实测背景叠加仿真干扰（如实标注为半实物），并至少适配一个公开的学习基线。

论文主线由结果决定。若目标与结构都有效，主题是“处理增益匹配的紧凑校正”，贡献依次为失效区间分析、增益匹配的校正器与目标缺席峰值目标、校准虚警下的评价。若只有训练目标在 U-Net 上有效，就写成与骨干无关的训练方法，U-Net 作为最终骨干。若都无效，如实报告区间分析与折中，不强行命名新网络。

## 6. 怎样证明有效并且超过 U-Net

1. 把 U-Net 当作自己的骨干候选。贡献若是机制，“U-Net 加机制优于 U-Net”就是充分的结论。
2. 比较前沿而不是单点。No target-increment 消融把检测率从 47.11% 提到 54.65%，目标增量误差翻倍，说明单点比较可以靠牺牲保真取胜。对每个模型扫描目标增量权重，比较检测率对目标增量误差的整条曲线。
3. 逐 JNR 报告，不只报平均。新机制应当在它声称解决的区间里起作用。
4. 三种预算分别比较：等参数、等运算量、等实测延迟；给出多个规模点。
5. 等虚警比较：独立 \(J+E\) 校准集冻结阈值，新确认集上同时报告检测率和实测虚警及区间；纯噪声与受扰背景分开。
6. 基线获得同等调参机会、同样的数据与标签；训练种子与场景的不确定性分开报告；不删失败种子。
7. 数据角色前移：已分析过的 TEST 和确认集不再充当最终证明。

## 7. 新颖性边界

本轮检索是定向的，不是穷尽查新。以“干扰抑制网络加距离–多普勒损失或虚警”“面向检测的雷达去噪损失”为概念的检索没有找到直接近邻；可微 NP 损失用于检测器而非恢复前端；CVEDNet 使用脉压域的能量型损失，其虚警按“已检出目标中的假目标比例”统计，不是目标缺席帧的虚警（这一条读过原文相应小节）。未检到不等于不存在。

谱分支在距离像域工作，与 ISRJ-FNet 的 HRRP 域滤波、CR-TCN 的时频域复数网络、MSMD-Net 的多域多阶段结构相邻；这些工作的全文方法尚未逐式比对，`PAPER2_NEAREST_WORK_COMPARISON.md` 中“全文差异尚未核实”的状态不变。可主张的差异应限定为：同噪声参考与四状态配对下的目标缺席评价、低干噪比区间、以及紧凑实现。

## 8. 文献入口

直接近邻（摘要层级，来自 OpenAlex 记录）：

- Complex-Valued FISTA-Net，IEEE TAES 2026，DOI 10.1109/TAES.2025.3646182
- MCA-ADMM-Net，IEEE TNNLS 2026，DOI 10.1109/TNNLS.2026.3736439
- ISRJ-FNet，IEEE SPL 2026，DOI 10.1109/LSP.2026.3726844
- MoRPCA，IEEE TAES 2026，DOI 10.1109/TAES.2026.3721809
- Time-Domain Adversarial Network，IEEE Sensors Journal 2026，DOI 10.1109/JSEN.2026.3725273（摘要报告单样本推理约 8 ms）
- 快慢时间联合假目标剔除，IEEE TAES 2026，DOI 10.1109/TAES.2026.3742311（摘要称 ISRJ 产生相参假目标）
- CR-TCN，Aerospace Science and Technology 2026，DOI 10.1016/j.ast.2026.113776（仅题录，出版方页面未能读取）
- CVEDNet，Sensors 2025，DOI 10.3390/s25133925（读过摘要、引言和评价指标小节）
- MSMD-Net，Remote Sensing 2022，DOI 10.3390/rs14143445

跨领域（题名、作者、出处、年份已核对）：

- Simple Baselines for Image Restoration，ECCV 2022，DOI 10.1007/978-3-031-20071-7_2
- RepVGG，CVPR 2021，DOI 10.1109/CVPR46437.2021.01352
- Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices，ACM MM 2021，DOI 10.1145/3474085.3475291
- Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks，CVPR 2023，DOI 10.1109/CVPR52729.2023.01157
- Speaker Recognition from Raw Waveform with SincNet，SLT 2018，arXiv 1808.00158
- DeepFilterNet，ICASSP 2022，arXiv 2110.05588
- GTCRN，ICASSP 2024，DOI 10.1109/ICASSP48485.2024.10448310
- TIGER，ICLR 2025，arXiv 2410.01469
- Conv-TasNet，IEEE/ACM TASLP 2019，DOI 10.1109/TASLP.2019.2915167
- KalmanNet，IEEE TSP 2022，DOI 10.1109/TSP.2022.3158588
- ALISTA，ICLR 2019
- Neurally Augmented Deep Unfolding for Automotive Radar Interference Mitigation，IEEE TRS 2024，DOI 10.1109/TRS.2024.3442692
- DeepSIC，IEEE TWC，DOI 10.1109/TWC.2020.3032663
- Algorithm Unrolling，IEEE SPM 2021，DOI 10.1109/MSP.2020.3016905
- Fast Fourier Convolution，NeurIPS 2020
- Resolution-robust Large Mask Inpainting with Fourier Convolutions，WACV 2022，DOI 10.1109/WACV51458.2022.00323
- Deep Filtering，IEEE SPL，DOI 10.1109/LSP.2019.2955818
- Neural network based spectral mask estimation for acoustic beamforming，ICASSP 2016，DOI 10.1109/ICASSP.2016.7471664
- False-Alarm-Controllable Radar Target Detection by Differentiable Neyman Pearson Criterion for Neural Network，IEEE TGRS 2023，DOI 10.1109/TGRS.2023.3302472
- Neural-Network-Based False Alarm Controllable Detector for Weak Targets via a Differentiable NP Loss，IEEE TAES 2025，DOI 10.1109/TAES.2025.3572501
- Tilted Empirical Risk Minimization，arXiv 2007.01162
- The Perception-Distortion Tradeoff，CVPR 2018，DOI 10.1109/CVPR.2018.00652
- You Only Train Once: Loss-Conditional Training of Deep Networks，ICLR 2020
- Learning the Pareto Front with Hypernetworks，ICLR 2021，arXiv 2010.04104
- Improved Knowledge Distillation via Teacher Assistant，AAAI 2020，DOI 10.1609/aaai.v34i04.5963
- Dynamic Contrastive Knowledge Distillation for Efficient Image Restoration，AAAI 2025，DOI 10.1609/aaai.v39i10.33180
- Equivariance with Learned Canonicalization Functions，arXiv 2211.06489
- Making Convolutional Networks Shift-Invariant Again，arXiv 1904.11486

## 9. 可复核文件

- 脚本：`tools/jnr_regime_diagnostic.py`
- 输出：`evidence/outputs/JNR_REGIME_DIAGNOSTIC_20261117/FA_BY_JNR.csv`、`SUMMARY.json`（含两份输入表的 SHA-256）
- 输入：`evidence/outputs/TEST_CONDITION_TABLE_1103.csv`、`evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv`
- 生成器中逐脉冲独立抽取干扰参数的位置：`paper2/scenes_1103.py`
- 去斜参考的定义：`paper2/vendor/legacy_model.py` 中的 `FixedDechirp`

本备忘没有复制训练数据、权重、数组或完整日志。
