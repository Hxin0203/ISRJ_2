# ReCoVP-Hybrid：面向间歇采样转发干扰恢复的轻量结构化变量投影与复值等变精修网络

> **稿件状态**：研究草稿（2026-11-25）。本文档用于记录模型定义、数学推导、训练协议和当前证据边界。ReCoVP-Hybrid seed 17 尚未完成 256 轮训练和固定 DEV 评测，文中所有最终性能位置均明确标为待补，不将预检结果写成最终检测结论。

## 摘要

间歇采样转发干扰（interrupted-sampling repeater jamming，ISRJ）在相参脉冲雷达中会产生与目标回波相似的时延、相位和多普勒结构。一个恢复器既要抑制干扰，又要保留目标响应和后续脉压、慢时间处理所需的相参关系。本文提出 ReCoVP-Hybrid，一种将观察值驱动的结构化变量投影与轻量复值等变 U-Net 精修器串联的恢复模型。第一阶段从归一化原始 I/Q 中提取幅度、匹配滤波响应和滞后乘积等实值不变特征，以固定脉冲模板生成 16 个观察值驱动的候选锚点，预测有限时延修正和学习窗，并通过带自适应对角精度的复数变量投影得到显式校正量。第二阶段只处理第一阶段残差及其校正量，采用无偏置复卷积、径向尺度不变门控、残差连接、跳跃连接和空洞瓶颈生成精修增量。两阶段在输入幅度缩放和整体复相位旋转下保持数值近似等变，从而将可解释的结构投影与小容量残差学习结合起来。

预检显示，模型具有 57,525 个有效实参数，其中第一阶段 28,948 个、第二阶段 28,577 个，低于 59,700 的参数上限；受测缩放样本的最大相对等变误差为 \(2.54\times10^{-7}\)，锚点列置换误差为 0，100/100 个可训练参数张量均获得有限非零梯度。该证据仅说明当前实现通过了有限样本预检。seed 17 正在服务器 B 的 physical GPU 1 上训练，最终 DEV 检测概率、虚警率、JCR 和推理效率尚未产生。本文同时给出四状态配对监督、脉压和匹配滤波尾峰代理损失的数学定义，并说明增量损失、尾峰代理与严格 CFAR 保持之间的边界。

**关键词**：间歇采样转发干扰；波形恢复；变量投影；复值神经网络；等变性；轻量模型；脉压尾峰

## Abstract

Interrupted-sampling repeater jamming (ISRJ) creates delayed, phase-coherent and Doppler-like structures that can obscure a target in a pulse-Doppler radar. A practical restorer must suppress the jammer while preserving the target response and the coherence required by subsequent pulse compression and slow-time processing. We propose ReCoVP-Hybrid, a compact two-stage restorer that combines observation-driven structured variable projection with an equivariant complex-valued U-Net refiner. Stage 1 extracts real-valued invariant features from normalized raw I/Q, selects 16 observation-driven candidate anchors from a matched-filter response, predicts bounded delay corrections and learned windows, and obtains an explicit correction through a complex variable-projection solve with adaptive diagonal precision. Stage 2 operates only on the Stage-1 residual and correction, using bias-free complex convolutions, radial scale-invariant gates, residual and skip connections, and a dilated bottleneck to produce a refinement increment. Both stages are numerically approximately equivariant to global amplitude scaling and complex phase rotation, separating interpretable structure from a small residual learner.

The preflight implementation contains 57,525 effective real parameters: 28,948 in Stage 1 and 28,577 in Stage 2, below the 59,700 parameter budget. On the finite preflight samples, the maximum relative equivariance error under tested rescalings was \(2.54\times10^{-7}\), the Stage-1 anchor-column permutation error was zero, and all 100 trainable parameter tensors received finite nonzero gradients. These checks are implementation evidence rather than universal guarantees. Seed 17 is still training on physical GPU 1 of server B; the final fixed-DEV detection, false-alarm, JCR, and latency results are therefore pending. We define the four-state paired supervision, pulse-compression and matched-filter-tail surrogate objectives, and clarify why the increment loss and tail surrogate do not establish lossless restoration or strict CFAR preservation.

## 1 引言

ISRJ 通过间断采样、存储和转发截获的雷达脉冲，在距离和慢时间维度形成与目标响应相互耦合的伪峰。经典方法通常依赖干扰参数估计、稀疏分解、形态学分量分离或端到端波形映射[1–14]。这些方法分别利用了时频结构、稀疏性、优化展开或深度特征，但在原始复数 I/Q 上同时处理结构可解释性、目标保持和部署容量仍然困难。一个完整网络可以获得较大的表示能力，却可能需要更多参数、较高的内存访问量，并难以说明每一部分究竟在恢复什么。

第一份草稿采用四状态配对监督，将同一目标、干扰和噪声组合成可比较的输入，并通过端点、脉压和目标增量误差约束恢复量。ReCoVP-Hybrid 延续这一任务定义，但改变了网络组织方式：先用观察值中的少量候选位置构造结构化字典，再用一个只处理残差的复值 U-Net 进行精修。这里的“Hybrid”不是把两个完整的恢复器并联，而是让第一阶段承担可解释的低维投影，让第二阶段只学习投影未能解释的部分。

本文的贡献集中在四点。第一，给出一个 16 锚点、复数变量投影和学习窗组成的结构化第一阶段，使校正量可以写成有限字典上的投影系数。第二，设计无偏置复值 U-Net 和径向尺度不变门控，使精修器对输入幅度和整体复相位具有明确的数值约束。第三，在 59,700 个有效实参数上限内，将容量分配为 28,948 个 Stage 1 参数和 28,577 个 Stage 2 参数，总计 57,525 个有效实参数。第四，使用目标缺失状态的匹配滤波尾峰代理项作为训练辅助，同时明确其不能替代严格的 CFAR 约束或独立检测校准。

当前工作的证据状态需要先说明。结构、参数量、有限样本等变性和梯度流已经完成预检；seed 17 的 256 轮训练正在服务器 B 上运行，最终 DEV 评测尚未完成。因此，本文把结果分为已完成的实现证据、训练状态和待补的最终性能，不根据预检或 Base/RSDR 背景结果宣称 Hybrid 已经优于任何模型。

## 2 信号模型与恢复目标

### 2.1 相参脉冲与离散观测

设雷达发射基带脉冲为长度 \(T_p\) 的线性调频信号

\[
 s(t)=\operatorname{rect}\left(\frac{t}{T_p}\right)\exp\left(j\pi K t^2\right),
\]

其中 \(K=B/T_p\)，\(B\) 为带宽。实现中采用 \(F_s=60\,\mathrm{MHz}\)、\(B=30\,\mathrm{MHz}\)、\(T_p=10\,\mu\mathrm{s}\)，因此一个参考脉冲为 600 个采样点。模型的单次输入是一个长度为 \(T=1800\) 的复数观测向量

\[
 y\in\mathbb C^{T},
\]

输入文件以两个实通道存储 \([\Re(y),\Im(y)]\)。训练和检测指标将八个连续脉冲组成一个相参记录，以便计算慢时间 FFT、距离–多普勒图和尾峰统计；这不改变网络单向量输入的定义。

### 2.2 ISRJ 片段转发模型

令 \(S\) 为目标回波，\(J\) 为 ISRJ 干扰，\(E\) 为热噪声或背景噪声。对一个相参脉冲组，观测可写为

\[
Y=S+J+E.
\]

ISRJ 的间断采样使 \(J\) 由多个截获片段的时延、相位和转发周期共同决定。本文不把真实干扰支撑、时延、SNR 或 JSR 作为网络输入；这些量只在数据生成和分层评价中用于定义场景。模型只从观测 \(y\) 和固定匹配滤波器构造候选锚点。

### 2.3 四状态配对与恢复参考

同一个目标、干扰和噪声实例生成四个状态：

\[
Y_{00}=E,\qquad
Y_{10}=S+E,\qquad
Y_{01}=J+E,\qquad
Y_{11}=S+J+E.
\]

网络输出记为校正量 \(C(Y)\)，恢复波形为

\[
R(Y)=Y-C(Y).
\]

校正标签和恢复参考分别为

\[
L=[0,0,J,J],
\qquad
D=[E,S+E,E,S+E].
\]

因此，网络在 \(Y_{01}\) 和 \(Y_{11}\) 上学习移除 \(J\)，在 \(Y_{00}\) 和 \(Y_{10}\) 上保持无干扰输入。四状态配对的价值是保持 \(S\)、\(J\) 和 \(E\) 相同，只改变目标是否存在，从而可以对目标响应的有限差分进行直接测量。

### 2.4 非线性恢复中的目标响应

定义恢复算子

\[
T_\theta(y)=y-C_\theta(y).
\]

在加入目标增量 \(S\) 的两个状态之间，有严格的有限差分恒等式

\[
T_\theta(v+S)-T_\theta(v)-S
=-[C_\theta(v+S)-C_\theta(v)].
\]

于是

\[
\left\|T_\theta(v+S)-T_\theta(v)-S\right\|_2^2
=
\left\|C_\theta(v+S)-C_\theta(v)\right\|_2^2.
\]

本文将右侧归一化后作为目标增量损失。该项度量的是加入目标后校正量的变化是否过大，并不等于干扰恢复正确性。例如 \(C_\theta\equiv0\) 时该项为零，但干扰完全保留。因此端点误差、目标增量误差和尾峰代理必须联合解释。

### 2.5 可识别性边界

仅凭 \(Y=S+J+E\) 一条观测，若不引入先验或配对参考，\(S\) 与 \(J\) 的分解一般不是唯一的。本文的可识别性来自三类附加信息：一是数据生成器提供的四状态配对；二是 LFM 匹配滤波和 ISRJ 结构带来的候选位置先验；三是端点、脉压和目标增量联合监督。因此，模型输出不能被解释为对任意未知干扰的无条件分解定理。特别是当干扰形态超出训练分布、目标与干扰完全重叠或噪声不再近似热噪声时，恢复性能需要重新验证。

## 3 ReCoVP-Hybrid 方法

### 3.1 RMS 归一化与输入约定

对非零输入，先计算复数 RMS

\[
\sigma_y=\sqrt{\frac1T\sum_{n=0}^{T-1}|y_n|^2},
\qquad
x=\frac{y}{\sigma_y}.
\]

所有可学习的结构特征先在 \(x\) 上计算，最终校正量再乘回 \(\sigma_y\)。零输入在实现中用 \(10^{-8}\) 下限保护。该处理把整体幅度从特征学习中分离出来，同时保留原始量纲用于输出。整体相位不通过显式相位估计校正，而是由复数表示和不变特征共同处理。

### 3.2 Stage 1：结构化变量投影

Stage 1 的目标不是直接生成最终波形，而是从观察值中构造一个小规模复数字典，并给出显式校正量。首先从归一化输入提取四个实值通道：

\[
|x_n|,
\quad |(x*h)_n|,
\quad \Re(x_nx_{n-1}^*),
\quad \Im(x_nx_{n-1}^*),
\]

其中 \(h\) 是固定 LFM 匹配滤波器。三尺度实值编码器–解码器的通道宽度为

\[
(13,26,44).
\]

编码器只处理上述观察值特征，不读取真实目标、干扰支撑、SNR 或 JSR 标签。其输出用于在候选位置处采样局部特征，并为每个候选位置预测有限的时延修正。

### 3.3 观察值驱动的 16 锚点选择

对固定矩形窗匹配滤波响应，按峰值能量选择 \(K=16\) 个位置，并要求相邻选中位置至少相隔 6 个采样点。选择过程只依赖当前归一化观测和固定脉冲频谱；训练时不把真实干扰支撑作为选择器输入。为便于消融和复现，实现允许外部传入锚点覆盖值，但正式前向默认使用观察值驱动选择。

16 个锚点是容量和覆盖范围之间的折中。锚点数太少时，多个干扰片段可能共享同一字典列；锚点数增加则会使复数正规方程和学习窗头的容量上升。本文固定 \(K=16\)，并将“16 个锚点足以覆盖本数据生成分布”视为需要通过 DEV 证据验证的工程假设，而不是普遍结论。

对每个锚点 \(a_k\)，局部特征、深层共享特征、归一化位置和匹配滤波峰值经过一个有界头得到

\[
\Delta a_k=2\tanh(f_\mathrm{delay}(\text{local}_k,
\text{shared},a_k/T,\log(1+p_k))),
\qquad
\tau_k=a_k+\Delta a_k.
\]

用 \(\tau_k\) 对固定 LFM 频谱进行相位平移，得到复数模板 \(t_k\)。模板不是由真实时延直接写入，而是由观察值候选位置和有限修正共同决定。

### 3.4 复模板、学习窗与变量投影

为适应 ISRJ 的片段化结构，模型使用 5、15 和 31 点平滑特征计算每个模板的学习窗。对

\[
z_{k,n}=x_n t_{k,n}^*,
\]

分别计算平滑复数序列的模长以及一阶滞后乘积的实部和虚部，再由窗头输出 \(m_{k,n}\in(0,1)\)。窗经过 3 点平滑和峰值归一化，形成原子

\[
a_{k,n}=m_{k,n}t_{k,n}.
\]

令 \(b_k=a_k/\|a_k\|_2\)，将所有列组成

\[
B=[b_1,\ldots,b_K]\in\mathbb C^{T\times K}.
\]

模型在归一化输入上求解复数变量投影

\[
\widehat\beta
=\left(B^HB+\Lambda\right)^{-1}B^Hx,
\]

其中 \(\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_K)\) 是正的自适应对角精度。精度头根据局部特征、共享特征、\(\log(1+\rho_k)\)、窗均值、位置和峰值生成 \(\lambda_k\)；

\[
\rho_k=\frac{|b_k^Hx|^2}{\sigma_x^2}
\]

用于衡量当前字典列与观测的匹配程度，\(\sigma_x^2\) 由稳健离差估计得到。实现使用复数双精度求解正规方程，再将结果转回复数单精度。

Stage 1 校正量和残差分别为

\[
\widehat c_1=\sigma_y B\widehat\beta,
\qquad
r_1=y-\widehat c_1.
\]

这个表达式使 Stage 1 的主要动作可以解释为“观察值驱动字典上的校正投影”，而不是一个没有结构约束的黑盒映射。由于 \(B\)、\(\Lambda\) 和 \(x\) 都由网络特征产生，投影并不等价于已知干扰参数下的最优估计；其作用是为后续精修提供低维、可审计的起点。

### 3.5 Stage 2：复值等变 U-Net 精修

Stage 2 的输入是 \(r_1\) 和 \(\widehat c_1\)，而不是原始 \(y\) 的再次完整编码。其六类不变特征为

\[
\log|r_1|,
\quad \log|\widehat c_1|,
\quad \Re(r_{1,n}r_{1,n-1}^*),
\quad \Im(r_{1,n}r_{1,n-1}^*),
\quad \Re(r_{1,n}\widehat c_{1,n}^*),
\quad \Im(r_{1,n}\widehat c_{1,n}^*),
\]

其中滞后和交叉项按尺度平方归一化。复值主干包含三层编码器、三层解码器、残差连接和跳跃连接，通道宽度为

\[
(9,13,17,21).
\]

瓶颈连续使用空洞率 \((1,2,4)\)，以在不增加大量层数的情况下扩大局部感受野。

复值卷积采用无偏置复线性映射。对输入 \(a+ib\)，权重 \(W_R+iW_I\) 的输出为

\[
(W_R+iW_I)(a+ib)
=(W_Ra-W_Ib)+i(W_Rb+W_Ia).
\]

无偏置设计避免在全局复相位旋转后引入固定的加性方向。每个复值块之后使用实值径向门控；门控输入由归一化模长和六类不变特征组成，输出为

\[
\operatorname{gate}=\operatorname{sigmoid}
\left(g(\log(|u|/s+\epsilon),\,\text{invariants})\right),
\qquad
u_\mathrm{out}=\operatorname{gate}\odot u.
\]

这里 \(s\) 与当前输入幅度同阶，因此门控本身不依赖绝对幅度。网络最后生成未约束的复增量 \(\delta_\mathrm{raw}\)。此外，模型从原始输入、Stage 1 校正量和 Stage 1 残差中计算两个物理统计量：校正量与观测能量比，以及残差匹配滤波峰值与平均能量的对数比。它们与六类不变特征的均值、标准差和 95% 分位数一起预测标量门控

\[
\alpha=\operatorname{sigmoid}(f_\alpha(\text{global statistics},\text{physical statistics})).
\]

精修增量为

\[
\delta=\alpha\,\delta_\mathrm{raw}\,\sigma_y.
\]

因此 Stage 2 的职责是对 Stage 1 残差做小幅、受统计量调节的修正，而不是重新承担完整的干扰分解。

### 3.6 两阶段合成与实现等变性

最终输出写为

\[
\widehat y=r_1-\delta
=y-\widehat c_1-\delta.
\]

当输入整体变为 \(y'=\gamma e^{i\phi}y\)，若固定脉冲模板、锚点选择和浮点误差允许相应变换，则期望

\[
\widehat y(\gamma e^{i\phi}y)
\approx \gamma e^{i\phi}\widehat y(y).
\]

Stage 1 的幅度等变来自 RMS 归一化和最后的尺度恢复；列选择只依据模值峰和最小间隔，因此对全局相位不敏感。Stage 2 的复卷积、归一化不变特征和标量门控共同保持同样的变换关系。预检对四个幅度因子和随机整体相位进行有限测试，最终最大相对误差为 \(2.54\times10^{-7}\)。这是一项受测样本的实现检查，不是对所有输入、所有浮点路径或训练后权重的严格数学定理。

### 3.7 参数量、复杂度与部署边界

有效实参数按可训练实标量计数，复数参数按两个实参数计数。当前结构的参数分配为：

| 模块 | 有效实参数 |
|---|---:|
| Stage 1 结构化变量投影 | 28,948 |
| Stage 2 复值等变 U-Net | 28,577 |
| **总计** | **57,525** |
| 预算上限 | 59,700 |
| 预算余量 | 2,175 |

同一计数口径下，ReCoVP-Base 为 58,943，参数匹配 U-Net 为 59,714，RSDR-Net 为 3,017,877。Hybrid 比 Base 少 1,418 个有效实参数，比参数匹配 U-Net 少 2,189 个；相对于 RSDR-Net，规模约为其 1/52。这里的比较说明结构容量和预算位置，不能直接推出延迟、显存、吞吐、能耗或芯片部署优势。Stage 1 的变量投影还包含 \(K=16\) 的复数正规方程求解，其代数求解量随 \(K^3\) 增长；在本模型中 \(K\) 固定且较小，但最终硬件收益必须通过统一设备和批大小实测。

## 4 联合训练目标

### 4.1 B4 端点、脉压与目标增量

对四状态输出 \(C=[C_{00},C_{10},C_{01},C_{11}]\)，定义端点误差

\[
\mathcal L_\mathrm{endpoint}
=\operatorname{mean}\frac{\|C-L\|_2^2}
{\max(\|D\|_2^2,\varepsilon)}.
\]

为约束恢复波形进入脉压链后的误差，令 \(M(\cdot)\) 表示固定匹配滤波器，定义

\[
\mathcal L_\mathrm{PC}
=\operatorname{mean}\frac{\|M(C-L)\|_2^2}
{\max(\|M(D)\|_2^2,\varepsilon)}.
\]

B4 的目标增量项使用目标存在与目标缺失状态的校正差：

\[
\mathcal L_\mathrm{damage}
=\operatorname{mean}\frac{\|C_{11}-C_{01}\|_2^2}
{\max(\|S\|_2^2,\varepsilon)}.
\]

基础目标为

\[
\mathcal L_\mathrm{new}
=\mathcal L_\mathrm{endpoint}
+0.1\mathcal L_\mathrm{PC}
+\mathcal L_\mathrm{damage}.
\]

端点项规定“校正了什么”，脉压项规定“进入相参处理后误差如何表现”，目标增量项规定“加入目标后校正量不要随目标一起变化过大”。三项互补，任何一项都不能单独证明无损恢复。

### 4.2 匹配滤波尾峰代理损失

为减少目标缺失状态中干扰残留形成的距离–多普勒尾峰，本文加入匹配滤波尾峰代理损失。对八脉冲的 \(Y_{01}=J+E\) 校正输出，构造残差

\[
r_{01}=J-C(Y_{01}).
\]

对 \(r_{01}\) 做 600 点 LFM 脉压和八点慢时间 FFT，得到距离–多普勒功率。以热噪声 \(E\) 的距离–多普勒平均功率归一化，在有效距离单元 `[36,1165)` 内取 top-8 尾峰。理想化 CA-CFAR 参考单元数为 \(N=436\)，名义 \(P_\mathrm{fa}=10^{-6}\)，对应门限系数

\[
\alpha_\mathrm{CFAR}=N\left(P_\mathrm{fa}^{-1/N}-1\right).
\]

训练中使用温度为 3 dB 的平滑 softplus 超额项，最终以 0.05 加权：

\[
\mathcal L_\mathrm{tail}
=\frac1{8}\sum_{q=1}^{8}
\frac{\operatorname{softplus}_\tau(p_q-\eta)-\operatorname{softplus}_\tau(0)}
{\eta},
\quad \tau=3\,\mathrm{dB},
\]

其中 \(p_q\) 是 top-8 尾峰的 dB 值，\(\eta\) 是理想化门限。本文称其为“匹配滤波尾峰代理损失”。它只在目标缺失状态上对残余伪峰提供平滑训练信号，不能称为严格 CFAR 约束，也不能推出等虚警率检测优势。脉压距离单元之间可能相关，恢复输出也未必服从独立同分布指数模型，因此实际虚警仍需在固定评价协议下测量。

### 4.3 Stage 1 辅助监督

最终输出和 Stage 1 输出分别计算同一组 B4 基础项与尾峰代理项。总目标为

\[
\begin{aligned}
\mathcal L_\mathrm{total}
={}&\mathcal L_\mathrm{new}(\mathrm{final})
+0.05\mathcal L_\mathrm{tail}(\mathrm{final})\\
&+0.25\left[
\mathcal L_\mathrm{new}(\mathrm{Stage1})
+0.05\mathcal L_\mathrm{tail}(\mathrm{Stage1})
\right].
\end{aligned}
\]

Stage 1 辅助监督防止第一阶段退化为无意义的预处理，并使第二阶段面对一个已经满足端点任务的残差。0.25 的权重只规定当前训练协议中的优化折中，不代表 Stage 1 在所有数据分布上应占四分之一的最终贡献。

### 4.4 优化流程

训练使用 512 个 parent，并为每个 parent 生成 direct、repeat 和 cyclic 三种干扰变体；每个 epoch 共 1,536 条记录。批大小为 32，每个批由四个 8-record microbatch 累积，梯度裁剪阈值为 5。优化器为 Adam[21]，

\[
\beta_1=0.9,\quad \beta_2=0.999,\quad \epsilon=10^{-8},\quad
\text{weight decay}=0.
\]

学习率首轮线性 warmup 至 \(10^{-3}\)，随后余弦下降至第 128 轮的 \(2\times10^{-5}\)，再下降至第 256 轮的 \(2\times10^{-6}\)。每个完整 epoch 保存一个恢复点，每 32 个 epoch 保存一个里程碑，同时记录优化器、随机数状态、源码和数据收据的 SHA-256 绑定。

## 5 实验设置与评价指标

### 5.1 数据、设备和冻结范围

训练和 DEV 数据绑定到 `data/paper2_1103/train.npz`、`dev.npz` 及对应收据。seed 17 的训练固定在服务器 B 的 `D:/code/Xin/ISRJ_222`，当前使用 physical GPU 1；本机不执行长训练。训练和固定 DEV 核心评测不读取 CONFIRM 或历史 TEST 数据。最终评测前会重新核对源码、数据、检查点和设备 UUID。

本稿使用的训练计划为 256 epochs、512 training parents、1536 records/epoch、batch size 32。预检阶段验证了输入输出形状 `[1,2,1800]`、复数变量投影可反向传播、四状态目标函数能运行以及训练参数均获得有限非零梯度。

### 5.2 指标

波形恢复使用端点误差和归一化复数均方误差；脉压和慢时间处理使用目标附近峰值、JCR、目标检测概率 \(P_D\)、帧虚警率和目标增量失真。任何检测结果都必须同时注明数据状态、样本数、CFAR 参考窗口和阈值校准方式。模型大小使用有效实参数；延迟、显存和吞吐只有在同一设备、同一批大小、同一预热和同步规则下测量后才报告。

尾峰相关指标的训练版本是匹配滤波尾峰代理，不等同于评价阶段的经验 CFAR。评价阶段将单独报告尾峰、帧虚警率和检测概率，不把代理损失数值直接转译为检测率。

### 5.3 对照和证据分层

ReCoVP-Base、参数匹配 U-Net 和 RSDR-Net 只能作为同一 DEV 协议下的描述性背景比较。它们的训练 seed、预算、结构和数据协议不完全相同，因此不能据此形成严格的结构因果结论。Hybrid 的最终 DEV 结果必须使用同一冻结检查点和预先规定的记录格式，不能用 Base 或 RSDR 的数值替代。

## 6 结果

### 6.1 预检与轻量性

当前预检状态为 PASS。模型有效实参数为 57,525，其中 Stage 1 和 Stage 2 分别为 28,948 和 28,577，低于 59,700 的上限。四个幅度因子和随机整体相位组合的有限测试中，Stage 1 残差、Stage 2 增量和最终输出均保持数值近似等变，最大相对误差为 \(2.5422161797905574\times10^{-7}\)。锚点列置换测试的相对误差为 0。真实训练 microbatch 中 100/100 个可训练参数张量获得有限非零梯度，没有缺失、零值或非有限梯度张量。

这些结果回答的是三个工程问题：当前前向图能运行；参数计数在预算内；训练信号能到达全部可训练张量。它们不回答最终检测率、虚警率、跨种子稳定性或实测硬件效率。

**表 1. 当前结构和同口径参数背景。**

| 模型 | 有效实参数 | 当前证据状态 |
|---|---:|---|
| ReCoVP-Hybrid | 57,525 | 预检通过，seed 17 训练中 |
| ReCoVP-Base | 58,943 | 已有 DEV 描述性结果 |
| 参数匹配 U-Net | 59,714 | 受控对照计划/参数背景 |
| RSDR-Net | 3,017,877 | 已有 DEV 描述性结果 |

### 6.2 当前训练状态

截至 2026-11-25 09:22:58 UTC，seed 17 状态文件记录为 epoch 21/256、update 24/48，最近一个完整提交轮次为 epoch 20。epoch 20 的训练集联合损失为 0.5955245466708826，Final 损失为 0.1348079847327123，Stage1 损失为 1.8428662545823802，单轮耗时为 96.12274089991115 s。epoch 20 检查点 SHA-256 为 `8e831179df94f08dfb08676f419c42453d72633a8323f6387e9177c056b8fa75`。这些是训练过程记录，不能用于判断 DEV 检测率或 Stage 2 的泛化收益。

当前已验证 epoch 000–020 共 21 个恢复点，训练器继续保留每个完整 epoch 和每 32 个 epoch 的里程碑。最终固定 DEV 评测将在 epoch 256 完成且源码、数据和检查点身份重新核对后执行。

### 6.3 Base/RSDR 描述性背景比较

在同一 DEV 协议下已有的描述性结果为：ReCoVP-Base 的 JCR 为 12.2564 dB、\(P_D=84.8958\%\)、帧虚警率为 6.90104%、absCPL 为 0.175693、damage 为 0.0716197；RSDR-Net 的 JCR 为 17.7605 dB、\(P_D=86.9792\%\)、帧虚警率为 5.07812%、damage 为 0.0947035。RSDR-Net 同时具有 3,017,877 个有效实参数，远大于 ReCoVP-Hybrid 的预算规模。

这些数值可作为协议背景，不能写成 Hybrid 的结果，也不能据此宣称 Hybrid 将优于 Base 或 RSDR。两者的训练 seed、训练预算、网络结构和数据协议存在差异；当前没有等参数、同预算、同检查点的完整结果来区分结构贡献。

### 6.4 Hybrid 最终 DEV 待补

以下内容待 256 轮训练和固定 DEV256 核心评测完成后补入：

1. Hybrid Final 与同检查点 Stage1Only 的配对 JCR、\(P_D\)、帧虚警率、absCPL、damage 和尾峰统计；
2. parent 配对 bootstrap 区间、复缩放检查和无干扰改变量；
3. 原生 SNR/JSR 分层、门控分布和 CUDA 延迟；
4. 与 Base、参数匹配 U-Net、RSDR 的统一协议对照、Equal-PFA 结果和失败 parent 回查。

在这些结果完成前，本文不提供 Hybrid 的最终性能排名，不声称已经完成独立泛化验证，也不声称结构创新已经转化为检测优势。若最终结果与背景比较不一致，正文应同时报告不利结果及其不确定性。

## 7 讨论

### 7.1 结构创新的作用边界

ReCoVP-Hybrid 的第一个创新点是把干扰恢复的前端写成观察值驱动的变量投影。16 个锚点、有限时延修正、平滑学习窗和复数正规方程使第一阶段具有可检查的中间量：锚点位置、字典列、精度和投影系数都可以单独记录。与直接输出校正波形相比，这种结构有助于说明网络首先在匹配滤波域中寻找什么，再如何把这些候选成分投影回原始域。

第二个创新点是残差精修的职责收缩。Stage 2 不再重新编码完整观测，而只处理 \(r_1\) 与 \(\widehat c_1\)。这使得第二阶段的容量用于处理投影误差、局部形态和残余尾峰，而不是重复学习 Stage 1 已经提供的结构。该叙事需要最终 Stage1Only 与 Final 的配对 DEV 结果支持；在结果完成前，它是设计意图而不是已证实的性能机制。

第三个创新点是复值等变约束和轻量门控的组合。无偏置复卷积负责复平面线性变换，径向门控负责根据归一化模长调节幅度，两个物理统计量则让全局门控能区分校正量占比和残差匹配滤波形态。预检证明当前有限样本上这些机制协同工作，但不证明所有训练后输入都严格等变。

### 7.2 轻量性不等于实际加速

57,525 个有效实参数让 Hybrid 落在与 Base 和参数匹配 U-Net 相近的容量区间，并远小于 RSDR-Net。参数减少可降低权重存储上限，但实际延迟还受复数运算、FFT、变量投影求解、张量布局和设备内存访问影响。特别是 Stage 1 的 FFT 和 \(16\times16\) 复数线性系统不能仅用参数量估算。因而本文只把轻量性作为可审计的模型容量属性，待统一 CUDA 和 CPU 计时完成后再讨论部署收益。

### 7.3 尾峰代理与检测评价的区别

尾峰代理损失把目标缺失状态的 top-8 距离–多普勒尾峰压到理想化门限附近，目的是在训练时提供可微的背景残留信号。它没有对每个距离单元使用独立参考窗，也没有在训练中完成经验虚警校准。真实输出的距离单元可能相关，目标附近也可能改变参考窗统计，因此代理项不能推出“严格 CFAR 保持”或“等虚警率下检测优势”。最终检测结论必须依赖固定 CFAR 实现、独立校准和经验虚警率。

### 7.4 需要继续完成的受控验证

为了判断当前结构是否真正贡献收益，至少需要四类对照：参数匹配的常规 U-Net；移除固定去斜或 RMS 归一化的版本；移除脉压项或目标增量项的损失消融；以及固定最终检查点后在新 parent 集上的一次性确认。每个对照都应保留完整 epoch 恢复点和源码、数据 SHA，避免以不同训练预算或不同数据身份混淆结构效应。

当前所有结果来自单目标仿真。多目标、目标–干扰近邻、波形失配、非高斯杂波、硬件转发链路和实测数据仍未验证。外推到这些条件前，需要新的数据收据、独立评价和硬件计时。

## 8 结论

本文提出 ReCoVP-Hybrid，一种面向 ISRJ 波形恢复的轻量两阶段模型。它先以 16 个观察值驱动锚点构造复数字典，通过带自适应精度的变量投影得到结构化校正量，再以 28,577 个有效实参数的复值等变 U-Net 对 Stage 1 残差进行精修。RMS 归一化、无偏置复卷积、径向尺度不变门控和四状态配对监督共同提供了幅度、相位和目标响应的结构约束。当前实现共含 57,525 个有效实参数，低于 59,700 的预算上限。

现阶段证据只支持预检通过、参数量达标、有限样本等变误差很小且训练梯度完整。seed 17 仍在服务器 B 上进行 256 轮训练，最终 DEV 结果和实际效率尚未完成。因此，本文的主要方法结论是结构设计和证据链已经形成，性能结论需要在固定 DEV 评测、等参数对照和独立确认完成后更新。

## 9 证据边界与局限

- 有限样本等变测试不能证明所有浮点输入上的严格等变。
- 16 锚点的覆盖率来自观察值驱动的匹配滤波候选；当前 DEV 指标是代理覆盖率，不是潜在干扰分量的完整真值标注。
- 目标增量损失控制的是校正量对目标加入的敏感性，不单独保证干扰被恢复正确。
- 匹配滤波尾峰项是代理损失，不是严格 CFAR 约束，也不证明等虚警率下的检测优势。
- 参数量不能直接推出延迟、显存、吞吐或能耗。
- 当前 seed 17 训练未完成，Hybrid 没有最终 DEV 性能排名。
- Base 和 RSDR 结果只是同一 DEV 协议下的描述性背景，不能替代 Hybrid 的统一对照。
- 数据仅覆盖当前仿真生成分布，尚无实测雷达、硬件转发、杂波和多目标验证。

## 10 数据、代码与可重复性

模型实现位于 `tools/vp_1103_recovp_hybrid.py`，训练目标位于 `tools/vp_1103_recovp_hybrid_objective.py`，尾峰代理实现位于 `tools/unet_tail_loss/objective.py`。训练状态、预检参数量、源码和数据 SHA 记录在 `evidence/outputs/recovp_hybrid_seed17/`。仓库只提交源代码、稿件和证据摘要；训练数据、权重、checkpoint、raw arrays、缓存、凭据和完整日志不上传到 GitHub。

训练运行在服务器 B 的 `D:/code/Xin/ISRJ_222`，使用 physical GPU 1，固定设备 UUID 和数据绑定见预检收据。每个完整 epoch 和每 32 轮里程碑均保留恢复点。最终发布时应按期刊政策说明数据和代码的可获得方式；本文不声称已完成实测数据许可或硬件验证。

## 参考文献

[1] Wang X, Liu J, Zhang W, Fu Q, Liu Z, Xie X. Mathematic principles of interrupted-sampling repeater jamming (ISRJ). *Science in China Series F: Information Sciences*, 2007, 50(1): 113–123. DOI: [10.1007/s11432-007-2017-y](https://doi.org/10.1007/s11432-007-2017-y).

[2] Chen J, Wu W, Xu S, Chen Z, Zou J. Band pass filter design against interrupted-sampling repeater jamming based on time-frequency analysis. *IET Radar, Sonar & Navigation*, 2019, 13(10): 1646–1654. DOI: [10.1049/iet-rsn.2018.5658](https://doi.org/10.1049/iet-rsn.2018.5658).

[3] Liu Z, Liang W, Fu N, Qiao L, Zhang J. Interrupted sampling repeater jamming suppression based on Gaussian mixture model and sparse reconstruction. *Digital Signal Processing*, 2025, 161: 105132. DOI: [10.1016/j.dsp.2025.105132](https://doi.org/10.1016/j.dsp.2025.105132).

[4] Huan S, Dai G, Luo G, Ai S. Bayesian Compress Sensing Based Countermeasure Scheme Against the Interrupted Sampling Repeater Jamming. *Sensors*, 2019, 19(15): 3279. DOI: [10.3390/s19153279](https://doi.org/10.3390/s19153279).

[5] Zhang Y, Jiu B, Peng X, Liu H, Jiang W. A new scheme of target detection for pulse Doppler radar in interrupted sampling repeater jamming environment. *IET Radar, Sonar & Navigation*, 2022, 16(11): 1836–1850. DOI: [10.1049/rsn2.12300](https://doi.org/10.1049/rsn2.12300).

[6] Wang H, Chen B, Ye Q, Huang F. MREDNet: A Multi-Residual Encoder-Decoder Network for recovering LFM interfered by intra-pulse retransmission interferences. *Signal Processing*, 2024, 217: 109329. DOI: [10.1016/j.sigpro.2023.109329](https://doi.org/10.1016/j.sigpro.2023.109329).

[7] He X, Liao K, Peng S, Tian Z, Huang J. Interrupted-Sampling Repeater Jamming-Suppression Method Based on a Multi-Stages Multi-Domains Joint Anti-Jamming Depth Network. *Remote Sensing*, 2022, 14(14): 3445. DOI: [10.3390/rs14143445](https://doi.org/10.3390/rs14143445).

[8] Meng Y, Yu L, Wei Y. Interrupted sampling repeater jamming suppression based on multiple extended complex-valued convolutional auto-encoders. *IET Radar, Sonar & Navigation*, 2024, 18(8): 1274–1290. DOI: [10.1049/rsn2.12568](https://doi.org/10.1049/rsn2.12568).

[9] Dai G, Yang X, Huan S, Chen Z, Peng C, Xu Y. End-to-End Interrupted Sampling Repeater Jamming Countermeasure Network Under Low Signal-to-Noise Ratio. *Sensors*, 2025, 25(13): 3925. DOI: [10.3390/s25133925](https://doi.org/10.3390/s25133925).

[10] Huang Q, Zhang L. Interrupted Sampling Repeater Jamming Suppression Using Complex-Valued FISTA-Net. *IEEE Transactions on Aerospace and Electronic Systems*, 2026, 62: 3270–3286. DOI: [10.1109/TAES.2025.3646182](https://doi.org/10.1109/TAES.2025.3646182).

[11] Wei S, Han H, Wei S, Zhang L, Zhou Y. ISRJ Suppression Using Joint Morphological Component Analysis and Signal Reconstruction Based on Jamming Parameters Estimation. *IEEE Transactions on Aerospace and Electronic Systems*, 2026, 62: 3773–3797. DOI: [10.1109/TAES.2025.3648973](https://doi.org/10.1109/TAES.2025.3648973).

[12] Wei S, Han H, Zhang H, Wei S, Zhou Y, Zhang L. Interrupted Sampling Repeater Jamming Suppression for Wideband Radar With MCA-ADMM-Net. *IEEE Transactions on Neural Networks and Learning Systems*, 2026, early access. DOI: [10.1109/TNNLS.2026.3736439](https://doi.org/10.1109/TNNLS.2026.3736439).

[13] Liu M, Gao X, Qiu X, Zhao Y. Structure-Informed Deep Filtering for Radar HRRP ISRJ Mitigation. *IEEE Signal Processing Letters*, 2026, 33: 3601–3605. DOI: [10.1109/LSP.2026.3726844](https://doi.org/10.1109/LSP.2026.3726844).

[14] Chen L, Wang K, Pan X, Gu Z, Xiao S. Interrupted Sampling Repeater Jamming Suppression-Based on Point-Wise Temporal Classification and Sparse Signal Reconstruction. *IET Radar, Sonar & Navigation*, 2026. DOI: [10.1049/rsn2.70251]. 在线发表条目，卷页待核验。

[15] 作者前期稿件. 可靠度引导的间歇采样转发干扰选择性波形恢复. 未发表研究稿，2026 年 11 月版本。本文将其作为方法承袭与冻结系统比较的来源，不视为已发表同行评议文献。

[16] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: *Medical Image Computing and Computer-Assisted Intervention—MICCAI 2015*. 2015: 234–241. DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).

[17] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016: 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).

[18] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. *IEEE Transactions on Image Processing*, 2017, 26(7): 3142–3155. DOI: [10.1109/TIP.2017.2662206](https://doi.org/10.1109/TIP.2017.2662206).

[19] Wu Y, He K. Group Normalization. In: *Computer Vision—ECCV 2018*. 2018: 3–19. DOI: [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1).

[20] Hendrycks D, Gimpel K. Gaussian Error Linear Units (GELUs). arXiv preprint, 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).

[21] Kingma D P, Ba J. Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).

[22] Efron B. Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 1979, 7(1): 1–26. DOI: [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).

## 附录 A：四状态损失的补充推导

### A.1 端点误差与目标增量误差不等价

令 \(r_{11}=C(Y_{11})-J\)、\(r_{01}=C(Y_{01})-J\)，则

\[
\mathcal L_\mathrm{damage}
=\frac{\|r_{11}-r_{01}\|_2^2}{\|S\|_2^2+\varepsilon}.
\]

若 \(r_{11}=r_{01}\ne0\)，则目标增量误差仍为零，但两个状态都含有相同的校正偏差。反过来，即使两个端点误差都较小，当目标能量很低时，差值除以 \(|S|_2^2\) 也可能较大。因此端点项和增量项回答不同问题。

### A.2 变量投影的正则化作用

当字典列 \(B\) 高度相关时，未经正则化的 \(B^HB\) 可能病态。正对角精度 \(\Lambda\) 使

\[
B^HB+\Lambda
\]

保持可逆，并抑制与当前观测匹配很弱或相互重复的列。该正则化是数值稳定和容量控制手段，不等价于已知干扰先验的贝叶斯后验，也不保证投影系数具有唯一物理解释。

### A.3 理想化 CA-CFAR 门限的适用条件

若待检功率 \(X\) 与 \(N\) 个参考功率独立同分布且均为均值 \(\sigma^2\) 的指数变量，参考均值的缩放门限可写成

\[
\Pr(X>\alpha\bar X_r)=\left(1+\frac{\alpha}{N}\right)^{-N}.
\]

令其等于 \(P_\mathrm{fa}\) 得

\[
\alpha=N\left(P_\mathrm{fa}^{-1/N}-1\right).
\]

本文使用该式只构造尾峰代理的参考尺度。恢复后的距离单元存在脉压旁瓣和慢时间相关，故实际帧虚警率仍需由经验评价确认。

### A.4 参数量口径

实卷积和线性层按可训练实标量计数；复数权重的实部和虚部各计一个实参数。固定脉冲频谱、匹配滤波器、锚点选择规则和常数阈值不计入有效实参数。参数量是结构容量指标，不包括运行时临时张量和 FFT 工作区。
