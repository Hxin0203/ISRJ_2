
### 3.7 复值残差精修

**输入与复值卷积。** 第二阶段在归一化坐标中工作，输入为两路复数信号$\boldsymbol r_1=\widehat{\boldsymbol u}_1/r$与$\boldsymbol j_1=\widehat{\boldsymbol j}_1/r$。网络主干由无偏置复值卷积构成[48]。复数权重$\boldsymbol W=\boldsymbol W_R+\mathrm j\boldsymbol W_I$与复数特征$\boldsymbol z=\boldsymbol z_R+\mathrm j\boldsymbol z_I$的卷积按实部与虚部展开为

$$
\boldsymbol W*\boldsymbol z=(\boldsymbol W_R*\boldsymbol z_R-\boldsymbol W_I*\boldsymbol z_I)+\mathrm j(\boldsymbol W_R*\boldsymbol z_I+\boldsymbol W_I*\boldsymbol z_R). \tag{23}
$$

去掉偏置项是为了让每一层都与全局复数缩放交换：若输入乘以复数$c$，输出也恰好乘以$c$。

**相位不变的实值门。** 复值网络的难点在于非线性。对实部和虚部分别施加ReLU或tanh等逐分量非线性，会使输出依赖于输入的绝对相位，破坏上述交换性质。本文把非线性全部放在一个相位不变的实值分支中，再把其输出作为乘性门作用回复数特征。每一层复卷积的输出$\boldsymbol v$乘以门$\boldsymbol g$：

$$
\boldsymbol z^{+}=\boldsymbol g\odot\boldsymbol v,\qquad
\boldsymbol g=\operatorname{sigmoid}\Big(\operatorname{SiLU}\big(\mathcal K[\log(|\boldsymbol v|_\varepsilon+10^{-3}),\,\boldsymbol\psi]\big)\Big),\qquad
|v|_\varepsilon=\sqrt{|v|^2+10^{-14}}, \tag{24}
$$

其中$\mathcal K$为实值卷积，$\boldsymbol\psi$为6路相位不变条件特征：两路输入的逐点对数幅值，$\boldsymbol r_1$相邻样点共轭积的实部与虚部，以及$\boldsymbol r_1\odot\boldsymbol j_1^{*}$的实部与虚部。最后两路衡量初步恢复与已撤销干扰在每个时刻的相关程度，是识别式（20）中收缩偏差的主要依据；低分辨率层的$\boldsymbol\psi$由平均池化得到。门只缩放复数特征的模，不改变其相位。这种以条件特征调制中间特征的做法在图像恢复中有类似形式[41]，区别在于本文的调制只有乘性项，没有会破坏相位关系的加性偏移。由于SiLU有下界$-0.278$，门的取值范围约为$(0.431,1)$，它是一个连续的强度调节器，不能把某个位置完全关闭。

**网络层级。** 精修网络为四级复值U-Net：通道宽度依次为$9,13,17,21$，长度依次为$T,T/2,T/4,T/8$，每级包含复值残差块[37]；最低分辨率处设置膨胀率为1、2、4的三个复值卷积块，以扩大时间感受野而不进一步降低分辨率，使网络能够覆盖重复转发产生的相隔数十个样点的相似片段；解码端经三次上采样与跳跃连接回到全分辨率，最后由一个无偏置复卷积输出单通道修正提案$\widetilde{\boldsymbol\delta}$。上采样采用对实部与虚部分别做线性插值的方式，下采样采用平均池化，两者都与复数缩放交换。

**脉冲级标量门。** 层内门在每个位置、每个通道上调节特征，但不能决定整个脉冲是否需要精修。例如无干扰输入的第一阶段输出已经接近参考，此时最好几乎不采用精修提案。为此设置一个独立的标量头：它汇总6路条件特征在全脉冲上的均值、标准差与第95百分位数（18维），再加上第一阶段干扰估计能量与观测能量之比、初步恢复结果的脉压峰均功率比，共20维输入，经$20\to4\to1$的多层感知机输出

$$
\alpha=\operatorname{sigmoid}\big(f_\alpha(\boldsymbol h_\alpha)\big)\in(0,1),\qquad
\boldsymbol\delta_2=r\,\alpha\,\widetilde{\boldsymbol\delta}. \tag{25}
$$

前一个能量比反映第一阶段撤销了多少，后一个峰均比反映初步恢复的脉压结果是否仍有明显的假峰，两者都是判断“是否需要继续修正”的直接线索。$\alpha$只限制对当前提案的采用比例，不限制提案本身的能量。

**精修何时有效。** 记第一阶段误差为$\boldsymbol e_1=\widehat{\boldsymbol u}_1-\boldsymbol u$。由$\widehat{\boldsymbol u}-\boldsymbol u=\boldsymbol e_1-\boldsymbol\delta_2$得

$$
\|\widehat{\boldsymbol u}-\boldsymbol u\|_2^2-\|\boldsymbol e_1\|_2^2=\|\boldsymbol\delta_2\|_2^2-2\operatorname{Re}\big(\boldsymbol e_1^H\boldsymbol\delta_2\big). \tag{26}
$$

因此，单个样本的恢复误差下降当且仅当$2\operatorname{Re}(\boldsymbol e_1^H\boldsymbol\delta_2)>\|\boldsymbol\delta_2\|_2^2$，即修正量与第一阶段误差在方向上足够一致，且幅度不超过该方向上误差投影的两倍。式（26）对单个样本精确成立，不依赖于独立性或线性化假设。它也说明了标量门的作用：给定方向，$\alpha$的最优值为$\operatorname{Re}(\boldsymbol e_1^H\widetilde{\boldsymbol\delta})/(r\|\widetilde{\boldsymbol\delta}\|_2^2)$截断到$(0,1)$，当提案方向不可靠时，较小的$\alpha$可以限制误差增大的幅度。同样的分析适用于目标损伤，只需把$\boldsymbol\delta_2$换为两种受扰状态精修量之差。

表1列出了网络各部分的接口与规模。

**表1 ReCoVP-Hybrid的模块与接口**（批量维省略）

|模块|输入→输出|结构|作用|
|---|---|---|---|
|RMS归一化|$\mathbb C^{1800}\to\mathbb C^{1800}$，尺度$r$|式（12）|统一幅度尺度，保留物理输出尺度|
|特征编码—解码|实$4\times1800\to13\times1800$|宽度13/26/44，两次下采样，GN+ReLU，跳连|融合能量、模板匹配和局部相位变化|
|候选选择|相关幅值$\to$16个整数位置|13点最大池化，间隔≥6|定位可能的延迟波形|
|时延细化头|$59\to64\to1$|MLP，$\pm2$点有界输出|吸收非整数时延与峰值偏移|
|窗网络|实$22\times1800\to1\times1800$（每个候选）|$22\to12\to1$逐点卷积，sigmoid，平滑|估计每个原子的时间支撑|
|字典与正则头|$\mathbb C^{1800\times16}$；$61\to16\to1$|式（16）、（17）|构造原子，确定每个候选的正则强度|
|系数求解|$16\times16$正定线性方程|式（19），双精度|求干扰复幅度，得到$\widehat{\boldsymbol j}_1$|
|复值编码器|2路复数$\to$复$9\times1800\to13\times900\to17\times450\to21\times225$|无偏置复卷积，残差块，相位不变实门|聚合多尺度残余结构|
|膨胀瓶颈|复$21\times225$|膨胀率1/2/4|扩大时间感受野|
|复值解码器|复$21\times225\to1\times1800$|三次上采样，跳连|恢复全分辨率修正提案$\widetilde{\boldsymbol\delta}$|
|标量门|实20维$\to$标量$\alpha$|$20\to4\to1$ MLP，sigmoid|决定整个脉冲对提案的采用比例|

整个网络共有57,525个有效实参数（每个复数参数计为两个实参数）。参数主要集中在特征编码器与复值精修网络中，字典构造与系数求解本身没有可训练参数。

〔图3占位：第一阶段细节图。(a) 一个受扰脉冲的相关幅值及选出的16个候选；(b) 某一候选的多尺度匹配特征与估计窗$\boldsymbol m_k$，叠加真实采样窗作对照；(c) 正则强度$\lambda_k$与相关强度$\rho_k$的散点关系；(d) 第一阶段干扰估计$\widehat{\boldsymbol j}_1$与真实干扰的实部对比。〕

〔图4占位：第二阶段复值精修网络结构。画出四级复值编码器、膨胀瓶颈与解码器，标注每级通道数和长度；单独画出一层“复卷积→相位不变实门→复数特征”的计算单元，以及标量门的20维输入构成。〕

### 3.8 复数缩放等变性与计算复杂度

**命题3（复数缩放等变）。** 设所有计算均为精确算术，且候选选择在幅值并列时采用确定的规则。对任意非零复数$c$，有

$$
F_\theta(c\boldsymbol y)=c\,F_\theta(\boldsymbol y). \tag{27}
$$

证明思路如下（细节见附录A）。由式（12），$r(c\boldsymbol y)=|c|\,r(\boldsymbol y)$，归一化输入变为$e^{\mathrm j\arg c}\boldsymbol x$，只发生全局相位旋转。式（13）的特征、相关幅值以及确定性的峰选择均不变，故候选、窗、正则与字典不变，$\widehat{\boldsymbol c}$乘以$e^{\mathrm j\arg c}$，$\widehat{\boldsymbol j}_1$乘以$c$。第二阶段中，无偏置复卷积、复数线性插值与平均池化都与相位旋转交换，相位不变条件特征和门保持不变，故$\boldsymbol\delta_2$也乘以$c$。命题3的意义在于，目标幅度与初始相位的变化不会改变处理器“如何解释”观测，网络不必从数据中学习这种对称性。群等变网络[51]讨论了更一般的对称结构，本文只涉及全局复数缩放。

需要注意的是，命题3依赖精确算术与确定的并列规则。在单精度实现中，相关幅值的舍入误差可能改变峰值排序，从而引起候选切换。5.9节给出了实测的数值偏差。

**计算复杂度。** 设$K$为候选数，$L$为模板长度。模板相关与非整数时延模板生成均可通过FFT实现，复杂度为$O(KN\log N)$；构造$\boldsymbol B^H\boldsymbol B$需要$O(TK^2)$，求解$K\times K$线性方程需要$O(K^3)$；两阶段卷积网络的复杂度与$T$成线性关系。在$T=1800$、$K=16$的设置下，字典相关计算与卷积网络在总计算量中处于同一量级。

## 4 四状态联合训练

第3节的结构限定了修正量的形式，但没有规定如何在撤销干扰与保留目标之间取舍。训练目标在四状态（式（6））上同时约束这两件事：端点损失要求每种状态的修正量逼近该状态应当撤销的干扰，目标增量损失则要求目标加入后修正量不发生不必要的变化。

### 4.1 波形与脉压约束

对状态$ab$，应撤销的修正量为$\boldsymbol j^\star_{ab}=b\,\boldsymbol j$，参考为$\boldsymbol u_{ab}=\boldsymbol\epsilon+a\boldsymbol s$。记$\langle\cdot\rangle$为在训练记录、脉冲和四种状态上的算术平均。对任一阶段的修正量$C$，波形损失为

$$
\mathcal L_{\mathrm{wav}}(C)=\Big\langle\frac{\|C(\boldsymbol y_{ab})-\boldsymbol j^\star_{ab}\|_2^2}{\max(\|\boldsymbol u_{ab}\|_2^2,10^{-12})}\Big\rangle. \tag{28}
$$

两种无干扰状态的监督目标为零修正，式（28）因而直接惩罚处理器对干净输入的改动。以$\boldsymbol u_{ab}$的能量归一化，使不同信噪比的样本对损失的贡献处于同一量级。

波形误差在时域上平均分布，但检测只关心脉压后的峰值。参照CVEDNet[19]在波形域与脉压域同时施加约束的做法，加入脉压损失

$$
\mathcal L_{\mathrm{pc}}(C)=\Big\langle\frac{\|\mathcal H[C(\boldsymbol y_{ab})-\boldsymbol j^\star_{ab}]\|_2^2}{\max(\|\mathcal H\boldsymbol u_{ab}\|_2^2,10^{-12})}\Big\rangle, \tag{29}
$$

其中$\mathcal H$为已知LFM模板的零填充相关算子。

### 4.2 目标增量约束

由$F(\boldsymbol y)=\boldsymbol y-C(\boldsymbol y)$与式（6），两种受扰状态的目标增量误差恰好等于修正量之差的相反数：

$$
F(\boldsymbol y_{11})-F(\boldsymbol y_{01})-\boldsymbol s=-\big[C(\boldsymbol y_{11})-C(\boldsymbol y_{01})\big]. \tag{30}
$$

目标损伤损失因此写为

$$
\mathcal L_{\mathrm{tar}}(C)=\Big\langle\frac{\|C(\boldsymbol y_{11})-C(\boldsymbol y_{01})\|_2^2}{\max(\|\boldsymbol s\|_2^2,10^{-12})}\Big\rangle. \tag{31}
$$

式（31）惩罚的是“加入目标后干扰估计的变化”，它与式（21）中的两项都直接相关：既约束当前算子对目标的修改，也约束目标引起的算子变化。该损失只在训练时需要四状态样本，推理时不需要。

### 4.3 残余尾峰约束

虚警主要来自无目标单元中的残余峰值，而这些峰值在总残余能量中所占比例很小，波形损失对其不敏感。对无目标受扰状态，令残余$\boldsymbol v=\boldsymbol j-C(\boldsymbol y_{01})$（不含噪声），对一个相参处理间隔的$M$个脉冲做脉冲压缩与$M$点慢时间FFT，得到距离—多普勒功率图。以同次噪声在全部距离—多普勒单元上的平均功率为分母，取距离索引$[36,1165)$内残余功率与之的比值（dB），记前8个最大值为$q_l$，$l=1,\ldots,8$。以理想CA-CFAR的门限因子$\gamma=N_r(p_0^{-1/N_r}-1)$定义参考水平$q_0=10\log_{10}\gamma$，其中$N_r=436$、$p_0=10^{-6}$。尾峰损失为

$$
\mathcal L_{\mathrm{tail}}(C)=\frac1{q_0}\Big\langle\Big[3\log\{1+e^{(q_l-q_0)/3}\}-3\log2\Big]_+\Big\rangle_{l=1,\ldots,8}, \tag{32}
$$

其中$[x]_+=\max(x,0)$。该项只对超过检测门限水平的少数峰值施加惩罚，softplus使惩罚在门限附近平滑增长。由于前8选择与截断，该损失分段光滑；它约束的是已知干扰残余相对于噪声的峰值，并非对实际虚警率的可微替代。

### 4.4 联合目标与优化设置

每个阶段的损失为

$$
\mathcal L_{\mathrm{stage}}(C)=\mathcal L_{\mathrm{wav}}(C)+0.1\,\mathcal L_{\mathrm{pc}}(C)+\mathcal L_{\mathrm{tar}}(C)+0.05\,\mathcal L_{\mathrm{tail}}(C), \tag{33}
$$

总损失为$\mathcal L=\mathcal L_{\mathrm{stage}}(C_{\mathrm{final}})+0.25\,\mathcal L_{\mathrm{stage}}(C_1)$，其中$C_1=\widehat{\boldsymbol j}_1$为第一阶段修正，$C_{\mathrm{final}}=\widehat{\boldsymbol j}_1+\boldsymbol\delta_2$为总修正。第一阶段辅助项使字典分支即使在精修分支存在时也保持有意义的干扰估计，避免两阶段退化为“第一阶段任意、第二阶段全包”。两个阶段联合优化，第一阶段不单独预训练或冻结。各项权重为固定值；GradNorm[52]等自适应加权方法可用于平衡多个损失的梯度尺度，本文未采用。

优化器为Adam[53]（$\beta_1=0.9$，$\beta_2=0.999$，$\varepsilon=10^{-8}$，无权重衰减）。学习率在第一轮内线性预热至$10^{-3}$，按余弦计划在第128轮降至$2\times10^{-5}$，第256轮降至$2\times10^{-6}$。有效批量为32条记录，由4个8条记录的微批次梯度累积得到，梯度范数裁剪为5。每轮48次更新，共训练256轮（12,288次更新），取最后一轮的模型参数，不依据评价集挑选轮次。
