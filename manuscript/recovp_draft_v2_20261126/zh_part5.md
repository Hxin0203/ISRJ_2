
## 6 讨论

### 6.1 证据门控为何降低背景虚警

本文方法最稳定的优势不在目标保真，而在背景一侧：对无干扰输入的改动比12类同规模卷积网络低一个数量级以上，受扰背景的整帧虚警约为它们的一半。两点都可以追溯到修正的组织方式。对无干扰输入，推论1的$K/T$上界只对固定字典成立；候选取自噪声本身时，纯投影会移除约5.6%的噪声能量（5.7节）。第一阶段实际依靠式（17）中依赖证据的正则：缺少匹配证据的原子得到极大的正则，移除的能量被压低约5个数量级（推论2）。标量门又在无干扰状态下几乎关闭精修分支（5.9节），整个处理器因而近似于恒等映射，CFAR参考窗的统计没有被改变。对受扰输入，虚警来自残余中仍与发射波形相干的成分：只有这部分会在脉冲压缩后重新聚成峰值。第一阶段的原子就是开窗后的时延LFM，其系数由最小二乘精确对齐幅度和相位，针对的正是会在脉压后聚焦的成分。图7(b)显示，完整模型高相干残余的帧占比低于全部12类同规模卷积网络，而仅用第一阶段时反而高于其中多数，说明这一优势是两阶段分工的结果：第一阶段撤销大部分干扰而不扰动背景，第二阶段在门控范围内清除剩余的相干残余。通用卷积网络按均方误差回归修正波形，训练目标并不直接约束残余是否相干。

同一机制也解释了代价。真实目标同样是时延LFM，当候选落在目标时延附近时，式（20）的泄漏项会把部分目标能量一并撤销，平均幅度收缩由此而来。推理期诊断另表明，证据门本身也在保护目标：去掉它后目标损伤由0.060升至1.13，幅度降至0.46（5.7节），所以现存的收缩主要来自那些与目标重叠、匹配证据又足以通过门控的候选。去除去斜的CorrectionNet和U-Net类没有这种归纳偏置，目标损伤更低，却以更多的背景改动和虚警为代价。两者的取舍不是网络容量的差别，而是修正被允许落在哪里的差别。由此看，降低目标收缩的方向应是让候选选择和正则强度对目标时延更敏感，而不是放开修正的自由度。

### 6.2 结构化估计与残差补偿的分工

ReCoVP-Hybrid把干扰修正拆成两部分：能够由“局部窗×时延LFM”表示的部分由字典与闭式系数解决，其余部分由复值网络补偿。与把迭代算法展开为固定层数的网络[4,34]相比，本文在给定字典时直接消去线性系数，网络只需估计时延、窗和正则这三类非线性参数；与先分类片段再重构目标的串联方法[32,33]相比，本文不对观测做硬性的片段取舍，而是在全长观测上估计应当减去的修正量。ISRJ-FNet利用目标的结构先验约束滤波网络[35]，本文则直接以目标增量误差约束修正量。上述区别体现在计算接口和约束对象上，其性能意义需要由同协议的对比实验检验。

表3与表4从两个角度说明了两阶段的作用。只有第一阶段的ReCoVP-Base对干净输入的改动最小，但恢复误差最大；加入精修分支后，恢复误差下降一个数量级，背景改动仍明显小于通用卷积网络。这与命题2的分析一致：正则收缩偏差与表示残差是第一阶段的主要误差来源，而它们恰好是精修分支能够从$(\widehat{\boldsymbol u}_1,\widehat{\boldsymbol j}_1)$中识别的成分。另一方面，精修在降低平均误差的同时增加了平均幅度衰减，且至少一半场景的目标损伤有所增加，说明精修分支对目标并非完全透明。式（26）给出了一个解释：精修提案在受扰状态与含目标受扰状态之间的差异，即使很小，也会直接成为目标增量误差的一部分。

### 6.3 目标保真的度量

式（8）与式（10）说明，恢复误差、目标损伤和幅度保持是三个相互关联但不能互相推出的量。表4中各模型排序不一致，正是这一点的体现：通用卷积网络的目标损伤较低而背景改动较大，RSDR-Net的恢复误差较低而目标损伤较高，ReCoVP-Hybrid则处在背景改动与虚警较低、目标幅度有一定收缩的位置。若只报告恢复误差或SJRIF，这些差别都无法被观察到。共享背景四状态提供了一种在仿真条件下单独度量目标响应的方法；对实测数据，目标真值不可得，需要通过已知目标注入或合作目标实验近似实现。

式（21）—（22）还指出，自适应处理器的目标损伤可以分为直接修改与算子变化两部分。本文目前只报告了完整输出的目标损伤，尚未区分这两部分。记录目标加入前后的候选集合与$\|\boldsymbol P_1-\boldsymbol P_0\|_2$，可以判断损伤是否集中在候选切换的场景中；局部归因图[55]与表示相似性分析[56]等可解释性工具也可用于检查两阶段特征对目标区域的响应。

### 6.4 恢复与检测

恢复损失、尾峰损失与CFAR检测作用于不同的对象：恢复损失约束全部样点，尾峰损失约束剥离噪声后的干扰残余峰值，而CFAR面对的是含噪输出及其自适应门限。表3中SJRIF与整帧虚警率均有改善，但检测概率与单元虚警率的差值区间都包含零，说明恢复指标的改善并不自动转化为相同虚警水平下的检测增益。比较检测性能时，应在相同的实测虚警工作点上进行（5.6节）。

### 6.5 局限性

本文结果存在以下局限。第一，数据全部来自仿真，信号模型为单目标、窄带、组内参数不变，未包含多目标、距离走动、接收机非线性与实测干扰；对多目标与扩展目标，共享背景下的目标增量需要按目标分别定义。第二，第5.4—5.5节中Hybrid的定量结果来自单个训练实例在开发集上的评价，基线已覆盖三个训练实例，置信区间只反映场景抽样的不确定性，不包含Hybrid自身的训练随机性与方法选择带来的偏差。第三，与传统方法和代表性学习方法的同协议比较仍不完整。第四，在当前的GPU实现中，单个相参处理间隔的推理耗时约为30 ms，远大于30 μs的脉冲重复间隔，按脉冲到达节奏的在线处理需要进一步的工程优化。

## 7 结论

本文研究了固定LFM波形下ISRJ的接收端复数波形恢复问题。以同次噪声实现下的无干扰观测为恢复参考，恢复误差等价于干扰估计误差，处理器对干净输入与真实目标的改动可以被单独度量。在此基础上，本文提出了ReCoVP-Hybrid：第一阶段由相位不变特征估计候选时延、局部窗和正则强度，构造复数字典后以正则化变量投影闭式求取干扰系数；第二阶段读取初步恢复与已估计干扰，以带相位不变实门的复值残差网络补偿剩余误差。

理论分析表明，固定字典下的系数子问题总有唯一解，对应算子在任何方向上都只撤销部分能量，从白噪声中带走的平均能量不超过候选数与信号长度之比；第一阶段误差可分解为正则收缩偏差、表示残差与目标噪声泄漏三项，这为引入精修阶段提供了依据；输入自适应的字典变化会使目标增量误差中出现与背景能量成比例的附加项，因此目标保真必须在完整输出上评价。整个映射在精确算术下对输入的复数缩放保持等变。

仿真实验表明，精修阶段显著降低了平均恢复误差与平均目标损伤，降低了受扰整帧虚警率，同时增加了平均幅度衰减。与12类参数量相当、每类3个训练实例的卷积恢复网络相比，本文方法的恢复误差、无干扰改动与受扰背景整帧虚警均更低，其中无干扰改动低一个数量级以上，整帧虚警约减半；其代价是对部分基线而言更高的目标损伤和更大的幅度收缩。把各方法的实测帧虚警率标定到同一水平后，这种更干净的背景转化为检测增益：本文方法在参数量相当的模型中检测概率最高，对12类卷积网络的配对差值区间均不含零，并远高于传统的Max-TF与GMM-OMP；参数规模更大的模型只有付出数十至数百倍的参数量，或连同无干扰输入在内改写整段波形，才得到更高的检测概率。在受控的SNR–JSR扫描中，本文方法的受扰整帧虚警率在全部SNR下维持在约6%至12%，同规模卷积网络则上升到20%至28%。这一结果说明，对ISRJ抑制而言，把修正限制在可解析的干扰子空间内并按匹配证据门控，比增大网络容量更能维持检测背景的洁净。后续工作将在匹配训练条件下完成结构消融与全部训练实例的统计，补充采样窗宽度与受扰脉冲数的扫描，并将方法推广到多目标与实测数据。

## 参考文献

{{REFERENCES}}

## 附录A 证明

**命题1的证明。** 令$\boldsymbol A=\boldsymbol B\boldsymbol\Lambda^{-1/2}$，则$\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda=\boldsymbol\Lambda^{1/2}(\boldsymbol A^H\boldsymbol A+\boldsymbol I)\boldsymbol\Lambda^{1/2}$，于是
$$
\boldsymbol P=\boldsymbol A(\boldsymbol A^H\boldsymbol A+\boldsymbol I)^{-1}\boldsymbol A^H .
$$
设$\boldsymbol A=\boldsymbol U\boldsymbol\Sigma\boldsymbol V^H$为奇异值分解，则$\boldsymbol P=\boldsymbol U\boldsymbol\Sigma(\boldsymbol\Sigma^H\boldsymbol\Sigma+\boldsymbol I)^{-1}\boldsymbol\Sigma^H\boldsymbol U^H$，其特征值为$\sigma_k^2/(1+\sigma_k^2)\in[0,1)$及$T-K$个零。$\boldsymbol P$显然为Hermitian半正定矩阵，迹等于非零特征值之和。由于特征值严格小于1，$\|\boldsymbol P\boldsymbol y\|_2\le\frac{\sigma_1^2}{1+\sigma_1^2}\|\boldsymbol y\|_2$，且$\boldsymbol I-\boldsymbol P$的特征值位于$(0,1]$，为非扩张算子。若某个$\sigma_k\in(0,\infty)$，则对应特征值$\sigma_k^2/(1+\sigma_k^2)\neq\{\sigma_k^2/(1+\sigma_k^2)\}^2$，故$\boldsymbol P^2\neq\boldsymbol P$。证毕。

**命题2的证明。** 由$\widehat{\boldsymbol u}_1=(\boldsymbol I-\boldsymbol P)\boldsymbol y$和$\boldsymbol u=\boldsymbol s+\boldsymbol\epsilon$，
$$
\widehat{\boldsymbol u}_1-\boldsymbol u=(\boldsymbol I-\boldsymbol P)\boldsymbol j-\boldsymbol P(\boldsymbol s+\boldsymbol\epsilon).
$$
以$\boldsymbol c^\star$表示$\boldsymbol j$在$\operatorname{span}(\boldsymbol B)$上的最小二乘系数（$\boldsymbol B$列不满秩时取最小范数解），则$\boldsymbol j_\perp=\boldsymbol j-\boldsymbol B\boldsymbol c^\star$满足$\boldsymbol B^H\boldsymbol j_\perp=\boldsymbol0$，从而$\boldsymbol P\boldsymbol j_\perp=\boldsymbol0$。又
$$
(\boldsymbol I-\boldsymbol P)\boldsymbol B=\boldsymbol B-\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol B^H\boldsymbol B
=\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol\Lambda,
$$
代入即得式（20）。证毕。

**命题3的证明。** 设$c=|c|e^{\mathrm j\varphi}\neq0$。(i) 由式（12），$r(c\boldsymbol y)=|c|r(\boldsymbol y)$，$\boldsymbol x'=e^{\mathrm j\varphi}\boldsymbol x$。(ii) 式（13）中$|x_n'|=|x_n|$，$a_n'=a_n$，$\ell_n'=x_n'x_{n-1}'^{*}=\ell_n$，故特征编码器输出、候选位置（在相同并列规则下）、时延、窗与$\boldsymbol B$均不变；$\rho_k'=|\boldsymbol b_k^He^{\mathrm j\varphi}\boldsymbol x|^2/\widehat\sigma'^2=\rho_k$（稳健离差只依赖距离，不受旋转影响），故$\boldsymbol\Lambda$不变。(iii) 正规方程右端乘以$e^{\mathrm j\varphi}$，故$\widehat{\boldsymbol c}'=e^{\mathrm j\varphi}\widehat{\boldsymbol c}$，$\widehat{\boldsymbol j}_1'=|c|r\boldsymbol B e^{\mathrm j\varphi}\widehat{\boldsymbol c}=c\widehat{\boldsymbol j}_1$，$\widehat{\boldsymbol u}_1'=c\widehat{\boldsymbol u}_1$。(iv) 第二阶段输入$\boldsymbol r_1'=e^{\mathrm j\varphi}\boldsymbol r_1$，$\boldsymbol j_1'=e^{\mathrm j\varphi}\boldsymbol j_1$；条件特征$\boldsymbol\psi$中的对数幅值、共轭积均不变；无偏置复卷积、复数线性插值、平均池化与相位旋转交换；门$\boldsymbol g$的输入$|\boldsymbol v|$不变，故每层输出乘以$e^{\mathrm j\varphi}$。标量门输入中的能量比与峰均比不变，故$\alpha'=\alpha$。于是$\boldsymbol\delta_2'=|c|r\alpha e^{\mathrm j\varphi}\widetilde{\boldsymbol\delta}=c\boldsymbol\delta_2$，$F_\theta(c\boldsymbol y)=c\widehat{\boldsymbol u}_1-c\boldsymbol\delta_2=cF_\theta(\boldsymbol y)$。证毕。

## 数据与代码可用性

〔作者填写：代码与配置的公开地址、数据生成方式及访问限制。〕
