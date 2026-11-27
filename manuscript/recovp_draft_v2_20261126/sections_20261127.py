"""Replace sections 5.3/5.6/5.10 (zh) and V-C/V-F/V-J (en) with the baseline-suite results.

Numbers come from evidence/outputs/paper2_baseline_suite_20261126/SUITE_SUMMARY.json and
E5/SWEEP_SUMMARY.json (commit b14b840). Backups are written before editing.
"""
import pathlib, re, shutil, hashlib

D = pathlib.Path(__file__).parent

ZH = {}
EN = {}

ZH["### 5.3 对比方法"] = r"""### 5.3 对比方法

对比方法分为四组。

**（1）同一模型的两个阶段输出。** 仅使用第一阶段的输出$\widehat{\boldsymbol u}_1$（记为Stage1）和最终输出$\widehat{\boldsymbol u}$（记为Final）。二者来自同一组参数，比较的是推理时加入精修分支带来的变化。

**（2）参数量相近的学习型恢复网络。** 包括仅含结构化字典求解的ReCoVP-Base、不含去斜处理的卷积残差修正网络P0-NoDechirp-Tail，以及12类卷积恢复网络：CorrectionNet及其去掉去斜、去掉RMS归一化、去掉脉压损失、去掉目标损伤损失的四种变体，原损失下重新训练的对照与加入匹配滤波尾峰损失的版本；普通实值I/Q U-Net及其加尾峰损失、观测噪声归一化、频域坐标和膨胀卷积四种变体。12类网络的有效参数均为59,714个，与本文方法的57,525个相当，训练数据、更新次数与本文方法相同，每类包含3个独立训练实例。

**（3）传统方法。** Max-TF在去斜后的时频平面上识别并抑制受扰时频单元[9]，窗长65点、帧移16点、频率平滑5点、能量分位0.7；另以窗长51点、帧移12点、平滑3点的配置作参数敏感性对照，二者结论一致，正文只报告前者。GMM-OMP用两分量高斯混合模型区分受扰与未受扰样本，再在未受扰样本上以正交匹配追踪重构目标回波[12]，稀疏度取8。两种方法只使用观测与公开的LFM参数，不需要训练。

**（4）参数规模较大的学习方法。** CVEDNet是复值编解码网络，以时域与脉压域联合损失、无噪目标为监督[19]，采用官方实现，参数量11.02 M。RSDR-Net是作者先前提出的可靠度引导复数恢复模型，参数量3.02 M。两者的权重均在另一套仿真数据上按各自原训练流程得到，本文未在当前训练集上重新训练，因此这一组比较反映的是直接使用已训练模型时的表现，与第（2）组的同预算比较性质不同。

表4c汇总了各方法的参数量与训练方式。

表4c　对比方法概况

|组别|方法|有效参数量|是否训练|训练数据|
|---|---|---:|---|---|
|本文|ReCoVP-Hybrid|57,525|是|本文训练集|
|同预算|12类卷积恢复网络、ReCoVP-Base、P0-NoDechirp-Tail|58,943–59,714|是|本文训练集|
|传统|Max-TF[9]、GMM-OMP[12]|—|否|—|
|大规模|RSDR-Net|3.02 M|是|另一套仿真数据|
|大规模|CVEDNet[19]|11.02 M|是|另一套仿真数据|

"""

ZH["### 5.6 相同实测虚警率下的检测性能"] = r"""### 5.6 相同实测虚警率下的检测性能

名义虚警参数相同时，不同处理器输出的背景分布不同，实测虚警率也不同，直接比较检测概率并不公平。为此对每个方法单独标定检测门限：在独立的校准集（1024个物理场景，每个场景含三种转发方式）上取该方法受扰无目标输出的距离–多普勒帧统计量，即有效单元功率与其局部CA参考均值之比的帧内最大值，以其经验99%分位作为该方法自身的门限；再在另一组2048个物理场景上统计受扰整帧虚警率与检测概率。校准集与评价集互不重叠，也与训练集和5.5节的评价集无重叠。按此标定后，各方法在评价集上的实测帧虚警率在0.51%至1.00%之间。表5给出结果，差值为本文方法减对比方法，区间为评价场景的配对bootstrap 95%区间，多实例方法先在场景内对训练实例取平均。

表5　受扰整帧虚警率标定为1%时的检测性能

|方法|有效参数量|训练实例|检测概率/%|实测帧虚警/%|$E_0$|$D_S$|检测概率差/百分点|
|---|---:|---:|---:|---:|---:|---:|---|
|**ReCoVP-Hybrid**|57,525|4|**55.8±2.1**|0.91|$1.0\times10^{-6}$|0.059|—|
|Stage1（同一模型）|—|4|53.9±3.9|0.96|$2.7\times10^{-7}$|0.142|+1.90 [+1.25, +2.53]|
|CN-NoDamage|59,714|3|54.7±0.8|1.00|$2.5\times10^{-4}$|0.125|+1.16 [+0.62, +1.71]|
|UNet-Tail|59,714|3|52.7±0.2|0.85|$8.2\times10^{-5}$|0.047|+3.10 [+2.63, +3.62]|
|CN-NoDechirp|59,714|3|51.9±0.7|0.90|$1.2\times10^{-4}$|0.044|+3.90 [+3.36, +4.45]|
|UNet-Dilation|59,714|3|51.6±0.2|0.87|$7.2\times10^{-5}$|0.049|+4.20 [+3.65, +4.75]|
|UNet|59,714|3|51.0±1.0|0.82|$7.3\times10^{-5}$|0.046|+4.78 [+4.19, +5.37]|
|CN-NoRMS|59,714|3|49.6±0.5|0.61|$4.8\times10^{-5}$|0.062|+6.24 [+5.54, +6.99]|
|CorrectionNet-Tail|59,714|3|47.9±1.2|0.75|$1.2\times10^{-4}$|0.061|+7.94 [+7.12, +8.78]|
|UNet-NoiseNorm|59,714|3|47.8±1.0|0.80|$1.3\times10^{-4}$|0.057|+7.98 [+7.18, +8.80]|
|CorrectionNet（复训对照）|59,714|3|47.2±0.6|0.80|$1.2\times10^{-4}$|0.060|+8.58 [+7.69, +9.47]|
|CorrectionNet|59,714|3|47.1±1.2|0.86|$1.2\times10^{-4}$|0.061|+8.71 [+7.84, +9.61]|
|CN-NoPC|59,714|3|46.5±0.7|0.83|$1.3\times10^{-4}$|0.057|+9.35 [+8.42, +10.30]|
|UNet-Freq|59,714|3|32.2±0.3|0.51|$2.2\times10^{-4}$|0.136|+23.57 [+22.04, +25.04]|
|Max-TF[9]|—|—|18.2|0.98|$1.6\times10^{-3}$|18.5|+37.62 [+35.83, +39.39]|
|GMM-OMP[12]|—|—|21.6|0.98|1.00|8.78|+34.23 [+32.19, +36.36]|
|RSDR-Net|3.02 M|1|73.5|0.93|$1.1\times10^{-5}$|0.091|−17.69 [−19.02, −16.34]|
|CVEDNet[19]|11.02 M|1|91.8|0.94|1.00|0.646|−35.97 [−38.00, −33.98]|

注：检测概率一栏的“±”为训练实例间的标准差；$E_0$为无干扰输入的相对改动量，$D_S$为目标损伤，均在同一评价集上计算。

在参数量相当的一组中，本文方法的检测概率最高，相对全部12类卷积恢复网络的配对差值均为正且区间不含零，幅度为1.2至23.6个百分点。与此同时，它对无干扰输入的改动比这些网络低约50至250倍。5.5节在名义CFAR下观察到的背景侧优势，即受扰帧虚警明显更低，在门限对齐以后转化成了检测增益：背景残余越少形成假峰，同样1%的帧虚警所需的门限越低，目标越容易越过门限。差值最小的CN-NoDamage去掉了目标损伤损失，它的目标损伤是本文方法的2.1倍，检测概率的接近是以放弃目标保护换来的。与同一模型的Stage1相比，最终输出的检测概率高1.9个百分点，目标损伤由0.142降至0.059，精修阶段的作用主要体现在目标保真上。

两种传统方法在门限对齐后检测概率只有18%至22%，目标损伤为8.8至18.5，GMM-OMP的$E_0$接近1，说明它的重构结果替换了整段观测。

参数规模更大的两种方法检测概率更高，但代价不同。RSDR-Net参数量约为本文方法的52倍，检测概率高17.7个百分点，对无干扰输入的改动约为本文方法的10倍，目标损伤也更高（配对差值区间为[−0.038, −0.023]）。CVEDNet检测概率达91.8%，但$E_0\approx1$、目标损伤0.646、5.5节评价集上的幅度比为0.52：它输出的是去噪后的目标估计，接收机噪声与目标幅度都不再保留，与本文“只撤除干扰、背景保持不变”的设定不同，它的检测增益来自非线性去噪带来的信噪比提升。这两者的训练数据也与本文不同，比较结论只说明在各自的使用方式下所处的位置。综合来看，本文方法的定位是同等参数预算下、对接收链透明的干扰撤除器；在这一约束下它的检测性能最好，更大模型的检测增益需要以参数量或波形保真为代价。

〔图7占位：受扰整帧虚警率与检测概率的关系曲线（各模型一条）。各方法均在自身校准输出上按目标帧虚警率0.1%、0.2%、0.5%、1%、2%、5%、10%取门限，在评价集上报告检测概率。〕

"""

ZH["### 5.10 条件分析"] = r"""### 5.10 条件分析

按基础场景的组内平均SNR分箱，检测概率由[−20, −15) dB内的46.2%上升至[−5, 0) dB内的99.4%。按平均JSR分箱时，样本集中于[10, 20) dB（162个基础场景），[30, 40) dB仅有3个基础场景。评价集为随机抽样，各箱内其他参数并未固定，上述结果描述的是条件关联，不是单因素响应。

为得到单因素响应，另行生成了受控扫描集：SNR取−25、−20、−15、−10、−5、0、5、10 dB，JSR取0、10、20、30 dB，共32个条件，每个条件64个物理场景，各含三种转发方式；扫描集只用于评价，不参与训练、门限标定或模型选择，与其他各数据集均无场景重叠。全部方法使用名义CFAR设置。表9列出各SNR下对四档JSR取平均的受扰整帧虚警率。

表9　受控扫描下的受扰整帧虚警率（%，名义CFAR，四档JSR平均）

|方法|−25 dB|−20 dB|−15 dB|−10 dB|−5 dB|0 dB|5 dB|10 dB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|**ReCoVP-Hybrid**|6.8|6.9|8.0|10.2|10.6|11.6|9.9|6.2|
|CorrectionNet|14.2|16.5|21.3|24.0|24.0|28.0|25.7|26.4|
|UNet|12.9|14.9|16.0|19.1|20.6|21.4|16.5|10.5|
|RSDR-Net|2.1|3.1|3.3|7.0|8.5|15.4|16.9|16.2|
|CVEDNet[19]|100.0|99.9|99.7|96.0|91.2|84.5|78.8|72.9|
|Max-TF[9]|11.2|17.3|29.8|39.8|46.5|60.2|67.6|74.9|
|GMM-OMP[12]|100.0|100.0|100.0|100.0|100.0|100.0|99.9|99.9|

注：本文方法为4个训练实例的平均，CorrectionNet与UNet为3个训练实例的平均。

检测概率方面，SNR不低于−15 dB时本文方法、CorrectionNet、UNet与RSDR-Net均为100%，−20 dB时为96%至97%，−25 dB时为32%至34%，彼此接近；CVEDNet在SNR不高于−15 dB时检测概率低于8%，与其在低SNR下输出接近零的现象一致。差异主要在虚警一侧。本文方法的受扰帧虚警率在整个SNR范围内保持在6.2%至11.6%之间，在每个SNR下都不到CorrectionNet的一半，也都低于UNet。RSDR-Net在低SNR时虚警更低，但随SNR升高而上升，SNR不低于0 dB时反而高于本文方法（15.4%至16.9%对6.2%至11.6%）。在JSR为30 dB、SNR为−15、−10、0 dB的强干扰条件下，本文方法的受扰帧虚警率为6.4%、9.5%、4.2%，CorrectionNet为26.2%、15.6%、20.7%。目标损伤方面，本文方法在全部SNR下不超过0.08，在SNR不低于0 dB时低于CorrectionNet（0.08至0.12）与RSDR-Net（0.12至0.13）。Max-TF的虚警随SNR升高而显著上升，GMM-OMP与CVEDNet在多数条件下几乎每帧都报警。

〔图9占位：受控扫描下各模型的检测概率、受扰整帧虚警率与目标损伤随SNR（分JSR）的变化曲线；三种转发方式分别绘制。〕

〔图10占位：采样窗宽度与受扰脉冲数变化时的结果。〕

"""

EN["### C. Compared Methods"] = r"""### C. Compared Methods

The compared methods fall into four groups.

**(1) The two stage outputs of the same model.** The first-stage output $\widehat{\boldsymbol u}_1$ (Stage1) and the final output $\widehat{\boldsymbol u}$ (Final) come from one set of parameters, so their difference isolates the effect of the refinement branch at inference.

**(2) Learned restoration networks of comparable size.** These are ReCoVP-Base, which contains only the structured dictionary solve; P0-NoDechirp-Tail, a convolutional residual-correction network without dechirping; and twelve families of convolutional restorers: CorrectionNet, its four variants without dechirping, without RMS normalization, without the pulse-compression loss and without the target-damage loss, a retrained control under the original loss and a version with the matched-filter tail loss; and a plain real-valued I/Q U-Net with four variants (tail loss, observation-noise normalization, frequency-domain coordinates, dilated convolution). Each of the twelve families has 59,714 effective parameters, close to the 57,525 of the proposed model, uses the same training data and number of updates, and is represented by three independently trained instances.

**(3) Classical methods.** Max-TF identifies and suppresses jammed time-frequency cells on the dechirped spectrogram [9], with a 65-sample window, a 16-sample hop, 5-bin frequency smoothing and an energy quantile of 0.7; a second setting (window 51, hop 12, smoothing 3) was run as a sensitivity check and leads to the same conclusions, so only the first is reported. GMM-OMP separates jammed from unjammed samples with a two-component Gaussian mixture and reconstructs the target echo from the unjammed samples by orthogonal matching pursuit [12], with sparsity 8. Both use only the observation and the public LFM parameters and need no training.

**(4) Larger learned methods.** CVEDNet is a complex-valued encoder-decoder trained with a joint time-domain and pulse-compression loss against the noise-free target [19]; the official implementation with 11.02 M parameters is used. RSDR-Net is the reliability-guided complex restoration model previously proposed by the authors, with 3.02 M parameters. Both sets of weights were obtained on a different simulation with their own training procedures and were not retrained on the present training set, so this group reflects off-the-shelf use and differs in nature from the matched-budget comparison of group (2).

Table IVc summarizes the methods.

TABLE IVc
Overview of the Compared Methods

|Group|Method|Effective parameters|Trained|Training data|
|---|---|---:|---|---|
|Proposed|ReCoVP-Hybrid|57,525|Yes|This work|
|Matched budget|12 convolutional families, ReCoVP-Base, P0-NoDechirp-Tail|58,943–59,714|Yes|This work|
|Classical|Max-TF [9], GMM-OMP [12]|—|No|—|
|Larger|RSDR-Net|3.02 M|Yes|Different simulation|
|Larger|CVEDNet [19]|11.02 M|Yes|Different simulation|

"""

EN["### F. Detection at Equal Measured False-Alarm Rate"] = r"""### F. Detection at Equal Measured False-Alarm Rate

With identical nominal false-alarm parameters, different processors leave different background distributions and therefore different measured false-alarm rates, so their detection probabilities are not directly comparable. Each method is therefore given its own threshold. On an independent calibration set (1024 physical scenes, each with three repeater modes), the frame statistic of the jammed target-absent output, namely the maximum over valid range-Doppler cells of the cell power divided by its local CA reference mean, is computed, and its empirical 99% quantile is taken as the method's threshold. Jammed frame false alarms and detections are then counted on another 2048 physical scenes. The two sets do not overlap with each other, with the training set, or with the evaluation set of Section V-E. After calibration the measured frame false-alarm rates on the evaluation set lie between 0.51% and 1.00%. Table V reports the results. Differences are the proposed method minus the comparator, with paired bootstrap 95% intervals over evaluation scenes; for multi-instance methods the instances are averaged within each scene first.

TABLE V
Detection With the Jammed Frame False-Alarm Rate Calibrated to 1%

|Method|Effective parameters|Instances|$P_d$ (%)|Measured frame $P_{fa}$ (%)|$E_0$|$D_S$|$\Delta P_d$ (points)|
|---|---:|---:|---:|---:|---:|---:|---|
|**ReCoVP-Hybrid**|57,525|4|**55.8±2.1**|0.91|$1.0\times10^{-6}$|0.059|—|
|Stage1 (same model)|—|4|53.9±3.9|0.96|$2.7\times10^{-7}$|0.142|+1.90 [+1.25, +2.53]|
|CN-NoDamage|59,714|3|54.7±0.8|1.00|$2.5\times10^{-4}$|0.125|+1.16 [+0.62, +1.71]|
|UNet-Tail|59,714|3|52.7±0.2|0.85|$8.2\times10^{-5}$|0.047|+3.10 [+2.63, +3.62]|
|CN-NoDechirp|59,714|3|51.9±0.7|0.90|$1.2\times10^{-4}$|0.044|+3.90 [+3.36, +4.45]|
|UNet-Dilation|59,714|3|51.6±0.2|0.87|$7.2\times10^{-5}$|0.049|+4.20 [+3.65, +4.75]|
|UNet|59,714|3|51.0±1.0|0.82|$7.3\times10^{-5}$|0.046|+4.78 [+4.19, +5.37]|
|CN-NoRMS|59,714|3|49.6±0.5|0.61|$4.8\times10^{-5}$|0.062|+6.24 [+5.54, +6.99]|
|CorrectionNet-Tail|59,714|3|47.9±1.2|0.75|$1.2\times10^{-4}$|0.061|+7.94 [+7.12, +8.78]|
|UNet-NoiseNorm|59,714|3|47.8±1.0|0.80|$1.3\times10^{-4}$|0.057|+7.98 [+7.18, +8.80]|
|CorrectionNet (retrained control)|59,714|3|47.2±0.6|0.80|$1.2\times10^{-4}$|0.060|+8.58 [+7.69, +9.47]|
|CorrectionNet|59,714|3|47.1±1.2|0.86|$1.2\times10^{-4}$|0.061|+8.71 [+7.84, +9.61]|
|CN-NoPC|59,714|3|46.5±0.7|0.83|$1.3\times10^{-4}$|0.057|+9.35 [+8.42, +10.30]|
|UNet-Freq|59,714|3|32.2±0.3|0.51|$2.2\times10^{-4}$|0.136|+23.57 [+22.04, +25.04]|
|Max-TF [9]|—|—|18.2|0.98|$1.6\times10^{-3}$|18.5|+37.62 [+35.83, +39.39]|
|GMM-OMP [12]|—|—|21.6|0.98|1.00|8.78|+34.23 [+32.19, +36.36]|
|RSDR-Net|3.02 M|1|73.5|0.93|$1.1\times10^{-5}$|0.091|−17.69 [−19.02, −16.34]|
|CVEDNet [19]|11.02 M|1|91.8|0.94|1.00|0.646|−35.97 [−38.00, −33.98]|

Note: “±” in the $P_d$ column is the standard deviation across training instances. $E_0$ is the relative change of jammer-free inputs and $D_S$ the target damage, both on the same evaluation set.

Among models of comparable size, the proposed method has the highest detection probability. Its paired differences against all twelve convolutional families are positive with intervals excluding zero, ranging from 1.2 to 23.6 points, while it changes jammer-free inputs about 50 to 250 times less than those networks. The background-side advantage observed under nominal CFAR in Section V-E, a clearly lower jammed frame false-alarm rate, turns into a detection gain once the thresholds are aligned: when fewer residual components form false peaks, the threshold required for a 1% frame false-alarm rate is lower and targets cross it more easily. The closest comparator, CN-NoDamage, drops the target-damage loss and has 2.1 times the target damage of the proposed method; its near-equal detection is bought by giving up target protection. Relative to Stage1 of the same model, the final output gains 1.9 points in detection and lowers the target damage from 0.142 to 0.059, so the refinement stage acts mainly on target fidelity.

After threshold alignment the two classical methods detect only 18% to 22% of targets with target damage of 8.8 to 18.5, and the $E_0$ of GMM-OMP is close to one, meaning that its reconstruction replaces the whole observation.

The two larger methods detect more targets, at different costs. RSDR-Net has about 52 times as many parameters, detects 17.7 points more, changes jammer-free inputs about 10 times as much, and has higher target damage (paired interval [−0.038, −0.023]). CVEDNet reaches 91.8%, but with $E_0\approx1$, a target damage of 0.646 and an amplitude ratio of 0.52 on the evaluation set of Section V-E. It outputs a denoised target estimate in which neither the receiver noise nor the target amplitude is preserved, which is a different setting from the jamming-only removal with an unchanged background considered here; its detection gain comes from the SNR improvement of nonlinear denoising. Both were also trained on different data, so the comparison only locates them under their own mode of use. Overall, the proposed method is positioned as a jamming remover that is transparent to the receive chain at a matched parameter budget. Under this constraint it gives the best detection, and the gains of larger models are paid for in parameters or in waveform fidelity.

[Fig. 7 placeholder: jammed frame false-alarm rate versus detection probability, one curve per model. Each method uses thresholds from its own calibration outputs at target frame false-alarm rates of 0.1%, 0.2%, 0.5%, 1%, 2%, 5% and 10%, and detection is reported on the evaluation set.]

"""

EN["### J. Condition Analysis"] = r"""### J. Condition Analysis

When base scenes are binned by their mean in-group SNR, the detection probability rises from 46.2% in [−20, −15) dB to 99.4% in [−5, 0) dB. When binned by mean JSR, the scenes concentrate in [10, 20) dB (162 base scenes), and [30, 40) dB contains only 3. Because the evaluation set is randomly drawn and other parameters vary within each bin, these figures describe conditional associations rather than single-factor responses.

To obtain single-factor responses, a separate controlled sweep set was generated with SNR at −25, −20, −15, −10, −5, 0, 5 and 10 dB and JSR at 0, 10, 20 and 30 dB, giving 32 conditions with 64 physical scenes each, every scene containing the three repeater modes. The sweep set is used only for evaluation, not for training, threshold calibration or model selection, and shares no scenes with any other set. All methods use the nominal CFAR setting. Table IX lists the jammed frame false-alarm rate at each SNR, averaged over the four JSR levels.

TABLE IX
Jammed Frame False-Alarm Rate in the Controlled Sweep (%, Nominal CFAR, Averaged Over JSR)

|Method|−25 dB|−20 dB|−15 dB|−10 dB|−5 dB|0 dB|5 dB|10 dB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|**ReCoVP-Hybrid**|6.8|6.9|8.0|10.2|10.6|11.6|9.9|6.2|
|CorrectionNet|14.2|16.5|21.3|24.0|24.0|28.0|25.7|26.4|
|UNet|12.9|14.9|16.0|19.1|20.6|21.4|16.5|10.5|
|RSDR-Net|2.1|3.1|3.3|7.0|8.5|15.4|16.9|16.2|
|CVEDNet [19]|100.0|99.9|99.7|96.0|91.2|84.5|78.8|72.9|
|Max-TF [9]|11.2|17.3|29.8|39.8|46.5|60.2|67.6|74.9|
|GMM-OMP [12]|100.0|100.0|100.0|100.0|100.0|100.0|99.9|99.9|

Note: the proposed method is averaged over four training instances, CorrectionNet and UNet over three.

In detection probability, the proposed method, CorrectionNet, UNet and RSDR-Net all reach 100% for SNR of −15 dB and above, 96% to 97% at −20 dB and 32% to 34% at −25 dB, close to one another. CVEDNet detects fewer than 8% of targets at SNR of −15 dB and below, consistent with its near-zero output at low SNR. The differences lie on the false-alarm side. The jammed frame false-alarm rate of the proposed method stays between 6.2% and 11.6% over the whole SNR range, less than half that of CorrectionNet and below that of UNet at every SNR. RSDR-Net has fewer false alarms at low SNR, but its rate grows with SNR and exceeds that of the proposed method for SNR of 0 dB and above (15.4% to 16.9% against 6.2% to 11.6%). Under strong jamming, at a JSR of 30 dB and SNR of −15, −10 and 0 dB, the proposed method has jammed frame false-alarm rates of 6.4%, 9.5% and 4.2%, against 26.2%, 15.6% and 20.7% for CorrectionNet. The target damage of the proposed method stays at or below 0.08 at every SNR and, for SNR of 0 dB and above, is lower than that of CorrectionNet (0.08 to 0.12) and RSDR-Net (0.12 to 0.13). The false alarms of Max-TF grow markedly with SNR, and GMM-OMP and CVEDNet raise an alarm in almost every frame in most conditions.

[Fig. 9 placeholder: detection probability, jammed frame false-alarm rate and target damage versus SNR (one panel per JSR) in the controlled sweep, plotted separately for the three repeater modes.]

[Fig. 10 placeholder: results as the sampling-window width and the number of jammed pulses vary.]

"""


def replace_sections(path, table):
    text = path.read_text(encoding="utf-8")
    bak = path.with_suffix(".md.pre_20261127.bak")
    if not bak.exists():
        shutil.copyfile(path, bak)
    for header, new in table.items():
        start = text.index(header)
        nxt = re.search(r"^### ", text[start + len(header):], re.M)
        end = start + len(header) + (nxt.start() if nxt else len(text) - start - len(header))
        text = text[:start] + new + text[end:]
    path.write_text(text, encoding="utf-8")
    print(path.name, "ok", hashlib.sha256(text.encode()).hexdigest()[:12])


replace_sections(D / "zh_part4.md", ZH)
replace_sections(D / "en_part4.md", EN)
