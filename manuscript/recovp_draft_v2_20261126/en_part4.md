
## V. Simulation Results

### A. Simulation Setup

Each CPI contains eight pulses. Target delay, amplitude and Doppler are constant within the CPI and the slow-time phase evolves according to (2); noise is generated independently for each pulse. The number of jammed pulses is drawn from $\{1,2,4,6,8\}$, and the sampling and forwarding parameters of each jammed pulse are drawn independently. Each base scene is generated in direct, repeated and cyclic forwarding variants that share the target and noise and differ only in the jammer. The main parameters are listed in Table II.

**TABLE II. Radar and Jammer Simulation Parameters**

|Parameter|Value|
|---|---|
|Sampling rate / bandwidth / pulse width|60 MHz / 30 MHz / 10 μs|
|Samples per pulse / template length|1800 / 600|
|Pulses per CPI / PRI|8 / 30 μs|
|Target amplitude / delay|0.15–2.0 / [64, 1136) samples|
|Target Doppler|−200 to 200 kHz|
|Forwarding schemes|direct, repeated, cyclic|
|Sampling width / period|6–72 / 36–260 samples|
|Forwarding delay / repetitions / maximum replay gap|1–96 samples / 1–10 / 64 samples|
|Jammer frequency offset|−200 to 200 kHz|
|Training set|512 base scenes: 256 regular (SNR [−20, 15] dB, JSR [−12, 35] dB) and 256 extreme (SNR [−30, −20) dB, JSR [−12, 45] dB)|
|Evaluation set|256 base scenes, SNR [−30, 15] dB, JSR [−12, 45] dB|

SNR is the ratio of the mean power over the effective target support to the noise power, and JSR the ratio of the mean power over the nonzero jammer support to the mean power over the effective target support. SNR is drawn per pulse and JSR per jammed pulse, so a base scene does not correspond to a single SNR and JSR. The slow-time sampling rate is about 33.33 kHz and large Doppler values alias; in evaluation they are folded into one PRF period and the nearest Doppler cell is used.

The evaluation set contains 256 base scenes and 768 records. Because the three forwarding variants share target and noise, the base scene is the independent unit: each base scene is first averaged with equal weights over its three variants and then averaged over the 256 base scenes. This set was used during method development, so the results in this section are development-set evaluations.

### B. Metrics and Detection Setup

Using the Frobenius norm over the eight-pulse matrix, restoration error, target-increment damage and jammer-free error are defined as

$$
E_U=\frac{\|\widehat U_{11}-(S+E)\|_F^2}{\|S+E\|_F^2},\quad
D_S=\frac{\|\widehat U_{11}-\widehat U_{01}-S\|_F^2}{\|S\|_F^2},\quad
E_0=\frac{\|\widehat U_{00}-E\|_F^2+\|\widehat U_{10}-(S+E)\|_F^2}{\|E\|_F^2+\|S+E\|_F^2}, \tag{34}
$$

where $\widehat U_{ab}$ is the processor output in state $ab$ and denominators are floored at $10^{-24}$. $E_0$ measures the change made to the two jammer-free inputs and is ideally zero. The magnitude of the target complex gain $|g|$ is computed by (9) on the eight-pulse matrix and is ideally one.

Pulse compression uses the 600-point template and keeps the valid delay cells 0–1200. With $P_v[k]$ the pulse-averaged range power and $k_\star$ the target cell given by the compressed peak of the jammer-free target, the signal-to-jammer ratio improvement factor is

$$
\mathrm{SJRIF}=10\log_{10}\frac{P_{\widehat U}[k_\star]}{\max_{k\neq k_\star}P_{\widehat U}[k]}-10\log_{10}\frac{P_{Y}[k_\star]}{\max_{k\neq k_\star}P_{Y}[k]}, \tag{35}
$$

where the non-target maximum includes sidelobes, residual jammer and noise.

Detection uses an eight-point slow-time FFT and two-dimensional CA-CFAR. The guard and reference half-widths are 12 and 24 in range and 1 and 2 in Doppler, with periodic boundaries in slow time, giving 436 reference cells and 9032 valid detection cells; the nominal cell false-alarm probability is $10^{-6}$. A target is detected if at least one detection occurs in the nearest aliased Doppler cell within the target range cell or one cell on either side. For the target-free jammed state two false-alarm measures are reported: the cell false-alarm rate, i.e., the fraction of valid cells declared, and the frame false-alarm rate, i.e., the fraction of CPIs with at least one false alarm. Processing changes the statistics of the background, so the measured false-alarm rate generally differs from the nominal value.

Confidence intervals are obtained by a base-scene bootstrap [54] with 4000 resamples, using the 2.5% and 97.5% quantiles of the mean; when two outputs are compared, paired differences are computed on the same resampling indices.

### C. Compared Methods

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

### D. Effect of the Refinement Stage

Table III compares the two stage outputs of the same trained model.

**TABLE III. First-Stage Output Versus Final Output** (probabilities in percent; probability differences in percentage points)

|Metric|Stage1|Final|Paired difference [95% CI]|
|---|---:|---:|---|
|$E_U$ ↓|0.40185|0.04312|−0.35873 [−0.46821, −0.26552]|
|$D_S$ ↓|0.12559|0.05967|−0.06591 [−0.13931, −0.01359]|
|SJRIF / dB ↑|10.4168|11.0968|+0.6801 [+0.5301, +0.8342]|
|Jammed frame false-alarm rate / % ↓|7.422|5.599|−1.823 [−3.125, −0.521]|
|Detection probability / % ↑|85.42|87.11|+1.693 [−0.130, +3.646]|
|Target gain magnitude $\lvert g\rvert$ (ideal 1)|0.98582|0.91518|−0.07064 [−0.07636, −0.06482]|
|Jammer-free error $E_0$ ↓|$2.551\times10^{-6}$|$4.566\times10^{-6}$|$+2.016\times10^{-6}$ [$1.248\times10^{-6}$, $3.265\times10^{-6}$]|
|Jammed cell false-alarm rate ↓|$5.781\times10^{-5}$|$6.026\times10^{-5}$|$+2.451\times10^{-6}$ [$-5.767\times10^{-6}$, $1.283\times10^{-5}$]|

Refinement reduces the mean restoration error from 0.402 to 0.043, about one tenth, roughly halves the mean target damage, raises SJRIF by 0.68 dB and lowers the jammed frame false-alarm rate by 1.8 percentage points; none of the four confidence intervals contains zero. This agrees with Proposition 2: shrinkage bias and representation residual account for most of the first-stage error, and the refinement branch compensates them effectively.

Refinement also has a measurable cost. The target gain magnitude falls from 0.986 to 0.915, i.e., the mean attenuation grows from about 1.4% to about 8.5%; the jammer-free error stays of order $10^{-6}$ in absolute terms, but its paired interval is positive. The paired difference of target damage has median +0.0079 and 95th percentile +0.0496, opposite in sign to the mean. A positive paired median means that the target damage increases in at least half of the scenes, and the decrease of the mean comes from large improvements in a minority. Together with (10), the simultaneous decrease of mean damage and mean gain magnitude admits one possible explanation: refinement strongly reduces the shape error in a few scenes while introducing slight amplitude shrinkage in many; confirming it requires a per-scene complex-gain decomposition (Fig. 6).

The point estimate of detection probability rises by 1.7 percentage points, but its interval contains zero; the interval for the jammed cell false-alarm rate also contains zero, unlike the frame rate. The frame rate measures whether any false alarm occurs, whereas the cell rate measures how many cells are false alarms; refinement reduces the CPIs with false alarms but not clearly the total number of false-alarm cells.

Stage1 is an intermediate output of a jointly trained model whose parameters are also shaped by the final loss. Table III thus answers what changes when the refinement branch is applied at inference, not whether the two-stage structure outperforms a single-stage structure with the same budget; the latter is addressed by the matched ablation in Section V-G.

[Figure 5 placeholder: Complex waveforms and pulse compression in typical scenes. One example each at the 10th, 50th and 90th percentile of the paired damage difference (a rule fixed in advance); for each example, the real parts of the observation, noisy reference, Stage1 output, Final output and refinement $\boldsymbol\delta_2$, and the corresponding compressed outputs, all on a common absolute power reference.]

[Figure 6 placeholder: Distribution of stage differences. (a) Histogram of paired restoration-error differences; (b) histogram of paired target-damage differences, with zero, mean and median marked; (c) per-scene change of complex-gain deviation $|g-1|^2$ and shape error $\|\boldsymbol q\|_2^2/\|\boldsymbol s\|_2^2$ (10) between the two stages.]

### E. Comparison With Other Models

Table IV lists the results of all models on the same evaluation set.

**TABLE IV. Comparison of Models** (point estimates)

|Model|Effective real parameters|$E_U$ ↓|$D_S$ ↓|$E_0$ ↓|$P_d$ / % ↑|Jammed frame FA / % ↓|SJRIF / dB ↑|
|---|---:|---:|---:|---:|---:|---:|---:|
|ReCoVP-Base|58,943|0.49665|0.07162|$1.94\times10^{-6}$|84.90|6.90|10.244|
|P0-NoDechirp-Tail|59,714|0.04866|0.04479|$9.19\times10^{-5}$|87.63|10.29|10.807|
|UNet-Tail|59,714|0.05449|0.05078|$1.17\times10^{-4}$|86.46|10.81|10.657|
|ReCoVP-Hybrid|57,525|0.04312|0.05967|$4.57\times10^{-6}$|87.11|5.60|11.097|
|RSDR-Net|3,017,877|0.03343|0.09470|$1.35\times10^{-5}$|86.98|5.08|11.338|

Adding the refinement branch to ReCoVP-Base reduces the restoration error from 0.497 to 0.043 and also lowers the target damage. The dictionary-only model changes clean inputs the least but has the largest restoration error, consistent with the analysis of shrinkage and representation residual in Proposition 2.

Compared with the two convolutional correction networks of similar size, ReCoVP-Hybrid has lower restoration error, jammer-free error and jammed frame false-alarm rate: the jammer-free error is more than an order of magnitude lower, the frame false-alarm rate about 5 percentage points lower and SJRIF 0.29–0.44 dB higher. Its target damage is higher, however, and its detection probability is close to that of P0-NoDechirp-Tail. The difference reflects two ways of correcting: a generic convolutional network regresses the correction directly, modifying clean inputs and targets more but producing a target increment closer to the true target, whereas ReCoVP-Hybrid confines most of the correction to the span of the dictionary, modifying the background less and leaving lower residual peaks, but shrinking the target when an atom overlaps it.

Compared with RSDR-Net, which has about 52 times more parameters, ReCoVP-Hybrid has lower target damage, a close frame false-alarm rate and SJRIF, and slightly higher restoration error. In other words, the low false-alarm level that the large model reaches with about three million parameters is approached here with 57.5 thousand (paired intervals for this pair remain to be added).

Table IV gives single-instance point estimates. To check whether the differences are due to training randomness, and to cover more convolutional restoration designs, ReCoVP-Hybrid is compared scene by scene with 12 families of convolutional restoration networks of similar size. They comprise the authors' earlier CorrectionNet, its retrained control, a version with the tail loss and four structural or loss ablations, as well as a plain U-Net and four variants with the tail loss, noise normalization, frequency coordinates and dilated convolutions. Each family is trained with random seeds 17, 29 and 41 for 256 epochs on the same data and number of updates as the proposed method, using the last-epoch parameters. For each scene the three instances are averaged, and the paired difference Hybrid minus family is computed. Table IV-b lists six representative families; full results are given in the supplementary material.

**TABLE IV-b. Per-Scene Paired Comparison With Convolutional Restoration Networks** (baselines averaged over three trained instances; in parentheses, the paired difference Hybrid minus baseline and its 95% CI; probability differences in percentage points)

|Model|$E_U$ ↓|$D_S$ ↓|$E_0$ ↓|Jammed frame FA / % ↓|SJRIF / dB ↑|$P_d$ / % ↑|
|---|---|---|---|---|---|---|
|ReCoVP-Hybrid|0.0431|0.0597|$4.6\times10^{-6}$|5.60|11.10|87.11|
|CorrectionNet|0.0599 (−0.0168 [−0.0208, −0.0128])|0.0634 (−0.0037 [−0.0100, +0.0047])|$1.25\times10^{-4}$|15.32 (−9.72 [−12.0, −7.64])|10.30 (+0.80 [+0.64, +0.97])|85.98 (+1.13 [+0.26, +2.04])|
|CorrectionNet + tail|0.0582 (−0.0151 [−0.0190, −0.0113])|0.0636 (−0.0039 [−0.0102, +0.0043])|$1.28\times10^{-4}$|14.54 (−8.94 [−11.1, −6.90])|10.40 (+0.69 [+0.54, +0.85])|86.46 (+0.65 [−0.17, +1.56])|
|CorrectionNet w/o dechirp|0.0509 (−0.0077 [−0.0109, −0.0045])|0.0452 (+0.0144 [+0.0083, +0.0226])|$1.23\times10^{-4}$|10.85 (−5.25 [−7.03, −3.52])|10.70 (+0.40 [+0.28, +0.53])|87.33 (−0.22 [−1.22, +0.74])|
|U-Net|0.0545 (−0.0114 [−0.0148, −0.0080])|0.0481 (+0.0115 [+0.0056, +0.0198])|$7.6\times10^{-5}$|11.11 (−5.51 [−7.34, −3.82])|10.61 (+0.49 [+0.37, +0.62])|86.81 (+0.30 [−0.56, +1.13])|
|U-Net + tail|0.0536 (−0.0105 [−0.0139, −0.0072])|0.0488 (+0.0109 [+0.0045, +0.0187])|$8.6\times10^{-5}$|10.63 (−5.04 [−6.81, −3.30])|10.67 (+0.43 [+0.31, +0.55])|86.63 (+0.48 [−0.39, +1.39])|
|U-Net + dilation|0.0571 (−0.0139 [−0.0176, −0.0104])|0.0505 (+0.0092 [+0.0031, +0.0171])|$7.5\times10^{-5}$|11.46 (−5.86 [−7.68, −4.17])|10.58 (+0.52 [+0.39, +0.65])|87.11 (0.00 [−1.00, +0.96])|

The paired intervals for $E_0$ all exclude zero and are omitted for space. Across all 12 families, Hybrid has lower restoration error, lower jammer-free modification and a lower jammed frame false-alarm rate, and a higher SJRIF; none of the paired intervals for these four metrics contains zero. The most stable difference is the jammer-free modification: Hybrid is 10 to 65 times lower than the baseline families and lower in at least 99% of the scenes. The jammed frame false-alarm rate of the baselines ranges from 9.1% to 16.5%, against 5.6% for Hybrid, about half that of the U-Net families of the same size and one third that of the CorrectionNet families. The standard deviation of the baselines' frame false-alarm rate across trained instances is only 0.3 to 0.8 percentage points, far smaller than this gap, so the gap is not caused by baseline training fluctuations; the training fluctuation of Hybrid itself will be given in Table VII.

The cost appears on the target side. Compared with the U-Net families and CorrectionNet without dechirping, Hybrid's target damage is about 0.01 higher, with intervals excluding zero; against the full CorrectionNet families there is no significant difference. Ten of the twelve families preserve the target amplitude better than Hybrid. Detection probability is significantly higher only against five families (CorrectionNet, its retrained control, the version without the target-damage loss, noise normalization and frequency coordinates); for the other seven the intervals contain zero. In other words, at the nominal CFAR setting Hybrid achieves a detection probability no lower than the baselines with about half the jammed-frame false alarms; the corresponding comparison at equal measured false-alarm rate is given in Section V-F.

These results agree with the analysis of Section III. A convolutional network regresses a $T$-sample correction directly and can keep its change to clean inputs small only through the training loss; the bulk of Hybrid's correction is bounded by the degrees of freedom of Corollary 1 and clean inputs remain nearly unchanged. False alarms in jammed backgrounds are mainly caused by residual components that are still coherent with the transmitted waveform and refocus into peaks after pulse compression; the first-stage atoms are exactly windowed delayed LFMs whose coefficients are aligned by least squares, so these coherent components are removed first. The higher target damage is the other side of the same mechanism: the target is also a delayed LFM, and when an atom coincides with the target delay the leakage term in (20) shrinks the target.

[Supplementary table placeholder: three-instance means, standard deviations, paired differences and intervals of all metrics for the 12 baseline families; per-scene paired results for ReCoVP-Base, P0-NoDechirp-Tail and RSDR-Net.]

[Figure placeholder: distribution of the ratio of compressed residual peak to residual energy (residual coherence) for each model in the target-free jammed state, to test the mechanism described above.]

### F. Detection at Equal Measured False-Alarm Rate

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

### G. Ablation Study

[Table VI placeholder: matched ablation. Each row is trained with the same data, losses and number of updates: (1) first stage only (same structure as the first stage of Hybrid, trained alone to convergence); (2) complex refinement network only (input: the observation); (3) refinement without the $\widehat{\boldsymbol j}_1$ input; (4) without the phase-invariant real gates (component-wise nonlinearity instead); (5) without the pulse-level scalar gate ($\alpha=1$); (6) without the target-increment loss; (7) without the tail loss; (8) full model. Columns: effective parameters, $E_U$, $D_S$, $|g|$, $E_0$, jammed frame false-alarm rate and detection probability.]

The existing structural ablation of the ReCoVP-Base family shows that removing the complex-scaling-equivariant structure changes the detection probability by about −1.04 percentage points (interval about [−2.08, 0.00]) and the target damage by −0.0021 (interval about [−0.0084, +0.0043]); neither shows a stable direction.

### H. Training Randomness

[Table VII placeholder: results of eight trained instances (random seeds 17, 29, 41, 53, 67, 79, 97 and 109) on the evaluation set. Columns: $E_U$, $D_S$, $|g|$, jammed frame false-alarm rate, detection probability and SJRIF; the last two rows give the mean and standard deviation across instances.]

[Figure 8 placeholder: Training-loss curves of the eight instances, and $E_U$ and $D_S$ on the evaluation set every 32 epochs.]

### I. Gate Behavior and Numerical Equivariance

The mean of the scalar gate $\alpha$ in the states $\boldsymbol y_{00},\boldsymbol y_{10},\boldsymbol y_{01},\boldsymbol y_{11}$ is 0.00045, 0.00053, 0.02481 and 0.02301, respectively. Jammer-free inputs hardly use the refinement proposal, while jammed inputs use about two orders of magnitude more, so the gate distinguishes the two kinds of input. Because $\alpha$ is only the applied fraction, the actual correction also depends on the magnitude of the proposal $\widetilde{\boldsymbol\delta}$.

To measure the deviation from Proposition 3 in the implementation, a random complex factor (magnitude in $[0.2,5.0]$, phase in $[-\pi,\pi]$) is drawn for each base scene, and the output for the scaled input is compared with the original output multiplied by the same factor. In a small check the maximum relative error is $2.54\times10^{-7}$ and permuting candidate columns gives zero error; over the full evaluation set the mean relative error is $2.33\times10^{-5}$ and its 95th percentile $8.72\times10^{-5}$, while the aggregate detection and false-alarm rates are unchanged. The larger error over the full set may come from occasional candidate switches caused by single-precision rounding of the correlation magnitude.

### J. Condition Analysis

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

### K. Computational Cost

Measured with CUDA events on an NVIDIA GeForce RTX 5060 Ti after 5 warm-up runs and 40 timed runs, single-pulse inference has a median of 27.86 ms and a 95th percentile of 29.26 ms, and eight-pulse batched inference 29.28 ms and 30.66 ms. Batching eight pulses costs about the same as one pulse, which suggests that fixed launch and synchronization overheads account for a large share of the time. The timing covers only the forward pass and excludes data transfer and post-detection processing. [Table VIII placeholder: single-pulse and batched times of all models on the same hardware, with separate times for the first and second stages.]
