
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

Two groups are compared. The first consists of the two stage outputs of one trained ReCoVP-Hybrid: the first-stage output $\widehat{\boldsymbol u}_1$ (Stage1) and the final output $\widehat{\boldsymbol u}$ (Final). They share the same parameters, so the comparison isolates the change caused by applying the refinement branch at inference. The second group consists of models of different structure:

- **ReCoVP-Base**: structured dictionary solve only, without the complex refinement branch;
- **P0-NoDechirp-Tail**: a convolutional residual correction network taking RMS-normalized I/Q as input and outputting the correction, without dechirping;
- **UNet-Tail**: a plain real-valued I/Q U-Net with widths 18/42/74 that outputs the correction;
- **RSDR-Net**: the reliability-guided complex restoration model previously proposed by the authors, about 52 times larger than the proposed method.

Except for RSDR-Net, the models have similar parameter counts, and P0-NoDechirp-Tail and UNet-Tail use the same waveform, pulse-compression, target-damage and tail losses as the proposed method. In addition, Section V-E compares the proposed method scene by scene with 12 families of convolutional restoration networks of similar size, each with three independently trained instances. [Table placeholder: results of traditional methods (e.g., time–frequency filtering [9] and Gaussian-mixture segment selection [12]) and representative learning methods (e.g., MREDNet [17] and CVEDNet [19]) under the same data and evaluation protocol.]

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

With the same nominal false-alarm setting, different processors produce different background distributions and different measured false-alarm rates, so comparing detection probabilities directly is unfair. [Table V placeholder: detection probability of each model and its paired difference from ReCoVP-Hybrid at jammed frame false-alarm rates of 0.05 and 0.10; thresholds are set on a separate calibration set and reported on the evaluation set.]

[Figure 7 placeholder: Jammed frame false-alarm rate versus detection probability for each model, obtained by sweeping the CFAR threshold factor.]

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

When base scenes are binned by their within-CPI mean SNR, the detection probability rises from 46.2% in [−20, −15) dB to 99.4% in [−5, 0) dB. When binned by mean JSR, the samples concentrate in [10, 20) dB (162 base scenes), and [30, 40) dB contains only 3. The evaluation set is randomly sampled and the other parameters are not fixed within a bin, so these results describe conditional associations rather than single-factor responses.

[Figure 9 placeholder: restoration error, target damage and detection probability of each model in controlled SNR and JSR sweeps with other parameters fixed, plotted separately for the three forwarding schemes.]

[Figure 10 placeholder: results as the sampling width and the number of jammed pulses vary.]

### K. Computational Cost

Measured with CUDA events on an NVIDIA GeForce RTX 5060 Ti after 5 warm-up runs and 40 timed runs, single-pulse inference has a median of 27.86 ms and a 95th percentile of 29.26 ms, and eight-pulse batched inference 29.28 ms and 30.66 ms. Batching eight pulses costs about the same as one pulse, which suggests that fixed launch and synchronization overheads account for a large share of the time. The timing covers only the forward pass and excludes data transfer and post-detection processing. [Table VIII placeholder: single-pulse and batched times of all models on the same hardware, with separate times for the first and second stages.]
