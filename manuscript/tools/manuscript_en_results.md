## 4 Experimental design and evaluation

### 4.1 Data generation and data roles

The radar settings and scan paths follow the earlier study [15], while the actual data manifests define the splits used here. The sampling frequency is 60 MHz, bandwidth 30 MHz, pulse width 10 μs and pulse repetition interval 30 μs. Each record contains eight coherent pulses of 1800 samples. Training contains one true target, with amplitude drawn from \([0.15,2]\), initial phase from \([0,2\pi)\), and Doppler from \([-200,200]\) kHz. Fast- and slow-time phases are retained. The broad Doppler sampling is scored modulo the PRF rather than treated as unambiguous velocity coverage.

There are 512 independent training parents. Half use SNR in \([-20,15]\) dB and JSR in \([-12,35]\) dB; the low-SNR half use \([-30,-20)\) dB and \([-12,45]\) dB, respectively. Each parent has direct, repeated and cyclic variants, yielding 1536 pulse-group records. The number of jammed pulses is selected from 1, 2, 4, 6 and 8. SNR is drawn per pulse, and jammer intensity and structure are drawn for the jammed pulses. The four states share the realized components.

DEV consists of another 256 parents with SNR drawn over \([-30,15]\) dB and JSR over \([-12,45]\) dB. It does not reproduce the training stratification quotas. TEST contains separate base scenes evaluated at the fixed conditions in Table 3. DEV is used for model selection and convergence analysis; TEST is used for the frozen-model results. Their averages are not pooled.

**Table 2. Training and optimization settings.**

| Item | Setting |
|---|---|
| Radar | \(f_s=60\) MHz; \(B=30\) MHz; \(T_p=10\) μs; \(T_r=30\) μs |
| Record | \(P=8\), \(N=1800\); four paired states |
| Sampling width | 6–72 samples |
| Sampling period | Lower bound \(\max(36,w+1)\); upper bound 260 samples |
| Replay delay, repetition count, additional gap | 1–96 samples; 1–10; 0–64 samples |
| Additional frequency shift | \([-200,200]\) kHz, with pulse-dependent phase variation |
| Training records | 512 parents × 3 modes; 8 pulses per record |
| Batch and updates | 32 records; 48 updates per epoch |
| Optimizer | Adam [21]; \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\) |
| Learning rate | First-epoch warmup to \(10^{-3}\); at epoch 128, cosine decay reaches \(2\times10^{-5}\); at epoch 256, the extension reaches \(2\times10^{-6}\) |
| Regularization | Weight decay 0; gradient-norm limit 5 |
| Seeds | 17, 29, 41, 53, 67, 79, 97, 109 |
| Retention | Every-epoch recovery; separate milestones every 32 epochs, 64 milestones in total |

### 4.2 Frozen model and comparison methods

The initial three seeds, 17, 29 and 41, were trained for 128 epochs under the same schedule, retaining epochs 32, 64, 96 and 128. These are intermediate checkpoints on one trajectory, not four separately optimized training durations. Epoch 128 was selected from the three-seed mean DEV waveform NMSE, and the fixed middle-seed rule selected seed 29 before TEST execution. This selection followed inspection of DEV; it was not a model choice made before seeing DEV.

The three runs were then extended to 256 epochs and five additional seeds followed the same two-stage schedule. All eight are evaluated on the same DEV every 32 epochs. The extension does not replace the epoch-128 TEST identity. A final model selected using these results requires a fresh confirmation set.

Comparisons include the identity output (KEEP), the same-noise jammer-free reference \(S+E\), Max-TF [2], GMM-OMP [3], BCS with oracle support [4], and frozen RSDR-Net [15]. The \(S+E\) reference requires inaccessible components and is not a deployable algorithm. The classical methods are local formula-based reproductions with frozen parameters, not the original authors' executables or TEST-tuned variants. Max-TF includes local choices for discrete smoothing and ties. GMM-OMP uses a two-component amplitude EM fit, at most 100 EM iterations and eight OMP iterations; its variance update uses the standard responsibility-weighted form, correcting an apparently omitted factor in the printed expression. Such choices can affect the comparison.

BCS uses a smoothed \(L_1\) MAP objective, outer reweighting and inner complex conjugate gradients. It receives the true unjammed support and is always labeled **BCS (oracle support)**. It is an extra-information reference, not an observation-only competitor. Systematic DEV calibration of the classical baselines remains incomplete.

The RSDR-Net comparator is the complete F-R7 system, with training seed 40312102 and a final-stage epoch-32 checkpoint. Its multistage components used 32 epochs per stage and 512 updates per epoch, whereas CorrectionNet uses 48 updates per epoch. Testing inputs, downstream processing and metrics are unified; training budgets are not. The comparison is therefore between frozen systems, not an isolated architectural ablation. RSDR-Net inference was added after the original TEST results and did not trigger retraining or checkpoint replacement.

### 4.3 Test grids and statistical units

**Table 3. Completed test suites. Conditions within a suite share parents and are not additional independent samples.**

| Suite | Conditions | Number of conditions | Independent parents |
|---|---|---:|---:|
| Three-mode main grid | SNR: −30, −28, −26, −25, −24, −23, −22, −20, −15, −10, −5, 0, 5, 10 dB; JSR: 0, 10, 20, 35 dB; three modes | 168 | 256 |
| Dense detection grid | SNR −30 to 0 dB, 1-dB steps; JSR −10 to 35 dB, 5-dB steps | 310 | 512 |
| Unjammed retention | SNR −30 to −20 dB, 1-dB steps, plus −15, −10, 0 dB; three scene groups | 42 | 1536, 512 per group |
| Width scan A | Period 260 samples, \(M=2\); widths 3, 6, 18, 36, 54, 72, 84; three intensity settings | 21 | 256 |
| Width scan B | \(M=2,4\); widths 6, 18, 36, 54, 72; period \((M+1)w+1\) | 20 | 128 |
| Coherent processing | (SNR, JSR) = (−20,20), (0,20), (0,35) dB; target Doppler within ±14 kHz | 3 | 16 |

Scan A uses \((\mathrm{SNR},\mathrm{JSR}_E)=(-10,20),(-20,35),(0,10)\) dB, and scan B uses the first two. Here \(\mathrm{JSR}_E\) is the jammer-to-target energy ratio over the whole pulse group. Main-grid JSR uses effective-support mean power. Changing support occupancy changes the relationship, so width-scan values are not pooled with main-grid means. Widths 3 and 84 are outside the training interval; this limited check does not establish generalization to arbitrary jamming.

Metrics are first averaged equally across specified conditions within each parent, and then across parents. Method differences use paired parent values and 4000 parent-bootstrap resamples for pointwise 95% intervals [22]. The three unjammed groups are resampled within strata. Pulses, states, modes and repeated conditions are not independent replicates. The intervals are not multiplicity-adjusted and do not include training-seed uncertainty. Convergence shading is the sample standard deviation across training seeds, a separate source of variation.

### 4.4 Waveform and target-increment metrics

With \(\widehat U_{ab}=T_\theta(Y_{ab})\), full-group errors are

{{EQ35}}

The implementation floors tiny denominators at \(10^{-24}\). The first metric measures total deviation from the same-noise reference; the second measures the change in target response at a fixed background. For \(V=\widehat U_{11}-\widehat U_{01}\), define

{{EQ36}}

Orthogonality gives

{{EQ37}}

Thus \(D_S\) combines complex gain error with deformation that a scalar gain cannot explain. It does not measure all interference remaining in \(\widehat U_{01}\): KEEP has \(D_S=0\) while retaining the jammer. Unjammed retention is measured by

{{EQ38}}

which includes both noise-only and target-plus-noise states.

### 4.5 Pulse compression, SJRIF and detection

Evaluation uses a 600-sample matched template and retains valid lags 0–1200, giving 1201 range cells. For compressed pulse \(Z_p(r)\), the average power is

{{EQ39}}

The target-only compressed signal determines \(r_t\). Following the existing experiment implementation,

{{EQ40}}

The numerical floor is defined in Section 2.7. The denominator includes adjacent mainlobe samples, sidelobes, noise and residual jamming. SJRIF here is therefore an improvement in a target-to-strongest-other-cell ratio, not a true-component physical signal-to-jammer improvement. At twice-bandwidth sampling, the ideal nearest mainlobe sample is only approximately 3.92 dB below the peak, and can dominate the denominator once interference is suppressed. A separate maximum false-peak suppression measure outside the mainlobe is retained in the evidence, but (40) is not treated as residual-jammer energy.

An eight-point slow-time FFT produces the range–Doppler power map. Two-dimensional CA-CFAR uses guard/reference half-widths of 12/24 in range and 1/2 in Doppler, with periodic Doppler boundaries. The number of reference cells is

{{EQ41}}

There are 9032 valid cells under test per frame. At nominal cell false-alarm probability \(P_{\rm FA}^{\rm nom}=10^{-6}\), the reference mean is multiplied by

{{EQ42}}

This coefficient assumes independent identically distributed exponential background powers. Matched filtering and nonlinear restoration can change both dependence and distribution, so empirical false alarms are measured separately. Detection is successful when at least one range cell within \(r_t\pm1\), at the nearest periodic Doppler bin, exceeds threshold. \(P_D\) is the frequency of this event over parents.

Cell and frame false-alarm rates are evaluated on \(E\) and \(J+E\). A frame alarm means at least one valid cell exceeds threshold. All methods share the nominal CFAR rule, but their empirical false alarms differ; this is not a matched-empirical-false-alarm ROC comparison.

### 4.6 Coherent-processing metrics

At the true target range, let \(z_p^S\) and \(z_p\) be the target-only and processed slow-time sequences. Define \(\chi_p=z_p(z_p^S)^*/|z_p^S|^2\) and remove the common phase \(\phi_0=\arg\sum_p\chi_p\). Then

{{EQ43}}

Phase RMSE is reported in degrees. \(G_{\rm coh}\) is a coherent power ratio relative to the target-only signal; noise can make it positive without implying a true target-energy gain. Peak frequency error uses 4096-point zero-padded spectra of the eight-pulse sequences and periodic frequency distance. Zero padding interpolates the spectrum without increasing physical Doppler resolution.

## 5 Results

### 5.1 Eight-seed training and convergence

All eight seeds completed 256 epochs, yielding 2048 epoch records and 64 DEV milestone evaluations. The training objective falls rapidly early and improves more slowly later (Fig. 2). Between epochs 32 and 128, mean DEV NMSE decreases from 0.08268 to 0.06044 and target-increment distortion from 0.09991 to 0.06465. From 128 to 256, NMSE decreases to 0.05944 while detection changes from 86.003% to 86.100%. The evidence indicates diminishing gains, rather than strictly monotone improvement or no further learning after epoch 128.

![Eight-seed training and DEV convergence](figures/fig02_convergence.png)

**Fig. 2.** Training objective and DEV metrics over eight seeds. Lines and shading show the mean and sample standard deviation across seeds. The training objective uses a logarithmic axis; DEV is evaluated every 32 epochs. The vertical line marks the schedule boundary at epoch 128. All seeds use the same DEV, so shading is not a population confidence interval for new scenes.

**Table 4. Mean DEV performance across eight seeds. Probabilities are percentages.**

| Epoch | \(\mathrm{NMSE}_U\) ↓ | \(D_S\) ↓ | SJRIF/dB ↑ | \(P_D\)/% ↑ | \(J+E\) frame alarms/% ↓ |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.082677 | 0.099914 | 9.5507 | 83.936 | 21.484 |
| 64 | 0.068933 | 0.079982 | 9.9558 | 85.335 | 18.180 |
| 96 | 0.062933 | 0.067263 | 10.1948 | 85.872 | 16.309 |
| 128 | 0.060439 | 0.064651 | 10.2786 | 86.003 | 15.283 |
| 160 | 0.060170 | 0.063376 | 10.2951 | 86.035 | 15.332 |
| 192 | 0.059729 | 0.062455 | 10.3120 | 86.084 | 15.153 |
| 224 | 0.059637 | 0.061939 | 10.3140 | 86.084 | 15.104 |
| 256 | 0.059440 | 0.061786 | 10.3221 | 86.100 | 15.120 |

At epoch 256, the seed standard deviations are 0.00165 for NMSE, 0.00395 for \(D_S\), 0.0726 dB for SJRIF and 0.285 percentage points for detection. This describes initialization variation on a fixed DEV, not eight independent generalization studies. DEV and the main grid have different condition distributions, preventing direct interpretation of their mean SJRIF or detection differences as training gains.

### 5.2 Frozen main-grid comparison with classical methods

Table 5 averages the 168 conditions equally within parents. Relative to KEEP, CorrectionNet reduces same-noise waveform error, raises SJRIF by 3.65 dB and increases detection from 59.20% to 67.07%. Relative to the local Max-TF reproduction, the paired SJRIF difference is 1.572 dB (95% interval 1.496–1.651 dB), and detection increases by 4.211 percentage points (3.764–4.653). Jammed-background frame alarms also decrease, but remain 20.42%, well above the jammer-free reference.

**Table 5. Main-grid results across three jamming modes. Reference rows are not ranked as deployable algorithms.**

| Method | \(\mathrm{NMSE}_U\) ↓ | \(D_S\) ↓ | SJRIF/dB ↑ | \(P_D\)/% ↑ | \(J+E\) frame alarms/% ↓ |
|---|---:|---:|---:|---:|---:|
| KEEP | 30.32106 | 0 | 0 | 59.203 | 34.924 |
| Max-TF, local reproduction | 1.12729 | 4.56331 | 2.0819 | 62.858 | 33.963 |
| GMM-OMP, local reproduction | 1.42492 | 8.99357 | −9.1752 | 39.651 | 99.923 |
| BCS (oracle support) | 0.10696 | 0.18506 | 4.6260 | 62.714 | 0.911 |
| CorrectionNet, epoch 128 | 0.08741 | 0.04731 | 3.6540 | 67.069 | 20.424 |
| \(S+E\), inaccessible reference | 0 | 0 | 5.1026 | 69.420 | 0.391 |

GMM-OMP produces very high jammed-background frame alarms in this implementation and fixed configuration. Different denoising targets can affect its same-noise NMSE, but do not by themselves explain a frame alarm rate near 100%. The result requires independent DEV calibration and failure-case reproduction before attributing it to the method class. BCS uses additional support information and has stronger SJRIF and lower background alarms than CorrectionNet, but lower detection. Neither observation establishes an oracle performance limit or identifies a single causal mechanism.

![Classical-method comparison for repeated retransmission at JSR 20 dB](figures/fig03_repeat_jsr20.png)

**Fig. 3.** Detection, jammed-background frame alarms, SJRIF and target-increment distortion versus SNR for repeated retransmission at JSR 20 dB. Each condition uses 256 paired parents. Lines join tested points. The distortion axis uses a symmetric logarithmic scale with a linear region around zero to display KEEP and larger errors together. Oracle and inaccessible references do not have the same input information as deployable methods.

### 5.3 Unified comparison with RSDR-Net

Under identical test inputs and processing, CorrectionNet has lower main-grid NMSE and target-increment distortion, with paired difference intervals excluding zero. The same comparison shows lower SJRIF and detection, and substantially more jammed-background alarms (Table 6). Simplification preserves useful waveform recovery but loses some control of residual interference.

**Table 6. Unified main-grid comparison over 256 independent parents. Differences are CorrectionNet minus RSDR-Net; brackets are paired pointwise 95% bootstrap intervals.**

| Metric | CorrectionNet | RSDR-Net | Difference [interval] |
|---|---:|---:|---|
| \(\mathrm{NMSE}_U\) ↓ | 0.087409 | 0.102302 | −0.014892 [−0.021168, −0.008966] |
| \(D_S\) ↓ | 0.047309 | 0.071769 | −0.024459 [−0.026527, −0.022480] |
| SJRIF/dB ↑ | 3.654024 | 4.736024 | −1.082001 [−1.150363, −1.011990] |
| \(P_D\)/% ↑ | 67.0689 | 67.9688 | −0.8998 [−1.0579, −0.7510] percentage points |
| \(J+E\) frame alarms/% ↓ | 20.4241 | 7.0754 | +13.3487 [12.0443, 14.6508] percentage points |

The waveform-error ranking is condition dependent. On the dense 310-condition grid, NMSE is 0.10515 for CorrectionNet and 0.07551 for RSDR-Net, reversing the main-grid order. Target-increment distortion remains lower, at 0.05294 versus 0.08844. Detection is 78.729% versus 79.170%, and jammed-background frame alarms are 21.326% versus 4.817%. Lower target-increment distortion is therefore more consistent across the two grids than lower total waveform error.

![Dense-grid slice comparing frozen models](figures/fig04_dense_jsr20.png)

**Fig. 4.** Dense-grid slice at JSR 20 dB with 1-dB SNR steps and 512 paired parents per condition. The four panels show how lower target-increment distortion can coexist with more jammed-background alarms. This slice does not replace the full-grid summary.

The complete maps in Fig. 5 locate detection differences mainly around the low-SNR transition and stronger jamming, while background-alarm differences extend more broadly. The maps are descriptive condition means and paired point differences, without cellwise significance claims.

![Complete SNR–JSR detection and false-alarm maps](figures/fig08_full_detection_grid.png)

**Fig. 5.** All 310 dense-grid conditions. The upper row shows detection probability and the lower row shows target-absent jammed-background frame alarms. Columns show CorrectionNet, RSDR-Net and their difference in percentage points. There are 512 paired parents per condition, with SNR/JSR steps of 1/5 dB. Absolute panels share a 0–100% scale; difference panels use symmetric limits separately for each metric. No smoothing or interpolation is applied. Thresholds follow the common nominal CFAR rule rather than matched empirical false alarms.

Because training, capacity and architecture differ together, this comparison does not identify which RSDR-Net branch accounts for its stronger suppression. The parameter-matched development comparison in Section 5.7 provides a separate check on the compact backbone.

### 5.4 Pulse compression and coherent processing

Figures 6 and 7 use the fixed parent 0 at SNR 0 dB and JSR 20 dB, rather than selecting a favourable output. Both networks preserve the principal response near the target, but amplitudes differ from the target-only reference and some CorrectionNet residual peaks are higher. The plotted target-only reference is \(S\), whereas training uses \(S+E\). Differences from the plot reference cannot all be assigned to network error, and peak-position retention is not distortion-free restoration.

![Fixed-scene pulse compression, Doppler and phase](figures/fig05_fixed_parent_pc_doppler.png)

**Fig. 6.** Pulse compression and coherent processing for fixed parent 0 at SNR 0 dB and JSR 20 dB. “Paper2” denotes frozen CorrectionNet at epoch 128; “RSDR” denotes the complete earlier system. The target-only reference and visible amplitude changes should be interpreted with the multi-scene measurements in Table 7.

![Fixed-scene range–Doppler maps](figures/fig06_fixed_parent_rd.png)

**Fig. 7.** Range–Doppler outputs for the same scene as Fig. 6. This single example illustrates spatial structure; detection and false-alarm conclusions use the paired statistics.

Across the 16-parent coherent check, at SNR 0 dB and JSR 35 dB, CorrectionNet reduces phase RMSE from 23.26° to 6.40° and peak frequency error from 317.38 to 69.68 Hz, while detection rises from 37.5% to 100%. The coherent power ratio is −3.79 dB, indicating substantial attenuation alongside that improvement. At the low-SNR condition (−20,20) dB, phase and frequency errors do not improve appreciably.

**Table 7. Coherent-processing means over 16 independent parents. Power ratios use the target-only reference. Detection estimates have limited sample size.**

| SNR/JSR, dB | Processing | Phase RMSE/° ↓ | Coherent power ratio/dB | Peak frequency error/Hz ↓ | \(P_D\)/% |
|---|---|---:|---:|---:|---:|
| −20/20 | Input | 16.429 | +0.488 | 116.984 | 93.75 |
| −20/20 | CorrectionNet | 16.460 | +0.322 | 120.036 | 93.75 |
| 0/20 | Input | 5.075 | +0.098 | 65.104 | 100 |
| 0/20 | CorrectionNet | 4.362 | −2.896 | 44.759 | 100 |
| 0/35 | Input | 23.260 | +0.496 | 317.383 | 37.5 |
| 0/35 | CorrectionNet | 6.402 | −3.795 | 69.682 | 100 |

Equally averaging the three conditions gives phase RMSEs of 9.075° and 8.955° for the new and old models, with a difference interval crossing zero. Mean peak frequency errors are 78.159 and 101.217 Hz: the paired difference is −23.058 Hz, with interval [−38.825, −8.647] Hz. This result is limited to the tested scenes and does not establish reliable velocity superiority in complex measured environments.

### 5.5 Unjammed retention and sampling-width sensitivity

The combined unjammed retention NMSE is \(1.6800\times10^{-4}\) for CorrectionNet and \(1.9396\times10^{-5}\) for RSDR-Net. Both are small, but the compact model modifies the input more. Unjammed detection is 55.6641% versus 55.6687%, with a difference interval crossing zero; noise-only cell false alarms are approximately \(1.442\times10^{-6}\), and frame alarms are 1.302% for both. These are averages over low-SNR conditions, not high-SNR detection ceilings. No equivalence margin was prespecified, so failure to resolve a detection difference is not proof of equivalence.

In fixed-period width scan A, the new and old NMSEs are 0.06744 and 0.11627, with nearly identical detection of 98.419% and 98.400%. Jammed-background frame alarms remain 16.592% and 2.958%. In scan B, where period varies with width, CorrectionNet NMSE rises to 0.30403 versus 0.10241 for RSDR-Net. Detection is 84.102% versus 92.344%, a difference of −8.242 percentage points with interval [−9.883, −6.563]. Target-increment distortion is still lower, 0.16088 versus 0.25874, without producing better detection.

![Width sensitivity at fixed repetition counts](figures/fig07_width_snr_minus20_jsre35.png)

**Fig. 8.** Scan B at SNR −20 dB and \(\mathrm{JSR}_E=35\) dB, with separate columns for \(M=2\) and \(M=4\). Each condition uses 128 paired parents. The period is \((M+1)w+1\); changing the horizontal coordinate changes both width and period, so it is not a width-only intervention.

These results locate a failure boundary of the simplified model. Preserving the target increment does not compensate for residual interference when fragment structure changes.

### 5.6 Why lower target-increment error can coexist with more alarms

Equations (16)–(17) measure the change in correction when a target is added. A residual retained similarly in the two states can cancel in that finite difference while producing false peaks in the \(J+E\) output. NMSE also averages energy over a pulse group: a few concentrated residuals and a diffuse error can have similar total energy but different CFAR consequences. SJRIF introduces another dependence on the strongest non-target cell and adjacent mainlobe samples. Different metric rankings therefore need not be contradictory.

The data support lower target-increment distortion in several scans and weaker residual-interference control than the complete RSDR-Net. They do not identify whether loss weights, backbone capacity or missing reliability mechanisms cause that tradeoff.

### 5.7 Parameter-matched U-Net development comparison

A conventional one-dimensional I/Q U-Net [16] has exactly 59,714 active parameters. Its channel widths are 18, 42 and 74, with two length-3 convolutions, GroupNorm and ReLU per scale, max-pooling downsampling, linear-interpolation upsampling and concatenated skips. It uses neither fixed dechirp nor local residual blocks. Widths were selected by parameter count, without using validation performance. RMS normalization, correction output, four-state supervision and all three losses are retained. Data, seeds 17/29/41 and the 256-epoch schedule match CorrectionNet. Several architectural choices change together, so this is a capacity-matched backbone comparison rather than a single-factor ablation.

All three U-Net runs completed training. Table 8 uses the same three seeds and epoch for both models; it does not compare an eight-seed CorrectionNet average with a three-seed U-Net average. DEV data hashes and shared evaluation-source hashes were checked. Values are means ± sample standard deviations across seeds on the same 256-parent DEV. They are descriptive development results, not independent confirmation or significance tests.

**Table 8. Same-seed, epoch-256 DEV comparison. Standard deviations of percentage metrics are in percentage points.**

| Model | Active parameters | \(\mathrm{NMSE}_U\) ↓ | \(D_S\) ↓ | SJRIF/dB ↑ | \(P_D\)/% ↑ | \(J+E\) frame alarms/% ↓ |
|---|---:|---:|---:|---:|---:|---:|
| CorrectionNet | 59,714 | 0.059888 ± 0.001727 | 0.063397 ± 0.005935 | 10.3013 ± 0.0842 | 85.981 ± 0.457 | 15.321 ± 0.601 |
| U-Net | 59,714 | 0.054511 ± 0.001022 | 0.048145 ± 0.001424 | 10.6052 ± 0.0266 | 86.806 ± 0.398 | 11.111 ± 0.328 |

U-Net performs better on all five aggregate metrics. This does not negate the feasibility of compact correction, but provides no evidence of an advantage for the dechirped residual backbone. Since DEV has been used repeatedly, the differences require the planned fresh confirmation. Switching model choice after this comparison would not make the existing TEST unseen again. The four structural/loss ablations are incomplete, so the result cannot attribute the difference to dechirp, normalization, residual blocks or optimization individually.

## 6 Discussion

### 6.1 Contribution in relation to prior methods

Unlike optimization-unfolding approaches [10–12], CorrectionNet does not map each layer to an iterative solver step. Its interpretable elements are the correction task, invertible phase rotation and paired finite difference. Unlike HRRP-domain structured filtering [13], it processes raw I/Q waveforms and feeds conventional pulse-compression and Doppler processing, while constraining the response of the full correction. Relative to RSDR-Net, it retains the task and part of the supervision but removes the separate reliability, expert and gating roles.

The contribution is consequently a compact implementation and an evaluation of its cost, rather than a new four-state supervision principle. The finite-difference formulation makes target response measurable; the unified tests show that this response can improve while jammed-background alarms worsen. The same-capacity result further limits a backbone-superiority claim. A stronger method contribution would require a reproducible improvement over that baseline or a clearly demonstrated efficiency–accuracy operating point.

### 6.2 What the current evidence establishes

Without oracle support, a single correction path improves the main-grid SJRIF and detection relative to KEEP and the local Max-TF configuration. Compared with complete RSDR-Net, insufficient background suppression recurs across scans. The eight-seed curves show only modest DEV improvement after epoch 128, but do not establish whether an epoch-256 model closes that gap on independent data.

All conclusions remain conditional on the generator, frozen checkpoint and processing chain. Three modes and four states derived from 512 training parents do not provide the same diversity as that many independent scenes. Repeated development choices can become dependent on the fixed DEV. Fresh confirmation must evaluate the eventual frozen choice rather than provide another opportunity to select favourable conditions.

### 6.3 Limitations and required validation

The matched-capacity comparison is the main limitation of the current method claim: conventional U-Net already performs better on the same DEV. Completed ablations of dechirp, RMS normalization, pulse-compression supervision and target-increment supervision, followed by fresh confirmation, are needed for structural attribution. More epochs on the same DEV are not a substitute.

Detection calibration is another limitation. Empirical false alarms are reported, but thresholds have not been calibrated on independent data to match actual false-alarm operating points. A detection comparison should select thresholds on calibration samples and evaluate both detection and realized false alarms on a separate frozen set. This prevents a more permissive output background from being mistaken for better detection capability.

All current results use single-target simulations. There are no measured radar echoes, hardware retransmission tests or real-clutter validation for this model. Some prior studies use measured data [10,11,13], but measured echoes with injected jamming must be distinguished from hardware-generated retransmission. Multiple targets, closely spaced targets, waveform mismatch, non-Gaussian clutter and unseen jammer combinations remain untested. Classical baselines also require DEV calibration and failure review; the coherent check has only 16 parents; and parameter savings lack same-hardware latency, memory and operation-count measurements. These limitations bound the current conclusions.

## 7 Conclusion

This work studies a compact single-corrector network for ISRJ waveform restoration. Under a same-noise reference, RMS normalization, fixed dechirp, a three-scale residual backbone and input subtraction are trained with endpoint, pulse-compression and target-increment losses. The increment loss has an exact algebraic relation to finite-difference target distortion, but does not guarantee distortion-free recovery for arbitrary inputs.

Frozen tests show improved pulse-compression and detection behaviour relative to the jammed input. Compared with the earlier RSDR-Net, the compact model has lower waveform error on the main grid and lower target-increment distortion, but the waveform-error ranking reverses on the dense grid. SJRIF and jammed-background false-alarm control are weaker, with a marked detection deficit in one width scan. Eight-seed training shows limited additional DEV improvement through epoch 256. A same-capacity U-Net also performs better on the three-seed DEV comparison, leaving the proposed backbone's advantage unestablished. Compact correction is feasible, but target preservation and residual-interference suppression must be improved jointly. Controlled attribution and fresh confirmation are needed before making a stronger independent method claim.

## Data, code and reproducibility

Code and lightweight evidence are maintained in the authors' dedicated repository, `Hxin0203/ISRJ_2`. Training data, weights and raw arrays are not uploaded there; current repository access is not a public-release commitment. Data identities, checkpoint hashes, condition summaries and figure provenance are recorded in the accompanying evidence. Access arrangements and required author statements must be finalized for the target journal. No measured-data permission or hardware validation is claimed.

## References

{{REFERENCES}}

## Appendix A. Additional interpretation of the paired objectives

### A.1 Pulsewise and groupwise normalization

Let \(e_p\) and \(u_p\) be error and reference energies for pulse \(p\). The pulsewise training average is \(P^{-1}\sum_p e_p/u_p\), whereas the group metric is \((\sum_p e_p)/(\sum_p u_p)=\sum_p\omega_p(e_p/u_p)\), with \(\omega_p=u_p/\sum_q u_q\). The former weights pulses equally; the latter weights their ratios by reference energy. Since the generator varies noise and interference across pulses, these quantities need not agree.

### A.2 Increment distortion and common residuals

With \(r_{11}=C(Y_{11})-J\) and \(r_{01}=C(Y_{01})-J\), \(D_S=\|r_{11}-r_{01}\|_F^2/\|S\|_F^2\) before denominator clipping. Equal nonzero residuals give zero increment distortion while leaving endpoint error. Conversely, a weak target can have appreciable normalized increment distortion despite small absolute endpoint errors. This is why both measurements are retained.

### A.3 Nominal CA-CFAR coefficient

If the cell power \(X\) and the \(N_r\) reference powers are independent exponentials with mean \(\sigma^2\), the reference sum is Gamma distributed. Integrating \(\Pr(X>\alpha\bar X_r\mid\bar X_r)\) over that distribution gives \((1+\alpha/N_r)^{-N_r}\). Solving for \(\alpha\) yields (42). Correlated matched-filter outputs and nonlinear restoration need not satisfy these assumptions. The derivation explains the nominal coefficient, not preservation of CFAR by the network.
