# A Compact Waveform Correction Network for Interrupted-Sampling Repeater Jamming: Target-Increment Constraints and Recovery Tradeoffs

> Complete manuscript revision, 16 November 2026 (UTC). Independent tests use the frozen seed-29, epoch-128 model. Eight-seed results through epoch 256 and the three-seed U-Net comparison are development-set analyses. The ablation and fresh-confirmation programme is not yet complete. Author names, affiliations and funding information remain for the authors to supply.

## Abstract

{{ABSTRACT}}

**Index terms:** Interrupted-sampling repeater jamming; waveform restoration; residual learning; target-increment consistency; pulse compression; constant false alarm rate detection.

## 1 Introduction

Interrupted-sampling repeater jamming (ISRJ) intercepts short portions of a radar pulse and retransmits delayed copies that can produce false peaks after matched filtering [1]. The interference retains the local chirp structure of the transmitted waveform and may overlap the desired echo. A receiver therefore has to reduce interference that can trigger downstream detections while preserving a target with similar local structure. A visually smoother waveform, or a lower average reconstruction error, does not establish that both requirements have been met.

Existing receiver-side methods exploit observable differences between the target and retransmitted fragments. Time–frequency filtering uses their distributions to construct rejection filters, with Max-TF providing a representative implementation [2]. Other methods identify usable samples and reconstruct the echo in a sparse representation. Gaussian-mixture-based processing estimates usable segments from amplitude statistics before sparse reconstruction [3], whereas Bayesian compressed sensing recovers the signal from incomplete observations [4]. These approaches express structural assumptions explicitly. Their performance nevertheless depends on how well sample selection, the reconstruction dictionary and estimated parameters match the observation, particularly when noise or temporal overlap obscures fragment boundaries.

Learning-based methods offer different output contracts. A multipulse detection method applies model-driven Doppler-channel accumulation and preprocessing before convolutional target detection [5], but its output need not replace a waveform-processing front end. For waveform recovery, MREDNet uses a multi-residual encoder–decoder [6], MSMD-Net combines stages and signal domains [7], and extended complex-valued convolutional autoencoders integrate sample discrimination, denoising and reconstruction [8]. An end-to-end approach for low-SNR operation also includes pulse-compression supervision [9]. These studies motivate combining signal structure and downstream processing requirements, while making the differences in supervision and model complexity important to interpretation.

Recent work incorporates physical priors and optimization structure more directly. Complex-valued FISTA-Net unfolds sparse reconstruction iterations [10]. Joint morphological component analysis and reconstruction use estimated jamming parameters to adapt the decomposition [11], and MCA-ADMM-Net organizes the network around a dual-dictionary model and unfolded ADMM iterations [12]. In the HRRP domain, ISRJ-FNet learns a structured filtering mask and constrains its consistency under target perturbations [13]. Point-wise temporal classification followed by sparse reconstruction provides another separation between structural identification and recovery [14]. These approaches constrain different objects: an optimization trajectory, a filtering mask or a sample classification. Their interpretation depends on the signal domain and the desired output.

Our earlier RSDR-Net manuscript defines the reference as the jammer-free received waveform with the same noise realization, and uses four paired states to assess the incremental response to a target [15]. The system combines reliability estimation, several restoration experts, a fixed convex combination and bounded gating. The present work retains the task definition and asks whether a single compact correction path can provide useful restoration when those separate branches are removed. This question concerns both simplification and its cost: a reduction in waveform error may coexist with stronger residual peaks and more alarms in target-absent jammed backgrounds.

CorrectionNet first normalizes the I/Q observation and applies a fixed dechirp transform. A three-scale residual encoder–decoder predicts the waveform correction, which is transformed back and subtracted from the observation. The training reference is not a noise-free target. Instead, the network should leave unjammed inputs largely unchanged and remove changes caused by the jammer. A paired target-increment loss additionally discourages the correction from changing when only the target is added to a fixed background. We derive its exact relation to the finite-difference distortion of the restored target response.

The contributions are a compact correction path with 59,714 active parameters; an explicit distinction between endpoint, pulse-compression and target-increment constraints; and a frozen evaluation covering waveform fidelity, detection, false alarms, coherent processing and sampling-width sensitivity. These contributions build on established encoder–decoders, residual learning and the earlier paired supervision [15–18]. The study does not establish a new general identifiability theorem or uniform superiority over RSDR-Net. The additional parameter-matched U-Net development comparison also provides no evidence that the proposed backbone is superior. It instead helps locate the remaining problem: obtaining target preservation and residual-interference control together within a compact model.

## 2 Signal model and restoration objective

### 2.1 Coherent pulses and sampled observations

Let the fast-time sampling frequency be \(f_s\), the sampling interval \(T_s=1/f_s\), the observation length \(N\), the number of coherent pulses \(P\), and the pulse repetition interval \(T_r\). The transmitted baseband LFM pulse is

{{EQ1}}

where \(T_p\), \(B\), and \(\mu\) denote pulse width, bandwidth and chirp rate. Sampling occurs at \(t_n=nT_s\). For a single target with delay \(\tau\), amplitude \(a\), reference phase \(\phi\), and Doppler frequency \(f_D\), pulse \(p\) is

{{EQ2}}

The resulting pulse-to-pulse phase relation is

{{EQ3}}

Stacking the sampled pulses gives \(S\in\mathbb C^{P\times N}\). On the effective target support, the sampled signal is

\[
s_p[n]=a\,\mathbf1_{\{0\le nT_s-\tau<T_p\}}
\exp\!\left\{j\left[\phi+\pi\mu(nT_s-\tau-T_p/2)^2
+2\pi f_D(nT_s+pT_r)\right]\right\}.
\]

The generator represents any fractional delay through waveform sampling. Slow-time samples cannot distinguish \(f_D\) from \(f_D+k/T_r\), because the additional pulse-dependent phase is \(2\pi kp\). Doppler scoring therefore uses the frequency folded by the PRF; a wide simulated Doppler range is not an unambiguous velocity range.

The additive noise \(E\in\mathbb C^{P\times N}\) is zero-mean complex Gaussian. A noise realization is generated once per base scene and shared by the paired states and associated jamming variants. Its real representation is

{{EQ4}}

For the main grid, let \(\Omega_S\) and \(\Omega_J\) be the nonzero target and jammer supports. Define \(P_S\) and \(P_J\) as their mean powers over those supports, and \(P_E=\mathbb E|E_p[n]|^2\). The generator uses

{{EQ5}}

The nominal SNR controls the noise variance; finite realizations fluctuate around that expectation. The synthesized jammer is scaled using its actual effective-support mean power. This convention differs from the whole-group energy ratio used in the width scans in Section 4.3.

### 2.2 Fragment retransmission model

Let \(w_{p,k}(t)\) select fragment \(k\) of pulse \(p\), \(d_{p,k,m}\) be its retransmission delay, \(\beta_{p,k,m}\) its complex gain, and \(\Delta f_{p,k,m}\) an additional frequency shift. A fragment model is

{{EQ6}}

The four factors describe sample selection, delay, gain/phase and frequency modulation. The simulator uses the current target waveform as a proxy for the intercepted LFM signal. It does not separately model all radar–target and radar–jammer propagation paths. Direct retransmission replays a selected fragment once; repeated retransmission replays it at several delays; cyclic retransmission replays a finite set of buffered fragments in reverse order. The modes share the target and noise but draw their own fragment and modulation parameters.

Using a diagonal selection matrix \(W_{p,k}\), a zero-padded integer-delay matrix \(P_d\), and a diagonal modulation matrix \(M(\nu)\), the discrete representation follows below. The delay subscript in (7) is measured in samples, corresponding to the continuous delay divided by \(T_s\) and represented by the generator's integer delay. The notation \(d_{p,k,m}\) is retained for this discrete delay.

{{EQ7}}

ISRJ is thus a sum of selection–delay–modulation operators rather than a fixed convolution. Since retransmissions can overlap each other and the target, support cannot generally be inferred from a unique amplitude threshold. The observation is

{{EQ8}}

or \(Y=S+J+E\) for the pulse group. The deployed network receives the observed I/Q samples and fixed transmit-waveform constants. True supports, target delay and SNR/JSR labels are not inputs.

For \(A_{p,k,m}=M(\Delta f_{p,k,m})P_{d_{p,k,m}}W_{p,k}\), the identities \(W^{\mathrm H}W=W\), \(M^{\mathrm H}M=I\), and \(P_d^{\mathrm H}P_d\preceq I\) give

\[
\|\mathbf j_p\|_2\le\sum_{k,m}|\beta_{p,k,m}|\,\|W_{p,k}\mathbf s_p\|_2,
\qquad
\|\mathbf j_p\|_2^2=\sum_{u,v}\beta_u^*\beta_v
\langle A_u\mathbf s_p,A_v\mathbf s_p\rangle.
\]

The cross terms show why unscaled jammer energy is not determined by width and repetition count alone. Moreover, after support-power scaling, the same nominal JSR can still produce different full-window energies as support occupancy changes.

### 2.3 Pulse compression and error propagation

Let \(\mathcal H\) denote linear matched filtering with the transmitted template, with out-of-window samples set to zero:

{{EQ9}}

The implementation uses zero-padded FFT correlation with \(L\ge N+N_p-1\), where the template length is \(N_p=600\). The training loss retains all linear-correlation lags. Linearity gives

{{EQ10}}

and, under the common DFT convention, Parseval's identity gives

{{EQ11}}

for a time-domain error \(r\). Pulse-compression supervision therefore weights spectral error by the transmit-template power spectrum. It penalizes errors that the correlator can amplify, but remains an energy criterion, not a direct objective on the largest false peak or a CFAR detection event. Zero padding embeds the signal isometrically, and selecting correlation lags cannot increase its norm. Hence

\[
\|\mathcal H r\|_2\le\|\mathcal H\|_{2\to2}\|r\|_2,
\qquad
\|\mathcal H\|_{2\to2}\le\max_k|\widehat s_0[k]|.
\]

For an individual lag, Cauchy–Schwarz also yields \(|(\mathcal Hr)[\ell]|\le\|h_\ell\|_2\|r\|_2\), with \(h_\ell\) the corresponding template. These bounds limit total and per-cell perturbations without ordering the maximum residual peaks of two different outputs. The range-bin spacing is \(c_0/(2f_s)\), while the ideal LFM compressed-mainlobe scale is approximately \(c_0/(2B)\).

### 2.4 Four paired states and the same-noise reference

A parent is one independent base scene with fixed \((S,J,E)\). The target and jammer presence indicators \(a,b\in\{0,1\}\) define the following paired states.

| State | Input \(Y_{ab}\) | Desired correction \(Q_{ab}\) | Output reference \(U_{ab}\) |
|---|---|---|---|
| 00 | \(E\) | \(0\) | \(E\) |
| 10 | \(S+E\) | \(0\) | \(S+E\) |
| 01 | \(J+E\) | \(J\) | \(E\) |
| 11 | \(S+J+E\) | \(J\) | \(S+E\) |

Equivalently, \(Y_{ab}=aS+bJ+E\), \(Q_{ab}=bJ\), and \(U_{ab}=aS+E\). The same components must be shared across states, so that

{{EQ12}}

Resampling either noise or jamming when adding the target would introduce a nuisance difference and invalidate the target-only interpretation. The learned correction and restoration maps are

{{EQ13}}

with desired endpoint conditions

{{EQ14}}

For \(r_{ab}=C_\theta(Y_{ab})-Q_{ab}\), direct substitution yields \(T_\theta(Y_{ab})-U_{ab}=-r_{ab}\). Correction supervision therefore also constrains the restored output. The reference preserves the realized receiver noise; it separates jammer removal from denoising and determines the meaning of the endpoint error. It is not a claim that noise-free supervision used elsewhere is invalid.

### 2.5 Target response of a nonlinear restoration map

A nonlinear output has no unique decomposition into a restored target and restored jammer. We instead define a finite difference at fixed background \(v=J+E\):

{{EQ15}}

**Proposition 1.** For any correction map and any \(v,S\),

{{EQ16}}

and therefore

{{EQ17}}

**Proof.** Substitution of \(T_\theta(x)=x-C_\theta(x)\) into (15) cancels \(v\) and \(S\), giving (16). Taking squared norms gives (17). For the paired data, \(v=Y_{01}\) and \(v+S=Y_{11}\). Thus the squared difference of the two corrections is exactly the distortion of the incremental target response. This is an algebraic identity, not an approximation or a recovery guarantee.

Where the real-coordinate mapping is twice differentiable along the relevant segment, a local expansion is

{{EQ18}}

The Jacobian is defined in \(2N\)-dimensional real coordinates. For the vector-valued second derivative, the norm is \(\|H_C(x)\|=\sup_{\|u\|_2=\|w\|_2=1}\|H_C(x)[u,w]\|_2\). The triangle inequality gives

\[
\|J_{C_\theta}(v)S\|_2
\le\|C_\theta(v+S)-C_\theta(v)\|_2+\|R_2(v,S)\|_2.
\]

A small increment error therefore supports a small first-order response along the target direction only when the perturbation is sufficiently small and the second-order remainder is controlled. At finite amplitudes, cancellation between first- and higher-order terms can produce a small difference without a small Jacobian. The identities (16)–(17) themselves require no differentiability. None of these statements guarantees preservation in all directions, at arbitrary amplitudes, or under multiple targets and distribution shift.

### 2.6 Identifiability, degenerate solutions and normalization

If all four endpoint residuals vanish for a scene, the restoration is correct at those four input points. This finite set of constraints does not establish a globally unique decomposition. For example, \(C\equiv0\) has zero target-increment distortion while leaving all jamming intact. Conversely,

\[
\|c_{11}-c_{01}\|_2^2
\le2\|c_{11}-J\|_2^2+2\|c_{01}-J\|_2^2
\]

bounds the unnormalized increment error by the endpoint errors. Dividing by a small target energy can make the normalized distortion appreciable even when absolute endpoint errors are small. The two objectives are complementary.

Let \(A_v=J_C(v)\). For nonzero \(S\), the theoretical metric without a denominator floor satisfies

\[
\frac{\|C(v+\alpha S)-C(v)\|_2^2}{\|\alpha S\|_2^2}
\longrightarrow\frac{\|A_vS\|_2^2}{\|S\|_2^2},\qquad\alpha\to0^+.
\]

Both leading terms scale as \(\alpha^2\). With a fixed positive denominator floor, however, sufficiently small amplitudes inevitably activate that floor and the implemented metric tends to zero. The directional-sensitivity interpretation applies only in the amplitude range where the denominator is not clipped.

To illustrate identifiability, consider the unnormalized squared risk \(\mathbb E\|C(Y)-Q\|_2^2\), assuming a square-integrable correction label. This illustrative risk omits network parameterization, per-state normalization, pulse-compression supervision and the target-increment penalty. The conditional decomposition

\[
\mathbb E[\|c-Q\|_2^2\mid Y=y]
=\|c-\mathbb E[Q\mid Y=y]\|_2^2
+\mathbb E[\|Q-\mathbb E[Q\mid Y=y]\|_2^2\mid Y=y]
\]

gives \(C^\star(y)=\mathbb E[Q\mid Y=y]\). Conditional variance remains when different component realizations admit the same observation. Pairing supplies finite-difference constraints but does not eliminate this ambiguity. Adding a target-increment penalty couples the two inputs \(J+E\) and \(J+E+S\); pulse-compression weighting and per-state normalization further change the optimum. Their combined risk generally has no pointwise closed-form solution.

### 2.7 Connection to downstream detection

For the jammed observation, define \(\widehat J=C_\theta(Y)\) and \(\rho=J-\widehat J\). Then \(\widehat U-(S+E)=\rho\) and \(\mathcal H\widehat U-\mathcal H(S+E)=\mathcal H\rho\). Endpoint and pulse-compression losses thus control errors in quantities used downstream, but a small total residual can remain concentrated in a false peak. Target-increment error can also cancel a residual shared by the two target states.

For the pulse-averaged compressed power \(P_x[\ell]=P^{-1}\sum_p|(\mathcal Hx_p)[\ell]|^2\), the implemented peak ratio is

\[
R(x)=10\log_{10}\frac{\max(P_x[\ell_t],\varepsilon_{64})}
{\max(\max_{\ell\ne\ell_t}P_x[\ell],\varepsilon_{64})},
\qquad \mathrm{SJRIF}=R(\widehat U)-R(Y).
\]

Here \(\ell_t\) is located from the target-only compressed power, and \(\varepsilon_{64}\) is the smallest positive normal float64 value, used to make the ratio and logarithm defined at zero power. The denominator excludes only the target cell. A mainlobe guard is used solely in the separately recorded false-peak suppression metric. SJRIF therefore includes the effects of noise, sidelobes and residual interference; it is not a component-separated physical signal-to-jammer energy ratio.

Let \(Z_x[\nu,\ell]\) be the slow-time Fourier transform of the compressed pulses. The detection event is

\[
\mathsf D(x)=\mathbf1\!\left\{
\max_{(\nu,\ell)\in\mathcal G_t}
\left(|Z_x[\nu,\ell]|^2-\alpha\widehat P_{\mathrm{ref}}[\nu,\ell]\right)>0\right\},
\quad
\mathcal G_t=\{(\nu_t,\ell):|\ell-\ell_t|\le1\}\cap\mathcal V.
\]

The bin \(\nu_t\) is nearest to the true Doppler frequency after folding modulo the pulse repetition frequency; \(\mathcal V\) contains cells with complete CFAR reference windows. The evaluator requires the target center to be valid. Target truth is used only for simulation scoring, not as a network input. The power-difference form also covers a zero reference power. Restoration changes both the cell-under-test power and its estimated background. Consequently, neither an energy bound nor SJRIF determines the probability of this threshold event. Detection and target-absent false alarms must be measured separately on frozen outputs, without treating nominal CFAR calibration as an empirical guarantee.

## 3 Compact waveform correction network

The network assigns the known LFM phase rotation to a fixed operator and the unknown fragment boundaries and amplitude/phase changes to a compact learned backbone. The output is a correction in observation coordinates, with a subtraction bypass implementing restoration (Fig. 1).

![CorrectionNet architecture and paired training](figures/CorrectionNet_Visio_20261115/CorrectionNet_editable.png)

**Fig. 1.** CorrectionNet inference and training. (a) Normalization, fixed dechirp, correction prediction, inverse transformation and subtraction. (b) Three-scale residual encoder–decoder. (c) Residual convolutional block. (d) Four states sharing the target, jammer and noise. The diagram abbreviates the text's \(\delta_z''\) as \(\delta_z\). Component labels are used only for training. Waveform icons are schematic.

### 3.1 I/Q representation, dechirp and amplitude normalization

The real representation and its inverse are

{{EQ19}}

Pointwise complex multiplication can be represented by

{{EQ20}}

Thus \(\|\mathcal R(x)\|_F^2=\|x\|_2^2\), and \(M(q)^{\mathsf T}M(q)=|q|^2I_2\). Known phase rotations need not be learned, although the subsequent real-valued I/Q convolutions still learn cross-channel coupling.

Define the fixed dechirp reference by

{{EQ21}}

Since \(|q[n]|=1\),

{{EQ22}}

On the target support, the transformed phase becomes

{{EQ23}}

where \(\phi_p\) absorbs terms independent of fast time. The common quadratic phase becomes linear. Delayed fragments retain different boundaries and frequencies, so dechirping simplifies coordinates without separating target and jammer by itself.

The input RMS scale and normalized signal are

{{EQ24}}

With \(F_\theta\) denoting the I/Q network, interpreted as a complex output, the complete maps are

{{EQ25}}

If neither the original nor the scaled input activates the RMS floor, then for \(\gamma>0\),

{{EQ26}}

This follows because the normalized input is unchanged and the external scale contributes \(\gamma\). It is positive-real amplitude equivariance, not arbitrary complex-phase equivariance. Real I/Q convolutions and GELU do not impose the latter. Adding a target changes the RMS, so Proposition 1 concerns the full correction map, including normalization and inverse transformation.

Away from the RMS floor, its differential is

\[
\mathrm da=\frac{\langle y,\mathrm dy\rangle_{\mathbb R}}{Na(y)},
\qquad
\mathrm dz=D\left(\frac{\mathrm dy}{a(y)}-\frac{y\,\mathrm da}{a(y)^2}\right).
\]

The second term removes the pure amplitude direction from normalized coordinates. At the floor, the denominator is fixed and this differential must be replaced by the corresponding piecewise expression.

### 3.2 Three-scale residual encoder–decoder

Let \(x_0=\mathcal R(z)\in\mathbb R^{2\times1800}\). Reliability inputs, reliability features and gating are disabled in the deployed configuration. A residual block is

{{EQ27}}

where \(W_{l,1}\) and \(W_{l,2}\) are convolutions of lengths 5 and 3, \(N_{l,i}\) are group-normalization operators, \(P_l\) is an identity or a \(1\times1\) projection, and \(\sigma\) is GELU. Downsampling uses stride 2 in the first convolution and projection. The forward recursion is

{{EQ28}}

Here \(U_1,U_0\) are linear interpolation to the skip-connection lengths, brackets denote channel concatenation, and \(W_o\) is a \(1\times1\) output convolution. The correction and restored output are

{{EQ29}}

**Table 1. Actual forward tensor dimensions.**

| Node | Channels | Fast-time length | Role |
|---|---:|---:|---|
| \(x_0\) | 2 | 1800 | Normalized, dechirped I/Q |
| \(f_0\) | 16 | 1800 | Full-resolution encoding |
| \(f_1\) | 32 | 900 | First encoding with stride 2 |
| \(f_2\) | 64 | 450 | Second encoding with stride 2 |
| \(g_1\) | 32 | 900 | Upsampling and \(f_1\) concatenation |
| \(g_0\) | 16 | 1800 | Upsampling and \(f_0\) concatenation |
| \(\delta_z''\) | 2 | 1800 | Dechirped I/Q correction |

The convolutional length rule is \(L_{\rm out}=\lfloor(L_{\rm in}+2p-k)/s\rfloor+1\). Kernel length 5, padding 2 and stride 2 produce lengths 900 and 450 from 1800. Decoder interpolation explicitly matches the skip lengths.

Ignoring normalization statistics, the convolutional support obeys \(j_{l+1}=s_lj_l\) and \(r_{l+1}=r_l+(k_l-1)j_l\). The encoder outputs have \((j,r)=(1,7),(2,15),(4,31)\). Counting the maximum two-point interpolation support gives approximately 47 input samples after the first decoder and 55 at the output head. This is a path-support calculation, not the full network receptive field: GroupNorm uses statistics across time and the input RMS uses the whole window, allowing global dependence. Boundary padding and interpolation alignment also affect individual paths. The model contains no slow-time convolution or cross-pulse attention; coherent tests assess preservation by a pulsewise front end.

GroupNorm [19] uses eight channel groups. For \(x\in\mathbb R^{C\times L}\),

\[
\mu_g=\frac{\sum_{c\in\mathcal C_g,n}x_{c,n}}{|\mathcal C_g|L},\qquad
\sigma_g^2=\frac{\sum_{c\in\mathcal C_g,n}(x_{c,n}-\mu_g)^2}{|\mathcal C_g|L},
\quad
\operatorname{GN}(x)_{c,n}=\gamma_c\frac{x_{c,n}-\mu_{g(c)}}{\sqrt{\sigma_{g(c)}^2+\epsilon}}+\beta_c.
\]

Statistics are recomputed per input at inference; no running batch averages are used. GELU [20] is the exact error-function form \(x[1+\operatorname{erf}(x/\sqrt2)]/2\). Together with the convolutions, these operators make the learned map nonlinear and input dependent.

### 3.3 Active parameters and deployment scope

The encoder blocks contain 32,464 parameters, decoder blocks 27,216, and output convolution 34, totaling 59,714 active parameters. For a biased convolution, \(n_{\rm conv}=C_o(C_ik+1)\); an affine GroupNorm has \(2C_o\) parameters. Each of the five blocks has two convolutions, two normalization layers and a projection:

\[
n_{\rm block}=C_o(5C_i+1)+2C_o+C_o(3C_o+1)+2C_o+C_o(C_i+1).
\]

The channel pairs \((2,16),(16,32),(32,64),(96,32),(48,16)\) give 1,072, 6,368, 25,024, 21,728 and 5,488 parameters. The registry also retains 21,840 disabled gating parameters and 31,202 parameters in an inactive reliability propagator, giving 112,756 registered parameters. They are not counted as inference capacity. The earlier RSDR-Net registers 3,017,877 parameters; the reduction in registered count is 96.26%. Parameter count alone does not establish latency, memory, throughput or energy savings.

If the learned correction vanishes, the outer bypass is exactly the identity. The zero labels for unjammed states therefore constrain the actual waveform modification. The fixed reference supports 1800 samples; longer windows require a new reference and dimensional validation. For fixed channel widths, convolution, interpolation and phase rotation scale linearly with \(N\), with convolutional order \(\mathcal O(N\sum_l k_lC_{l-1}C_l)\). No measured real-time deployment claim is made.

### 3.4 Training losses

For \(c_{ab}=C_\theta(Y_{ab})\), the normalized endpoint objective is

{{EQ30}}

This is also the normalized output error because the output and correction residuals differ only by sign. The two unjammed states penalize unnecessary modification, and the two jammed states supervise subtraction of \(J\). For \(r_{ab}=c_{ab}-q_{ab}\), pulse-compression supervision is

{{EQ31}}

The target-increment objective is

{{EQ32}}

where Proposition 1 gives the equality. The implemented objective is

{{EQ33}}

The three terms constrain correction accuracy, template-weighted error and target-direction response, respectively. The zero-correction degeneracy prevents using the increment term alone. Equations (11) and (17), and the bound in Section 2.6, explain their relations without turning any term into a detection-rate objective. Their empirical contributions require controlled ablations.

The implementation expands each record into eight pulses and four states, calculates denominators per pulse and state, and averages over the batch. Group-level test NMSE instead sums energies over the pulse group before division. These different normalization orders explain why the training loss is not numerically interchangeable with the test metric (Appendix A).

### 3.5 Optimization and gradient paths

An epoch traverses \(512\times3\) parent–jamming-mode records. A batch contains 32 records, giving 48 updates per epoch. Adam [21] uses \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\), no weight decay, and global gradient-norm clipping at 5. The learning rate warms linearly from zero to \(10^{-3}\) in epoch 1, decays by the registered cosine schedule to \(2\times10^{-5}\) at epoch 128, and then decays to \(2\times10^{-6}\) at epoch 256. The training procedure is

{{EQ34}}

For update index \(k\ge1\), initialize \(m_0=v_0=0\) and write

\[
\begin{aligned}
g_k&=\nabla_\theta\widehat{\mathcal L}_k,\qquad
\widetilde g_k=g_k\min\!\left(1,\frac{5}{\|g_k\|_2+\epsilon_g}\right),\\
m_k&=\beta_1m_{k-1}+(1-\beta_1)\widetilde g_k,\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)\widetilde g_k^{\,2},\\
\theta_{k+1}&=\theta_k-\eta_k\frac{m_k/(1-\beta_1^k)}{\sqrt{v_k/(1-\beta_2^k)}+\epsilon}.
\end{aligned}
\]

The positive clipping stabilizer \(\epsilon_g\) is distinct from Adam's \(\epsilon\); a zero gradient remains zero. Squares, square roots and divisions are elementwise, whereas clipping uses the global gradient norm. The common scaling acts on the gradient supplied to Adam and does not bound the norm of Adam's final parameter update.

Every completed epoch has a recovery point, and every 32nd epoch has a separate milestone. The frozen TEST model remains seed 29 at epoch 128. The eight-seed extension is used for development-set convergence analysis, not to replace that identity retrospectively.

Writing a block as \(B_l(x)=\sigma(G_l(x)+P_lx)\), its local Jacobian is

\[
J_{B_l}(x)=J_\sigma(G_l(x)+P_lx)\,[J_{G_l}(x)+P_l].
\]

The projection supplies a direct path, but does not prove nonexpansiveness. Group statistics and the GELU derivative \(\Phi(x)+x\varphi(x)\) vary with the features. Interpolation at fixed dimensions is a fixed linear operator whose norm need not be one. For \(\delta=F_\theta(z)\), the complete correction differential away from the RMS floor is

\[
\mathrm d\widehat c=D^{\mathrm H}\left(\delta\,\mathrm da+aJ_{F_\theta}(z)\,\mathrm dz\right).
\]

The normalization scale affects both the internal coordinates and the correction written back to the observation. The proven structural properties are dechirp isometry, positive-real scale equivariance under the stated condition, and the finite-difference identity. None establishes global unbiasedness, distortion-free recovery or CFAR preservation.

{{REST}}
