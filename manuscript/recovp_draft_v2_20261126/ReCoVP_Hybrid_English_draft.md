# Waveform Restoration under Interrupted-Sampling Repeater Jamming via Regularized Variable Projection and Complex-Valued Residual Refinement

## Abstract

Interrupted-sampling repeater jamming (ISRJ) replays fragments of the radar's own waveform, so after pulse compression it produces false targets with coherent processing gain. Because the jammer and the target echo share the same chirp structure, suppressing the jammer can also change the amplitude, phase and shape of the true target. This paper formulates ISRJ suppression as complex waveform restoration against a noisy reference, namely the observation with the jammer removed but the realized receiver noise retained, and proposes ReCoVP-Hybrid, a two-stage method that combines regularized variable projection with complex-valued residual refinement. The first stage selects candidate delays from the observation using the known linear-frequency-modulated template, estimates a local window and a regularization weight for each candidate from phase-invariant features, and obtains the jammer coefficients in closed form on the resulting complex dictionary. The second stage reads both the preliminary restoration and the estimated jammer, compensates in complex-valued convolution channels the residual that the finite dictionary cannot represent, and controls the correction strength with phase-invariant real gates. Training is carried out on four states that share the same noise and jammer background and jointly constrains waveform error, pulse-compression error, target-increment error and residual tail peaks. We establish uniqueness and contraction properties of the coefficient subproblem for a fixed dictionary, decompose the first-stage error into regularization shrinkage bias, representation residual and target leakage, and show how input-adaptive changes of the dictionary enter the target response. Because the correction is confined to a low-dimensional subspace spanned by a few structured atoms whose complex amplitudes are solved in closed form, the processor leaves jammer-free inputs almost untouched, and what it removes is precisely the component coherent with the transmitted waveform, so the residual is less prone to refocus into false peaks after pulse compression. Simulations show that, at a comparable parameter count, the method attains lower restoration error with smaller background modification and fewer false alarms in jammed backgrounds, at the cost of some attenuation of the target amplitude.

**Index Terms**—Interrupted-sampling repeater jamming, waveform restoration, variable projection, complex-valued neural networks, target fidelity, model-based deep learning.

## I. Introduction

Pulse-compression radars transmit signals with a large time–bandwidth product, such as linear frequency modulation (LFM), to obtain range resolution and processing gain. With mature digital radio-frequency memory (DRFM), a jammer can intercept, store and retransmit fragments of a pulse while the pulse is still being received. Interrupted-sampling repeater jamming (ISRJ) alternates between sampling and forwarding, needs only a single antenna, and remains coherent with the transmitted signal [1]. After matched filtering, the forwarded fragments produce a train of false peaks with processing gain, whose positions, number and amplitudes depend on the sampling width, the sampling period and the forwarding scheme [1,2]. When these peaks fall near the true target or inside its main lobe, detection, tracking and imaging are all affected [3,4].

The difficulty of ISRJ lies in its similarity to the target echo. Unlike noise jamming, every ISRJ fragment lies on the chirp trajectory of the transmitted signal in the time–frequency plane; it differs only by an incomplete time support and a forwarding delay. Removing samples on the basis of amplitude or energy alone is therefore likely to remove part of the true target as well, and attenuation or phase offset of the target further degrades coherent integration over pulses. A suppression method should thus be judged not only by how much jammer it removes but also by how well it preserves the complex response of the true target.

Existing countermeasures fall roughly into three groups. The first group works at the transmitter or jointly at the transmitter and receiver, increasing separability through frequency agility, pulse staggering, or joint design of waveforms and mismatched filters [5–8]. These methods are effective but require cooperation from the waveform and the scheduler, and do not apply directly to receiver-side processing with a fixed waveform. The second group is receiver-side signal processing that exploits the interrupted time support, the time–frequency trajectory, or the sparsity of the jammer. Time–frequency analysis and band-pass filtering build filters from the location of jammer fragments in the time–frequency plane [2,9,10]; Su et al. construct a cumulative waveform coherence function and an adaptive matched filter in the waveform domain [11]; Liu et al. identify jammer-free segments with a Gaussian mixture model and then perform sparse reconstruction [12]; Wei et al. estimate jammer parameters to build a short-time Fourier dictionary and jointly reconstruct target and jammer by morphological component analysis [13], and further propose multiple-model fusion filtering [14]. These methods are physically interpretable, but their performance depends on how accurately fragment boundaries, forwarding parameters or dictionaries are estimated. At low SNR or under varying jammer parameters, missed fragments lose target energy, while misclassified ones leave jammer in the reconstruction [12,15].

The third group is data driven. He et al. build a multistage multidomain network [16]; Wang et al. propose MREDNet, a multi-residual encoder–decoder that restores LFM signals contaminated by intra-pulse retransmission interference [17]; a complex-valued Transformer [18] and the complex-valued encoder–decoder CVEDNet [19] process the received signal directly in the complex domain, the latter combining waveform-domain and pulse-compression-domain losses. Other studies combine recognition and suppression [20], separate and measure composite ISRJ [21], suppress compound repeater jamming with self-supervised learning [22], or detect dense jamming by fusing a Transformer with CFAR [23]. In automotive FMCW interference mitigation, complex-valued convolutional networks [24,25], prior-guided networks [26] and contrastive learning [27] likewise show that learning can exploit jammer features that are hard to describe by rules; a recent systematic evaluation also points out that conclusions about data-driven methods depend strongly on the metrics and test conditions [28].

Model-based learning lies between the two. These methods keep the computational steps of sparse reconstruction or iterative optimization explicit inside the network [29–31]. Meng et al. cascade complex-valued segment extraction, segment denoising and compressed-sensing reconstruction [32]; Lei et al. perform point-wise temporal classification, I/Q residual denoising and sparse reconstruction in sequence [33]; Huang and Zhang unfold complex-valued FISTA into a network [4]; Wei et al. represent the target and the jammer with a delayed-LFM dictionary and a local Fourier dictionary and unfold ADMM for two-component sparse recovery [34]; Liu et al. constrain a filtering network on high-resolution range profiles with structural priors of the target [35]. These works improve interpretability by combining physical and learned models, but most of them still take clean-target reconstruction or the jamming suppression ratio as the main objective and rarely examine the complex response of the processor to the true target separately.

Image and speech restoration offer useful ideas on this point. Conv-TasNet replaces a fixed time–frequency transform with a learnable encoder and mask estimation, showing that the representation and the treatment of phase directly affect separation quality [36]. Residual learning [37,38], multistage progressive restoration [39] and blind denoising of real noise [40] show that decomposing a hard restoration task into stages with clear responsibilities lowers the learning difficulty of each stage. Spatial feature transform modulates intermediate features with conditional information [41], and deformable alignment in video restoration aligns before restoring details [42]. Physics-informed learning [43] and data-driven learning of inverse transforms [44] further show that prior knowledge can enter the representation and the computation of a network rather than only a preprocessing step. These mechanisms cannot be transplanted to radar signals without verification, but their way of dividing work between structured estimation and residual compensation matches the needs of ISRJ suppression.

The two routes sit at opposite ends in the degrees of freedom of the correction. Model-based methods modify only a few parameterized components, so the correction is tightly constrained, but the constraint is only as good as the parameter estimates. Data-driven methods output a correction waveform as long as the observation; they are expressive, yet nothing prevents them from altering clean samples. For detection this matters: the residual left by a generic network often still carries part of the chirp structure of the forwarded fragments and refocuses into peaks in target-free cells after pulse compression, while small modifications of jammer-free inputs change the statistics of the CFAR reference window. We argue that ISRJ suppression needs not a larger network but a correction with a controlled budget: its bulk should lie in a low-dimensional subspace of LFM fragments identified from the observation, with complex amplitudes given analytically, and only what this subspace cannot represent should be passed to a network whose correction strength is itself constrained.

We therefore study receiver-side complex waveform restoration for a fixed LFM waveform and address three questions. First, since ISRJ fragments inherit the structure of the transmitted waveform, can the explicitly representable part of the jammer be handled by an analytically solvable model so that the network learns only what the model cannot describe? Second, a finite dictionary always leaves a representation residual and regularization introduces shrinkage bias; how can these errors be compensated while keeping the structured estimate? Third, how large is the incidental modification of the target made by an adaptive processor while it removes the jammer, and how can this modification be measured separately during training and evaluation?

To this end we propose ReCoVP-Hybrid (regularized complex variable projection with hybrid refinement). The method takes the jammer-free observation with the realized noise as the reference, so that the ideal correction is exactly the jammer. The first stage selects candidate delays, estimates local windows and regularization weights from phase-invariant features, builds a complex dictionary, and solves for the jammer coefficients by regularized variable projection. The second stage reads the preliminary restoration and the estimated jammer and compensates the remaining error with a complex-valued residual network. What distinguishes this structure from generic restoration networks is how the correction is organized: the first stage has only $K$ complex coefficients per pulse, and its effective degrees of freedom are further reduced by the regularization matrix; the second-stage correction is conditioned on the first-stage estimate and its applied fraction is limited by a pulse-level gate. Two consequences follow directly: the change made to jammer-free inputs is structurally bounded, and what is removed is mainly the component coherent with the transmitted waveform. The main contributions are as follows.

1. A budget-controlled two-stage structure for complex waveform restoration under ISRJ. In the first stage the network estimates where and which LFM fragments are present, while their complex amplitudes are solved analytically by regularized variable projection, so that the bulk of the correction lies in a low-dimensional, interpretable subspace. The second stage, conditioned on the preliminary restoration and the removed jammer, compensates in complex-valued channels the representation residual and the regularization shrinkage bias, with correction strength controlled by phase-invariant gates. The whole mapping is equivariant to complex scaling of the input.
2. An analysis that establishes uniqueness and contraction of the coefficient subproblem together with a degrees-of-freedom bound on the energy removed from noise for a fixed dictionary, explaining why the structured correction hardly touches clean inputs; decomposes the first-stage error into regularization shrinkage bias, representation residual and target–noise leakage to motivate the refinement stage; and derives the effect of the input-adaptive dictionary on the target-increment response, on which a shared-background four-state training objective is built.
3. An evaluation protocol that reports restoration error, target-increment damage, jammer-free modification and two-dimensional CA-CFAR detection side by side, applied in a per-scene paired comparison with 12 families of convolutional restoration networks, three trained instances each. At a comparable parameter count the proposed structure lowers restoration error, jammer-free modification and the frame false-alarm rate in jammed backgrounds simultaneously, and the cost in target-amplitude shrinkage is quantified.

The rest of the paper is organized as follows. Section II gives the signal model and defines the restoration problem and the target response. Section III describes the modules of ReCoVP-Hybrid and their mathematical properties. Section IV presents the four-state joint training objective. Section V reports simulation results. Section VI discusses the mechanisms and scope of the method, and Section VII concludes the paper.

## II. Signal Model and Problem Formulation

### A. Target Echo and Interrupted-Sampling Repeater Jamming

Let the baseband LFM pulse be

$$
p(t)=\operatorname{rect}\!\left(\frac{t}{T_p}\right)\exp(\mathrm j\pi\mu t^2),\qquad \mu=\frac{B}{T_p}, \tag{1}
$$

where $T_p$ is the pulse width, $B$ the bandwidth, and $\operatorname{rect}(\cdot)$ the rectangular window on $[-1/2,1/2)$. A coherent processing interval (CPI) contains $M$ pulses with pulse repetition interval $T_r$. Under narrowband, single-target conditions with target parameters constant within the CPI, the echo in the $m$th pulse is

$$
s_m(t)=a\,p(t-\tau)\exp\{\mathrm j2\pi f_D(t+mT_r)\},\qquad m=0,\ldots,M-1, \tag{2}
$$

where the complex $a$ contains the target amplitude and initial phase, $\tau$ is the target delay and $f_D$ the Doppler frequency.

The jammer samples the intercepted signal intermittently with period $T_s$ and width $T_w$; the $\ell$th sampling window is $w_\ell(t)$. Each fragment is forwarded once or several times; let $\mathcal R_\ell$ be the forwarding set and $d_{\ell r}$ the delay of the $r$th forwarding relative to the sampling instant. Covering direct, repeated and cyclic forwarding, the jammer in the $m$th received pulse can be written as

$$
j_m(t)=\sum_{\ell}\sum_{r\in\mathcal R_{m\ell}}\beta_{m\ell r}\,
w_{m\ell}(t-d_{m\ell r})\,p(t-d_{m\ell r}-\tau_J)\,
\exp\{\mathrm j2\pi\nu_{m\ell r}t\}, \tag{3}
$$

where $\tau_J$ is the reference delay of the jammer, $\beta_{m\ell r}$ the complex forwarding gain and $\nu_{m\ell r}$ an additional frequency offset. The three forwarding schemes differ only in the arrangement of $\mathcal R_{m\ell}$ and $d_{m\ell r}$ [1,2]. Equation (3) shows that each jammer fragment is the product of a window, a delayed LFM and a frequency offset; this structure is the basis for the dictionary in Section III. None of these parameters is known to the processor at inference.

[Figure 1 placeholder: Principle of ISRJ. (a) Transmitted LFM pulse and interrupted sampling windows; (b) timing of direct, repeated and cyclic forwarding; (c) time–frequency distributions of target and jammer; (d) pulse compression of the jammed echo, with the true peak and false peaks marked.]

### B. Observation Model and Restoration Reference

Sampling one pulse at rate $f_s$ gives $T$ samples:

$$
\boldsymbol y=\boldsymbol s+\boldsymbol j+\boldsymbol\epsilon\in\mathbb C^T,\qquad
\boldsymbol u\triangleq\boldsymbol s+\boldsymbol\epsilon, \tag{4}
$$

where $\boldsymbol\epsilon$ is complex Gaussian noise. We take $\boldsymbol u$ as the restoration reference: the processor must remove the jammer but is not required to remove receiver noise. There are two reasons. Jammer suppression and noise suppression are different tasks, and the subsequent pulse compression and coherent integration already suppress noise. Moreover, with $\boldsymbol u$ as the reference the ideal correction is exactly the jammer, so any change that the processor makes to a jammer-free input can be measured separately.

Let the processor be $F_\theta:\mathbb C^T\to\mathbb C^T$ and define its correction $C_\theta(\boldsymbol y)\triangleq\boldsymbol y-F_\theta(\boldsymbol y)$. From (4),

$$
F_\theta(\boldsymbol y)-\boldsymbol u=\boldsymbol j-C_\theta(\boldsymbol y). \tag{5}
$$

Equation (5) states that the restoration error against the noisy reference equals the jammer estimation error, so restoration against $\boldsymbol u$ is equivalent to estimating the jammer to be subtracted. This equivalence defines the supervision target; it does not imply that the jammer is uniquely identifiable from an arbitrary single observation, which remains ambiguous when target and jammer share the same local structure over some interval.

### C. Shared-Background Four States and Target Response

The restoration error describes only the overall behavior on a jammed observation and cannot tell whether the error comes from residual jammer or from target damage. For one realization $(\boldsymbol s,\boldsymbol j,\boldsymbol\epsilon)$ we therefore construct four states,

$$
\boldsymbol y_{00}=\boldsymbol\epsilon,\quad
\boldsymbol y_{10}=\boldsymbol s+\boldsymbol\epsilon,\quad
\boldsymbol y_{01}=\boldsymbol j+\boldsymbol\epsilon,\quad
\boldsymbol y_{11}=\boldsymbol s+\boldsymbol j+\boldsymbol\epsilon, \tag{6}
$$

where the subscripts indicate the presence of the target and the jammer. The four states share the same noise and the same jammer; removing the target leaves the jammer unchanged. Physically, the jammer forwards the transmitted signal before it is reflected by the target, so fixing the jammer and toggling the target is a reasonable offline intervention that isolates the processor's response to the target.

Under the jammed background, define the target increment and its error

$$
\boldsymbol d\triangleq F_\theta(\boldsymbol y_{11})-F_\theta(\boldsymbol y_{01}),\qquad
\boldsymbol\eta\triangleq\boldsymbol d-\boldsymbol s. \tag{7}
$$

A processor that is transparent to the target gives $\boldsymbol d=\boldsymbol s$; a nonlinear input-adaptive processor generally does not. With the residuals $\boldsymbol e_{11}=F_\theta(\boldsymbol y_{11})-\boldsymbol u$ and $\boldsymbol e_{01}=F_\theta(\boldsymbol y_{01})-\boldsymbol\epsilon$ of the two jammed states, $\boldsymbol\eta=\boldsymbol e_{11}-\boldsymbol e_{01}$ and

$$
\|\boldsymbol\eta\|_2^2=\|\boldsymbol e_{11}\|_2^2+\|\boldsymbol e_{01}\|_2^2
-2\operatorname{Re}\left(\boldsymbol e_{01}^{H}\boldsymbol e_{11}\right). \tag{8}
$$

Equation (8) shows that lowering the restoration error $\|\boldsymbol e_{11}\|_2$ alone does not guarantee a lower target-increment error; the correlation between the two residuals also matters. When they are highly correlated, residual jammer cancels in the difference and target damage can be small; when they are uncorrelated, the target-increment error can be large even if each residual is small. This is why we evaluate restoration error and target damage separately.

To distinguish types of target damage, assume $\boldsymbol s\neq\boldsymbol 0$ and project $\boldsymbol d$ onto the target direction:

$$
g\triangleq\frac{\boldsymbol s^{H}\boldsymbol d}{\|\boldsymbol s\|_2^2},\qquad
\boldsymbol q\triangleq\boldsymbol d-g\boldsymbol s,\qquad \boldsymbol s^{H}\boldsymbol q=0. \tag{9}
$$

Since $\boldsymbol\eta=(g-1)\boldsymbol s+\boldsymbol q$ with orthogonal terms,

$$
\frac{\|\boldsymbol\eta\|_2^2}{\|\boldsymbol s\|_2^2}=|g-1|^2+\frac{\|\boldsymbol q\|_2^2}{\|\boldsymbol s\|_2^2}. \tag{10}
$$

The first term is the complex-gain deviation, with $|g|<1$ corresponding to attenuation and $\arg g\neq0$ to a phase offset; the second is the shape error that a single complex gain cannot explain. They affect detection differently: gain deviation lowers the compressed peak directly, whereas shape error mainly raises sidelobes or broadens the peak. Equation (10) also shows that mean amplitude preservation and mean target damage can move in different directions and cannot substitute for each other.


## III. ReCoVP-Hybrid

### A. Overall Structure and Design Rationale

ReCoVP-Hybrid processes each pulse independently; input and output are complex vectors of length $T$. The design follows from (3): each ISRJ fragment has the form "local window × delayed LFM", and the shape of the delayed LFM is known at the receiver. Once the delays of the jammer fragments and their window shapes are estimated from the observation, the jammer lies in the span of a set of known structured atoms, and the complex amplitudes follow from linear least squares without regression by a network. In this stage the network only performs two kinds of nonlinear estimation: where to place atoms (delays) and what time support each atom has (windows).

This representation is incomplete. Candidate delays can be missed, windows can be misestimated, jammer frequency offsets are not modeled per atom, and regularization shrinks the coefficients toward zero; all of these leave residuals in the first-stage output (Section III-E gives the exact decomposition). The second stage is therefore not restricted to the span of the dictionary and estimates an additional correction with a complex-valued convolutional network. It reads two inputs: the preliminary restoration $\widehat{\boldsymbol u}_1$ shows what is left, and the estimated jammer $\widehat{\boldsymbol j}_1$ shows what has already been removed; together they indicate whether a residual has the same origin as the removed jammer. Both corrections are subtracted from the observation:

$$
\widehat{\boldsymbol j}_1=\mathcal S_1(\boldsymbol y),\qquad
\widehat{\boldsymbol u}_1=\boldsymbol y-\widehat{\boldsymbol j}_1,\qquad
\boldsymbol\delta_2=\mathcal S_2(\widehat{\boldsymbol u}_1,\widehat{\boldsymbol j}_1),\qquad
\widehat{\boldsymbol u}=\widehat{\boldsymbol u}_1-\boldsymbol\delta_2. \tag{11}
$$

The total correction is $C_\theta(\boldsymbol y)=\widehat{\boldsymbol j}_1+\boldsymbol\delta_2$. The pipeline uses only the observation and the known transmitted waveform; it never reads target positions, jammer supports or true coefficients, and the four-state ground truth is used only for training and evaluation.

The same principle decides the split between real-valued and complex-valued computation. Every decision that must be invariant to a global phase rotation (candidate positions, window shapes, regularization weights, gate strengths) is computed from phase-invariant real features, and every quantity that must preserve phase relations (atoms, coefficients, correction waveforms) is computed in the complex domain. Section III-H shows that this split makes the whole mapping equivariant to complex scaling of the input.

[Figure 2 placeholder: Overall framework of ReCoVP-Hybrid. Left, first stage: RMS normalization → phase-invariant feature encoder → candidate delay selection and refinement → local window estimation → complex dictionary and regularization matrix → regularized coefficient solve → $\widehat{\boldsymbol j}_1$. Right, second stage: $(\widehat{\boldsymbol u}_1,\widehat{\boldsymbol j}_1)$ → complex encoder with phase-invariant real gates → dilated bottleneck → skip-connected decoder → pulse-level scalar gate → $\boldsymbol\delta_2$. Use different colors for complex data flow, real-valued conditioning and training-only ground truth, and label channel numbers and lengths at each node.]

Algorithm 1 summarizes inference.

> **Algorithm 1: Single-pulse inference of ReCoVP-Hybrid**
> Input: jammed pulse $\boldsymbol y\in\mathbb C^T$; known LFM template $\boldsymbol p$.
> 1. Compute the RMS scale $r$ and the normalized input $\boldsymbol x=\boldsymbol y/r$ (12).
> 2. Extract the four phase-invariant features $\boldsymbol\phi$ (13) and obtain shared features $\boldsymbol h\in\mathbb R^{13\times T}$ from the feature encoder–decoder.
> 3. Select $K$ candidate peaks $a_k$ from the template correlation magnitude and refine them to delays $\tau_k$ by (14).
> 4. Generate delayed templates $\boldsymbol t_k$ by (15), obtain real windows $\boldsymbol m_k$ from the window network, and form the dictionary $\boldsymbol B$ by (16).
> 5. Obtain $\boldsymbol\Lambda$ from the regularization head (17), solve (19) for $\widehat{\boldsymbol j}_1$, and set $\widehat{\boldsymbol u}_1=\boldsymbol y-\widehat{\boldsymbol j}_1$.
> 6. Feed $(\widehat{\boldsymbol u}_1/r,\widehat{\boldsymbol j}_1/r)$ to the complex refinement network to obtain the proposal $\widetilde{\boldsymbol\delta}$ and the pulse-level gate $\alpha$ ((23)–(25)).
> 7. Output $\widehat{\boldsymbol u}=\widehat{\boldsymbol u}_1-r\alpha\widetilde{\boldsymbol\delta}$.

### B. Scale Normalization and Phase-Invariant Features

The amplitude of the received signal varies widely with target range and jammer power. A nonzero input is normalized by its root-mean-square scale,

$$
r(\boldsymbol y)=\Big(\tfrac1T\textstyle\sum_{n=0}^{T-1}|y_n|^2\Big)^{1/2},\qquad
\boldsymbol x=\boldsymbol y/r(\boldsymbol y), \tag{12}
$$

and the all-zero input is defined separately by $r=1$, $\boldsymbol x=\boldsymbol 0$. All corrections are estimated in normalized coordinates and multiplied by $r$, so the output keeps the absolute amplitude of the input.

With the unit-energy template $\widetilde{\boldsymbol p}$ and the correlation magnitude $\boldsymbol a=|\boldsymbol x\star\widetilde{\boldsymbol p}|$, four features are formed at each sample:

$$
\boldsymbol\phi_n=\big[\,|x_n|,\;a_n,\;\operatorname{Re}\ell_n,\;\operatorname{Im}\ell_n\,\big]^\top,\qquad
\ell_n=x_nx_{n-1}^{*},\ \ell_0=0. \tag{13}
$$

The three kinds of features play different roles. $|x_n|$ describes local energy, and abrupt energy changes often mark the boundaries of jammer fragments. $a_n$ describes how well the observation matches the known LFM, and its peaks indicate possible delayed waveforms. The phase of the lag product $\ell_n$ equals the local instantaneous frequency times the sampling interval; on an LFM it changes linearly with time, and the forwarding delays and frequency offsets of jammer fragments introduce jumps into this pattern. All three are unchanged when the input is multiplied by $e^{\mathrm j\varphi}$, so candidate selection and window estimation do not depend on the global phase.

The feature encoder is a one-dimensional U-Net [45] with widths $4\to13\to26\to44$ and lengths $T\to T/2\to T/4$; each convolution is followed by group normalization [46] and ReLU. The decoder returns to shared features $\boldsymbol h$ of size $13\times T$ through two upsampling steps with skip connections. The skips keep details near fragment boundaries, and the downsampling path gathers context over longer time spans, which helps to distinguish the multiple similar fragments produced by repeated forwarding.

### C. Candidate Delay Selection and Refinement

The correlation magnitude of the observation with the unnormalized LFM template is computed, local peaks are found by a 13-point max pooling over indices $2\le n\le T-3$, and the $K=16$ largest peaks are selected with a minimum spacing of 6 samples. If fewer than $K$ local peaks exist, the remainder is filled by magnitude from other positions, with ties broken by the smaller index. The candidates depend only on the observation and the known template.

Integer candidates cannot represent fractional delays or absorb peak shifts caused by frequency offsets. For each candidate $a_k$, the shared features in its neighborhood, globally pooled features, its normalized position and peak information form a 59-dimensional vector $\boldsymbol h_k$, and a $59\to64\to1$ multilayer perceptron gives a bounded refinement

$$
\tau_k=a_k+2\tanh\big(f_\tau(\boldsymbol h_k)\big). \tag{14}
$$

The refinement is limited to $\pm2$ samples so that the network adjusts near each candidate and cannot move a candidate onto another peak. The cost is that jammer fragments missed by pooling cannot be recovered by refinement; this error is left to the second stage.

### D. Local-Window Dictionary

To represent fractional delays, a linear phase is applied to the unit-energy template in an $N=4096$-point frequency domain and the first $T$ samples are kept:

$$
\boldsymbol t_k=\mathcal T_T\,\mathcal F_N^{-1}\Big[\widetilde{\boldsymbol P}\odot\exp(-\mathrm j2\pi\boldsymbol f\tau_k)\Big], \tag{15}
$$

where $\widetilde{\boldsymbol P}$ is the DFT of the zero-padded template, $\boldsymbol f$ the normalized frequency vector in DFT order, and $\mathcal T_T$ the truncation to the first $T$ samples.

A candidate delay identifies which delayed LFM is present but not which of its samples belong to a forwarded fragment. By (3) a fragment is a delayed LFM truncated by a sampling window, so a time window is estimated for each candidate. The window network takes two groups of inputs: the 13 shared feature channels $\boldsymbol h$, and 9 candidate-specific matching channels obtained by smoothing the conjugate product $\boldsymbol x\odot\boldsymbol t_k^{*}$ with 5-, 15- and 31-point moving averages and taking, at each scale, the magnitude and the real and imaginary parts of the lag product. A fragment that matches $\boldsymbol t_k$ appears in $\boldsymbol x\odot\boldsymbol t_k^{*}$ as a nearly constant complex envelope with large magnitude and near-zero adjacent phase difference, whereas mismatched parts appear as fast phase rotation; the multiscale smoothing lets the network see both narrow and wide fragments. The 22 input channels pass through $22\to12\to1$ point-wise convolutions, a sigmoid and three-point mean smoothing, and are normalized by their peak (with a floor of 0.05) to give a real window $\boldsymbol m_k\in[0,1]^T$. The atoms are

$$
\boldsymbol b_k=\frac{\boldsymbol m_k\odot\boldsymbol t_k}{\|\boldsymbol m_k\odot\boldsymbol t_k\|_2+10^{-12}},\qquad
\boldsymbol B=[\boldsymbol b_1,\ldots,\boldsymbol b_K]\in\mathbb C^{T\times K}. \tag{16}
$$

All candidates share one window network and obtain different windows because their template positions and matching features differ. Atoms are normalized to unit energy and their complex amplitudes are determined by the coefficient solve. A one-to-one correspondence between atoms and physical fragments is not required: one fragment may be represented by several atoms, and one window may cover several adjacent fragments.

### E. Regularized Variable Projection

**Coefficient subproblem.** Given $\boldsymbol B$, the jammer coefficients can be found by least squares. Because candidates are only a few samples apart, neighboring atoms are highly correlated and $\boldsymbol B^H\boldsymbol B$ may be ill-conditioned; a direct solve then produces large coefficients of opposite signs. Each candidate therefore receives a regularization weight $\lambda_k>0$. Let $\widehat\sigma^2$ be a robust spread of the normalized input (a center obtained by three inverse-distance-weighted updates, then the median squared distance to it divided by $\log2$), and let $\rho_k=|\boldsymbol b_k^H\boldsymbol x|^2/\widehat\sigma^2$ be the correlation strength of candidate $k$. The regularization head ($61\to16\to1$) outputs a real $v_k$, and

$$
z_k=v_k+2\big[\log 9-\log(1+\rho_k)\big],\qquad
\lambda_k=10^{-4}+\exp\{30\tanh(z_k/30)\},\qquad
\boldsymbol\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_K). \tag{17}
$$

The bias $-2\log(1+\rho_k)$ assigns weaker regularization by default to strongly correlated candidates and stronger regularization to weak ones, and the network output $v_k$ adjusts this default; the $\tanh$ limiter and the positive floor ensure $\lambda_k\in(10^{-4},10^{-4}+e^{30})$. The subproblem is

$$
\widehat{\boldsymbol c}=\arg\min_{\boldsymbol c\in\mathbb C^K}\;
\|\boldsymbol x-\boldsymbol B\boldsymbol c\|_2^2+\boldsymbol c^H\boldsymbol\Lambda\boldsymbol c. \tag{18}
$$

Setting the Wirtinger derivative with respect to $\boldsymbol c^{*}$ to zero gives the normal equations $(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)\widehat{\boldsymbol c}=\boldsymbol B^H\boldsymbol x$. For any $\boldsymbol v\neq\boldsymbol0$, $\boldsymbol v^H(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)\boldsymbol v=\|\boldsymbol B\boldsymbol v\|_2^2+\sum_k\lambda_k|v_k|^2>0$; the system matrix is positive definite and (18) has a unique solution whether or not the columns of the dictionary are correlated. The first-stage jammer estimate is

$$
\widehat{\boldsymbol j}_1=r\,\boldsymbol B\widehat{\boldsymbol c}=\boldsymbol P\boldsymbol y,\qquad
\boldsymbol P\triangleq\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol B^H. \tag{19}
$$

This follows the idea of variable projection [47–49]: in a separable nonlinear least-squares problem the linear parameters have a closed-form solution given the nonlinear ones and can be eliminated, so that only the nonlinear parameters need to be optimized. Here the nonlinear parameters (delays, windows, regularization) are estimated by networks, the linear coefficients are obtained analytically from (19), and gradients propagate to the networks through the solution of the linear system [31]. Compared with unfolding a fixed number of iterations [4,30,34], the coefficients are always exactly optimal for the given dictionary and the network does not have to learn the iteration itself. The implementation solves for $\widehat{\boldsymbol c}$ with a linear solver rather than forming an inverse; the solve uses double-precision complex arithmetic and the rest uses single precision.

**Proposition 1 (contraction for a fixed representation).** If $\boldsymbol B$ is fixed and $\boldsymbol\Lambda\succ0$, then $\boldsymbol P$ is Hermitian positive semidefinite with eigenvalues in $[0,1)$. With $\sigma_1\ge\cdots\ge\sigma_K$ the singular values of $\boldsymbol B\boldsymbol\Lambda^{-1/2}$, the nonzero eigenvalues of $\boldsymbol P$ are $\sigma_k^2/(1+\sigma_k^2)$ and $\operatorname{tr}\boldsymbol P=\sum_k\sigma_k^2/(1+\sigma_k^2)<K$.

The proof is given in Appendix A. Proposition 1 has two implications. First, $\|\boldsymbol P\boldsymbol y\|_2\le\frac{\sigma_1^2}{1+\sigma_1^2}\|\boldsymbol y\|_2<\|\boldsymbol y\|_2$: in every direction the first stage removes strictly less energy than the observation contains, and $\boldsymbol I-\boldsymbol P$ is nonexpansive. Second, $\boldsymbol P$ generally does not satisfy $\boldsymbol P^2=\boldsymbol P$ and is not an orthogonal projection; stronger regularization means stronger shrinkage. $\operatorname{tr}\boldsymbol P$ can be read as the effective degrees of freedom of the first stage and is smaller than the number of candidates $K$.

**Corollary 1 (degrees-of-freedom budget of the correction).** Under the conditions of Proposition 1, if the noise $\boldsymbol\epsilon\sim\mathcal{CN}(\boldsymbol0,\sigma^2\boldsymbol I_T)$ is independent of $\boldsymbol B$ and $\boldsymbol\Lambda$, then

$$
\frac{\mathbb E\|\boldsymbol P\boldsymbol\epsilon\|_2^2}{\mathbb E\|\boldsymbol\epsilon\|_2^2}=\frac{\operatorname{tr}(\boldsymbol P^2)}{T}\le\frac{\operatorname{tr}\boldsymbol P}{T}<\frac{K}{T}. \tag{19a}
$$

The proof uses $\mathbb E\|\boldsymbol P\boldsymbol\epsilon\|_2^2=\sigma^2\operatorname{tr}(\boldsymbol P^H\boldsymbol P)=\sigma^2\operatorname{tr}(\boldsymbol P^2)$ and $\mu_k^2\le\mu_k$ for eigenvalues $\mu_k\in[0,1)$. With $K=16$ and $T=1800$ the right-hand side is below 0.89%. The corollary states that the degrees of freedom of the first-stage correction do not grow with the signal length: whatever delays and windows the network produces, the average energy it can remove from white noise is below $K/T$. A generic network that outputs a $T$-sample correction has no such structural constraint and can keep its changes to clean inputs small only indirectly through the training loss. In practice the dictionary is selected from the observation and candidates lock onto the strongest correlation peaks of the noise, so independence does not hold exactly; on the other hand, (17) assigns strong regularization to weakly correlated candidates, which makes $\operatorname{tr}\boldsymbol P$ much smaller than $K$. Equation (19a) therefore indicates the order of magnitude of the degrees of freedom; the measured jammer-free modification is reported in Section V-E.

**Sources of first-stage error.** Decompose the true jammer into parts inside and outside the span of the dictionary, $\boldsymbol j=\boldsymbol B\boldsymbol c^\star+\boldsymbol j_\perp$, where $\boldsymbol c^\star$ is the least-squares coefficient of $\boldsymbol j$ on $\operatorname{span}(\boldsymbol B)$ and $\boldsymbol B^H\boldsymbol j_\perp=\boldsymbol 0$.

**Proposition 2 (decomposition of the first-stage error).** Under the conditions of Proposition 1,

$$
\widehat{\boldsymbol u}_1-\boldsymbol u=
\underbrace{\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol\Lambda\,\boldsymbol c^\star}_{\text{shrinkage bias}}
+\underbrace{\boldsymbol j_\perp}_{\text{representation residual}}
-\underbrace{\boldsymbol P(\boldsymbol s+\boldsymbol\epsilon)}_{\text{target and noise leakage}}. \tag{20}
$$

The proof uses $\boldsymbol P\boldsymbol j_\perp=\boldsymbol 0$ and $(\boldsymbol I-\boldsymbol P)\boldsymbol B=\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol\Lambda$; see Appendix A. The three terms correspond to three failure modes. The first grows with $\boldsymbol\Lambda$: regularization stabilizes the coefficients but leaves part of the representable jammer behind, and this part lies in the span of the dictionary, close in direction to $\widehat{\boldsymbol j}_1$. The second comes from missed candidates, window errors and unmodeled frequency offsets and cannot be removed by regularization. The third is the incidental modification of target and noise by the first stage and is most pronounced when an atom coincides with the target in delay and window. The three terms pull in different directions: reducing $\boldsymbol\Lambda$ lowers the first but amplifies the third when the dictionary is ill-conditioned, and enlarging the dictionary lowers the second but raises the chance that an atom overlaps the target. Tuning the first stage alone can hardly suppress all three at once, which is the direct reason for the second stage. The second stage reads both $\widehat{\boldsymbol u}_1$ and $\widehat{\boldsymbol j}_1$: the first term shares its origin with $\widehat{\boldsymbol j}_1$ and can be recognized from the correlation between the two inputs; the second remains in $\widehat{\boldsymbol u}_1$ as unremoved LFM-like fragments; and the third requires the second stage not to weaken the target further, which is enforced by the target-increment loss of Section IV.

### F. Effect of the Adaptive Dictionary on the Target Response

Propositions 1 and 2 hold for a fixed dictionary. In practice $\boldsymbol B$ and $\boldsymbol\Lambda$ are estimated from the observation, and adding the target changes the candidates, windows and regularization, hence the whole operator. Let $\boldsymbol y_0=\boldsymbol j+\boldsymbol\epsilon$ be the target-free jammed observation, $\boldsymbol P_0=\boldsymbol P(\boldsymbol y_0)$, $\boldsymbol P_1=\boldsymbol P(\boldsymbol y_0+\boldsymbol s)$, and $F_1(\boldsymbol y)=(\boldsymbol I-\boldsymbol P(\boldsymbol y))\boldsymbol y$ the first-stage output. The first-stage target-increment error follows directly from the definitions:

$$
\boldsymbol\eta_1=F_1(\boldsymbol y_0+\boldsymbol s)-F_1(\boldsymbol y_0)-\boldsymbol s
=-\boldsymbol P_1\boldsymbol s-(\boldsymbol P_1-\boldsymbol P_0)\boldsymbol y_0, \tag{21}
$$

and the triangle inequality gives

$$
\|\boldsymbol\eta_1\|_2\le\|\boldsymbol P_1\boldsymbol s\|_2+\|\boldsymbol P_1-\boldsymbol P_0\|_2\,\|\boldsymbol y_0\|_2. \tag{22}
$$

The first term on the right of (21) is the direct modification of the target by the current operator; the second is the additional term produced when the operator change acts on the background. Under strong jamming $\|\boldsymbol y_0\|_2$ is large, and even a small operator change caused by the target can make the second term dominate. Checking $\boldsymbol P\boldsymbol s$ with a frozen dictionary is therefore insufficient for assessing target fidelity: candidates switch discontinuously where the peak ordering changes, and no global Lipschitz bound on the operator change exists. For this reason the target increment (7) is computed on the fully re-estimated output in both training and evaluation, instead of on outputs with some modules frozen. Equation (22) also suggests a measurable diagnostic: recording $\|\boldsymbol P_1-\boldsymbol P_0\|_2$ together with the target damage shows whether the damage comes mainly from direct modification or from operator switching.


### G. Complex-Valued Residual Refinement

**Inputs and complex convolution.** The second stage operates in normalized coordinates on two complex inputs, $\boldsymbol r_1=\widehat{\boldsymbol u}_1/r$ and $\boldsymbol j_1=\widehat{\boldsymbol j}_1/r$. Its backbone consists of bias-free complex convolutions [50]. For complex weights $\boldsymbol W=\boldsymbol W_R+\mathrm j\boldsymbol W_I$ and features $\boldsymbol z=\boldsymbol z_R+\mathrm j\boldsymbol z_I$,

$$
\boldsymbol W*\boldsymbol z=(\boldsymbol W_R*\boldsymbol z_R-\boldsymbol W_I*\boldsymbol z_I)+\mathrm j(\boldsymbol W_R*\boldsymbol z_I+\boldsymbol W_I*\boldsymbol z_R). \tag{23}
$$

Biases are removed so that every layer commutes with global complex scaling: multiplying the input by $c$ multiplies the output by exactly $c$.

**Phase-invariant real gates.** The difficulty of complex networks lies in the nonlinearity. Applying ReLU or tanh separately to real and imaginary parts makes the output depend on the absolute phase of the input and breaks the commutation above. We place all nonlinearity in a phase-invariant real branch and apply its output as a multiplicative gate to the complex features. The output $\boldsymbol v$ of each complex convolution is multiplied by a gate $\boldsymbol g$:

$$
\boldsymbol z^{+}=\boldsymbol g\odot\boldsymbol v,\qquad
\boldsymbol g=\operatorname{sigmoid}\Big(\operatorname{SiLU}\big(\mathcal K[\log(|\boldsymbol v|_\varepsilon+10^{-3}),\,\boldsymbol\psi]\big)\Big),\qquad
|v|_\varepsilon=\sqrt{|v|^2+10^{-14}}, \tag{24}
$$

where $\mathcal K$ is a real convolution and $\boldsymbol\psi$ consists of six phase-invariant conditioning channels: the point-wise log magnitudes of the two inputs, the real and imaginary parts of the lag product of $\boldsymbol r_1$, and the real and imaginary parts of $\boldsymbol r_1\odot\boldsymbol j_1^{*}$. The last two measure, at each instant, how strongly the preliminary restoration is correlated with the removed jammer, and are the main cue for identifying the shrinkage bias in (20); at lower resolutions $\boldsymbol\psi$ is obtained by average pooling. The gate scales only the modulus of the complex features and leaves their phase unchanged. Modulating intermediate features with conditioning features has a counterpart in image restoration [41]; the difference is that the modulation here is purely multiplicative, without the additive offset that would break phase relations. Because SiLU is bounded below by $-0.278$, the gate takes values in about $(0.431,1)$; it is a continuous strength adjuster and cannot switch a position off completely.

**Network hierarchy.** The refinement network is a four-level complex U-Net with widths $9,13,17,21$ and lengths $T,T/2,T/4,T/8$, each level containing complex residual blocks [37]. At the lowest resolution, three complex convolution blocks with dilation rates 1, 2 and 4 enlarge the temporal receptive field without further downsampling, so that the network covers similar fragments tens of samples apart produced by repeated forwarding. The decoder returns to full resolution through three upsampling steps with skip connections, and a final bias-free complex convolution outputs a single-channel proposal $\widetilde{\boldsymbol\delta}$. Upsampling interpolates real and imaginary parts linearly and downsampling uses average pooling; both commute with complex scaling.

**Pulse-level scalar gate.** The in-layer gates adjust features at each position and channel but cannot decide whether a pulse needs refinement at all. For a jammer-free input, for example, the first-stage output is already close to the reference and the proposal should hardly be used. An independent scalar head therefore summarizes the mean, standard deviation and 95th percentile of the six conditioning channels over the pulse (18 values) together with the ratio of first-stage jammer energy to observation energy and the peak-to-average power ratio of the compressed preliminary restoration, 20 inputs in total, and a $20\to4\to1$ perceptron outputs

$$
\alpha=\operatorname{sigmoid}\big(f_\alpha(\boldsymbol h_\alpha)\big)\in(0,1),\qquad
\boldsymbol\delta_2=r\,\alpha\,\widetilde{\boldsymbol\delta}. \tag{25}
$$

The energy ratio indicates how much the first stage has removed, and the peak-to-average ratio indicates whether the compressed preliminary restoration still contains obvious false peaks; both are direct cues for whether further correction is needed. $\alpha$ limits only the fraction of the proposal that is applied, not the energy of the proposal itself.

**When refinement helps.** Let $\boldsymbol e_1=\widehat{\boldsymbol u}_1-\boldsymbol u$ be the first-stage error. Since $\widehat{\boldsymbol u}-\boldsymbol u=\boldsymbol e_1-\boldsymbol\delta_2$,

$$
\|\widehat{\boldsymbol u}-\boldsymbol u\|_2^2-\|\boldsymbol e_1\|_2^2=\|\boldsymbol\delta_2\|_2^2-2\operatorname{Re}\big(\boldsymbol e_1^H\boldsymbol\delta_2\big). \tag{26}
$$

The restoration error of a sample therefore decreases if and only if $2\operatorname{Re}(\boldsymbol e_1^H\boldsymbol\delta_2)>\|\boldsymbol\delta_2\|_2^2$, i.e., the correction is sufficiently aligned with the first-stage error and its magnitude is less than twice the projection of the error onto it. Equation (26) holds exactly for each sample and requires no independence or linearization. It also clarifies the role of the scalar gate: for a given direction the optimal $\alpha$ is $\operatorname{Re}(\boldsymbol e_1^H\widetilde{\boldsymbol\delta})/(r\|\widetilde{\boldsymbol\delta}\|_2^2)$ clipped to $(0,1)$, and when the direction of the proposal is unreliable a small $\alpha$ limits the possible increase of error. The same analysis applies to target damage if $\boldsymbol\delta_2$ is replaced by the difference of the refinements in the two jammed states.

Table I lists the interfaces and sizes of the network.

**TABLE I. Modules and interfaces of ReCoVP-Hybrid** (batch dimension omitted)

|Module|Input → output|Structure|Role|
|---|---|---|---|
|RMS normalization|$\mathbb C^{1800}\to\mathbb C^{1800}$, scale $r$|(12)|Common amplitude scale; physical output scale kept|
|Feature encoder–decoder|real $4\times1800\to13\times1800$|widths 13/26/44, two downsamplings, GN+ReLU, skips|Fuse energy, template matching and local phase change|
|Candidate selection|correlation magnitude → 16 integer positions|13-point max pooling, spacing ≥6|Locate possible delayed waveforms|
|Delay refinement head|$59\to64\to1$|MLP, bounded ±2 samples|Absorb fractional delays and peak shifts|
|Window network|real $22\times1800\to1\times1800$ per candidate|$22\to12\to1$ point-wise conv, sigmoid, smoothing|Estimate the time support of each atom|
|Dictionary and regularization head|$\mathbb C^{1800\times16}$; $61\to16\to1$|(16), (17)|Build atoms; set per-candidate regularization|
|Coefficient solve|$16\times16$ positive-definite system|(19), double precision|Complex amplitudes and $\widehat{\boldsymbol j}_1$|
|Complex encoder|2 complex inputs → complex $9\times1800\to13\times900\to17\times450\to21\times225$|bias-free complex conv, residual blocks, phase-invariant real gates|Aggregate multiscale residual structure|
|Dilated bottleneck|complex $21\times225$|dilation 1/2/4|Enlarge temporal receptive field|
|Complex decoder|complex $21\times225\to1\times1800$|three upsamplings, skips|Full-resolution proposal $\widetilde{\boldsymbol\delta}$|
|Scalar gate|real 20 → scalar $\alpha$|$20\to4\to1$ MLP, sigmoid|Fraction of the proposal applied to the pulse|

The network has 57,525 effective real parameters, a complex parameter counting as two. Most of them lie in the feature encoder and the complex refinement network; dictionary construction and the coefficient solve have no trainable parameters.

[Figure 3 placeholder: Details of the first stage. (a) Correlation magnitude of a jammed pulse and the 16 selected candidates; (b) multiscale matching features and the estimated window $\boldsymbol m_k$ of one candidate, overlaid with the true sampling window; (c) scatter of regularization weight $\lambda_k$ against correlation strength $\rho_k$; (d) real part of the first-stage estimate $\widehat{\boldsymbol j}_1$ against the true jammer.]

[Figure 4 placeholder: Complex refinement network. Four-level complex encoder, dilated bottleneck and decoder with channel numbers and lengths; a separate panel showing one "complex convolution → phase-invariant real gate → complex feature" unit and the composition of the 20 inputs of the scalar gate.]

### H. Complex-Scaling Equivariance and Complexity

**Proposition 3 (complex-scaling equivariance).** Assume exact arithmetic and a deterministic tie-breaking rule in candidate selection. For any nonzero complex $c$,

$$
F_\theta(c\boldsymbol y)=c\,F_\theta(\boldsymbol y). \tag{27}
$$

In outline (details in Appendix A): by (12), $r(c\boldsymbol y)=|c|\,r(\boldsymbol y)$ and the normalized input becomes $e^{\mathrm j\arg c}\boldsymbol x$, a global phase rotation. The features (13), the correlation magnitude and the deterministic peak selection are unchanged, so candidates, windows, regularization and dictionary are unchanged; $\widehat{\boldsymbol c}$ is multiplied by $e^{\mathrm j\arg c}$ and $\widehat{\boldsymbol j}_1$ by $c$. In the second stage, bias-free complex convolutions, complex linear interpolation and average pooling commute with phase rotation, while the phase-invariant conditioning features and gates are unchanged, so $\boldsymbol\delta_2$ is also multiplied by $c$. The proposition means that changes in target amplitude and initial phase do not change how the processor interprets the observation, and the network does not need to learn this symmetry from data. Group-equivariant networks [51] treat more general symmetries; only global complex scaling is considered here.

Proposition 3 relies on exact arithmetic and a fixed tie-breaking rule. In single precision, rounding of the correlation magnitude can change the peak ordering and switch a candidate. Section V-I reports the measured numerical deviation.

**Complexity.** Let $K$ be the number of candidates. Template correlation and fractional-delay template generation can be implemented with FFTs at cost $O(KN\log N)$; forming $\boldsymbol B^H\boldsymbol B$ costs $O(TK^2)$ and solving the $K\times K$ system $O(K^3)$; the convolutional networks of both stages scale linearly with $T$. With $T=1800$ and $K=16$, the dictionary computations and the convolutional networks are of the same order in total cost.

## IV. Four-State Joint Training

The structure of Section III restricts the form of the correction but does not specify how to trade jammer removal against target preservation. The training objective constrains both on the four states (6): endpoint losses require the correction in each state to approach the jammer that should be removed, and the target-increment loss requires the correction not to change unnecessarily when the target is added.

### A. Waveform and Pulse-Compression Losses

For state $ab$, the correction to be removed is $\boldsymbol j^\star_{ab}=b\,\boldsymbol j$ and the reference is $\boldsymbol u_{ab}=\boldsymbol\epsilon+a\boldsymbol s$. Let $\langle\cdot\rangle$ denote the arithmetic mean over training records, pulses and the four states. For the correction $C$ of either stage, the waveform loss is

$$
\mathcal L_{\mathrm{wav}}(C)=\Big\langle\frac{\|C(\boldsymbol y_{ab})-\boldsymbol j^\star_{ab}\|_2^2}{\max(\|\boldsymbol u_{ab}\|_2^2,10^{-12})}\Big\rangle. \tag{28}
$$

The target for the two jammer-free states is the zero correction, so (28) directly penalizes changes to clean inputs. Normalizing by the energy of $\boldsymbol u_{ab}$ keeps samples of different SNR on the same scale.

Waveform error is spread uniformly over time, whereas detection depends on compressed peaks. Following CVEDNet [19], which constrains both the waveform and the pulse-compression domain, a pulse-compression loss is added:

$$
\mathcal L_{\mathrm{pc}}(C)=\Big\langle\frac{\|\mathcal H[C(\boldsymbol y_{ab})-\boldsymbol j^\star_{ab}]\|_2^2}{\max(\|\mathcal H\boldsymbol u_{ab}\|_2^2,10^{-12})}\Big\rangle, \tag{29}
$$

where $\mathcal H$ is the zero-padded correlation with the known LFM template.

### B. Target-Increment Loss

From $F(\boldsymbol y)=\boldsymbol y-C(\boldsymbol y)$ and (6), the target-increment error between the two jammed states equals the negative difference of the corrections:

$$
F(\boldsymbol y_{11})-F(\boldsymbol y_{01})-\boldsymbol s=-\big[C(\boldsymbol y_{11})-C(\boldsymbol y_{01})\big]. \tag{30}
$$

The target-damage loss is therefore

$$
\mathcal L_{\mathrm{tar}}(C)=\Big\langle\frac{\|C(\boldsymbol y_{11})-C(\boldsymbol y_{01})\|_2^2}{\max(\|\boldsymbol s\|_2^2,10^{-12})}\Big\rangle. \tag{31}
$$

Equation (31) penalizes the change of the jammer estimate caused by adding the target and is directly related to both terms of (21): it constrains the modification of the target by the current operator and the operator change induced by the target. Four-state samples are needed only during training, not at inference.

### C. Residual Tail-Peak Loss

False alarms arise mainly from residual peaks in target-free cells, and these peaks carry a small fraction of the total residual energy, to which the waveform loss is insensitive. For the target-free jammed state, let $\boldsymbol v=\boldsymbol j-C(\boldsymbol y_{01})$ be the residual without noise. Pulse compression and an $M$-point slow-time FFT over the CPI give a range–Doppler power map. The denominator is the mean power of the same noise realization over all range–Doppler cells; the ratios (in dB) of residual power to this mean within range indices $[36,1165)$ are computed, and the eight largest are denoted $q_l$, $l=1,\ldots,8$. A reference level $q_0=10\log_{10}\gamma$ is defined from the ideal CA-CFAR threshold factor $\gamma=N_r(p_0^{-1/N_r}-1)$ with $N_r=436$ and $p_0=10^{-6}$. The tail loss is

$$
\mathcal L_{\mathrm{tail}}(C)=\frac1{q_0}\Big\langle\Big[3\log\{1+e^{(q_l-q_0)/3}\}-3\log2\Big]_+\Big\rangle_{l=1,\ldots,8}, \tag{32}
$$

with $[x]_+=\max(x,0)$. It penalizes only the few peaks that exceed the detection-threshold level, and the softplus makes the penalty grow smoothly near the threshold. Because of the top-8 selection and the truncation the loss is piecewise smooth; it constrains the peaks of the known jammer residual relative to noise and is not a differentiable substitute for the actual false-alarm rate.

### D. Joint Objective and Optimization

The loss of each stage is

$$
\mathcal L_{\mathrm{stage}}(C)=\mathcal L_{\mathrm{wav}}(C)+0.1\,\mathcal L_{\mathrm{pc}}(C)+\mathcal L_{\mathrm{tar}}(C)+0.05\,\mathcal L_{\mathrm{tail}}(C), \tag{33}
$$

and the total loss is $\mathcal L=\mathcal L_{\mathrm{stage}}(C_{\mathrm{final}})+0.25\,\mathcal L_{\mathrm{stage}}(C_1)$, where $C_1=\widehat{\boldsymbol j}_1$ is the first-stage correction and $C_{\mathrm{final}}=\widehat{\boldsymbol j}_1+\boldsymbol\delta_2$ the total correction. The auxiliary first-stage term keeps the dictionary branch producing a meaningful jammer estimate in the presence of the refinement branch, preventing the degenerate solution in which the first stage is arbitrary and the second does all the work. Both stages are optimized jointly; the first stage is neither pretrained nor frozen. The weights are fixed; adaptive weighting such as GradNorm [52] could balance the gradient scales of the losses but is not used here.

The optimizer is Adam [53] ($\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, no weight decay). The learning rate is warmed up linearly to $10^{-3}$ in the first epoch and follows a cosine schedule to $2\times10^{-5}$ at epoch 128 and $2\times10^{-6}$ at epoch 256. The effective batch is 32 records, accumulated from four micro-batches of 8, with gradient-norm clipping at 5. Each epoch has 48 updates and training runs for 256 epochs (12,288 updates); the parameters of the last epoch are used, without selecting epochs on the evaluation set.


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


## VI. Discussion

### A. Why a Controlled Correction Lowers Background False Alarms

The most stable advantage of the proposed method lies not in target fidelity but on the background side: its change to jammer-free inputs is more than an order of magnitude below that of 12 families of convolutional networks of the same size, and its frame false-alarm rate in jammed backgrounds is about half of theirs. Both can be traced to how the correction is organized. For jammer-free inputs, Corollary 1 shows that the first stage removes on average no more than $K/T$ of the noise energy, and the scalar gate almost switches off the refinement branch in jammer-free states (Section V-I), so the processor is close to the identity and the statistics of the CFAR reference window are left intact. For jammed inputs, false alarms come from residual components still coherent with the transmitted waveform, since only these refocus into peaks after pulse compression. The first-stage atoms are windowed delayed LFMs whose coefficients are aligned in amplitude and phase by least squares, so what is removed most completely is exactly the component that would focus after compression; a generic convolutional network regresses the correction waveform under a mean-squared error and is insensitive to whether the residual is coherent. This explanation can be tested directly through the ratio of compressed residual peak to residual energy (figure placeholder in Section V-E).

The same mechanism explains the cost. The true target is also a delayed LFM, and when a candidate falls near the target delay the leakage term in (20) removes part of the target energy as well, producing the mean amplitude shrinkage. CorrectionNet without dechirping and the U-Net families lack this inductive bias and have lower target damage, at the price of larger background changes and more false alarms. The trade-off is therefore not a matter of network capacity but of where the correction is allowed to lie. It follows that target shrinkage should be reduced by making candidate selection and regularization more sensitive to the target delay, rather than by relaxing the degrees of freedom of the correction.

### B. Division of Work Between Structured Estimation and Residual Compensation

ReCoVP-Hybrid splits the jammer correction into two parts: what can be represented as "local window × delayed LFM" is handled by the dictionary and closed-form coefficients, and the rest is compensated by a complex-valued network. Unlike unfolding an iterative algorithm into a fixed number of layers [4,34], the linear coefficients are eliminated exactly for the given dictionary and the network estimates only three kinds of nonlinear parameters: delays, windows and regularization. Unlike cascades that first classify segments and then reconstruct the target [32,33], no hard segment selection is applied to the observation; the correction to be subtracted is estimated over the full pulse. ISRJ-FNet constrains a filtering network with structural priors of the target [35], whereas here the correction is constrained directly by the target-increment error. These differences concern computational interfaces and the objects being constrained; their performance implications must be tested by comparisons under a common protocol.

Tables III and IV show the role of the two stages from two angles. The dictionary-only ReCoVP-Base changes clean inputs the least but has the largest restoration error; adding refinement lowers the restoration error by an order of magnitude while keeping background changes clearly below those of generic convolutional networks. This matches Proposition 2: shrinkage bias and representation residual are the main first-stage errors, and they are exactly the components that the refinement branch can recognize from $(\widehat{\boldsymbol u}_1,\widehat{\boldsymbol j}_1)$. On the other hand, refinement increases the mean attenuation while lowering the mean error, and target damage increases in at least half of the scenes, so the refinement branch is not fully transparent to the target. Equation (26) gives an explanation: any difference between the refinement proposals of the jammed and the target-plus-jammed states, however small, becomes part of the target-increment error.

### C. Measuring Target Fidelity

Equations (8) and (10) show that restoration error, target damage and amplitude preservation are related but cannot be inferred from one another. The inconsistent rankings in Table IV illustrate this: the generic convolutional networks have lower target damage but larger background changes, RSDR-Net has lower restoration error but higher target damage, and ReCoVP-Hybrid sits at lower background change and false alarms with some target shrinkage. If only restoration error or SJRIF were reported, none of these differences would be visible. The shared-background four states provide a way to measure the target response separately in simulation; for measured data, where target truth is unavailable, known-target injection or cooperative-target experiments are needed to approximate it.

Equations (21)–(22) further indicate that the target damage of an adaptive processor consists of direct modification and operator change. Only the damage of the complete output is reported here, without separating the two. Recording the candidate sets and $\|\boldsymbol P_1-\boldsymbol P_0\|_2$ before and after adding the target would show whether damage concentrates in scenes with candidate switching; interpretability tools such as local attribution maps [55] and representation-similarity analysis [56] could also be used to inspect how the features of the two stages respond to the target region.

### D. Restoration and Detection

The restoration loss, the tail loss and CFAR detection act on different objects: the restoration loss constrains all samples, the tail loss constrains the peaks of the noise-free jammer residual, and CFAR operates on the noisy output with an adaptive threshold. In Table III both SJRIF and the frame false-alarm rate improve, yet the intervals for detection probability and cell false-alarm rate contain zero; better restoration metrics do not automatically translate into a detection gain at the same false-alarm level. Detection performance should be compared at equal measured false-alarm operating points (Section V-F).

### E. Limitations

The results have the following limitations. First, all data are simulated, with a single narrowband target and parameters constant within the CPI; multiple targets, range migration, receiver nonlinearity and measured jamming are not included, and for multiple or extended targets the shared-background target increment must be defined per target. Second, the quantitative results for Hybrid in Sections V-D and V-E come from a single trained instance evaluated on the development set, while the baselines already cover three trained instances; the confidence intervals reflect only scene sampling and include neither Hybrid's own training randomness nor bias due to method selection. Third, comparisons with traditional and representative learning methods under a common protocol are incomplete. Fourth, in the current GPU implementation one CPI takes about 30 ms, far longer than the 30 μs PRI, so online processing at the pulse arrival rate requires further engineering optimization.

## VII. Conclusion

This paper studied receiver-side complex waveform restoration under ISRJ for a fixed LFM waveform. With the jammer-free observation containing the realized noise as the reference, restoration error equals jammer estimation error, and the changes that the processor makes to clean inputs and to the true target can be measured separately. On this basis we proposed ReCoVP-Hybrid: the first stage estimates candidate delays, local windows and regularization weights from phase-invariant features and solves for the jammer coefficients in closed form by regularized variable projection on the resulting complex dictionary; the second stage reads the preliminary restoration and the estimated jammer and compensates the remaining error with a complex-valued residual network controlled by phase-invariant real gates.

The analysis shows that, for a fixed dictionary, the coefficient subproblem always has a unique solution and the associated operator removes only part of the energy in every direction, removing on average no more than the ratio of the number of candidates to the signal length from white noise; the first-stage error decomposes into regularization shrinkage bias, representation residual and target–noise leakage, which motivates the refinement stage; and input-adaptive changes of the dictionary introduce into the target-increment error a term proportional to the background energy, so target fidelity must be evaluated on the complete output. The whole mapping is equivariant to complex scaling of the input under exact arithmetic.

Simulations show that the refinement stage markedly reduces the mean restoration error and mean target damage and lowers the jammed frame false-alarm rate, while increasing the mean attenuation. Compared with 12 families of convolutional restoration networks of similar size, each with three trained instances, the proposed method has lower restoration error, jammer-free modification and jammed frame false-alarm rate; the jammer-free modification is more than an order of magnitude lower and the frame false-alarm rate is roughly halved, at the cost of higher target damage than some baselines and stronger amplitude shrinkage. For ISRJ suppression, confining the correction to an analytically solvable jammer subspace thus keeps the detection background clean more effectively than enlarging network capacity. Future work will complete the structural ablation and multi-instance statistics under matched training conditions, add detection comparisons at equal measured false-alarm rate and common-protocol comparisons with traditional and external learning methods, and extend the method to multiple targets and measured data.

## References

[1] Wang X, Liu J, Zhang W, Fu Q, Liu Z, Xie X. Mathematic principles of interrupted-sampling repeater jamming (ISRJ). Science in China Series F: Information Sciences, 2007, 50(1): 113–123. https://doi.org/10.1007/s11432-007-2017-y

[2] Zhou C, Liu Q, Hu C. Time-frequency analysis techniques for recognition and suppression of interrupted sampling repeater jamming. Journal of Radars, 2019, 8(1): 100–106. https://doi.org/10.12000/JR18080

[3] Zhang Y, Jiu B, Peng X, Liu H, Jiang W. A new scheme of target detection for pulse Doppler radar in interrupted sampling repeater jamming environment. IET Radar, Sonar & Navigation, 2022, 16: 1836–1850. https://doi.org/10.1049/rsn2.12300

[4] Huang Q, Zhang L. Interrupted sampling repeater jamming suppression using complex-valued FISTA-Net. IEEE Transactions on Aerospace and Electronic Systems, 2026, 62: 3270–3286. https://doi.org/10.1109/TAES.2025.3646182

[5] Guo T, Zhan H, Su X, Wang T. Anti-interrupted sampling repeater jamming method for random pulse repetition interval and intra-pulse frequency agile radar. IET Radar, Sonar & Navigation, 2023, 17: 1796–1811. https://doi.org/10.1049/rsn2.12465

[6] Zhou Y, Zhou Z, Shi Q, Cui G, Fan P. Complementary ISAC waveform and filter design based on minimized AF sidelobes against interrupted sampling repeater jamming. Signal Processing, 2026, 245: 110574. https://doi.org/10.1016/j.sigpro.2026.110574

[7] Liu Y, Yu L, Wei Y. Parameter estimation and suppression of nonperiodic interrupted sampling repeater jamming based on joint transmit-receive design. Signal Processing, 2026: 110998. https://doi.org/10.1016/j.sigpro.2026.110998

[8] Li J, Liu H, Wang W. An anti-interrupted sampling repeater jamming method based on SSFA-MCPC waveform design combined with segmented mismatched filtering. Progress In Electromagnetics Research C, 2026, 167: 254–264. https://doi.org/10.2528/PIERC26012303

[9] Chen J, Wu W, Xu S, Chen Z, Zou J. Band pass filter design against interrupted-sampling repeater jamming based on time-frequency analysis. IET Radar, Sonar & Navigation, 2019, 13(10): 1646–1654. https://doi.org/10.1049/iet-rsn.2018.5658

[10] Li J-M, Li B-Z. Anti-interrupted sampling repeater jamming via linear canonical Wigner distribution lightweight LFM detection. Signal Processing, 2026, 239: 110230. https://doi.org/10.1016/j.sigpro.2025.110230

[11] Su H, Bao Q, Pan J, Guo F, Hu W. Waveform-domain adaptive matched filtering for suppressing interrupted-sampling repeater jamming. IEEE Transactions on Aerospace and Electronic Systems, 2024, 60(6): 9247–9260. https://doi.org/10.1109/TAES.2024.3439493

[12] Liu Z, Liang W, Fu N, Qiao L, Zhang J. Interrupted sampling repeater jamming suppression based on Gaussian mixture model and sparse reconstruction. Digital Signal Processing, 2025, 161: 105132. https://doi.org/10.1016/j.dsp.2025.105132

[13] Wei S, Han H, Wei S, Zhang L, Zhou Y. ISRJ suppression using joint morphological component analysis and signal reconstruction based on jamming parameters estimation. IEEE Transactions on Aerospace and Electronic Systems, 2026, 62. https://doi.org/10.1109/TAES.2025.3648973

[14] Wei S, Zhang H, Zhou Y, Zhang L. Multiple models fusion filtering for interrupted sampling repeated jamming suppression. Digital Signal Processing, 2026, 168: 105487. https://doi.org/10.1016/j.dsp.2025.105487

[15] Cai T T, Wang L. Orthogonal matching pursuit for sparse signal recovery with noise. IEEE Transactions on Information Theory, 2011, 57(7): 4680–4688. https://doi.org/10.1109/TIT.2011.2146090

[16] He X, Liao K, Peng S, Tian Z, Huang J. Interrupted-sampling repeater jamming-suppression method based on a multi-stages multi-domains joint anti-jamming depth network. Remote Sensing, 2022, 14(14): 3445. https://doi.org/10.3390/rs14143445

[17] Wang H, Chen B, Ye Q, Huang F. MREDNet: A multi-residual encoder-decoder network for recovering LFM interfered by intra-pulse retransmission interferences. Signal Processing, 2024, 217: 109329. https://doi.org/10.1016/j.sigpro.2023.109329

[18] Wang Y, Li Y, Zhou Z, Yu G, Li Y. An attention-guided complex-valued transformer for intra-pulse retransmission interference suppression. Remote Sensing, 2024, 16(11): 1989. https://doi.org/10.3390/rs16111989

[19] Dai G, Yang X, Huan S, Chen Z, Peng C, Xu Y. End-to-end interrupted sampling repeater jamming countermeasure network under low signal-to-noise ratio. Sensors, 2025, 25(13): 3925. https://doi.org/10.3390/s25133925

[20] Wu Y, Duan L, Yang L, Liu Z, Xing M, Quan Y. Fine-grained recognition and suppression of ISRJ based on UNet-A. IEEE Geoscience and Remote Sensing Letters, 2024, 21: 1–5. https://doi.org/10.1109/LGRS.2024.3448611

[21] Wang G, Lv Q, Liu L, Yang R, Bie B, Wu Y, Quan Y. An end-to-end deep learning framework for separation and parameter measurement of composite intermittent sampling repeater jamming. Aerospace Science and Technology, 2026, 168: 111218. https://doi.org/10.1016/j.ast.2025.111218

[22] Wu Z, Ji T, Cao Y, Zhang L, Zhou C, Yang L. Suppression of radar compound interrupted sampling and repeating jamming based on self-supervised learning method. IEEE Transactions on Aerospace and Electronic Systems, 2025, 61. https://doi.org/10.1109/TAES.2025.3555247

[23] Zhang H, Wei S, Wei S, Zhang L, Ren P, Zhou Y. Intensive interrupted sampling repeater jamming detection based on Transformer–CFAR fusion detection model. IEEE Transactions on Radar Systems, 2024, 2: 936–. https://doi.org/10.1109/TRS.2024.3465017

[24] Fuchs A, Rock J, Tóth M, Meissner P, Pernkopf F. Complex-valued convolutional neural networks for enhanced radar signal denoising and interference mitigation. In: Proc. IEEE Radar Conference (RadarConf21), 2021. https://doi.org/10.1109/RadarConf2147009.2021.9455296

[25] Ristea N-C, Anghel A, Ionescu R T. Estimating the magnitude and phase of automotive radar signals under multiple interference sources with fully convolutional networks. IEEE Access, 2021, 9: 153491–153507. https://doi.org/10.1109/ACCESS.2021.3128151

[26] Wang J, Li R, He Y, Yang Y. Prior-guided deep interference mitigation for FMCW radars. IEEE Transactions on Geoscience and Remote Sensing, 2022, 60: 5118316. https://doi.org/10.1109/TGRS.2022.3211605

[27] Wang J, Li R, Zhang X, He Y. Interference mitigation for automotive FMCW radar based on contrastive learning with dilated convolution. IEEE Transactions on Intelligent Transportation Systems, 2024, 25(1): 545–. https://doi.org/10.1109/TITS.2023.3306576

[28] Rameez M, Javadi S, Dahl M, Pettersson M I. Evaluating deep learning approaches for mitigating interference in automotive radar systems. Computers and Electrical Engineering, 2026, 140: 111496. https://doi.org/10.1016/j.compeleceng.2026.111496

[29] Shlezinger N, Whang J, Eldar Y C, Dimakis A G. Model-based deep learning. Proceedings of the IEEE, 2023, 111(5): 465–499. https://doi.org/10.1109/JPROC.2023.3247480

[30] Zhang J, Ghanem B. ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing. In: Proc. IEEE/CVF CVPR, 2018: 1828–1837.

[31] Kovács P, Bognár G, Huber C, Huemer M. VPNET: Variable projection networks. International Journal of Neural Systems, 2022, 32(1): 2150054. https://doi.org/10.1142/S0129065721500544

[32] Meng Y, Yu L, Wei Y. Interrupted sampling repeater jamming suppression based on multiple extended complex-valued convolutional auto-encoders. IET Radar, Sonar & Navigation, 2024, 18(8): 1274–1290. https://doi.org/10.1049/rsn2.12568

[33] Lei C, Ke W, Xiaoyi P, Zhaoyu G, Shunping X. Interrupted sampling repeater jamming suppression based on point-wise temporal classification and sparse signal reconstruction. IET Radar, Sonar & Navigation, 2026, 20: e70251. https://doi.org/10.1049/rsn2.70251

[34] Wei S, Han H, Zhang H, Wei S, Zhou Y, Zhang L. Interrupted sampling repeater jamming suppression for wideband radar with MCA-ADMM-Net. IEEE Transactions on Neural Networks and Learning Systems, 2026: 1–15. https://doi.org/10.1109/TNNLS.2026.3736439

[35] Liu M, Gao X, Qiu X, Zhao Y. Structure-informed deep filtering for radar HRRP ISRJ mitigation. IEEE Signal Processing Letters, 2026, 33: 3601–3605. https://doi.org/10.1109/LSP.2026.3726844

[36] Luo Y, Mesgarani N. Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019, 27(8): 1256–1266. https://doi.org/10.1109/TASLP.2019.2915167

[37] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proc. IEEE CVPR, 2016: 770–778. https://doi.org/10.1109/CVPR.2016.90

[38] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 2017, 26(7): 3142–3155. https://doi.org/10.1109/TIP.2017.2662206

[39] Zamir S W, Arora A, Khan S, Hayat M, Khan F S, Yang M-H, Shao L. Multi-stage progressive image restoration. In: Proc. IEEE/CVF CVPR, 2021: 14821–14831.

[40] Guo S, Yan Z, Zhang K, Zuo W, Zhang L. Toward convolutional blind denoising of real photographs. In: Proc. IEEE/CVF CVPR, 2019: 1712–1722.

[41] Wang X, Yu K, Dong C, Loy C C. Recovering realistic texture in image super-resolution by deep spatial feature transform. In: Proc. IEEE/CVF CVPR, 2018: 606–615.

[42] Wang X, Chan K C K, Yu K, Dong C, Loy C C. EDVR: Video restoration with enhanced deformable convolutional networks. In: Proc. IEEE/CVF CVPR Workshops, 2019.

[43] Karniadakis G E, Kevrekidis I G, Lu L, Perdikaris P, Wang S, Yang L. Physics-informed machine learning. Nature Reviews Physics, 2021, 3: 422–440. https://doi.org/10.1038/s42254-021-00314-5

[44] Zhu B, Liu J Z, Cauley S F, Rosen B R, Rosen M S. Image reconstruction by domain-transform manifold learning. Nature, 2018, 555: 487–492. https://doi.org/10.1038/nature25988

[45] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation. In: Proc. MICCAI, 2015: 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

[46] Wu Y, He K. Group normalization. In: Proc. ECCV, 2018: 3–19.

[47] Golub G H, Pereyra V. The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. SIAM Journal on Numerical Analysis, 1973, 10(2): 413–432. https://doi.org/10.1137/0710036

[48] Golub G, Pereyra V. Separable nonlinear least squares: the variable projection method and its applications. Inverse Problems, 2003, 19(2): R1–R26. https://doi.org/10.1088/0266-5611/19/2/201

[49] O'Leary D P, Rust B W. Variable projection for nonlinear least squares problems. Computational Optimization and Applications, 2013, 54: 579–593. https://doi.org/10.1007/s10589-012-9492-9

[50] Trabelsi C, Bilaniuk O, Zhang Y, et al. Deep complex networks. In: Proc. ICLR, 2018.

[51] Cohen T, Welling M. Group equivariant convolutional networks. In: Proc. ICML, PMLR 48, 2016: 2990–2999.

[52] Chen Z, Badrinarayanan V, Lee C-Y, Rabinovich A. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: Proc. ICML, PMLR 80, 2018: 794–803.

[53] Kingma D P, Ba J. Adam: A method for stochastic optimization. In: Proc. ICLR, 2015.

[54] Efron B. Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 1979, 7(1): 1–26. https://doi.org/10.1214/aos/1176344552

[55] Gu J, Dong C. Interpreting super-resolution networks with local attribution maps. In: Proc. IEEE/CVF CVPR, 2021: 9199–9208.

[56] Kornblith S, Norouzi M, Lee H, Hinton G. Similarity of neural network representations revisited. In: Proc. ICML, PMLR 97, 2019: 3519–3529.

## Appendix A. Proofs

**Proof of Proposition 1.** Let $\boldsymbol A=\boldsymbol B\boldsymbol\Lambda^{-1/2}$. Then $\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda=\boldsymbol\Lambda^{1/2}(\boldsymbol A^H\boldsymbol A+\boldsymbol I)\boldsymbol\Lambda^{1/2}$ and
$$
\boldsymbol P=\boldsymbol A(\boldsymbol A^H\boldsymbol A+\boldsymbol I)^{-1}\boldsymbol A^H .
$$
With the singular value decomposition $\boldsymbol A=\boldsymbol U\boldsymbol\Sigma\boldsymbol V^H$, $\boldsymbol P=\boldsymbol U\boldsymbol\Sigma(\boldsymbol\Sigma^H\boldsymbol\Sigma+\boldsymbol I)^{-1}\boldsymbol\Sigma^H\boldsymbol U^H$, whose eigenvalues are $\sigma_k^2/(1+\sigma_k^2)\in[0,1)$ together with $T-K$ zeros. $\boldsymbol P$ is clearly Hermitian positive semidefinite and its trace is the sum of its nonzero eigenvalues. Since all eigenvalues are strictly less than one, $\|\boldsymbol P\boldsymbol y\|_2\le\frac{\sigma_1^2}{1+\sigma_1^2}\|\boldsymbol y\|_2$, and the eigenvalues of $\boldsymbol I-\boldsymbol P$ lie in $(0,1]$, so it is nonexpansive. If some $\sigma_k>0$, the corresponding eigenvalue lies strictly between 0 and 1 and is not equal to its square, hence $\boldsymbol P^2\neq\boldsymbol P$. ∎

**Proof of Proposition 2.** From $\widehat{\boldsymbol u}_1=(\boldsymbol I-\boldsymbol P)\boldsymbol y$ and $\boldsymbol u=\boldsymbol s+\boldsymbol\epsilon$,
$$
\widehat{\boldsymbol u}_1-\boldsymbol u=(\boldsymbol I-\boldsymbol P)\boldsymbol j-\boldsymbol P(\boldsymbol s+\boldsymbol\epsilon).
$$
Let $\boldsymbol c^\star$ be the least-squares coefficient of $\boldsymbol j$ on $\operatorname{span}(\boldsymbol B)$ (the minimum-norm solution if $\boldsymbol B$ is rank deficient). Then $\boldsymbol j_\perp=\boldsymbol j-\boldsymbol B\boldsymbol c^\star$ satisfies $\boldsymbol B^H\boldsymbol j_\perp=\boldsymbol0$ and hence $\boldsymbol P\boldsymbol j_\perp=\boldsymbol0$. Moreover,
$$
(\boldsymbol I-\boldsymbol P)\boldsymbol B=\boldsymbol B-\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol B^H\boldsymbol B
=\boldsymbol B(\boldsymbol B^H\boldsymbol B+\boldsymbol\Lambda)^{-1}\boldsymbol\Lambda,
$$
and substitution gives (20). ∎

**Proof of Proposition 3.** Let $c=|c|e^{\mathrm j\varphi}\neq0$. (i) By (12), $r(c\boldsymbol y)=|c|r(\boldsymbol y)$ and $\boldsymbol x'=e^{\mathrm j\varphi}\boldsymbol x$. (ii) In (13), $|x_n'|=|x_n|$, $a_n'=a_n$ and $\ell_n'=x_n'x_{n-1}'^{*}=\ell_n$; hence the encoder output, the candidate positions (under the same tie-breaking rule), the delays, the windows and $\boldsymbol B$ are unchanged. Also $\rho_k'=|\boldsymbol b_k^He^{\mathrm j\varphi}\boldsymbol x|^2/\widehat\sigma'^2=\rho_k$, since the robust spread depends only on distances and is unaffected by rotation, so $\boldsymbol\Lambda$ is unchanged. (iii) The right-hand side of the normal equations is multiplied by $e^{\mathrm j\varphi}$, so $\widehat{\boldsymbol c}'=e^{\mathrm j\varphi}\widehat{\boldsymbol c}$, $\widehat{\boldsymbol j}_1'=|c|r\boldsymbol Be^{\mathrm j\varphi}\widehat{\boldsymbol c}=c\widehat{\boldsymbol j}_1$ and $\widehat{\boldsymbol u}_1'=c\widehat{\boldsymbol u}_1$. (iv) The second-stage inputs become $e^{\mathrm j\varphi}\boldsymbol r_1$ and $e^{\mathrm j\varphi}\boldsymbol j_1$; the log magnitudes and conjugate products in $\boldsymbol\psi$ are unchanged; bias-free complex convolution, complex linear interpolation and average pooling commute with phase rotation; and the gate input $|\boldsymbol v|$ is unchanged, so every layer output is multiplied by $e^{\mathrm j\varphi}$. The energy ratio and peak-to-average ratio in the scalar-gate input are unchanged, so $\alpha'=\alpha$. Hence $\boldsymbol\delta_2'=|c|r\alpha e^{\mathrm j\varphi}\widetilde{\boldsymbol\delta}=c\boldsymbol\delta_2$ and $F_\theta(c\boldsymbol y)=c\widehat{\boldsymbol u}_1-c\boldsymbol\delta_2=cF_\theta(\boldsymbol y)$. ∎

## Data and Code Availability

[To be completed by the authors: public address of code and configuration, data generation procedure and access restrictions.]
