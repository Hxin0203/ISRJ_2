# Waveform Restoration under Interrupted-Sampling Repeater Jamming via Regularized Variable Projection and Complex-Valued Residual Refinement

## Abstract

Interrupted-sampling repeater jamming (ISRJ) replays fragments of the radar's own waveform, so after pulse compression it produces false targets with coherent processing gain. Because the jammer and the target echo share the same chirp structure, suppressing the jammer can also change the amplitude, phase and shape of the true target. This paper formulates ISRJ suppression as complex waveform restoration against a noisy reference, namely the observation with the jammer removed but the realized receiver noise retained, and proposes ReCoVP-Hybrid, a two-stage method that combines regularized variable projection with complex-valued residual refinement. The first stage selects candidate delays from the observation using the known linear-frequency-modulated template, estimates a local window and a regularization weight for each candidate from phase-invariant features, and obtains the jammer coefficients in closed form on the resulting complex dictionary. The second stage reads both the preliminary restoration and the estimated jammer, compensates in complex-valued convolution channels the residual that the finite dictionary cannot represent, and controls the correction strength with phase-invariant real gates. Training is carried out on four states that share the same noise and jammer background and jointly constrains waveform error, pulse-compression error, target-increment error and residual tail peaks. We establish uniqueness and contraction properties of the coefficient subproblem for a fixed dictionary, decompose the first-stage error into regularization shrinkage bias, representation residual and target leakage, and show how input-adaptive changes of the dictionary enter the target response. Because the correction is confined to a low-dimensional subspace spanned by a few structured atoms, and the correction strength of each atom is set by its matched energy relative to a robust noise floor, the processor leaves jammer-free inputs almost untouched even though the candidates are selected from the observation itself; what it removes is mainly the component coherent with the transmitted waveform, so the residual is less prone to refocus into false peaks after pulse compression. Simulations show that, at a comparable parameter count, the method attains lower restoration error with smaller background modification and fewer false alarms in jammed backgrounds, at the cost of some attenuation of the target amplitude.

**Index Terms**—Interrupted-sampling repeater jamming, waveform restoration, variable projection, complex-valued neural networks, target fidelity, model-based deep learning.

## I. Introduction

Pulse-compression radars transmit signals with a large time–bandwidth product, such as linear frequency modulation (LFM), to obtain range resolution and processing gain. With mature digital radio-frequency memory (DRFM), a jammer can intercept, store and retransmit fragments of a pulse while the pulse is still being received. Interrupted-sampling repeater jamming (ISRJ) alternates between sampling and forwarding, needs only a single antenna, and remains coherent with the transmitted signal [1]. After matched filtering, the forwarded fragments produce a train of false peaks with processing gain, whose positions, number and amplitudes depend on the sampling width, the sampling period and the forwarding scheme [1,2]. When these peaks fall near the true target or inside its main lobe, detection, tracking and imaging are all affected [3,4].

The difficulty of ISRJ lies in its similarity to the target echo. Unlike noise jamming, every ISRJ fragment lies on the chirp trajectory of the transmitted signal in the time–frequency plane; it differs only by an incomplete time support and a forwarding delay. Removing samples on the basis of amplitude or energy alone is therefore likely to remove part of the true target as well, and attenuation or phase offset of the target further degrades coherent integration over pulses. A suppression method should thus be judged not only by how much jammer it removes but also by how well it preserves the complex response of the true target.

Existing countermeasures fall roughly into three groups. The first group works at the transmitter or jointly at the transmitter and receiver, increasing separability through frequency agility, pulse staggering, or joint design of waveforms and mismatched filters [5–8]. These methods are effective but require cooperation from the waveform and the scheduler, and do not apply directly to receiver-side processing with a fixed waveform. The second group is receiver-side signal processing that exploits the interrupted time support, the time–frequency trajectory, or the sparsity of the jammer. Time–frequency analysis and band-pass filtering build filters from the location of jammer fragments in the time–frequency plane [2,9,10]; Su et al. construct a cumulative waveform coherence function and an adaptive matched filter in the waveform domain [11]; Liu et al. identify jammer-free segments with a Gaussian mixture model and then perform sparse reconstruction [12]; Wei et al. estimate jammer parameters to build a short-time Fourier dictionary and jointly reconstruct target and jammer by morphological component analysis [13], and further propose multiple-model fusion filtering [14]. These methods are physically interpretable, but their performance depends on how accurately fragment boundaries, forwarding parameters or dictionaries are estimated. At low SNR or under varying jammer parameters, missed fragments lose target energy, while misclassified ones leave jammer in the reconstruction [12,15].

The third group is data driven. He et al. build a multistage multidomain network [16]; Wang et al. propose MREDNet, a multi-residual encoder–decoder that restores LFM signals contaminated by intra-pulse retransmission interference [17]; a complex-valued Transformer [18] and the complex-valued encoder–decoder CVEDNet [19] process the received signal directly in the complex domain, the latter combining waveform-domain and pulse-compression-domain losses. Other studies combine recognition and suppression [20], separate and measure composite ISRJ [21], suppress compound repeater jamming with self-supervised learning [22], or detect dense jamming by fusing a Transformer with CFAR [23]. In automotive FMCW interference mitigation, complex-valued convolutional networks [24,25], prior-guided networks [26] and contrastive learning [27] likewise show that learning can exploit jammer features that are hard to describe by rules; a recent systematic evaluation also points out that conclusions about data-driven methods depend strongly on the metrics and test conditions [28].

Model-based learning lies between the two. These methods keep the computational steps of sparse reconstruction or iterative optimization explicit inside the network [29–31]. Meng et al. cascade complex-valued segment extraction, segment denoising and compressed-sensing reconstruction [32]; Lei et al. perform point-wise temporal classification, I/Q residual denoising and sparse reconstruction in sequence [33]; Huang and Zhang unfold complex-valued FISTA into a network [4]; Wei et al. represent the target and the jammer with a delayed-LFM dictionary and a local Fourier dictionary and unfold ADMM for two-component sparse recovery [34]; Liu et al. constrain a filtering network on high-resolution range profiles with structural priors of the target [35]. These works improve interpretability by combining physical and learned models, but most of them still take clean-target reconstruction or the jamming suppression ratio as the main objective and rarely examine the complex response of the processor to the true target separately.

Image and speech restoration offer useful ideas on this point. Conv-TasNet replaces a fixed time–frequency transform with a learnable encoder and mask estimation, showing that the representation and the treatment of phase directly affect separation quality [36]. Residual learning [37,38], multistage progressive restoration [39] and blind denoising of real noise [40] show that decomposing a hard restoration task into stages with clear responsibilities lowers the learning difficulty of each stage. Spatial feature transform modulates intermediate features with conditional information [41], and deformable alignment in video restoration aligns before restoring details [42]. Physics-informed learning [43] and data-driven learning of inverse transforms [44] further show that prior knowledge can enter the representation and the computation of a network rather than only a preprocessing step. These mechanisms cannot be transplanted to radar signals without verification, but their way of dividing work between structured estimation and residual compensation matches the needs of ISRJ suppression.

The two routes sit at opposite ends in the degrees of freedom of the correction. Model-based methods modify only a few parameterized components, so the correction is tightly constrained, but the constraint is only as good as the parameter estimates. Data-driven methods output a correction waveform as long as the observation; they are expressive, yet nothing prevents them from altering clean samples. For detection this matters: the residual left by a generic network often still carries part of the chirp structure of the forwarded fragments and refocuses into peaks in target-free cells after pulse compression, while small modifications of jammer-free inputs change the statistics of the CFAR reference window. We argue that ISRJ suppression needs not a larger network but an evidence-gated correction: its bulk should lie in a low-dimensional subspace of LFM fragments identified from the observation, the amount of correction applied to each fragment should be set by its matched energy relative to the noise floor so that fragments without evidence shrink to zero automatically, and only what this subspace cannot represent should be passed to a network whose correction strength is itself gated.

We therefore study receiver-side complex waveform restoration for a fixed LFM waveform and address three questions. First, since ISRJ fragments inherit the structure of the transmitted waveform, can the explicitly representable part of the jammer be handled by an analytically solvable model so that the network learns only what the model cannot describe? Second, a finite dictionary always leaves a representation residual and regularization introduces shrinkage bias; how can these errors be compensated while keeping the structured estimate? Third, how large is the incidental modification of the target made by an adaptive processor while it removes the jammer, and how can this modification be measured separately during training and evaluation?

To this end we propose ReCoVP-Hybrid (regularized complex variable projection with hybrid refinement). The method takes the jammer-free observation with the realized noise as the reference, so that the ideal correction is exactly the jammer. The first stage selects candidate delays, estimates local windows and regularization weights from phase-invariant features, builds a complex dictionary, and solves for the jammer coefficients by regularized variable projection. The second stage reads the preliminary restoration and the estimated jammer and compensates the remaining error with a complex-valued residual network. What distinguishes this structure from generic restoration networks is how the correction is organized: the first stage has only $K$ complex coefficients per pulse, and the regularization weight of each candidate is set by its matched energy relative to the noise floor, so candidates without evidence are pushed to zero; the second-stage correction is conditioned on the first-stage estimate and its applied fraction is limited by a pulse-level gate. Two consequences follow directly: the change made to jammer-free inputs is governed by evidence rather than by the number of candidates, and what is removed is mainly the component coherent with the transmitted waveform. The main contributions are as follows.

1. An evidence-gated two-stage structure for complex waveform restoration under ISRJ. In the first stage the network estimates where and which LFM fragments are present, while their complex amplitudes are solved analytically by regularized variable projection with a regularization weight set by each candidate's matched energy relative to the noise floor, so that the bulk of the correction lies in a low-dimensional, interpretable subspace and fragments without evidence are switched off automatically. The second stage, conditioned on the preliminary restoration and the removed jammer, compensates in complex-valued channels the representation residual and the regularization shrinkage bias, with correction strength controlled by phase-invariant gates. The whole mapping is equivariant to complex scaling of the input.
2. An analysis that establishes uniqueness and contraction of the coefficient subproblem; shows that the fixed-dictionary $K/T$ bound on the energy removed from noise does not hold for candidates selected from the observation, estimating the removal of plain projection by order statistics; and gives a deterministic bound on the correction energy under evidence gating that needs no independence assumption, explaining why selecting candidates from noise still leaves the background untouched; decomposes the first-stage error into regularization shrinkage bias, representation residual and target–noise leakage to motivate the refinement stage; and derives the effect of the input-adaptive dictionary on the target-increment response, on which a shared-background four-state training objective is built.
3. An evaluation protocol that reports restoration error, target-increment damage, jammer-free modification and two-dimensional CA-CFAR detection side by side, comparing against 12 parameter-matched families of convolutional restoration networks with three trained instances each, the classical Max-TF and GMM-OMP, and the larger CVEDNet and RSDR-Net, under nominal CFAR, equal measured false-alarm rate and a controlled SNR–JSR sweep. At a comparable parameter count the proposed structure has the smallest jammer-free modification and jammed-background false-alarm rate and the highest detection probability at equal false-alarm rate; inference-time structural diagnostics confirm that the background transparency comes from evidence gating. The cost in target-amplitude shrinkage is quantified.

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
