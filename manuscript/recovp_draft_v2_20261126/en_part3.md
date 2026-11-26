
### G. Complex-Valued Residual Refinement

**Inputs and complex convolution.** The second stage operates in normalized coordinates on two complex inputs, $\boldsymbol r_1=\widehat{\boldsymbol u}_1/r$ and $\boldsymbol j_1=\widehat{\boldsymbol j}_1/r$. Its backbone consists of bias-free complex convolutions [48]. For complex weights $\boldsymbol W=\boldsymbol W_R+\mathrm j\boldsymbol W_I$ and features $\boldsymbol z=\boldsymbol z_R+\mathrm j\boldsymbol z_I$,

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
