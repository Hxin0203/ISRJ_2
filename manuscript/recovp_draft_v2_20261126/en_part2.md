
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

The feature encoder is a one-dimensional U-Net [49] with widths $4\to13\to26\to44$ and lengths $T\to T/2\to T/4$; each convolution is followed by group normalization [50] and ReLU. The decoder returns to shared features $\boldsymbol h$ of size $13\times T$ through two upsampling steps with skip connections. The skips keep details near fragment boundaries, and the downsampling path gathers context over longer time spans, which helps to distinguish the multiple similar fragments produced by repeated forwarding.

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

This follows the idea of variable projection [45–47]: in a separable nonlinear least-squares problem the linear parameters have a closed-form solution given the nonlinear ones and can be eliminated, so that only the nonlinear parameters need to be optimized. Here the nonlinear parameters (delays, windows, regularization) are estimated by networks, the linear coefficients are obtained analytically from (19), and gradients propagate to the networks through the solution of the linear system [31]. Compared with unfolding a fixed number of iterations [4,30,34], the coefficients are always exactly optimal for the given dictionary and the network does not have to learn the iteration itself. The implementation solves for $\widehat{\boldsymbol c}$ with a linear solver rather than forming an inverse; the solve uses double-precision complex arithmetic and the rest uses single precision.

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
