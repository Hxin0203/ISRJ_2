
## VI. Discussion

### A. Division of Work Between Structured Estimation and Residual Compensation

ReCoVP-Hybrid splits the jammer correction into two parts: what can be represented as "local window × delayed LFM" is handled by the dictionary and closed-form coefficients, and the rest is compensated by a complex-valued network. Unlike unfolding an iterative algorithm into a fixed number of layers [4,34], the linear coefficients are eliminated exactly for the given dictionary and the network estimates only three kinds of nonlinear parameters: delays, windows and regularization. Unlike cascades that first classify segments and then reconstruct the target [32,33], no hard segment selection is applied to the observation; the correction to be subtracted is estimated over the full pulse. ISRJ-FNet constrains a filtering network with structural priors of the target [35], whereas here the correction is constrained directly by the target-increment error. These differences concern computational interfaces and the objects being constrained; their performance implications must be tested by comparisons under a common protocol.

Tables III and IV show the role of the two stages from two angles. The dictionary-only ReCoVP-Base changes clean inputs the least but has the largest restoration error; adding refinement lowers the restoration error by an order of magnitude while keeping background changes clearly below those of generic convolutional networks. This matches Proposition 2: shrinkage bias and representation residual are the main first-stage errors, and they are exactly the components that the refinement branch can recognize from $(\widehat{\boldsymbol u}_1,\widehat{\boldsymbol j}_1)$. On the other hand, refinement increases the mean attenuation while lowering the mean error, and target damage increases in at least half of the scenes, so the refinement branch is not fully transparent to the target. Equation (26) gives an explanation: any difference between the refinement proposals of the jammed and the target-plus-jammed states, however small, becomes part of the target-increment error.

### B. Measuring Target Fidelity

Equations (8) and (10) show that restoration error, target damage and amplitude preservation are related but cannot be inferred from one another. The inconsistent rankings in Table IV illustrate this: the generic convolutional networks have lower target damage but larger background changes, RSDR-Net has lower restoration error but higher target damage, and ReCoVP-Hybrid sits at lower background change and false alarms with some target shrinkage. If only restoration error or SJRIF were reported, none of these differences would be visible. The shared-background four states provide a way to measure the target response separately in simulation; for measured data, where target truth is unavailable, known-target injection or cooperative-target experiments are needed to approximate it.

Equations (21)–(22) further indicate that the target damage of an adaptive processor consists of direct modification and operator change. Only the damage of the complete output is reported here, without separating the two. Recording the candidate sets and $\|\boldsymbol P_1-\boldsymbol P_0\|_2$ before and after adding the target would show whether damage concentrates in scenes with candidate switching; interpretability tools such as local attribution maps [55] and representation-similarity analysis [56] could also be used to inspect how the features of the two stages respond to the target region.

### C. Restoration and Detection

The restoration loss, the tail loss and CFAR detection act on different objects: the restoration loss constrains all samples, the tail loss constrains the peaks of the noise-free jammer residual, and CFAR operates on the noisy output with an adaptive threshold. In Table III both SJRIF and the frame false-alarm rate improve, yet the intervals for detection probability and cell false-alarm rate contain zero; better restoration metrics do not automatically translate into a detection gain at the same false-alarm level. Detection performance should be compared at equal measured false-alarm operating points (Section V-F).

### D. Limitations

The results have the following limitations. First, all data are simulated, with a single narrowband target and parameters constant within the CPI; multiple targets, range migration, receiver nonlinearity and measured jamming are not included, and for multiple or extended targets the shared-background target increment must be defined per target. Second, the quantitative results of Sections V-D and V-E come from a single trained instance evaluated on the development set; the confidence intervals reflect only scene sampling and include neither training randomness nor bias due to method selection. Third, comparisons with traditional and representative learning methods under a common protocol are incomplete. Fourth, in the current GPU implementation one CPI takes about 30 ms, far longer than the 30 μs PRI, so online processing at the pulse arrival rate requires further engineering optimization.

## VII. Conclusion

This paper studied receiver-side complex waveform restoration under ISRJ for a fixed LFM waveform. With the jammer-free observation containing the realized noise as the reference, restoration error equals jammer estimation error, and the changes that the processor makes to clean inputs and to the true target can be measured separately. On this basis we proposed ReCoVP-Hybrid: the first stage estimates candidate delays, local windows and regularization weights from phase-invariant features and solves for the jammer coefficients in closed form by regularized variable projection on the resulting complex dictionary; the second stage reads the preliminary restoration and the estimated jammer and compensates the remaining error with a complex-valued residual network controlled by phase-invariant real gates.

The analysis shows that, for a fixed dictionary, the coefficient subproblem always has a unique solution and the associated operator removes only part of the energy in every direction; the first-stage error decomposes into regularization shrinkage bias, representation residual and target–noise leakage, which motivates the refinement stage; and input-adaptive changes of the dictionary introduce into the target-increment error a term proportional to the background energy, so target fidelity must be evaluated on the complete output. The whole mapping is equivariant to complex scaling of the input under exact arithmetic.

Simulations show that the refinement stage markedly reduces the mean restoration error and mean target damage and lowers the jammed frame false-alarm rate, while increasing the mean attenuation. Compared with convolutional correction networks of similar size, the proposed method changes clean inputs less and produces fewer background false alarms, but causes higher target damage. Restoration error, target fidelity and background false alarms are three aspects that must be considered together when evaluating ISRJ suppression. Future work will complete the structural ablation and multi-instance statistics under matched training conditions, add common-protocol comparisons with traditional and learning methods, and extend the method to multiple targets and measured data.

## References

{{REFERENCES}}

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
