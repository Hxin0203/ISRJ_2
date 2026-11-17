# Residual and detection-failure diagnosis

Date: 2026-11-17
Scope: exploratory DEV diagnosis for the frozen epoch-256 models. No weights or training data were changed. DEV has already participated in model development, so these results are for hypothesis formation only.

## Protocol and audit trail

The analysis evaluated six models (`full`, `unet`, `no_dechirp`, `no_rms`, `no_pc`, `no_damage`) at seeds 17, 29, and 41 on the same 256 DEV parents. Each parent was the resampling unit; three jammer families were averaged within parent. The analysis read the verified frozen checkpoints, the existing DEV data, and the established pulse-compression, Doppler, and CA-CFAR evaluation path. It did not train, update, or export waveforms.

The completed run is `residual_failure_diagnostic_20261117_v2`. The status record binds DEV SHA-256 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`, frozen configuration SHA-256 `6db6fc300fe8fe76004c5a4d9e208f629cdc4f34c47014246c34e1186e95cd8b`, and diagnostic source SHA-256 `17d1e3521dcbf94d645268cfccc879724b13ccae3a7f16ecfa5958da9635300a`. All 18 checkpoint hashes are recorded in `STATUS.json`; all 18 per-model/seed results are retained in this directory.

## Findings

Three-seed descriptive means on DEV:

| Model | Target Pd | Jammer-frame Pfa | SJRIF (dB) | Background residual / jammer (dB) | MF peak / noise-residual median (dB) | Maximum RD / CA-CFAR threshold (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.860 | 0.153 | 10.30 | -15.52 | 40.72 | 0.42 |
| U-Net | 0.868 | 0.111 | 10.61 | -15.95 | 43.17 | -0.48 |
| No-dechirp | 0.873 | 0.109 | 10.70 | -16.10 | 38.95 | -0.43 |
| No-RMS | 0.872 | 0.146 | 10.34 | -15.99 | 48.05 | 1.28 |
| No-PC | 0.865 | 0.165 | 10.22 | -15.36 | 40.89 | 0.50 |
| No-damage | 0.836 | 0.091 | 10.86 | -15.91 | 34.77 | -0.14 |

The first column of residual evidence is not enough to rank detection behavior: background residual-to-jammer energy is similar across models, while jammer-frame Pfa ranges from 0.091 to 0.165. The no-damage model also illustrates the tradeoff: its lower Pfa coincides with lower Pd (0.836) and substantially greater target-response damage (0.130 versus 0.048 for U-Net). Pfa alone is not a suitable selection objective.

Parent-level associations were consistent across the three seeds. For the full model, the Spearman correlation between the matched-filter residual peak normalized by the median noise-only residual power and jammer-frame false alarm was 0.656–0.677 across seeds; the corresponding association for total background residual energy was only 0.015–0.071. For U-Net, the matched-filter association was 0.546–0.582, whereas total residual energy gave 0.101–0.117. The maximum range-Doppler power relative to its CA-CFAR threshold also tracked jammer-frame false alarms (Spearman 0.337–0.396 for full and 0.452–0.475 for U-Net). The full per-seed bootstrap intervals and top-versus-bottom-quartile event rates are in each result JSON under `failure_associations`.

These results support a narrow next hypothesis: a matched-filtered residual peak is more informative about jammer-frame false alarms than total residual energy in these DEV data. They do not show that the peak causes a false alarm or that optimizing a proxy will improve calibrated Pfa. The peak metric's denominator is the median output residual on the noise-only state, which can be very small; it is a ranking diagnostic, not a calibrated physical noise-power estimate. The RD/CFAR statistic is also an operating-point diagnostic, not an independent estimate of the requested false-alarm probability.

## Decision and next experiment

Proceed to a single-factor loss screen on the exact parameter-matched U-Net. Compare the already completed U-Net baseline with a fresh training that changes only the objective by adding a differentiable tail term on the target-absent jammer state. Form the residual as

\[
r_{j}(p,n)=\hat{x}_{j+e}(p,n)-e(p,n)=j(p,n)-C(j+e)(p,n),
\]

apply the same 600-sample LFM matched filter used by evaluation, retain the complete-waveform nonnegative range lags, then take the eight-pulse slow-time FFT. Normalize the resulting range-Doppler power by the mean target-free thermal-noise range-Doppler power for that parent. Penalize a smooth top-eight tail above the idealized CA-CFAR multiplier for the frozen 436-cell reference window. Keep the endpoint, pulse-compression, and target-damage terms unchanged. This proxy addresses the observed failure mode without directly penalizing total output power; it is not a differentiable substitute for a calibrated CFAR guarantee.

Use the same three seeds, training parents, U-Net initialization procedure, optimizer, learning-rate schedule, 12,288 updates per seed, precision, and epoch-32 recovery cadence. Reuse the existing completed U-Net as the baseline, preserving it unchanged. Report DEV only as screening evidence, with per-JSR/SNR strata, Pd, frame/cell Pfa, target damage, no-jammer distortion, and matched-filter/RD tail summaries. Do not select an epoch or seed from DEV. If the loss screen improves the detection-versus-target-preservation tradeoff consistently across seeds, move to the prespecified long-range/frequency and normalization experiments; otherwise stop this branch and record the negative result. The final combination must be evaluated on a newly generated confirmation parent set that has not been used to design or tune these experiments.

## Limitations

- The results use one development role and three training seeds; parent bootstrap intervals quantify parent sampling conditional on each fixed model, not seed-to-seed uncertainty.
- SNR/JSR quartiles are descriptive partitions of DEV, not the frozen Main/Extreme roles.
- Association does not establish causation, and the diagnostic peak is not the local CFAR statistic.
- No independent confirmation-set result exists for the proposed loss or the combined model.
