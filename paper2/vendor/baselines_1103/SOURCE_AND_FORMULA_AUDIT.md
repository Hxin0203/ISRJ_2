# 1103 baseline source and formula audit

These are local implementations reconstructed from the published methods cited below; no upstream implementation code is included. They are research reproductions and require parameter calibration on DEV before they can support a paper comparison. All numerical settings, deviations from the printed formulas, and non-oracle inputs must be reported with the results.

| Method | Source as currently identified | Implementation correspondence | Information available to the method |
|---|---|---|---|
| Max-TF | Chen et al., *IET Radar, Sonar & Navigation* (2019), equations 17–26 | Squared STFT, maximum-over-frequency time profile, global and per-segment quarter-peak masks, mask extension/inversion, normalized band-pass response, smoothing and filtering | Received waveform and known transmit waveform parameters only |
| GMM-OMP | Liu et al., *Digital Signal Processing* 161 (2025), article 105132 | Two-component amplitude GMM by EM, midpoint clean-sample selection, dechirping, Fourier OMP, inverse dechirping | Received waveform and known transmit waveform parameters only; no target, jammer, SNR, JSR, or clean-component labels |
| BCS | Sha et al., *Sensors* 19(15) (2019), article 3279, equations 14, 17–23 and Algorithm 1 | Smooth-L1 MAP objective, outer iterative reweighting and inner complex conjugate-gradient reconstruction | Exact non-jammer support is supplied; this is an **oracle reference**, not a blind baseline |

The Max-TF implementation makes explicit reproduction choices for discrete tie handling, segment smoothing and thresholding; these are not claimed as source-paper defaults. The GMM-OMP code uses the standard responsibility-weighted variance update because the printed update omits the responsibility multiplier. That deviation is recorded in the module docstring. BCS's support oracle is explicit in the adapter and must remain labelled in tables and figures.

Full article titles, DOI/URL, equation transcription, source-paper hyperparameters and final citation metadata still require bibliographic verification before manuscript submission. Do not present smoke tests as a performance comparison or treat oracle BCS as an equal-information competitor.
