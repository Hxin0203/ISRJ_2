# Paper 2 minimal model specification

Status: exploratory feasibility study; novelty unconfirmed.

This file describes the original four-state pilot. Its later frozen confirmation failed the empirical budget/tail requirements; see PAPER2_DEV_CONFIRMATION.md. The separate eight-state target-family specification is FAMILY_STUDY_CONTRACT.md. It retains the identical backbone and adds no inference modules.

Use the existing RGSSR three-scale convolutional encoder-decoder (16/32/64 channels), without reliability estimator, routing, attention or selective gates. Its known invertible LFM rotation is retained. Input and output are I/Q [B,2,1800]. The sole learned output is the actual correction C(Y); restoration is Y-C(Y). Per-observation RMS scaling is computed from that observation alone and inverted at output. No target location or clean-pair information enters inference.

For each parent and jammer type build four inputs E, S+E, J+E, S+J+E in this order with labels 0,0,J,J. Components remain at their physical scale and share the same E/J; removing S does not regenerate J. The target of restoration remains U=S+E. Train B1 (four-state normalized endpoint supervision), B2 (B1 plus 0.1 normalized linear matched-filter error), B4 (B2 plus 1.0 target-response damage). All use identical parameters, initialization for each seed, batches, optimizer and number of updates. All states receive equal weight. No warm starts or legacy weights.

The damage is ||C11-C01||²/||S||². Endpoint errors use each state's desired background energy (E or S+E), not J energy. Zero-target states have no damage denominator. FFT matched filtering uses the known transmitted chirp and linear zero padding; the same filter is applied to error and desired reference.

Development evaluates complete mixed-input U-NMSE, background R0 error, target response damage, amplitude/phase, d>0.25 frequency, 95th percentile, no-jammer distortion and the R0/RS cross term. Metrics use float64/complex128 before reductions. Physical parent is the statistical unit, averaging its three jammer variants. This first study has one target per parent; weak multi-target conclusions are prohibited.
