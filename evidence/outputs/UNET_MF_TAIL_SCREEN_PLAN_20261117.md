# U-Net matched-filter tail-loss screen

Date: 2026-11-17
Role: exploratory single-factor screening on the existing DEV set. The DEV set has participated in model development and will not be treated as independent confirmation.

## Question and comparison

The residual diagnosis found that matched-filtered residual peaks track jammer-frame false alarms more closely than total background residual energy in the current DEV runs. This screen tests whether adding a small matched-filter range-Doppler tail penalty to the parameter-matched U-Net objective can reduce jammer-frame false alarms while preserving target response.

The treatment changes one factor: add `0.05 × L_tail` to the existing endpoint, pulse-compression, and target-damage objective. The new term is computed on the target-absent jammer residual `j − C(j+e)`. It applies the same 600-sample LFM matched filter, 8-pulse Doppler FFT, target-free thermal-noise power normalization, and idealized 436-reference-cell CA-CFAR multiplier as the evaluation protocol. Its smooth top-eight hinge is evaluated only on valid CFAR range cells `36:1165`. It is a differentiable screening surrogate; it is not a CFAR implementation or a guarantee on empirical false-alarm probability.

## Frozen training and recovery

- Three seeds: 17, 29, and 41, paired with the already completed U-Net controls.
- Same 59,714 active parameters, data, parent/jammer sampling, seed-specific initialization, optimizer, learning-rate schedule, 12,288 updates, and FP32 settings as the frozen U-Net controls.
- Train through epoch 256. Save an immutable full recovery checkpoint and metrics after every epoch, plus immutable epoch-32 milestones.
- Existing U-Net checkpoints remain read-only. The new run has its own run identity and output directory.
- Execute only on server B, `D:/code/Xin/ISRJ_222`, with physical GPU 1 selected by `CUDA_VISIBLE_DEVICES=1`.

## DEV outcomes and screening rule

At epoch 256, evaluate the three baseline/treatment pairs on the same 256 DEV parents. Average the three jammer families within each physical parent and retain parent-level paired differences. Report U-NMSE, target damage, SJRIF, Pd, jammer-frame and cell Pfa, and the remaining CFAR-tail diagnostics when available. Seed is the training replicate; parent bootstrap intervals describe sampling conditional on each fitted seed and do not replace seed replication.

Advance to the next exploratory stage only if the median seed-level jammer-frame Pfa difference is below zero, every seed's Pd change is at least −0.02 absolute, and the mean target-damage change is no greater than +0.005. This prespecified screen is a resource-allocation rule, not a significance test. A failed screen stops this loss branch; a passed screen motivates, but does not establish, long-range/frequency and normalization hypotheses. No epoch or seed is selected from DEV.

## Confirmation boundary

Any combined factors must be evaluated only after screening is complete, on a newly generated independent confirmation parent set with frozen generator settings and no reuse for tuning. DEV findings cannot be reported as confirmation evidence. The confirmation run must retain the frozen checkpoints, data receipts, source/config hashes, per-parent metric arrays, and all required recovery points.
