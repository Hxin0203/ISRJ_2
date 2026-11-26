# Reproduction and evidence map

This review is a read-only synthesis of files already present in the workspace. It does not rerun evaluation or training.

## Hybrid seed 17 sources

- Fixed DEV result: `evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json`
- Preflight and parameter/equivariance checks: `evidence/outputs/recovp_hybrid_seed17/PREFLIGHT.json`
- Training completion and recovery-point receipts: `evidence/outputs/recovp_hybrid_seed17/TRAINING_COMPLETE.json`
- Completion state: `evidence/outputs/recovp_hybrid_seed17/POSTTRAINING_STATUS.json`
- Configuration and source/data bindings: `evidence/outputs/recovp_hybrid_seed17/CONFIG.json`
- Prior core report: `reports/RECoVP_HYBRID_SEED17_EXPERIMENT_REPORT_20261126.md`

The run used seed 17, 256 epochs, 12,288 updates, 257 saved recovery points, and the fixed DEV256 dataset with SHA-256 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`. The final checkpoint SHA-256 is `e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`. The Hybrid preflight records 57,525 effective real parameters.

## Frozen baseline summaries

The read-only archive is in the local recovery worktree at `.cindy-worktrees/recovp-hybrid/evidence/outputs/recovp_archive_readonly/results/formal_dev/`. The recovery worktree is at Git commit `5723605ccbde2f439a5f1e985e5ca050bf6ce67b`, but these archive output files are untracked in that worktree and are not present on the checked `paper2` GitHub branches. Their SHA-256 values are listed in `VERIFICATION.json`; the published review therefore carries the summarized values and hashes, not the source receipts themselves.

- `ReCoVP-Base_seed_17.json`
- `P0-NoDechirp-Tail_seed_17.json`
- `UNet-Tail_seed_17.json`
- `RSDR-Net_seed_40312102.json`
- `PAIRED_DELTAS.json`

Each model receipt identifies the same DEV dataset SHA and 256-parent unit. The archived files contain model-specific bootstrap summaries and strata. They do not provide Hybrid-versus-baseline parent-paired differences.

## Missing server-side inputs

Hybrid raw parent values are referenced by the result receipt at `D:/code/Xin/ISRJ_222/outputs/recovp_hybrid_seed17/seed_17/dev_epoch_256_parent_values.npz` with SHA-256 `b81774bf19c0ab8f833fc916637450ed43fbba42aa29466ccd2c5dba6b537bda`. The file is not present on this workstation, and the current host inventory reports `rsdr-b` as unreadable. Do not replace it with the much smaller aggregate JSON when computing paired CIs, equal-PFA calibration, parent-specific Top-10 outcomes, or extra radar metrics.

## Reproduction sequence after server access returns

1. Verify the server identity, root, final checkpoint SHA, DEV SHA and raw parent-value SHA against the receipts above.
2. Read the existing frozen evaluation sources and baseline raw-parent assets without changing model checkpoints or thresholds.
3. Use the repository's formal radar-metric and DEV-gap/equal-PFA implementations to calculate the Hybrid metrics and parent-paired comparisons. Preserve parent as the resampling unit and aggregate variants within parent as specified by the evaluation protocol.
4. Produce Hybrid direct/repeat/cyclic and `repeat / JSR ≥ 30 dB` tables, Top-10 parent comparisons, CPU/GPU timing, peak memory and Stage 1/Stage 2 timing under one fixed protocol.
5. Check every input/output hash and confirm the run remains DEV-only. Do not access CONFIRM or historical TEST.

No new evaluation command is run by this review because the required raw parent arrays and server-side evaluation environment are unavailable.
