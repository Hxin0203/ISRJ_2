# 当前状态 / Current state

Updated UTC: 2026-11-14T08:38:32.564895+00:00

## Paper 2, 1103-aligned run

- Active process: PID 41024, seed 41, epoch 40/128, step 1900/6144; latest finite training loss 0.151826. Seed 17 and seed 29 finished all 128 epochs with four checkpoints each. Seed 41 epoch-32 checkpoint is saved (SHA-256 `bc98387f8ad512222c840be9e469edbb88f9e5273f804caa44edc4eb41defbfb`).
- TRAIN has 512 parent groups with three within-parent jammer variants. DEV has 256 distinct parents. No independent TEST data has been generated or opened. Legacy held-out Gate arrays were not accessed.
- Seed 17 alone has v3 DEV results for epochs 32/64/96/128. These exploratory values do not select the final checkpoint. Seed 29/41 checkpoint DEV evaluations remain pending.
- Scope: CorrectionNet, Max-TF, GMM-OMP, exact-support BCS oracle, KEEP, and noisy no-jammer reference. MCA-ADMM-Net, ISRJ-FNet, and ablations are suspended.
- Full 168-cell comparison, 310-cell repeated-jammer Pd/Pfa, 8-pulse pulse-compression/Doppler, slice-width, no-jammer retention, and independent TEST runs remain incomplete.
- `python -m pytest -q`: 24 passed. Full progress and run boundaries are in `PAPER2_1103_PROGRESS_20261114.md` and `PAPER2_1103_NEAR_TERM_PLAN_20261114.md`.

Raw data, checkpoints, arrays, caches, and full logs remain local and excluded from Git. Lightweight aggregate evidence only is included.
