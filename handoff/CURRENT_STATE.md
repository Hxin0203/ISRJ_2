# 当前状态 / Current state

Updated UTC: 2026-11-14T08:11:01.002154+00:00

## Paper 2, 1103-aligned run

- Active process: PID 41024, seed 29, epoch 88/128, step 4200/6144; latest finite training loss 0.142749. Checkpoints at 32 and 64 epochs are saved. Seed 17 finished all four checkpoints; seed 41 has not started.
- TRAIN has 512 parent groups with three within-parent jammer variants. DEV has 256 distinct parents. No independent TEST data has been generated or opened. Legacy held-out Gate arrays were not accessed.
- Seed 17 alone has v3 DEV results for epochs 32/64/96/128. These exploratory values do not select the final checkpoint. Seed 29/41 checkpoint DEV evaluations remain pending.
- Scope: CorrectionNet, Max-TF, GMM-OMP, exact-support BCS oracle, KEEP, and noisy no-jammer reference. MCA-ADMM-Net, ISRJ-FNet, and ablations are suspended.
- Full 168-cell comparison, 310-cell repeated-jammer Pd/Pfa, 8-pulse pulse-compression/Doppler, slice-width, no-jammer retention, and independent TEST runs remain incomplete.
- `python -m pytest -q`: 24 passed. Full progress and run boundaries are in `PAPER2_1103_PROGRESS_20261114.md` and `PAPER2_1103_NEAR_TERM_PLAN_20261114.md`.

Raw data, checkpoints, arrays, caches, and full logs remain local and excluded from Git. Lightweight aggregate evidence only is included.
