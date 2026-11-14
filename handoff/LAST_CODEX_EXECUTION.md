# Last Codex execution

- Formal training remains active as PID 41024; latest observed seed 41 status: epoch 40/128, step 1900/6144, finite loss 0.15182551741600037, updated 2026-11-14T08:38:32Z.
- Seed 29 completed all four milestones; checkpoint hashes match its completion receipt. Seed 41 epoch-32 checkpoint is saved at SHA-256 `bc98387f8ad512222c840be9e469edbb88f9e5273f804caa44edc4eb41defbfb`.
- Seed 17 has all four checkpoints and v3 DEV results; independent TEST has not been generated or opened.
- Scope remains CorrectionNet, Max-TF, GMM-OMP, BCS-oracle, KEEP, and no-jammer noisy reference. MCA-ADMM-Net, ISRJ-FNet, and ablations remain suspended.
- Tests: python -m pytest -q — 24 passed.
- Protocol, code, and screened aggregate evidence are on `paper2/codex/1103-experiment`, currently at `9c6e0d1a99276e03103b4ff975420e3593b50f08`; this live status update is pending commit. Raw data and checkpoints excluded.

Captured UTC: 2026-11-14T08:38:32.564895Z
