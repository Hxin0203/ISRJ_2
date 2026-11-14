# Last Codex execution

- Formal training remains active as PID 41024; latest observed seed 41 status: epoch 65/128, step 3100/6144, finite loss 0.11107458174228668, updated 2026-11-14T08:47:07Z.
- Seed 29 completed all four milestones; checkpoint hashes match its completion receipt. Seed 41 epoch-32/64 checkpoints are saved at SHA-256 `bc98387f8ad512222c840be9e469edbb88f9e5273f804caa44edc4eb41defbfb` and `e9074fae56683bd8eb84e191f4d22c2a824fd2c6b3f02dc9d0a20009a4529305`.
- Seed 17 has all four checkpoints and v3 DEV results; independent TEST has not been generated or opened.
- Scope remains CorrectionNet, Max-TF, GMM-OMP, BCS-oracle, KEEP, and no-jammer noisy reference. MCA-ADMM-Net, ISRJ-FNet, and ablations remain suspended.
- Tests: python -m pytest -q — 24 passed.
- Protocol, code, and screened aggregate evidence are on `paper2/codex/1103-experiment`, now at `55134cb7b9f40c78ca27986b594165e46e86f1e0`; remote SHA was verified. Raw data and checkpoints excluded.

Captured UTC: 2026-11-14T08:47:07.909564Z
