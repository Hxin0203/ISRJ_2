# Output evidence index

Progress refreshed on 2026-11-15 after controller completion at 2026-11-15T04:29:14.525505+00:00. No active project training/test process.

Current progress report: `PAPER2_PROGRESS_CHECK_20261115.md`.

Eight-seed epoch-256 DEV: `evidence/outputs/CONVERGENCE_256_8SEED_20261114/CONVERGENCE_SUMMARY.json`, `DEV_METRICS_BY_CHECKPOINT.csv`, `TRAINING_LOSS_BY_EPOCH.csv`, and `MANIFEST.json` in the same directory. All 64 checkpoint evaluations complete. Original curves are retained; readable PNG/PDF exports and their provenance are in `evidence/outputs/PROGRESS_CHECK_20261115/`. Full new integrity audit: `INTEGRITY_AUDIT.json` in that directory.

Controller completion, stage history and training completion receipts: `evidence/outputs/paper2_long_pipeline_20261114/`. TEST pulse-compression, Doppler and scan figures are in its `test_figures/` directory, with `TEST_FIGURE_MANIFEST.json`. The frozen TEST uses seed 29 epoch 128; it is separate from the eight-seed DEV convergence results. A matched RSDR-Net comparison and new independent confirmation of the expanded models have not been produced.

Current TEST coverage: `evidence/outputs/TEST_COVERAGE_1103.json`; condition table: `evidence/outputs/TEST_CONDITION_TABLE_1103.csv` (only when results exist).

DEV v4: `evidence/outputs/paper2_1103_retry_20261114/DEV_SUMMARY_v4.json`; checkpoint integrity: `CHECKPOINT_VERIFICATION.json`; selection table: `DEV_SELECTION_TABLE_v4.csv`.

Local raw parent blocks and checkpoints are intentionally excluded. A missing condition result means not yet produced, never a fabricated scientific value. Earlier v2/v3 results are historical.

| Suite | Expected conditions | Completed method counts |
|---|---:|---|
| main | 168 | model=168, keep=168, reference=168, max_tf=168, gmm_omp=168, bcs_oracle=168 |
| pd | 310 | model=310, keep=310, reference=310 |
| retention | 42 | model=42, keep=42, reference=42 |
| coherent | 3 | model=3, keep=3, reference=3 |
| sliceA | 21 | model=21, keep=21, reference=21, max_tf=21 |
| sliceB | 20 | model=20, keep=20, reference=20, max_tf=20 |
