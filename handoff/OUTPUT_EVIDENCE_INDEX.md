# Output evidence index

Updated UTC: 2026-11-14T17:05:39.095835+00:00

Current TEST coverage: `evidence/outputs/TEST_COVERAGE_1103.json`; condition table: `evidence/outputs/TEST_CONDITION_TABLE_1103.csv` (only when results exist).

DEV v4: `evidence/outputs/paper2_1103_retry_20261114/DEV_SUMMARY_v4.json`; checkpoint integrity: `CHECKPOINT_VERIFICATION.json`; selection table: `DEV_SELECTION_TABLE_v4.csv`.

Local raw parent blocks and checkpoints are intentionally excluded. A missing condition result means not yet produced, never a fabricated scientific value. Earlier v2/v3 results are historical.

| Suite | Expected conditions | Completed method counts |
|---|---:|---|
| main | 168 | model=168, keep=168, reference=168, max_tf=168, gmm_omp=168, bcs_oracle=168 |
| pd | 310 | model=208, keep=208, reference=208 |
| retention | 42 | model=0, keep=0, reference=0 |
| coherent | 3 | model=3, keep=3, reference=3 |
| sliceA | 21 | model=0, keep=0, reference=0, max_tf=21 |
| sliceB | 20 | model=0, keep=0, reference=0, max_tf=20 |


Long queue verified UTC: 2026-11-14T17:05:41.646122+00:00. Controller PID 21220, child PID 5016. TEST advanced beyond failed pd index 205; current index 208, 408/512 parents, method model.

User-authorized order: complete frozen TEST/audit/plots/report, continue seeds 17/29/41 to 256, then train seeds 53/67/79/97/109 to 256, and aggregate DEV convergence. 64 milestone models in total; every new training epoch has an independent full recovery archive. Original 12 models and all old histories remain.

Plan: `PAPER2_256_EPOCH_8_SEED_PLAN_20261114.md`. Queue: `configs/paper2_long_pipeline_20261114.json`. Runtime status: `outputs/paper2_long_pipeline_20261114/STATUS.json`. Restart: `powershell -NoProfile -File tools/start_long_pipeline.ps1`. Local input/source/architecture snapshot verified; training not yet started. Tests: 42 passed, recovery self-test passed. Evidence: `evidence/outputs/LONG_PIPELINE_ENGINEERING_VERIFICATION_20261114.json`; `evidence/outputs/paper2_256_epoch_8seed_20261114/`; `evidence/outputs/paper2_long_pipeline_20261114/`.
