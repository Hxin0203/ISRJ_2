# Current state

Verified UTC: 2026-11-14T17:05:39.095835+00:00.

All three 128-epoch training runs and all 12 DEV v4 evaluations complete. All checkpoint hashes and source/data bindings verified. Selected epoch 128 using three-seed mean DEV eU; fixed seed 29. Rule finalized after DEV, before independent TEST; not a pre-DEV preregistration. Weights remain local.

Independent TEST scans are tracked in `RUNNING_EXPERIMENTS.json`; coverage in `evidence/outputs/TEST_COVERAGE_1103.json`. Primary report: `PAPER2_1103_PROGRESS_20261114.md`. Freeze: `configs/paper2_1103_scan_freeze.json` (SHA 8fed1450643ee71d8725daf0bb046a96c3d9301959c11000acb2d08c951c2930).

Only paper2 remote may be pushed. No ChatGPT collaboration; no old Gate access; no MCA-ADMM/FNet/ablations. Training was 48 updates/epoch, unlike 1103's 512. Preserve every checkpoint, failure and negative condition. Raw data, weights, arrays, caches and logs stay local. Baselines are formula reproductions, BCS oracle is labelled; bibliographic verification remains.


Long queue verified UTC: 2026-11-14T17:05:41.644870+00:00. Controller PID 21220, child PID 5016. TEST advanced beyond failed pd index 205; current index 208, 408/512 parents, method model.

User-authorized order: complete frozen TEST/audit/plots/report, continue seeds 17/29/41 to 256, then train seeds 53/67/79/97/109 to 256, and aggregate DEV convergence. 64 milestone models in total; every new training epoch has an independent full recovery archive. Original 12 models and all old histories remain.

Plan: `PAPER2_256_EPOCH_8_SEED_PLAN_20261114.md`. Queue: `configs/paper2_long_pipeline_20261114.json`. Runtime status: `outputs/paper2_long_pipeline_20261114/STATUS.json`. Restart: `powershell -NoProfile -File tools/start_long_pipeline.ps1`. Local input/source/architecture snapshot verified; training not yet started. Tests: 42 passed, recovery self-test passed. Evidence: `evidence/outputs/LONG_PIPELINE_ENGINEERING_VERIFICATION_20261114.json`; `evidence/outputs/paper2_256_epoch_8seed_20261114/`; `evidence/outputs/paper2_long_pipeline_20261114/`.
