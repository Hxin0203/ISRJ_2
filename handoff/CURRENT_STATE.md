# Current state

Verified UTC: 2026-11-14T11:53:44.392105+00:00.

All three 128-epoch training runs and all 12 DEV v4 evaluations complete. All checkpoint hashes and source/data bindings verified. Selected epoch 128 using three-seed mean DEV eU; fixed seed 29. Rule finalized after DEV, before independent TEST; not a pre-DEV preregistration. Weights remain local.

Independent TEST scans are tracked in `RUNNING_EXPERIMENTS.json`; coverage in `evidence/outputs/TEST_COVERAGE_1103.json`. Primary report: `PAPER2_1103_PROGRESS_20261114.md`. Freeze: `configs/paper2_1103_scan_freeze.json` (SHA 8fed1450643ee71d8725daf0bb046a96c3d9301959c11000acb2d08c951c2930).

Only paper2 remote may be pushed. No ChatGPT collaboration; no old Gate access; no MCA-ADMM/FNet/ablations. Training was 48 updates/epoch, unlike 1103's 512. Preserve every checkpoint, failure and negative condition. Raw data, weights, arrays, caches and logs stay local. Baselines are formula reproductions, BCS oracle is labelled; bibliographic verification remains.
