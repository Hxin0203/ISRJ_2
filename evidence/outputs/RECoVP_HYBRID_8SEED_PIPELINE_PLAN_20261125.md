# ReCoVP-Hybrid 8-seed unattended run

## Scope

Complete the predeclared seed set `17, 29, 41, 53, 67, 79, 97, 109`, with 256 epochs per seed. Seed17 remains frozen and seed29 remains under its existing runner. The continuation controller runs seed41 first, then seeds53, 67, 79, 97, and109 sequentially. It waits for the independently running seed29 completion receipt before finalizing the cohort, then verifies fixed DEV256 results for all eight seeds.

## Admission and recovery

Long training runs only on server B at `D:/code/Xin/ISRJ_222`. Seed41 requests physical GPU0. The launcher waits until that GPU reports utilization below 85%; later seeds choose the least-utilized physical GPU below 85%. Each worker rechecks host, repository root, GPU UUID, and utilization immediately before training. Existing jobs are not stopped or migrated. Every seed is limited to 256 epochs and must provide all 257 recovery-point hashes, including epoch zero, before the queue advances. Failed jobs receive up to eight automatic retries, then the controller stops in a fail-closed state and records the failure.

## Data and evaluation boundary

Each seed runs the frozen TRAIN/DEV preflight, which checks the pinned data hashes and source identity before training. Post-training comparison is fixed DEV256 only. The controller does not open CONFIRM or historical TEST data. Seed17 was verified complete at 256 epochs with 257 recovery hashes and an existing DEV256 receipt before launch preparation. At the latest server snapshot, seed29 had committed through epoch117 and was working on epoch118; it remains owned by its existing process.

## Launcher sources

- `tools/run_recovp_hybrid_8seed_pipeline.py`
- `tools/start_recovp_hybrid_extended.py`
- `tools/vp_1103_recovp_hybrid_extended.py`
- `tools/start_recovp_hybrid_extended_eval.py`
- `tools/evaluate_recovp_hybrid_extended.py`

Runtime state and logs are written under `evidence/outputs/recovp_hybrid_8seed_pipeline/` and `outputs/recovp_hybrid_8seed_pipeline/`. This document records the planned control flow; live status must be read from the server-generated `STATUS.json` and dated snapshots.
