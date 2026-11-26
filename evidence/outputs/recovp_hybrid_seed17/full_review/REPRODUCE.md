# Reproduction and current evidence map

This review uses saved DEV256 receipts and a read-only copy of the Hybrid parent summary array. It does not launch training, inference, scans, or access CONFIRM or historical TEST.

## Verified identities

- Fixed DEV256: `data/vp_1103/dev.npz`, SHA-256 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`.
- ReCoVP-Hybrid seed17 checkpoint: SHA-256 `e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`.
- Hybrid core result: `evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json`, SHA-256 `877cfe7f6baf4fb3a59bff58da4a344e5437e43f000b3b3a9808a47acbda7df1`.
- Parent summary array: server path `D:/code/Xin/ISRJ_222/outputs/recovp_hybrid_seed17/seed_17/dev_epoch_256_parent_values.npz`; server and local mirror SHA-256 `b81774bf19c0ab8f833fc916637450ed43fbba42aa29466ccd2c5dba6b537bda`.

## What the parent array contains

The NPZ contains 256 parent IDs and already computed summary vectors for Hybrid Final, Stage1Only, detection metrics, no-jammer behavior, gate alpha and scale-equivariance checks. It does not contain raw IQ components, restored waveforms, type-resolved Hybrid outputs, baseline per-parent scores, or threshold score distributions. Therefore it cannot supply the missing formal radar metrics, cross-model paired bootstrap, Equal-PFA recalibration, type strata, or Top-10 cross-model audit. Marginal JSON summaries cannot replace those inputs.

## ReCoVP-Hybrid seed plan

The multi-seed runner freezes seed17 and permits seeds 29 and 41. Two seeds have been started so far: seed17 completed 256 epochs, and seed29 remains in progress. Seed41 has not been started.

## Current independent training status

The already-running seed29 job is outside this review. At `2026-11-26T06:14:58Z`, its status receipt reported epoch 54 in progress, 36/48 updates, with epoch 53 as the last committed checkpoint. The latest checkpoint SHA-256 was `862ca8334e5e395faa91ad6ca17a1e4e4a42a2726bcd027a767465cd406b87dc`, verified against `latest.pt`. This review neither started nor stopped that job.

## Resource snapshot

At the latest server query (`2026-11-26T06:15Z`), GPU0 utilization was 98% and GPU1 utilization was 87%. Both exceed the project rule requiring the selected GPU to be below 85% before starting a new GPU job. No inference was launched. Do not interrupt unrelated server jobs.

## Remaining DEV-only work

To complete the task-book analyses, obtain the exact baseline per-parent score arrays and the Hybrid raw inputs/outputs required by `tools/vp_1103_recovp_radar_metrics.py` and the existing equal-PFA audit. Verify all checkpoint, source and DEV hashes before running. Use parent as the resampling unit and aggregate variants within parent. Recheck GPU utilization immediately before any GPU work; CPU latency may be measured under a fixed thread protocol when appropriate. Keep all results DEV-only. Do not access CONFIRM or historical TEST, start further seeds, or launch large scans under this review scope.
