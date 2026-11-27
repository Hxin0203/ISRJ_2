# ReCoVP-Hybrid resumed seeds: launch handoff

Snapshot: 2026-11-27 09:27 UTC. The long-running training jobs remain active on Server B (`D:\code\Xin\ISRJ_222`).

| Seed | Resumed from | Current latest recovery point | Physical GPU | State |
|---|---:|---:|---:|---|
| 79 | epoch 75 | epoch 78 | 1 | Running |
| 97 | epoch 139 | epoch 142 | 0 | Running |
| 109 | epoch 0 | not started | selected after first completion | Pending in controller |

The pipeline controller is running in the Server B service session. It resumes seeds 79 and 97 concurrently on their frozen GPU UUIDs. Once either completes, it admits seed 109 on a GPU whose live utilization is below 85%. It preserves the epoch recovery files and sidecars and retries an interrupted worker from its latest checkpoint (up to three attempts).

Before dispatch, the original configs, preflight receipts, source and data hashes, GPU UUID bindings, all existing recovery sidecars, and the resume checkpoint hashes were verified. At the launch snapshot, both workers were advancing and both GPUs were busy. No training results are claimed yet. Completion requires a matching `TRAINING_COMPLETE.json` receipt for each seed, epoch 256, and 257 verified recovery points per seed.

Server-side live status and logs are under `D:\code\Xin\ISRJ_222\evidence\outputs\recovp_hybrid_resume_20261127\`. The local `STATUS.json` is a dated launch snapshot; query the server copy for live state.
