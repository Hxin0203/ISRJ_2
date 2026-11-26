# ReCoVP-Hybrid 8-seed progress snapshot

Snapshot time: 2026-11-26 08:08 UTC on server B.

The detached pipeline remains active: wrapper PID `29872`, Python controller PID `38336`; seed41 worker PID `34364`. Seed41 is training on physical GPU0 and has completed epoch 1, with epoch 2 at update 36/48; epoch 1 is committed. Its TRAIN/DEV preflight passed and its epoch-0 recovery checkpoint was verified present at launch.

The pre-existing seed29 process remains active on GPU1 at epoch 126, update 12/48, with epoch 125 committed. The queue controller is synchronously waiting for seed41, then advances through seeds53, 67, 79, 97, and109 after each 256-epoch completion receipt. Seed29 will remain monitored as an independent run and must also have a valid completion receipt before the eight-seed DEV256 cohort is finalized.

At this snapshot GPU utilization was 86% on GPU0 and 65% on GPU1. No additional seed has been started; each queued launcher checks utilization again and admits a GPU only below 85%. Current work is unattended on server B, with per-seed training/evaluation logs and pipeline state under `evidence/outputs/recovp_hybrid_8seed_pipeline/`.
