# ReCoVP-Hybrid 8-seed pipeline launch snapshot

Snapshot time: 2026-11-26 08:06 UTC (server B, `DESKTOP-RFFERM3`).

The unattended controller is running from `D:/code/Xin/ISRJ_222` under the workspace CUDA environment (`.venv/Scripts/python.exe`). Its Windows process wrapper is PID `29872`; the Python controller is PID `38336`. The controller status is `RUNNING`, current seed `41`, attempt `1`, with a 60-second retry/poll policy and six remaining queued seeds after seed41: `53, 67, 79, 97, 109` (seed29 remains an independently running job).

Seed41 passed its TRAIN/DEV preflight and started training on physical GPU0, UUID `GPU-f12e73b1-3bed-5a1e-bcb0-f9726dd1621d`. At the snapshot, its status was epoch 1, update 12/48, with epoch 0 committed; the latest recovery checkpoint was present. The worker process was PID `34364`. Preflight verified the pinned data: TRAIN SHA256 `6148d097fecb0381e56227eb80cf1379e5f2750816ba65ce12f5457ee678f0fd`; DEV SHA256 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`. It also passed the finite-precision equivariance and finite/nonzero-gradient checks. The full receipt is on server B at `evidence/outputs/recovp_hybrid_seed41/PREFLIGHT.json`.

Seed29 remained active in its existing process on GPU1, at epoch 124, update 36/48, with epoch 123 committed. The queue controller waits for a valid 256-epoch completion receipt before advancing from each seed and waits for seed29's existing completion before final cohort finalization. It will run fixed DEV256 evaluation for each seed and stops with an explicit failure status after eight unsuccessful attempts.

Admission was respected at seed41 startup: GPU0 utilization was 2% at preflight admission (below the strict 85% threshold). During the later snapshot, GPU0 and GPU1 were 88% and 99%; the already admitted runs continue, and all later seeds will wait until a candidate GPU is again below 85%. No running job was stopped or moved.

The server's default `D:/Anaconda/python.exe` is CPU-only and was rejected before any seed41 checkpoint was written. The launcher was corrected to use the repository `.venv` (PyTorch `2.9.1+cu128`, CUDA 12.8, CUDA available); the failed preflight attempt remains in the controller log for audit. Only the approved second-paper branch was updated. No training data, weights, checkpoints, raw arrays, caches, or full logs were uploaded to GitHub.
