# Server experiment configuration

Verified 2026-11-19 03:15:42 UTC. The user restored networking and authorized configuring experiments on server B. Connectivity and the expected server identity were verified through Cindy SSH.

The isolated deployment is `D:/code/Xin/ISRJ_222/happy-liskov-structured-20261119`. It contains 13 byte-verified files: the nine frozen source bindings, three package initializers and the existing experiment CONFIG. CONFIG SHA-256 is `68d402746f85d5414b405997d38785dde6c48971c5a933978afee0b44a896619`. No historical project source was overwritten.

Use the existing parent project's `.venv/Scripts/python.exe`: Python 3.12.15, PyTorch 2.9.1+cu128, NumPy 2.2.6. The system Anaconda Python has CPU-only PyTorch 1.12.1 and is unsuitable for this run. Physical GPU 1 was admitted only after observing zero memory use and zero utilization, then selected by UUID before importing Torch. CUDA reported exactly one visible device.

All five methods passed the existing synthetic engineering smoke procedure: one warm-up optimizer step and two measured optimizer steps per method. Finite final losses were 0.702767, 0.724543, 0.712890, 0.713315 and 0.832130 for full_base, full_tail, no_dechirp_base, no_dechirp_tail and unet_tail respectively. These are engineering checks, not performance comparisons. A second read verified all deployed hashes and the five CUDA results. Server-local DEPLOYMENT.json, READINESS.json and outputs/structured_20261118/training/ENGINEERING.json retain receipts.

Status: CONFIGURED_GPU_SMOKE_VERIFIED. No production training was started, no real training data or checkpoints were migrated, and no held-out Gate data was accessed. The deployment portion of D02 is resolved for this isolated root; production handoff, existing local-process disposition and extension budgets remain separate decisions. The existing hourly server probe still targets the parent project; absence there does not imply absence of this isolated deployment.

At 03:15:30 UTC the independent local audit verified unet_tail_17 at saved epoch 390, four fixed 512 milestones, five started groups, zero training-plus-evaluation completions and no audit errors. The local controller remains running. Preserve that trajectory; avoid launching a duplicate production queue by treating the smoke pass as production authorization.
