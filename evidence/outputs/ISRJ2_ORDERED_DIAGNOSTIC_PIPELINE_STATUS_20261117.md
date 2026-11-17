# ISRJ_2 ordered diagnostic pipeline — status

Checked: 2026-11-17 11:35 UTC
Execution host: server B (`rsdr-b`), project `D:/code/Xin/ISRJ_222`

## Current position

The residual and detection-failure diagnosis is complete. Its DEV-only findings are recorded in `RESIDUAL_FAILURE_DIAGNOSTIC_REPORT_20261117.md`. Within the available development runs, matched-filtered residual peaks had a stronger ranking association with jammer-frame false alarms than aggregate background-residual energy. This is a screening observation; it does not establish causation or independent generalization.

The U-Net matched-filter-tail-loss screen is now training on server B, physical GPU 1. It uses the frozen 256-epoch U-Net comparison protocol, seeds 17/29/41, and an immutable recovery point after every completed epoch. At this check, seed 17 was in epoch 19; epochs 0–18 were present, and the epoch-18 checkpoint SHA-256 matched its sidecar. The saved run identity matched the frozen configuration (`cef0a79c03a022e09bce91237f909179a692952c5c44f34c19fd8315a404c1af`) and training-data identity. No model-performance conclusion is available yet: the other seeds and the paired DEV evaluation have not completed.

The launch is owned by the server's WMI process path so it survives the SSH session. GPU 1 showed 1,542 MiB in use and 72% utilization at the check; GPU 0 was not assigned to this run. The training output, checkpoints, and arrays remain on the server.

## Decision sequence

1. Finish the three-seed U-Net tail-loss screen and evaluate baseline/treatment pairs on the same DEV parents. Apply the preregistered screen: median seed-level jammer-frame Pfa improves, each seed's Pd change is at least −0.02, and mean target-damage change is at most +0.005. This gate allocates further experiment time; it is not a significance test.
2. If the screen passes, test long-range processing, frequency-domain processing, and normalization as separate factors under paired seeds and fixed DEV observations. Keep the existing model, data, training budget, and evaluation protocol fixed wherever the factor permits. Do not combine factors during screening.
3. Combine only factors that pass screening, freeze the resulting configuration, then evaluate it on a newly generated independent confirmation parent set. DEV remains development evidence and cannot serve as confirmation.

The current gate has not yet been evaluated, so stages 2 and 3 above remain unopened. Continue preserving each epoch recovery point and each 32-epoch milestone; do not change the frozen configuration mid-run.
