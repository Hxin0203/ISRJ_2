## CWC R1 authorization and recovery — 2026-11-19 UTC

The user's ISRJ2_CWC_AUTONOMOUS_SUPERVISOR_R1 explicitly supersedes earlier CWC suspension. This dated entry governs CWC only. Experiment budgets, host restrictions and frozen assets retain their existing constraints.

Workspace happy-liskov; connector Codex with ChatGPT · happy-liskov; all discussions in the codex-with-chatgpt Project. CLI 0.1.3 and upstream C2C protocol (no separate numbered version), INIT → PLAN → EXECUTED → implicit REVIEW → DONE/BLOCKED/ERROR. Read docs/CWC_COLLABORATION_POLICY.md and reports/CWC_COLLABORATION_LOG.md before new work, then c2c session for the current checkpoint.

Last manual read-only review received DONE on 2026-11-18 around 16:07 UTC. Health checks passed on 2026-11-19 around 02:11 UTC. Natural-trigger acceptance c2c_a719 is pending at 02:20 UTC. Verified state remains MANUAL_READ_ONLY_REVIEW_VERIFIED until all receipts exist. See reports/CWC_END_TO_END_ACCEPTANCE.md.

B access is unresolved and local-versus-server training authority remains a separate pending decision. CWC authorization does not resolve training budgets or deployment. Earlier handoff entries below retain their historical dates.

## Superseding own-model priority — 2026-11-17 15:03 UTC

- The user challenged the prolonged U-Net-only optimization schedule. Own-backbone validation now takes priority on physical GPU 1: current U-Net tail training + DEV -> fresh CorrectionNet original-loss control -> CorrectionNet + identical 0.05 tail -> original U-Net dilation. Both own-model arms use seeds 17/29/41 to 256; preserve all epoch recovery and 32-epoch milestones. Old eight-seed models remain intact.
- Both own-model controllers are deployed on server B only (`rsdr-b`, `D:/code/Xin/ISRJ_222`), PID 14952 (original) and 31212 (tail). At 15:03:52 UTC both are alive, WAIT_DEPENDENCY, no saved epochs, no source drift. GPU numerical recovery preflight is pending device release. CPU engineering checks passed; 53 frozen source/protocol hashes matched between local and server. No efficacy result exists.
- Dilation has an owned PAUSE with token `ISRJ2_CORRECTION_TAIL_STUDY_20261117_V1`; CONTROL_STATUS confirms PAUSED. Only the matching owned pause is automatically removed after both CorrectionNet arms, paired DEV and four-cell summary complete. Do not duplicate-launch. GPU 0 normalization/frequency scheduling is unchanged; unrelated jobs are untouched.
- Freeze `tools/correction_tail_study/`, its launch/deploy scripts and `CORRECTIONNET_TAIL_STUDY_PLAN_20261117.md`. Use independent read-only `tools.snapshot_correction_priority_20261117` to inspect status. The frozen operations snapshot reader touches a Windows-locked byte and may fail; do not edit frozen sources to repair this non-training reader.
- Evaluate own tail against its fresh original-loss control first: jammer-frame Pfa, Pd, SJRIF, target damage and component losses. Then compare architecture-by-loss four cells descriptively. U-Net screens cannot establish own-model efficacy. No-dechirp/tail combinations and transfer of useful normalization/long-range/frequency factors remain conditional, undeployed plans.
- Report: `evidence/outputs/CORRECTIONNET_PRIORITY_CORRECTION_REPORT_20261117.md`; priority overlay: `CORRECTIONNET_PRIORITY_PIPELINE_20261117.json` (CN0/CN1/CN2 avoid inherited C0/C1 ID collisions). Historical 18-task snapshots retain their dates and are superseded only for this execution priority. Latest own-queue snapshot: `CORRECTIONNET_PRIORITY_STATUS_20261117_150352.json`.
- No local training, first-paper writes, MCA-ADMM/FNet or ChatGPT collaboration. Push exact key code/report/light evidence paths only to `paper2`; never data, weights, raw arrays, caches, credentials or full logs.

Earlier snapshots below are historical; this dated priority entry takes precedence.

# Current state

## Superseding idea-to-experiment pipeline — 2026-11-17 14:29 UTC

Both original idea documents have been mapped to evidence, gaps, entry gates and stopping rules in `evidence/outputs/IDEA_TO_EXPERIMENT_PIPELINE_REVIEW_20261117.md`. Machine plan: `evidence/outputs/IDEA_EXPERIMENT_PIPELINE_20261117.json`. Only tasks T/N/L/F have deployed controllers; later combinations, distillation, eight-seed training and new confirmation remain conditional plans.

- Server B only, dedicated second-paper root. At 14:29:11 UTC, GPU 1 tail-loss seed 17/29/41 saved 256/228/not started; GPU 0 observation-scale saved 131/not started/not started. Neither has complete three-seed DEV results. Training-only remaining estimates are about 1.8/4.0 hours, excluding evaluation, waits and interruptions.
- Two persistent source-bound queues are already running: dilation on physical GPU 1 after tail training plus paired DEV, and frequency coordinates on physical GPU 0 after normalization training plus paired DEV. Both are independent factors, original RMS/loss, 59,714 active parameters, matching initialization, seeds 17/29/41 to epoch 256. Dependencies release devices and do not require efficacy-gate success. Do not duplicate-launch or edit frozen sources/protocols. Server CPU engineering checks passed; GPU numerical recovery preflight remains pending device release.
- All nine Full/No-dechirp/U-Net by seed phase/scale diagnostic combinations completed on fixed first 16 DEV parents, server CPU only. Phase inconsistency exists, common amplitude scaling is nearly exact, and endpoint changes have mixed signs. No causal detection conclusion, delay test or teacher audit exists yet. SUMMARY.json is the completion authority; the intermediate STATUS.json still says DIAGNOSING with nine completed pairs.
- Latest checkpoint SHA sidecars match; source drift is empty. Full archive/data audit at 14:22:15 UTC covered tail 256/209 and normalization 113; later light snapshots are not full archive audits. Preserve every epoch and every 32-epoch milestone. No new confirmation data was consumed.
- Next: denominator/delay/teacher diagnostics; four-factor DEV review; justified mean-PC/hard-parent controls; eligible minimal combinations with architecture-by-loss four cells; data scale and shared-device efficiency; final candidate and key U-Net controls at eight common seeds to 256; frozen H0 calibration and fresh confirmation. Phase/delay and selective-transfer ideas are still conditional research routes, not disproved or replaced by the four screens.
- Keep diagnostic MF-peak normalization (noise-only output residual RD median) distinct from tail training normalization (known training-noise RD power). Historical RSDR/JNR comparisons are not new three-seed epoch-256 results.
- Report text audit was offline with protected spans, backup and SHA. Only exact code/protocol/report/light JSON paths go to `paper2`, never `origin`, data, weights, arrays, caches, credentials or full logs. Preserve unrelated working changes and paper manuscripts.

Earlier snapshots below are historical; this dated entry takes precedence.

## Superseding parallel snapshot — 2026-11-17 13:42 UTC

The user explicitly authorized using otherwise idle server GPUs for independent experiments. GPU 1 continues the frozen matched-filter tail-loss screen; GPU 0 now runs a separate normalization-only screen. This supersedes historical instructions below that required normalization to wait for the tail-loss gate. Do not change frozen sources, share partial outcomes for tuning, or combine factors before screening completes.

- Server B only: `rsdr-b`, `D:/code/Xin/ISRJ_222`. GPU 1 tail loss: seed 17 completed 256; seed 29 saved 103; seed 41 pending. GPU 0 normalization: seed 17 saved 8; seeds 29/41 pending. Both branches automatically train all three seeds to 256, then run paired DEV evaluation against the same-seed frozen U-Net controls.
- Every existing recovery point passed SHA and identity checks; no missing completed epochs or due 32-epoch milestones. No frozen source drift; training and DEV data hashes match. No new DEV efficacy conclusion exists yet. No independent confirmation set has been opened for these optimization factors.
- Normalization changes only the observation-derived input/output scale; parameters, initial tensors, original loss, data and update budget match the plain U-Net. It adds no tail loss. Numerical preflight passed. Strong jammer leakage can inflate the outer-band scale proxy (train-first32 diagnostic q95 ratio 2.565063, max 12.617789); retain this limitation and any negative outcome.
- Approximate remaining training from this snapshot: tail 2.5 hours; normalization 4.7 hours, excluding evaluation, saving/auditing, waits and recovery. Do not treat this as a deadline. Long-range/frequency factors are not yet launched; freeze separate protocols before scheduling them on released devices. Only later combine eligible factors and use fresh independent confirmation parents.
- Report: `evidence/outputs/PARALLEL_EXPERIMENT_PROGRESS_20261117.md`; full light audit: `evidence/outputs/PARALLEL_GPU_STATUS_20261117.json`; normalization protocol: `evidence/outputs/UNET_NOISE_NORM_SCREEN_PLAN_20261117.md`. New guarded entry: `tools/start_unet_norm_screen.ps1`. All completed epoch recovery points, earlier models and failure history remain preserved on server. Push only key source/report/text evidence to `paper2`, never datasets, weights, raw arrays, caches, credentials or full logs.

Earlier snapshots below are historical; the dated parallel snapshot above takes precedence.


## Superseding snapshot — 2026-11-17 11:45 UTC

- The user-authorized sequence is active: residual/detection-failure diagnosis → U-Net matched-filter tail-loss screen → separate long-range, frequency-processing, and normalization screens → combine only factors that pass screening, then evaluate on a newly generated independent confirmation set.
- The residual diagnosis is complete as DEV-only hypothesis formation. Matched-filter residual peaks were more strongly associated with jammer-frame false alarms than total residual energy in the inspected DEV sample; this is not causal or independent confirmation evidence. See `evidence/outputs/RESIDUAL_FAILURE_DIAGNOSTIC_REPORT_20261117.md`.
- The matched-filter tail-loss U-Net screen is training on server B (`rsdr-b`, `D:/code/Xin/ISRJ_222`, physical GPU 1), run `unet_mf_tail_screen_20261117_v2`. Latest snapshot at 2026-11-17 11:50:00 UTC: seed 17, epoch 58 committed, epoch 59 in progress; checkpoint SHA-256 `b8364572e52205d8c734b32ff5ccc160bf1f866afecbc6409e4c4bf40d46c54b`. Epoch records 0–58 and 118 recovery archive files are present. The epoch-32 milestone SHA-256 is `6344699d7270a12955412526e3a071fdd7ce9b349aae17c0a2841374118f11e4`. The latest checkpoint sidecar matches its file hash. Frozen config SHA-256 is `cef0a79c03a022e09bce91237f909179a692952c5c44f34c19fd8315a404c1af`.
- The displayed losses are training diagnostics only. No screening performance outcome exists until all paired seeds 17/29/41 complete and the frozen DEV evaluation runs. Do not start the downstream long-range/frequency/normalization factors before applying the prespecified screening rule in `evidence/outputs/UNET_MF_TAIL_SCREEN_PLAN_20261117.md`.
- At the observed epoch rate (about 22.2 s/epoch), remaining work for the three-seed screen is roughly 4.4 compute-hours from this snapshot, before evaluation and queue overhead. This is a rough estimate, not a deadline.
- No local long training is running. No confirmation parents have been generated or consumed for this branch. Training data, checkpoints, raw arrays, caches, and full logs remain off GitHub. Only `paper2` may be pushed.

The earlier snapshots below are historical and superseded by this entry.

## Superseding snapshot — 2026-11-16 14:52 UTC

- The U-Net/ablation server queue is complete: 15 runs × 256 epochs; five variants have confirmation comparisons. Completion receipt: `evidence/outputs/unet_ablation_3seed_256_20261115/PIPELINE_COMPLETE.json`.
- Max-TF (9 candidates) and GMM-OMP (8 candidates) DEV pilots all failed the predeclared gate. No full-DEV rerun or independent confirmation was opened. BCS GCV selected `λ=0.001`; `λ=0.01` sensitivity remains a small DEV-only oracle comparison.
- A synthetic phase-origin diagnostic showed that the zero-origin and centered dechirp references differ by a constant phase and an exact 450-bin DFT shift under project parameters. The OMP equivariance test passes; this does not justify a revised baseline or another DEV run. See `evidence/outputs/GMM_OMP_PHASE_ORIGIN_DIAGNOSTIC_20261116.md` and `tests/test_gmm_phase_origin_equivalence.py`.
- Server B GPU 1 was idle at the latest check. The Python processes present belonged to unrelated projects. No local or server long training was started in this continuation.
- The next experiment-design task is source-equation and evaluation-target audit for the classical baselines. Only if it identifies a substantive correction should a new, versioned DEV-only calibration plan be frozen. No confirmation data may be used for parameter repair.

The snapshot below is historical and is superseded by the dated status above.

Server snapshot 2026-11-15T16:52:41.103691+00:00: TRAINING_QUEUE; job 1/15, unet seed 17, epoch 243. Independently SHA-verified saved epochs at 2026-11-15T16:52:45.331611+00:00: {'unet/seed_17': 242}.

U-Net seeds 17/29/41 run first, each 256 epochs; then no_dechirp, no_rms, no_pc and no_damage, each with the same three seeds and budget. 15 new runs in total. U-Net has exactly 59,714 active/registered parameters; the full model has 59,714 active plus 53,042 unused registered parameters. The three full epoch-256 checkpoints are reused and their DEV evaluations were rerun on the server. Every new epoch retains a complete recovery archive; every 32 epochs retains a milestone and DEV result. After all training, generate 512 fresh confirmation parents and compare all six models at fixed epoch 256 on the same observations. No best-seed/epoch selection. Report u_nmse, damage, SJRIF, Pd and empirical Pfa. Parent bootstrap intervals are conditional on the three trained seeds; seed SD is separate. Historical full training environment differs from the new server training; do not claim identical trajectories.

A planned saved-boundary launcher restart preserved epoch 1 and moved execution to a hidden WMI-owned launcher independent of SSH lifetime. The retained CONTROLLER_FAILURE.json for old PID 30264 records that planned stop; assess the current queue/PIDs and saved epochs. Initial TF32 numerical-preflight failure is retained; explicit TF32-off FP32 passed the unchanged tolerances. No U-Net performance conclusion is available yet.

Original training completed for all eight seeds (17,29,41,53,67,79,97,109) at epoch 256: 64 immutable milestones, 64 DEV v4 evaluations and 2048 epoch metric rows. Original independent TEST used frozen seed 29/epoch 128: 2237 method-condition receipts and 103826 audited parent blocks. The additive F-R7 RSDR-Net comparison completed 564/564 conditions on the same second-paper TEST observations; it compares one frozen seed per model with different training budgets/distributions. The original epoch-256 curves remain DEV evidence. Reports: `PAPER2_PROGRESS_CHECK_20261115.md`, `PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md`.

Long training is server-only: rsdr-b / DESKTOP-RFFERM3, `D:/code/Xin/ISRJ_222`, physical GPU 1. No local formal training or fallback. First-paper sources, environments and unrelated jobs are read-only. Frozen `paper2/`, `tools/unet_ablation/` and `tools/durable_io.py` must match config SHA bindings. Only the `paper2` remote (`Hxin0203/ISRJ_2`) may be pushed. Do not upload datasets, weights, raw arrays, caches, credentials or full logs. No old Gate access, MCA-ADMM/FNet or ChatGPT collaboration. The user's 2026-11-15 request explicitly authorizes these ablations and supersedes older pause statements.
