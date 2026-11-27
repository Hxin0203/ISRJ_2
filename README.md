# ISRJ_2: target-response-constrained restoration

<!-- HYBRID_5SEED_RESULTS_START -->
## Hybrid 五种子结果（固定更新入口）

本轮完成 seed **17、29、41、53、67** 的 epoch 256 汇总与零训练补测，共 **15 张图（PNG + PDF）**。覆盖训练收敛、Equal-PFA、32 格 SNR×JSR、12 类卷积对照、背景长尾、脉压/相位/多普勒和五种子冻结推理干预。

- 综合报告：[REPORT.md](reports/Hybrid_5seed_20261127/REPORT.md)。
- 数值与证据：同目录 `ANALYSIS.json`、`TRAINING.json`、各项 CSV、`VALIDATION.json` 和逐文件 SHA256 的 `MANIFEST.json`。
- 项目备份：`G:/Project Code/ISRJ_222/reports/Hybrid_5seed_20261127/`。
- 交接备份：`G:/Files/Markdown/ISRJ_2/opus对话/03_分析/Hybrid_5seed_20261127/`。
- GitHub 同步回执：`handoff/Hybrid_5seed_20261127/GITHUB_SYNC.json`（推送核验后写入）。

主要结论：Hybrid 在 5%/10% 校准虚警点超过两种 Tail 基线；对第一篇 RSDR-Net，5% 点略低、10% 点持平、1% 点明显落后。受控扫描中平均 E_U 和 E_S 更低；DEV 脉压目标相干功率少损失约 0.323 dB，但实际含噪多普勒估计未证实改善。seed 67 的 E0 异常已保留并单列。不同评测分布的排名不能混用；BVR01/JCR、独立确认集和公平计时尚未封口。这是五种子阶段报告，不是八种子最终结论。

维护规则：以后本项结果、路径或结论变更时，**直接更新本 README 的这个固定段落**，保留本说明入口，不另建新的主目录说明报告。历史备份放入对应备份目录。
<!-- HYBRID_5SEED_RESULTS_END -->


This repository contains the controlled source and evidence snapshot for the second ISRJ paper. The active server-only U-Net comparison and four ablations are specified in [PAPER2_UNET_ABLATION_PLAN_20261115.md](PAPER2_UNET_ABLATION_PLAN_20261115.md). Read [PAPER2_PROGRESS_CHECK_20261115.md](PAPER2_PROGRESS_CHECK_20261115.md) for the completed original training and TEST audit, and [PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md](PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md) for the additive comparison with the first-paper RSDR-Net. [ISRJ_2_CURRENT_STATE_REPORT_20261113.md](ISRJ_2_CURRENT_STATE_REPORT_20261113.md) preserves the earlier project history, model structure and mathematical formulation. Results concern the stated simulation distributions; novelty and field performance remain unestablished.

The 1103-aligned execution is documented in [PAPER2_1103_PROGRESS_20261114.md](PAPER2_1103_PROGRESS_20261114.md) and [PAPER2_1103_NEAR_TERM_PLAN_20261114.md](PAPER2_1103_NEAR_TERM_PLAN_20261114.md). The independent original TEST completed with the frozen seed-29/epoch-128 checkpoint: 2,237 method-condition receipts and 103,826 audited parent blocks. The additive RSDR-Net comparison has its own 564-condition freeze and completion receipts under `evidence/outputs/RSDR_UNIFIED_20261115/`; read its current report for audited coverage. Both models use the same second-paper TEST parents, pulse compression, Doppler processing and CFAR. This is one frozen seed per model with different training budgets; Pd must be interpreted alongside empirical false alarms. MCA-ADMM-Net and FNet remain paused. The user authorized the new U-Net comparison and structural/loss ablations on 2026-11-15.

The follow-up specified in [PAPER2_256_EPOCH_8_SEED_PLAN_20261114.md](PAPER2_256_EPOCH_8_SEED_PLAN_20261114.md) has completed: seeds 17, 29, 41, 53, 67, 79, 97 and 109 all reached epoch 256, with 64 immutable 32-epoch milestones, 64 DEV evaluations and 2,048 epoch metric rows. Earlier checkpoints and recovery archives are retained. Convergence data and figures are under `evidence/outputs/CONVERGENCE_256_8SEED_20261114/` and `evidence/outputs/PROGRESS_CHECK_20261115/`. These expanded-model results are DEV evidence; they do not replace the frozen epoch-128 TEST or justify TEST-based model reselection.

`PAPER2_EXPERIMENT_ROADMAP_20261114.md` records the earlier broader roadmap. The active addition uses seeds 17, 29 and 41, each trained for 256 epochs, for an ordinary U-Net and four interventions: no dechirp, no RMS, no pulse-compression loss, and no target-damage loss. The U-Net has exactly 59,714 active parameters, matching the full model's active graph. The full model also registers 53,042 unused parameters, which are excluded from capacity matching. Original full epoch-256 checkpoints are reused. Every new epoch retains a complete recovery checkpoint; every 32 epochs retains a milestone and DEV evaluation. Fresh independent confirmation follows all 15 training runs.

Read PAPER2_GO_NO_GO.md and PAPER2_PILOT_RESULTS.md for the initial pilot, and PAPER2_DEV_CONFIRMATION.md for its subsequent failed frozen confirmation. FAMILY_STUDY_CONTRACT.md and FAMILY_STUDY_FREEZE.json define the separate target-family experiment; its completed development results are in PAPER2_FAMILY_RESULTS.md and evidence/family/RESULTS.json. PAPER2_NEAREST_WORK_COMPARISON.md records literature access limits.

All long training runs only on server B (`rsdr-b`, `DESKTOP-RFFERM3`) in `D:/code/Xin/ISRJ_222`, on physical GPU 1. Python and PowerShell entry points reject local training. From this local workspace, inspect and retrieve server evidence with:

```powershell
python -m tools.deploy_unet_ablation status
python -m tools.update_unet_ablation_handoff
```

The queue is already running under a hidden launcher independent of SSH lifetime. Do not redeploy or prepare over its frozen sources. `PAUSE` and `STOP` files in the server run directory act at saved boundaries; no historical checkpoint is discarded. Local CPU engineering regression checks are available with:

```powershell
python -m pytest tests/test_unet_ablation.py tests/test_train_1103.py tests/test_evaluation_1103_v4.py tests/test_long_pipeline.py -q
```

The first pilot and its confirmation are complete and immutable. Do not rerun their evaluators to replace the updated go/no-go decision. Historical family-study commands and sources are retained as evidence; they are not the active training queue. Completed family results are preserved rather than automatically reselected.

Observed-only waveform restoration is available through `python -m tools.restore_waveform --help`. It accepts complex observations and a fixed checkpoint/global gain; it never accepts true target/jammer components. Do not describe these development weights as a validated production or field model.

The engineering command freezes training length from measured cost. Completed runs are preserved and source/config/checkpoint hashes must match. A partial scientific run deliberately requires investigation before restart. The evaluation wrapper checks supplemental dataset bindings, all six run identities, identical per-seed initialization and minibatches, then performs DEV-only analysis. Do not bypass it by calling paper2.evaluate directly.

Historical local data, weights and raw outputs stay in ignored data/ and outputs/. New training data, weights, raw arrays and full logs stay in the dedicated server project; only lightweight JSON evidence is fetched. Only screened source, contracts and small derived evidence belong in Git. No old held-out Gate arrays are read. The original pipeline and RSDR comparison are complete; current training progress is tracked in `handoff/RUNNING_EXPERIMENTS.json` and `evidence/outputs/unet_ablation_3seed_256_20261115/SERVER_STATUS.json`. Known waveforms and all old source snapshots have origin hashes in SOURCE_MANIFEST.json. Push this work only to the dedicated `paper2` remote (`Hxin0203/ISRJ_2`).

The user explicitly suspended codex-with-chatgpt for this task; Codex performs and labels its own checks. No historical ChatGPT review or PASS is fabricated.
