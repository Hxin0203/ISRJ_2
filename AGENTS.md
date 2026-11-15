# ISRJ_2 workspace rules

- Long training for this project must run on server B (`ssh rsdr-b`), never on this local workstation. The U-Net/ablation study uses the separate server directory `D:/code/Xin/ISRJ_222` and physical GPU 1. Its Python entry points and PowerShell launcher enforce the host, path and device. Do not bypass these checks or silently fall back to local execution.
- All second-paper assets remain in this workspace or its dedicated server counterpart. First-paper directories and existing frozen sources/checkpoints are read-only. Do not modify or stop unrelated server jobs.
- Only push the `paper2` remote (`Hxin0203/ISRJ_2`); `origin` belongs to the first paper. Stage exact paths. Do not upload training data, weights, raw arrays, caches, credentials or full logs.
- The user authorized U-Net comparison and structural/loss ablations on 2026-11-15. This supersedes older handoff statements prohibiting ablations. Do not start MCA-ADMM or FNet work. ChatGPT collaboration remains disabled; do not use `codex-with-chatgpt` or equivalent alternate collaboration flows.
- Preserve every completed epoch recovery point and every 32-epoch milestone, including prior models. Validate source, data and checkpoint SHA before resume or evaluation. Report saved epochs and dated status snapshots, not just process existence.
- Apply the user's `clean-user-facing-text` preference when writing scientific reports: offline audit, backup and SHA, protected formulas/values/paths, no indiscriminate whole-document rewrite or claims of authorship detection evasion.
