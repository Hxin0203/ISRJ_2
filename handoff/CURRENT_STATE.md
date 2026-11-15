# Current state

Progress checked on 2026-11-15. The seven-stage pipeline is COMPLETE (2026-11-15T04:29:14.525505+00:00, 2026-11-14 20:29 PST). No live project training/test process remains. Fresh full integrity audit: `evidence/outputs/PROGRESS_CHECK_20261115/INTEGRITY_AUDIT.json`.

All eight seeds (17, 29, 41, 53, 67, 79, 97, 109) reached epoch 256. Verified: 64 immutable milestone checkpoints, 64 DEV v4 evaluations, 2048 epoch metric rows, 1667 recovery archives, 1748 convergence input hashes, and all 12 original checkpoints. Curves and aggregates: `evidence/outputs/CONVERGENCE_256_8SEED_20261114/`; readable review figures: `evidence/outputs/PROGRESS_CHECK_20261115/`.

Frozen independent TEST is complete: 2237 method-condition results and 103826 audited parent blocks. It uses seed 29 epoch 128, chosen using the original three-seed mean DEV eU and frozen before TEST; not a pre-DEV preregistration. The eight-seed epoch-256 results are DEV, not new independent TEST confirmation. No TEST-based reselection. `reference` is the same-noise unjammed reference, not RSDR-Net. Matched RSDR-Net comparison remains outstanding. Freeze: `configs/paper2_1103_scan_freeze.json` (SHA 8fed1450643ee71d8725daf0bb046a96c3d9301959c11000acb2d08c951c2930).

Current report: `PAPER2_PROGRESS_CHECK_20261115.md`. Historical selection, formulas and TEST differences: `PAPER2_1103_PROGRESS_20261114.md`. In this schedule, gains diminish beyond epoch 128 and curves nearly plateau at 192–256. Seed SD describes training randomness on shared data, not independent datasets or a parent CI. Oracle SJRIF/Pfa and coherent attenuation limitations remain; do not claim universal superiority or submission readiness.

Only paper2 remote may be pushed. No ChatGPT collaboration; no old Gate access; no MCA-ADMM/FNet/ablations. Training was 48 updates/epoch, unlike 1103's 512. Preserve every checkpoint, failure and negative condition. Raw data, weights, arrays, caches and logs stay local. Baselines are formula reproductions, BCS oracle is labelled; bibliographic verification remains.
