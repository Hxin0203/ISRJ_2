# Independent code/contract review record

Reviewer: a separate Codex code-review agent in this session, not ChatGPT web and not an external experimental PASS.

## Initial four-state implementation

Confirmed by inspection: state labels and shared latent scale; target-response/error decomposition signs; nested B1/B2/B4 losses and same inference graph; observed-only per-input normalization and GroupNorm; parent grouping; real/complex constrained quadratic gain formulas; sufficient zero padding for linear matched filtering.

Two safeguards were initially missing: evaluation-time source/config/end-to-end identity checks and explicit dataset SHA in each training run. An external guard now verifies all six runs, frozen configuration, original source, checkpoint bytes and per-seed init/batch/exposure equality. RUN_DATA_BINDING.json transparently records the supplemental timing of dataset binding. No scientific run was repeated to conceal the omission. EVALUATION_AUDIT.json contains the actual verification outcome.

Review limits: four-state target removal retains an S-generated J, therefore is a controlled intervention rather than an independent physical absence experiment. Mean budget comparisons are constrained-frontier comparisons, not exact equal realized damage. Wrapped-phase arithmetic mean is not sufficient evidence of phase preservation; subsequent development checks include circular summaries. DEV selection renders bootstrap intervals descriptive rather than selection-adjusted.

## Completed pilot and frozen development check

Reviewer agrees the original candidate is FAILED_CONFIRMATION: new-ID empirical mean budgets exceeded; weak-target empirical tail thresholds failed. The large U-NMSE advantage remains an observation. Small-sample tail threshold failures do not estimate population risk precisely. All B1/B2 new-ID damages also drifted, so no claim that this drift is specific to B4.

A separate target-family study is justified by the user request and the earlier confirmation contract. The review required new roles, all-eight-state exposure in every method, explicit four-edge averaging, own-target denominators, shared controls and a six-run cap. FAMILY_STUDY_CONTRACT.md implements these restrictions. Neither adding modules nor redefining calibration as innovation is authorized by this scientific rationale.

## Frozen eight-state implementation

Fresh static review confirmed all eight indices, the four equal response edges, individual target denominators, separate-target mean constraints, parent-maximum tail order statistic, shared forward graph and freeze-before-confirmation ordering. No training-source correction was required. A new CPU test forces the tail bound to bind before the mean bound. tools/family_guard.py performs pre-DEV and post-evaluation provenance/namespace/component checks outside the immutable training source.

The target-family gate constrains each target's severe-parent rate separately, not the union event that either target is severely damaged. Noise-named helper fields in target-removal contexts represent other-target-plus-noise backgrounds; they must not be described as receiver-noise-only measurements.

## Final evidence review

The second study's two seeds pass its prospective empirical gate: U-NMSE improvement 51.08%/55.41%, suppression vs KEEP 81.46%/81.58%, maximum target mean damage .03735/.04004, maximum per-target severe-parent rate 3.125%/2.604%. Registry gains and checkpoint identities match DEV freeze, and result/gain SHA match final audit.

Material qualification: candidate weak-target preservation is not uniformly superior to the selected B1 control. Weak mean damage is about .0301/.0295 versus control .0224/.0219; weak severe-parent rate 3.125%/2.604% versus control 0%. The supported benefit is stronger restoration within shared empirical constraints, not dominance on every safety metric.
