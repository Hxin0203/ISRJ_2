# Paper 2 minimal screening implementation plan

Execution: inline by Codex, following writing-plans and executing-plans; the user explicitly requested autonomous decisions and no repeated approvals.

Goal: complete a fair, bounded six-run mechanism pilot and retain a defensible continue/revise/stop decision.
Architecture: source-bound simulator snapshot, shared residual model, separate role generation/training/evaluation/report modules.
Tech stack: existing Python 3.13, NumPy, PyTorch 2.9.1 CUDA, pytest, matplotlib.
Spec: PAPER2_MINIMAL_MODEL_SPEC.md and PAPER2_RESEARCH_CONTRACT.md.

1. Write scientific contract tests for paired states, physical scale, full-error decomposition, gain scaling, complex gain control and train-only loss; observe missing implementation failure. Snapshot only small simulator/model source files and record original paths/hashes.
2. Implement data and model/loss helpers. Verify shared physical components, disjoint role namespaces, no oracle inference inputs, nonzero gradients for all losses and gain math against direct complex arithmetic.
3. Run engineering-only <=30-update timing probe. Freeze steps based on measured runtime, memory and storage. No scientific DEV output before this freeze.
4. Generate only fresh TRAIN/DEV and fit B1/B2/B4 with seeds 17/29. Bind config, source, role manifests, checkpoint, update count and runtime. Write atomic status and per-run completion receipts; never overwrite completed runs silently.
5. Reuse frozen model outputs for all gains; compute parent metrics, matched-budget controls, seed-specific comparisons and parent bootstrap intervals. Record all results and limitations in five requested deliverables; leave TEST untouched.

Review focus: independent normalization per observation; removing target must not rescale J/noise; variants must not inflate sample count; scalar attenuation/amplification/phase gains must not impersonate learning a direction; final checkpoint must not be selected on TEST or DEV. All five receive direct tests or recorded runtime audits.
