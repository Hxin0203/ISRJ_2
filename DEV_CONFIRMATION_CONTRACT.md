# Frozen development confirmation — before new data generation

This is a second, explicitly exploratory development check after the six-run G2 signal. It is not final TEST and will be counted as development exposure. No new training, loss tuning, gain fitting or model selection is allowed here.

Freeze all six final checkpoints and each method's complex global gain at original DEV damage budget 0.05. Original DEV selected B1 seed17 / B2 seed29 as the strongest controls; retain these controls and B4 for the comparison. Also retain all six method results, even if another control happens to be better on new data. Freeze prediction implementations and checkpoint hashes in DEV_CONFIRMATION_FREEZE.json before generation.

New ID development role: 192 parents, seed 2026102011, namespace 220050000; same simulator parameters as TRAIN/DEV. New two-target role: 96 parents, seed 2026102012, namespace 220060000; same parameters, with a second independently delayed/phased/Doppler target at amplitude 0.25 times the first target (−12.04 dB power). The jammer and receiver noise remain those of the first target's parent; never rescale them after adding/removing either target. Three jammer variants per parent remain grouped.

For each target k, the counterfactual removes only Sk and preserves the other target, J and E. Full restoration reference always remains S1+S2+E. Evaluate each target's response damage separately; neither the total target energy nor strong-target energy may normalize weak-target damage. Multi-target scenes are an extra-synthetic stress condition, not a claim to reproduce field jamming.

Continue the current mechanism only if on new ID both seeds still improve U-NMSE at the frozen gains, empirical mean damage remains <=0.05 and severe-parent rate <=0.1. Failure does not authorize retuning on this role. New ID includes no selection-adjusted guarantee; DEV-only uncertainty remains descriptive. For two-target scenes, report strong/weak actual damage, amplitude, circular phase and severe-parent rate. If weak-target mean damage exceeds 0.05 or severe-parent rate exceeds 0.1, do not claim H2; record that a new preregistered target-family training study would be required before any formal TEST.

Only 288 new parents and inference on the existing six models are authorized for this check. Final TEST seed 2026102004 / namespace 220040000 remains ungenerated. No OOD or extra parameter sweep is added.
