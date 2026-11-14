# Exploratory research contract — 2026-10-20

User request: pursue a feasible new ISRJ model using the two supplied plans; autonomy granted. Latest user instruction temporarily disables ChatGPT collaboration. Local evidence governs decisions. Prior old-workspace checkpoint is historical and untouched.

H1: at equal empirical mean target-response damage, B4 reduces full U-NMSE beyond same-data B1/B2 and globally rescaled corrections. This pilot can reject or motivate H1, never confirm novelty.

Fresh simulator roles: engineering 16 parents (seed 2026102001, namespace 220010000), TRAIN 512 (2026102002, 220020000), DEV 192 (2026102003, 220030000). Each parent has direct/repeat/cyclic jammer variants and four states. No TEST generated or accessed. Reserve future TEST namespace 220040000 and seed 2026102004; do not materialize before method freeze. Distinct parents, not variant rows, are independent units. TRAIN/DEV parameters: length 1800, fs 60 MHz, pulse 10 us, bandwidth 30 MHz, SNR [-10,5] dB, JSR [0,25] dB; all remaining parameters inherited explicitly in generator manifest. This is a restricted feasibility distribution, not the old Extreme distribution.

Training: seeds 17 and 29, one backbone, B1/B2/B4, at most six scientific short runs. Adam lr=0.001, batch 32 parent-variant groups (128 state observations). Engineering timing determines short-run length before TRAIN/DEV training. Finite loss/gradient, source identity, role disjointness and numeric identities required. Fixed final checkpoint; no DEV checkpoint selection. No adaptive lambda search.

Before DEV inspection freeze: damage budgets [0.01,0.05,0.1]; real alpha candidates 0 to 2 in steps of 0.025, plus analytic constrained optimum in that finite range; DEV-fitted global complex gain control on the same radius bound 2. Each method gets the same controls. KEEP is always included. Mechanism signal requires >=5% relative U-NMSE improvement over the B1/B2 control envelope in both seeds at a common budget, actual suppression >=10% relative to KEEP, and damage>0.25 frequency <=10%. If no budget qualifies, REVISE/STOP this candidate; no extra modules or unbounded search. These are empirical exploratory criteria, not population guarantees. Report parent bootstrap uncertainty and seed-specific values; small differences are inconclusive.

Resource ceilings: engineering probe <=30 updates; each short run <=1200 updates and <=15 minutes; total six-run training <=90 minutes; new local assets <=5 GB; one GPU process. Do not change OS settings or interrupt the existing migration process. Abort on nonfinite values or identity drift. Preserve failure evidence and resume only incomplete engineering work. A failed scientific configuration counts toward the six-run ceiling.

Formal TEST remains unopened, especially while ISRJ-FNet and MCA-ADMM-Net full methods remain unavailable. Do not use a useful DEV result as proof of novelty, risk control, field performance or weak-target safety.
