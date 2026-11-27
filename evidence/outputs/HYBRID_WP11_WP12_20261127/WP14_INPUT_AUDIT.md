# WP14 probe input provenance audit

- Audited file: `WP14_probe_SCORES.npz`
- SHA256: `1ae952c7ddcd572073d715e045893e0281cc56bfe974c11dc94a8626077d5b01`
- Container inventory: `cal_parent_id` (1024), `eval_parent_id` (2048), and 11 score/diagnostic arrays. The score arrays have shape `(2048, 3)` and dtype `float64`.
- The file contains no method or seed labels, source/config/checkpoint hashes, evaluation registration, generation receipt, or code identity binding the three columns to a model/run.
- Result: **not admissible as a formal WP14 result**. Do not compute or report BVR01/JCR from these columns until provenance and metric definitions are independently recovered. Preserve the file unchanged for traceability.

This is a provenance finding, not evidence that the numerical arrays are invalid. The original first-paper source/manuscript searches performed for this continuation did not locate an authoritative BVR01/JCR formula or executable implementation. WP14 therefore remains open pending same-definition metric recovery.

## Follow-up source lookup (2026-11-27)

The first-paper repository contains a `.codegraph` index, so a targeted CodeGraph lookup was attempted before text search. `codegraph explore`, `status`, and a direct `query` produced no output within bounded waits (20–30 seconds). A bounded text search of the first-paper `markdown/` and `docs/` folders for `BVR01`, `JCR_dB`, “Jamming Cancellation Ratio”, and cancellation-ratio wording returned no hits. This limited negative search does not establish that the definition is absent from every source or archived document. No formula was inferred from similarly named metrics, and no values were computed from the unauthenticated NPZ. WP14 remains open.
