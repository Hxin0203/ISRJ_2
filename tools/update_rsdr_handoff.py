"""Refresh lightweight handoff evidence for the additive RSDR benchmark."""
from __future__ import annotations
import subprocess

from paper2.io import ROOT, now, read, sha
from tools.compare_rsdr_unified import LIGHT, DEST, FREEZE
from tools.durable_io import atomic_copy, atomic_json, atomic_write, process_alive
from tools.status_rsdr_unified import snapshot


def write(path,text):
    if path.exists():
        backup=DEST/'handoff_history'/f'{path.name}_{sha(path)}'
        if not backup.exists():atomic_copy(path,backup)
    atomic_write(path,lambda stream:stream.write(text.encode('utf8')))


def main():
    s=snapshot(save=True);a=read(LIGHT/'PAIRED_RESULTS.json') if (LIGHT/'PAIRED_RESULTS.json').exists() else None
    state=s['status']
    h=ROOT/'handoff';running=read(h/'RUNNING_EXPERIMENTS.json')
    running.update(checked_utc=now(),rsdr_unified=s,active_experiment_processes=[dict(pid=s['pid'],name='frozen RSDR-Net benchmark')] if process_alive(s['pid']) else [],
        note='Original training/TEST pipeline complete. Additive frozen RSDR benchmark status is reported separately; no model reselection.',
        uncompleted_scientific_work=([] if state=='COMPLETE' else ['Complete and audit additive RSDR benchmark'])+['Independent confirmation plan for expanded models','Baseline bibliography/discrete implementation verification','Final manuscript'])
    write(h/'RUNNING_EXPERIMENTS.json',__import__('json').dumps(running,indent=2,ensure_ascii=False)+'\n')
    text=f'''# Current state

Updated {now()}. Additive frozen RSDR-Net comparison: **{state}**, {s['complete_conditions']}/{s['expected_conditions']} condition results saved. Primary repeat main-grid scope complete: {s['primary_complete']}. Live PID and heartbeat evidence: `evidence/outputs/RSDR_UNIFIED_20261115/RUN_STATUS.json`. A result exists only after a complete condition receipt; partial blocks remain resumable.

Current comparison report: `PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md`. Audited snapshot: {a['complete_conditions'] if a else 0} conditions. Comparison freeze: `configs/rsdr_unified_20261115.json` (SHA {sha(FREEZE)}). Immutable inference/evaluation sources must not be edited. Postprocessing is managed by `python -u -m tools.finalize_rsdr_unified`; it audits, plots and refreshes the report at complete primary/main/all milestones, with no Git writes.

Second paper: frozen seed 29 epoch 128, SHA `7e05f594c26a9c24061216ab4c9cc1e9f264269fb9cbca5e10c27b7690e53cf3`. First paper: complete F-R7 RSDR-Net, seed 40312102 epoch 32, two experts/three routers/fixed anchor/bounded group gate. Native inference and local/remote numerical parity passed on fresh engineering scenes. Same second-paper TEST observations, parent IDs, pulse compression, Doppler and CA-CFAR. No external input normalization. One frozen seed per model, different training budgets and training distributions. Same nominal CFAR does not imply equal empirical Pfa. SJRIF is a pulse-compression peak proxy. Do not claim universal superiority or equal-budget superiority.

The original seven-stage pipeline completed at 2026-11-15T04:29:14.525505+00:00. All eight seeds (17,29,41,53,67,79,97,109) reached epoch 256: 64 immutable milestone checkpoints, 64 DEV v4 evaluations, 2048 epoch metric rows, 1667 recovery archives, 1748 convergence input hashes and all 12 ancestral checkpoints verified. Prior integrity audit: `evidence/outputs/PROGRESS_CHECK_20261115/INTEGRITY_AUDIT.json`. Independent original TEST: 2237 method-condition results, 103826 audited parent blocks. Epoch-256 results remain DEV; no TEST reselection.

Only the `paper2` remote (`Hxin0203/ISRJ_2`) may be pushed. No ChatGPT collaboration, old Gate access, MCA-ADMM/FNet or ablations. All new persistent assets stay here; remote inference uses in-memory streaming and reads original assets only. Preserve failures, negative outcomes and checkpoints. Raw arrays, datasets, weights, caches and full logs must not enter GitHub.
'''
    write(h/'CURRENT_STATE.md',text)
    write(h/'LAST_CODEX_EXECUTION.md',f'''# Last execution

Updated {now()}. User requested a direct unified comparison with the first-paper RSDR-Net. Identified and hash-bound the complete deployed F-R7 graph; preserved original vendor sources byte-for-byte. Validated inference against the original portable runtime without invoking its old-data audit/validation entry points. Fresh engineering parity receipts are separate from TEST results. Adapter persistence bug was corrected and validation rerun before freeze; event retained.

Frozen new comparison before RSDR TEST inference. Reused the existing seed-29 epoch-128 TEST blocks after SHA, metadata and parent-identity checks. Running {s['complete_conditions']}/564 conditions with 16-parent atomic recovery blocks on the local GPU and two read-only remote streaming GPU workers. Status: {s['status']}. The full planned evaluation count is 231216 parent-condition observations, with repeated physical parents across grid cells. Report only complete scopes as complete; do not infer population sample size from this count.

Added paired aggregation, pointwise physical-parent bootstrap intervals, event counts and exact binomial per-cell intervals, figures with common scales, fixed-parent examples, report generation and durable postprocessing. Report/prose audit is offline detect-only with protected math/values/paths/citations. Current evidence is in `evidence/outputs/RSDR_UNIFIED_20261115/`; comparison sources are in `tools/`, leaving original freeze-bound `paper2/` untouched.

Original training and independent TEST pipeline remains complete. No epoch/seed reselection, threshold tuning, new training, old Gate-array access or ChatGPT collaboration. No original directory modified. Only lightweight evidence/source/config/report/figures are candidates for the dedicated paper2 GitHub repository; model weights, raw arrays and logs stay local.
''')
    write(h/'OUTPUT_EVIDENCE_INDEX.md',f'''# Output evidence index

Updated {now()}. Direct RSDR comparison is {s['status']}: {s['complete_conditions']}/564 saved conditions; audited aggregate currently {a['complete_conditions'] if a else 0}. Primary complete: {s['primary_complete']}. The aggregate/report can trail live saved progress while auditing.

Current comparison report: `PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md`. In `evidence/outputs/RSDR_UNIFIED_20261115/`: `RUN_STATUS.json`, `PAIRED_RESULTS.json`, `CONDITION_COMPARISONS.csv`, `RESULT_INTEGRITY_AUDIT.json`, `MODEL_IDENTITIES.json`, `ADAPTER_PARITY.json`, `EXTENDED_PARITY.json`, `ENGINEERING_EVENTS.json`, `FIGURE_MANIFEST.json`, `CURVE_SOURCE_SUMMARIES.csv`, `REPORT_RECEIPT.json`, `TEXT_AUDIT.json`, and completion milestone receipts as produced. No missing file represents a scientific zero. Freeze: `configs/rsdr_unified_20261115.json`.

Original progress report: `PAPER2_PROGRESS_CHECK_20261115.md`; original method/selection report: `PAPER2_1103_PROGRESS_20261114.md`. Eight-seed epoch-256 DEV convergence: `evidence/outputs/CONVERGENCE_256_8SEED_20261114/`; readable review figures and verified training receipts: `evidence/outputs/PROGRESS_CHECK_20261115/`. These are distinct from frozen epoch-128 TEST and do not establish expanded-model independent TEST confirmation.

Original seven-stage controller and completion receipts: `evidence/outputs/paper2_long_pipeline_20261114/`; original TEST figures: its `test_figures/`. Original coverage and condition tables: `evidence/outputs/TEST_COVERAGE_1103.json` and `TEST_CONDITION_TABLE_1103.csv`. Original TEST complete at 2237 method-condition receipts / 103826 audited parent blocks. Original DEV/checkpoint verification: `evidence/outputs/paper2_1103_retry_20261114/`.

All raw arrays, datasets, model checkpoints, caches and full logs remain local under ignored `outputs/` or `data/`. Original failure history retained. Only the dedicated paper2 remote is authorized for this work.
''')
    p=subprocess.run(['git','status','--short','--untracked-files=no'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0
    write(h/'GIT_STATUS.txt',f'Checked {now()}\nTracked changes only; unrelated untracked evidence omitted.\n{p.stdout}\nOnly paper2 remote may be pushed. No raw arrays/weights/logs.\n')
    print(dict(status=s['status'],saved=s['complete_conditions'],audited=a['complete_conditions'] if a else 0),flush=True)


if __name__=='__main__':main()
