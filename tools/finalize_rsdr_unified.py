"""Durable postprocessing for the already running, finite benchmark job.

No model/source/threshold selection and no Git writes. Receipts are generated
at complete primary, full main, and full benchmark milestones. Resume safely
by checking receipts bound to the current postprocessing source hashes.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
import traceback

from paper2.io import ROOT, now, read, sha
from tools.durable_io import ProcessLock, atomic_json, atomic_write, atomic_copy
from tools.compare_rsdr_unified import LIGHT, DEST, FREEZE
from tools.status_rsdr_unified import snapshot


def prose_audit(report):
    # Detect-only. Never apply whole-Markdown normalization to protected math.
    dest=LIGHT/'TEXT_AUDIT.json'
    skill=ROOT.__class__('C:/Users/admin/.codex/skills/clean-user-facing-text/scripts/inspect_text.py')
    backup=DEST/'report_history'/f'{report.stem}_{sha(report)}.md'
    if not backup.exists():atomic_copy(report,backup)
    p=subprocess.run([sys.executable,str(skill),'--audit',str(report)],capture_output=True,text=True,encoding='utf8',errors='replace')
    atomic_json(dest,dict(checked_utc=now(),report_sha256=sha(report),backup_sha256=sha(backup),
        mode='Offline detect-only; formulas, values, code, paths and citations protected. No automatic cleanup.',
        chinese_scoring_limit='English stylometry heuristic is not calibrated for Chinese; not an authorship or authenticity test.',
        exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode in (0,1),(p.returncode,p.stderr)


def publish_snapshot(stage):
    from tools.aggregate_rsdr_unified import aggregate
    from tools.plot_rsdr_unified import main as figures
    from tools.report_rsdr_unified import report
    result,records=aggregate(require_complete=stage=='all',return_records=True)
    figures(with_examples=stage=='all',audited=records)
    path=report();prose_audit(path)
    from tools.update_rsdr_handoff import main as handoff
    handoff()
    atomic_json(LIGHT/f'POSTPROCESS_{stage.upper()}_COMPLETE.json',dict(completed_utc=now(),stage=stage,
        conditions=result['complete_conditions'],status=result['status'],freeze_sha256=sha(FREEZE),
        report_sha256=sha(path),aggregate_sha256=sha(LIGHT/'PAIRED_RESULTS.json'),
        figure_manifest_sha256=sha(LIGHT/'FIGURE_MANIFEST.json'),
        postprocessing_source_sha256={name:sha(ROOT/name) for name in (
            'tools/aggregate_rsdr_unified.py','tools/plot_rsdr_unified.py',
            'tools/report_rsdr_unified.py','tools/status_rsdr_unified.py',
            'tools/update_rsdr_handoff.py','tools/finalize_rsdr_unified.py')}))


def main(once=None):
    with ProcessLock(DEST/'POSTPROCESS.lock'):
        if once:
            publish_snapshot(once);return
        while True:
            s=snapshot(save=True)
            if s['primary_complete'] and not (LIGHT/'POSTPROCESS_PRIMARY_COMPLETE.json').exists():publish_snapshot('primary')
            if s['coverage']['main']['complete']==s['coverage']['main']['expected'] and not (LIGHT/'POSTPROCESS_MAIN_COMPLETE.json').exists():publish_snapshot('main')
            if s['status']=='COMPLETE':
                if not (LIGHT/'POSTPROCESS_ALL_COMPLETE.json').exists():publish_snapshot('all')
                return
            if s['status']!='RUNNING':raise RuntimeError('Benchmark stopped without completion; saved blocks remain resumable.')
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',choices=('primary','main','all'));a=p.parse_args()
    try:main(a.once)
    except BaseException:
        atomic_json(DEST/f'POSTPROCESS_FAILURE_{time.time_ns()}.json',dict(failed_utc=now(),traceback=traceback.format_exc()))
        raise
