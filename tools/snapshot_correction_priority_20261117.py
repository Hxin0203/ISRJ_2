"""Light status inspection; skip the locked first byte of Windows process locks."""
import json
import platform
import statistics
from pathlib import Path
from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json, process_alive


def main():
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Server B only')
    jobs = []
    for run in ('correction_original_control_20261117_v1','correction_mf_tail_20261117_v1'):
        base = ROOT/'outputs'/run
        binding = read(base/'QUEUE_BINDING.json') if (base/'QUEUE_BINDING.json').exists() else None
        pid = None
        if (base/'CONTROLLER.lock').exists():
            with (base/'CONTROLLER.lock').open('rb') as stream:
                stream.seek(1)
                pid = json.loads(stream.read())['pid']
        seeds = []
        for seed in (17,29,41):
            folder = base/'arm'/f'seed_{seed}'
            receipts = sorted((folder/'recovery').glob('epoch_*.json'))
            latest = read(receipts[-1]) if receipts else None
            times = [p.with_suffix('.pt').stat().st_mtime for p in receipts[-21:]]
            intervals = [b-a for a,b in zip(times,times[1:]) if b>a]
            seeds.append({'seed':seed,'saved_epoch':latest['epoch'] if latest else None,
                'recovery_receipts':len(receipts),
                'latest_sha_matches':sha(receipts[-1].with_suffix('.pt'))==latest['checkpoint_sha256'] if latest else None,
                'complete':(folder/'TRAINING_COMPLETE.json').exists(),
                'recent_wall_seconds_per_epoch':statistics.median(intervals) if intervals else None})
        jobs.append({'run':run,'physical_gpu':1,'controller_pid':pid,'controller_alive':process_alive(pid),
            'status':read(base/'STATUS.json') if (base/'STATUS.json').exists() else {'stage':'NOT_STARTED'},
            'queue_binding_sha256':sha(base/'QUEUE_BINDING.json') if binding else None,
            'source_drift':[p for p,h in binding['source_sha256'].items() if sha(ROOT/p)!=h] if binding else None,
            'seeds':seeds,'numerical_preflight_passed':(base/'NUMERICAL_PREFLIGHT.json').exists()})
    dilation = ROOT/'outputs/unet_dilation_screen_20261117_v1'
    pause = dilation/'PAUSE'
    result = {'snapshot_utc':now(),'jobs':jobs,
        'dilation_pause_owned':pause.exists() and pause.read_text(encoding='utf8').strip()=='ISRJ2_CORRECTION_TAIL_STUDY_20261117_V1',
        'dilation_control_status':read(dilation/'CONTROL_STATUS.json') if (dilation/'CONTROL_STATUS.json').exists() else None,
        'scope':'Lightweight queue snapshot; no new efficacy or full recovery archive audit',
        'snapshot_entry':'tools.snapshot_correction_priority_20261117; frozen operations snapshot reads a locked byte on Windows; use this reader'}
    atomic_json(ROOT/'evidence/outputs/CORRECTIONNET_PRIORITY_STATUS_20261117.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
