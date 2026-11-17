"""Dated, lightweight pipeline status; no dataset/weight export."""
import json
import platform
import statistics
from pathlib import Path

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json


def main():
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Server B only')
    rows=[]
    for run, factor, gpu in (
        ('unet_mf_tail_screen_20261117_v2','tail',1),
        ('unet_noise_norm_screen_20261117_v1','norm',0),
        ('unet_dilation_screen_20261117_v1','factor',1),
        ('unet_frequency_screen_20261117_v1','factor',0)):
        base=ROOT/'outputs'/run
        status=read(base/'STATUS.json')
        cfg=read(base/'CONFIG.json') if (base/'CONFIG.json').exists() else None
        queue=read(base/'QUEUE_BINDING.json') if (base/'QUEUE_BINDING.json').exists() else None
        binding=cfg if cfg else queue
        drift=[n for n,h in binding['source_sha256'].items() if sha(ROOT/n)!=h]
        seeds=[]
        for seed in (17,29,41):
            folder=base/factor/f'seed_{seed}'
            receipts=sorted((folder/'recovery').glob('epoch_*.json'))
            last=read(receipts[-1]) if receipts else None
            checkpoints=sorted((folder/'recovery').glob('epoch_*.pt'))
            recent=checkpoints[-21:]
            periods=[(b.stat().st_mtime_ns-a.stat().st_mtime_ns)/1e9 for a,b in zip(recent,recent[1:])]
            periods=[v for v in periods if v>0]
            seeds.append({'seed':seed,'saved_epoch':last['epoch'] if last else None,
                'recovery_receipts':len(receipts),'latest_sha_matches':sha(receipts[-1].with_suffix('.pt'))==last['checkpoint_sha256'] if last else None,
                'complete':(folder/'TRAINING_COMPLETE.json').exists(),
                'recent_wall_seconds_per_epoch':statistics.median(periods) if periods else None})
        row={'run':run,'physical_gpu':gpu,'status':status,'seeds':seeds,'source_drift':drift,
             'numerical_preflight_passed':(
                 read(base/'NUMERICAL_PREFLIGHT.json').get('stage')=='NUMERICAL_PREFLIGHT_PASSED'
                 if (base/'NUMERICAL_PREFLIGHT.json').exists() else None),
             'preflight_note':('Tail run used a separate finite-gradient smoke check; absence of this later receipt format is not a failed check.'
                               if factor=='tail' else 'Pending until GPU release if no receipt exists.'),
             'queue_binding_sha256':sha(base/'QUEUE_BINDING.json') if queue else None,
             'config_sha256':sha(base/'CONFIG.json') if cfg else None,
             'dev_summary_exists':(base/'DEV_SCREEN_SUMMARY.json').exists()}
        periods=[s['recent_wall_seconds_per_epoch'] for s in seeds if s['recent_wall_seconds_per_epoch']]
        row['estimated_training_hours_remaining_current_speed']=(
            sum(256-(s['saved_epoch'] or 0) for s in seeds)*statistics.median(periods)/3600 if periods else None)
        row['eta_limitation']='Training only, recent wall-clock median; excludes DEV evaluation, interruptions and waiting. New factors have no speed measurement yet.'
        for name in ('STATUS.json','QUEUE_BINDING.json','NUMERICAL_PREFLIGHT.json','DEV_SCREEN_SUMMARY.json','PIPELINE_STATUS.json'):
            if (base/name).exists():
                atomic_json(ROOT/'evidence/outputs'/run/name,read(base/name))
        rows.append(row)
    diag=ROOT/'evidence/outputs/phase_scale_diagnostic_20261117_v1'
    diagnostic={'summary_exists':(diag/'SUMMARY.json').exists(),
                'status':read(diag/'STATUS.json') if (diag/'STATUS.json').exists() else None}
    if (diag/'SUMMARY.json').exists():
        diagnostic['summary_sha256']=sha(diag/'SUMMARY.json')
        diagnostic['authoritative_stage']=read(diag/'SUMMARY.json')['stage']
    result={'checked_utc':now(),'timezone_for_user':'America/Los_Angeles','host':platform.node(),
            'jobs':rows,'diagnostic':diagnostic,'scope':'Development pipeline; no new confirmation result'}
    target=ROOT/'evidence/outputs/IDEA_PIPELINE_STATUS_20261117.json'
    atomic_json(target,result)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
