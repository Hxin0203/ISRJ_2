"""Supplemental capacity control with bounded GPU memory and durable resumption."""
import gc
import os
import subprocess
import sys
import time
import traceback
from vp_1103_common import WORK,atomic_json,read,sha,provenance,validate_role,now
from tools.durable_io import ProcessLock

OUT=WORK/'evidence/outputs/unet_matched58943';PRIVATE=WORK/'outputs/unet_matched58943'

def gpu_sample():
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    return {'free_mib':int(row[0]),'utilization_percent':int(row[1])}

def run():
    os.environ['ISRJ2_CORRECTION_ARM']='tail';PRIVATE.mkdir(parents=True,exist_ok=True)
    with ProcessLock(PRIVATE/'CONTROLLER.lock'):
        from vp_1103_unet_matched_preflight import run as preflight
        check=preflight();assert check['status']=='PASS'
        training,tb=validate_role('train');_,db=validate_role('dev')
        files=[WORK/'tools'/n for n in ('vp_1103_unet_matched_model.py','vp_1103_unet_matched_train.py',
            'vp_1103_unet_matched_run.py','vp_1103_unet_matched_score.py','vp_1103_unet_matched_preflight.py',
            'vp_1103_recovp_formal_score.py','vp_1103_recovp_formal_stats.py','vp_1103_recovp_radar_metrics.py')]
        config={'run':'unet_matched58943','model':'UNet-Matched58943','parameters':58943,'widths':[21,40,74],
            'train':tb,'dev':db,'epochs':256,'micro_batch':8,'seeds':[17,29,41],'coordinates':'RAW_IQ',
            'loss':'Same endpoint + 0.1 PC + damage + 0.05 tail','control':'ReCoVP-NoEquiv same three seeds, fixed epoch256',
            'source_sha256':provenance(*files)['source_sha256'],'checkpoint_selection':'Final epoch only; no seed pruning'}
        path=PRIVATE/'CONFIG.json'
        if path.exists():assert read(path)==config,'Frozen configuration changed'
        else:atomic_json(path,config)
        state={**provenance(*files),'stage':'PREFLIGHT_PASS','started_utc':now(),'pid':os.getpid(),
            'parameters':58943,'seeds':[17,29,41],'micro_batch':8,'epochs':256,'confirmation_unlocked':False}
        def update(**fields):state.update(fields,updated_utc=now());atomic_json(OUT/'STATUS.json',state)
        update()
        def ready():
            root=WORK/'outputs/recovp_formal_dev'
            return (root/'ReCoVP-NoEquiv/41/TRAINING_COMPLETE.json').exists() and all(
                (root/'scoring/ReCoVP-NoEquiv'/str(seed)/'SCORE.json').exists() for seed in (17,29,41))
        stable=0
        while True:
            if (PRIVATE/'STOP').exists():update(stage='STOPPED_BY_USER',ended_utc=now());return
            sample=gpu_sample();deps=ready()
            stable=stable+1 if deps and sample['free_mib']>=4096 and sample['utilization_percent']<70 else 0
            update(stage='WAITING_SAFE_GPU_SLOT',original_three_seed_scores_ready=deps,gpu_sample=sample,
                admission_rule='Original NoEquiv training/scoring complete; >=4GiB free and <70% utilization twice. Never stop original controller.')
            if stable>=2:break
            time.sleep(10)
        import torch
        if not torch.cuda.is_available():raise RuntimeError('No CPU training fallback')
        # Bound only this process. Original GPU context and queue remain untouched.
        torch.cuda.set_per_process_memory_fraction(.25)
        from vp_1103_unet_matched_train import train
        try:
            for seed in (17,29,41):
                try:
                    train(seed,'RAW_IQ',training,None,config,update)
                except torch.cuda.OutOfMemoryError as error:
                    attempt={'utc':now(),'seed':seed,'error':repr(error),'retained_last_committed_epoch':state.get('last_committed_epoch',0),
                        'micro_batch_unchanged':8,'policy':'Preserve attempt; retry same seed/RNG checkpoint only after original suite finishes'}
                    atomic_json(OUT/f'GPU_MEMORY_ATTEMPT_{seed}.json',attempt)
                    gc.collect();torch.cuda.empty_cache()
                    while read(WORK/'evidence/outputs/recovp_formal_dev/STATUS.json')['stage']!='COMPLETE_GATE_REVIEW':
                        update(stage='WAITING_ORIGINAL_SUITE_AFTER_MEMORY_ATTEMPT',seed=seed);time.sleep(30)
                    torch.cuda.set_per_process_memory_fraction(.5)
                    train(seed,'RAW_IQ',training,None,config,update)
            update(stage='EVALUATING_THREE_SEEDS')
            from vp_1103_unet_matched_score import run as score
            score();update(stage='COMPLETE_DEV_COMPARISON',ended_utc=now(),summary_sha256=sha(OUT/'SUMMARY.json'))
        except Exception as error:
            update(stage='FAILED_EXECUTION',ended_utc=now(),error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':run()
