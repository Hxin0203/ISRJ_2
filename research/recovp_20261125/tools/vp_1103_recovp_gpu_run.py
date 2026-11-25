"""Explicit user release: two seed17 models, frozen winner suite, conditional seeds."""
import os
import subprocess
import sys
import traceback
from pathlib import Path
from vp_1103_common import WORK,atomic_json,read,sha,now,provenance,validate_role
from tools.durable_io import ProcessLock

OUT=WORK/'evidence/outputs/recovp_formal_dev'
PRIVATE=WORK/'outputs/recovp_formal_dev'

def run():
    os.environ['ISRJ2_CORRECTION_ARM']='tail'
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with ProcessLock(PRIVATE/'CONTROLLER.lock'):
        check=read(OUT/'FP32_QUICKCHECK.json');assert check['status']=='PASS'
        for name,digest in check['source_sha256'].items():assert sha(WORK/name)==digest,name
        training,train_binding=validate_role('train');_,dev_binding=validate_role('dev')
        files=[Path(__file__),*[WORK/'tools'/f for f in ('vp_1103_recovp_gpu_train.py','vp_1103_recovp_fp32.py',
            'vp_1103_recovp_fp32_ablation.py','vp_1103_recovp_fp32_anchors.py')]]
        config={'coordinates':'RAW_IQ','extractor':{'method':'E0','window':'rect','rounds':1,'k':12},
            'epochs':256,'micro_batch':8,'seeds':[17,29,41],'train':train_binding,'dev':dev_binding,
            'arithmetic':'float32 real features/CNN; complex64 FFT; complex128 Gram and solve; float32 output',
            'source_sha256':provenance(*files)['source_sha256']}
        cpath=PRIVATE/'CONFIG.json'
        if cpath.exists():assert read(cpath)==config,'Immutable training config differs'
        else:atomic_json(cpath,config)
        atomic_json(OUT/'AUTHORIZATION.json',{'utc':now(),'explicit_user_release':True,
            'supersedes_performance_gate':'NO_GO_EXP1_PREFLIGHT',
            'prior_receipts_preserved':True,'independent_confirmation_unlocked':False,
            'base_and_noequiv_seed17_required':True,'suite_required_even_if_seed17_uncompetitive':True,
            'softshare_and_cvar':'NOT_AUTHORIZED_THIS_ROUND'})
        state={**provenance(*files),'stage':'RELEASED','started_utc':now(),'pid':os.getpid(),
            'micro_batch':8,'epochs':256,'coordinates':'RAW_IQ','independent_confirmation_unlocked':False}
        def update(**values):
            state.update(values,updated_utc=now());atomic_json(OUT/'STATUS.json',state)
        def command(script,*args):
            update(stage='EVALUATING',script=script)
            subprocess.run([sys.executable,'-B',str(WORK/'tools'/script),*map(str,args)],cwd=WORK,check=True)
        try:
            from vp_1103_recovp_gpu_train import train
            for arm in ('ReCoVP-Base','ReCoVP-NoEquiv'):
                train(17,'RAW_IQ',training,None,config,update,arm)
            command('vp_1103_recovp_formal_score.py')
            winner=read(OUT/'WINNER.json')
            if winner['extend_seeds']:
                for seed in (29,41):train(seed,'RAW_IQ',training,None,config,update,winner['arm'])
                command('vp_1103_recovp_formal_score.py')
            command('vp_1103_recovp_isrj1_suite.py')
            update(stage='COMPLETE_GATE_REVIEW',ended_utc=now(),winner=winner['arm'])
        except Exception as error:
            update(stage='FAILED_EXECUTION',ended_utc=now(),error=repr(error),traceback=traceback.format_exc())
            raise

if __name__=='__main__':run()
