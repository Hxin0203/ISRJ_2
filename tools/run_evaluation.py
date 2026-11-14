"""Supplemental provenance guard. Does not alter frozen training code."""
import argparse
from pathlib import Path
import numpy as np
import torch
from paper2.io import ROOT,read,write,sha,source_bindings,now


def validate_record(record,config_sha,source,steps):
    for label,actual,expected in [('config',record['config_sha256'],config_sha),
                                  ('source',record['source'],source),('steps',record['steps'],steps)]:
        if actual!=expected:
            raise RuntimeError(f'training {label} identity mismatch')


def bind_data():
    path=ROOT/'RUN_DATA_BINDING.json'
    roles={}
    for role in ('engineering','train','dev'):
        data=ROOT/'data'/f'{role}.npz'
        receipt=ROOT/'data'/f'{role}.receipt.json'
        saved=read(receipt)
        assert saved['sha256']==sha(data)
        roles[role]=dict(data_sha256=sha(data),receipt_sha256=sha(receipt),
                         simulator_manifest_sha256=sha(ROOT/'data'/f'{role}.manifest.json'),
                         role_created_utc=saved['created_utc'],bytes=data.stat().st_size)
    binding=dict(roles=roles,config_sha256=sha(ROOT/'PILOT_FROZEN_CONFIG.json'),source=source_bindings())
    if path.exists():
        old=read(path)
        assert old['binding']==binding,'supplemental data identity drift'
        return old
    old=dict(bound_utc=now(),binding=binding,
             limitation='Supplemental binding added during training, not an original pre-training receipt. Original dataset SHA receipts existed before all scientific runs. No rerun performed.')
    write(path,old)
    return old


def verify_training():
    cfg=read(ROOT/'PILOT_FROZEN_CONFIG.json')
    config_sha=sha(ROOT/'PILOT_FROZEN_CONFIG.json')
    source=source_bindings()
    completion=read(ROOT/'outputs'/'TRAINING_COMPLETE.json')
    assert completion['config_sha256']==config_sha and completion['source']==source
    runs=[]
    for seed in cfg['seeds']:
        shared=[]
        for method in cfg['methods']:
            folder=ROOT/'outputs'/f'{method}_seed{seed}'
            record=read(folder/'COMPLETE.json')
            validate_record(record,config_sha,source,cfg['steps'])
            assert sha(folder/'model.pt')==record['checkpoint_sha256']
            payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            assert payload['source']==source and payload['config']==cfg
            assert payload['method']==method and payload['seed']==seed
            shared.append(tuple(record[k] for k in ('initial_parameters_sha256','batch_sequence_sha256','exposure_state_observations')))
            runs.append({k:record[k] for k in ('tag','steps','elapsed_seconds','checkpoint_sha256','exposure_state_observations')})
        assert len(set(shared))==1,'per-seed initialization/batch/exposure inequivalence'
    return runs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bind-only',action='store_true')
    args=parser.parse_args()
    binding=bind_data()
    if args.bind_only:
        print('Supplemental dataset binding recorded/verified.',flush=True)
        return
    runs=verify_training()
    before=ROOT/'evidence'/'BEFORE_DEV.json'
    guard=dict(binding_sha256=sha(ROOT/'RUN_DATA_BINDING.json'),source=source_bindings(),
               evaluator_guard_sha256=sha(Path(__file__)),runs=runs,numpy=np.__version__,torch=torch.__version__)
    if before.exists():
        assert read(before)['guard']==guard,'evaluation implementation or training identity changed'
    else:
        assert not list((ROOT/'outputs').glob('*/dev_predictions.npz')),'unbound previous inference exists'
        write(before,dict(created_utc=now(),guard=guard))
    from paper2.evaluate import evaluate
    evaluate()
    assert bind_data()==binding
    assert verify_training()==runs
    write(ROOT/'evidence'/'EVALUATION_AUDIT.json',dict(completed_utc=now(),all_checks_pass=True,
          config_source_dataset_checkpoint_binding=True,same_seed_initialization_batches_exposure=True,
          dev_parent_count=192,test_opened=False,results_sha256=sha(ROOT/'evidence'/'PILOT_RESULTS.json'),
          prediction_receipts={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'outputs').glob('*/DEV_PREDICTIONS.json')}))
    print('Evaluation and all supplemental provenance checks completed.',flush=True)


if __name__=='__main__':
    main()
