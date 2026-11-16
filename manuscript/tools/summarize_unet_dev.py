"""Descriptive comparison at the same seeds/epoch/DEV identity; no inference."""
from pathlib import Path
import hashlib, json, statistics
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'manuscript'
keys=['u_nmse','damage','sjrif_db','pd','frame_pfa_jammer']
rows=[]; bindings=[]
for seed in [17,29,41]:
    files=[ROOT/f'evidence/outputs/paper2_256_epoch_8seed_20261114/seed_{seed}/dev_epoch_256_v4.json',
           ROOT/f'evidence/outputs/unet_ablation_3seed_256_20261115/unet/seed_{seed}/dev_epoch_256.json']
    a,b=[json.loads(p.read_text(encoding='utf-8')) for p in files]
    assert a['data_sha256']==b['data_sha256'] and a['parents']==b['parents']==256
    assert a['seed']==b['seed']==seed and a['epoch']==b['epoch']==256
    assert b['scope']=='DEV'
    for key in ['paper2/evaluation_1103_v4.py','paper2/radar_eval.py','paper2/data_1103.py']:
        assert a['evaluator_source_sha256'][key]==b['source_sha256'][key]
    for name,d,m in [('CorrectionNet',a,a['metrics']['model']),('U-Net',b,b['metrics'])]:
        rows.append({'model':name,'seed':seed,'epoch':256,**{k:m[k]['mean'] for k in keys}})
    for p,d in zip(files,[a,b]):
        bindings.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'data_sha256':d['data_sha256'],'checkpoint_sha256':d['checkpoint_sha256']})
stats={name:{k:{'mean':statistics.mean(r[k] for r in rows if r['model']==name),
                   'sample_sd':statistics.stdev(r[k] for r in rows if r['model']==name)} for k in keys}
       for name in ['CorrectionNet','U-Net']}
report={'scope':'DEV descriptive only; not independent confirmation; no significance test',
        'seeds':[17,29,41],'epoch':256,'parents':256,'rows':rows,'summary':stats,'source_bindings':bindings,
        'evaluator_check':'Same data SHA and shared v4 waveform/radar/data source SHA; different wrappers verified by source inspection',
        'architecture_control':'59714 active parameters each; U-Net changes blocks, widths, pooling, activation and dechirp together; not a single-factor ablation'}
(OUT/'UNET_DEV_3SEED_20261116.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(stats,indent=2))
