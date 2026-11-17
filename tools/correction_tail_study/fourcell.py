"""Describe backbone-by-loss cells without treating DEV as confirmation."""
import numpy as np
from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json
from tools.unet_ablation.protocol import BASE as ABLATION_BASE
from paper2.evaluation_1103_v4 import summarize
from .settings import TAIL_RUN, CONTROL_RUN


def summarize_four_cells():
    from .train import validate_config, EVIDENCE, BASE, SEEDS
    config = validate_config()
    if BASE.name != TAIL_RUN:
        raise RuntimeError('Only the final original-backbone tail arm can form four cells')
    unet = ROOT/'outputs/unet_mf_tail_screen_20261117_v2'
    uc = read(unet/'CONFIG.json')
    if uc['dev_data_sha256'] != config['dev_data_sha256'] or uc['training_data_sha256'] != config['training_data_sha256']:
        raise RuntimeError('Four-cell dataset identity mismatch')
    for name, digest in uc['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('U-Net source drift before four-cell summary')
    rows = []
    evidence = {}
    for seed in SEEDS:
        u_path = unet/'tail'/f'seed_{seed}'/'dev_paired_epoch_256.json'
        c_path = BASE/'arm'/f'seed_{seed}'/'dev_paired_epoch_256.json'
        u, c = read(u_path), read(c_path)
        if u['config_sha256'] != sha(unet/'CONFIG.json') or c['config_sha256'] != sha(BASE/'CONFIG.json'):
            raise RuntimeError('Four-cell configuration binding mismatch')
        cps = [(ABLATION_BASE/'unet'/f'seed_{seed}'/'epoch_256.pt', u['baseline_checkpoint_sha256']),
               (unet/'tail'/f'seed_{seed}'/'epoch_256.pt', u['tail_checkpoint_sha256']),
               (ROOT/'outputs'/CONTROL_RUN/'arm'/f'seed_{seed}'/'epoch_256.pt', c['baseline_checkpoint_sha256']),
               (BASE/'arm'/f'seed_{seed}'/'epoch_256.pt', c['study_checkpoint_sha256'])]
        if any(sha(path) != digest for path,digest in cps):
            raise RuntimeError('Four-cell checkpoint identity mismatch')
        for path, row in ((u_path,u),(c_path,c)):
            if sha(path.with_suffix('.npz')) != row['parent_values_sha256']:
                raise RuntimeError('Four-cell parent-array binding mismatch')
            evidence[path.relative_to(ROOT).as_posix()] = sha(path)
        with np.load(u_path.with_suffix('.npz'),allow_pickle=False) as ua, np.load(c_path.with_suffix('.npz'),allow_pickle=False) as ca:
            if not np.array_equal(ua['parent_id'],ca['parent_id']):
                raise RuntimeError('Four-cell DEV parent roster mismatch')
            interaction = {k: summarize(ca[f'delta__{k}']-ua[f'delta__{k}']) for k in u['baseline']}
        rows.append({'seed': seed, 'unet_original': u['baseline'], 'unet_tail': u['tail'],
            'correction_original': c['baseline'], 'correction_tail': c['arm'],
            'loss_effect_unet': u['tail_minus_baseline'], 'loss_effect_correction': c['arm_minus_baseline'],
            'loss_effect_correction_minus_unet': interaction})
    result = {'stage': 'FOUR_CELL_DEV_DESCRIPTION', 'rows': rows, 'source_evidence_sha256': evidence,
        'dev_data_sha256': config['dev_data_sha256'], 'created_utc': now(),
        'limitation': 'Three training seeds and reused DEV parents; interaction is descriptive, not a significance test or independent confirmation. Backbone comparison includes each backbone preprocessing.'}
    atomic_json(BASE/'FOUR_CELL_DEV_SUMMARY.json',result)
    atomic_json(EVIDENCE/'FOUR_CELL_DEV_SUMMARY.json',result)
    return result
