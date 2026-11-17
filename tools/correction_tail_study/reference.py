"""Read-only reference validation; historical controls are descriptive only."""
from paper2.io import ROOT, read, sha
from tools.unet_ablation.protocol import ORIGINAL
from .settings import ARM, CONTROL_RUN


def reference_folder(seed):
    return ((ORIGINAL / f'seed_{seed}') if ARM == 'original' else
            (ROOT / 'outputs' / CONTROL_RUN / 'arm' / f'seed_{seed}'))


def reference_binding(seed, train_sha):
    folder = reference_folder(seed)
    path = folder / 'TRAINING_COMPLETE.json'
    complete = read(path)
    cp = folder / 'epoch_256.pt'
    digest = sha(cp)
    if complete['milestones']['256'] != digest or read(cp.with_suffix('.json'))['checkpoint_sha256'] != digest:
        raise RuntimeError(f'Reference checkpoint SHA mismatch: {cp}')
    identity = complete['identity']
    data_key = 'dataset_sha256' if ARM == 'original' else 'data_sha256'
    if identity[data_key] != train_sha:
        raise RuntimeError('Reference training data differs')
    for name, expected in identity['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Reference source drift: {name}')
    if ARM == 'tail':
        cfg_path = ROOT / 'outputs' / CONTROL_RUN / 'CONFIG.json'
        if identity['run'] != CONTROL_RUN or sha(cfg_path) != identity['config_sha256']:
            raise RuntimeError('Fresh control identity changed')
    return {'checkpoint_sha256': digest, 'complete_sha256': sha(path),
            'folder': folder.relative_to(ROOT).as_posix(),
            'comparison_scope': 'historical descriptive only' if ARM == 'original' else 'same-protocol paired loss contrast'}
