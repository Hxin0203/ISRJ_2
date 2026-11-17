"""Server engineering checks before a queued arm receives a training budget."""
from __future__ import annotations

import hashlib
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch

from paper2.io import ROOT, now, read, sha
from paper2.train_1103 import load_arrays, make_batch
from paper2.train_256_8seed import optimizer_for, seed_rng, rng_snapshot, restore_rng
from tools.durable_io import atomic_json
from .objective import study_objective
from .reference import reference_binding, reference_folder
from paper2.network import CorrectionNet as StudyModel
from .settings import ARM, GPU_INDEX
from .runtime import require_server, configure_precision


def geometry_check():
    """Server CPU forward/backward only; no optimization or DEV access."""
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Engineering checks also run on server B only')
    configure_precision()
    torch.set_num_threads(2)
    seed_rng(17)
    control = StudyModel()
    seed_rng(17)
    model = StudyModel()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items())
    x = torch.randn(2, 2, 1800)
    y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    active = sum(p.numel() for p in model.parameters() if p.grad is not None)
    total = sum(p.numel() for p in model.parameters())
    assert active == 59714 and total == 112756
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        scale_error = float((model(x * 10) / 10 - y).abs().max())
    assert scale_error < 2e-5
    synthetic = {name: torch.randn(8, 2, 1800) for name in ('s', 'j', 'e')}
    synthetic_c = model(torch.randn(32, 2, 1800)).reshape(8, 4, 2, 1800)
    terms = study_objective(synthetic_c, synthetic, 1)
    assert all(torch.isfinite(value).all() for value in terms.values())
    expected = terms['endpoint'] + .1 * terms['pc'] + terms['damage']
    if ARM == 'tail':
        expected = expected + .05 * terms['tail']
    assert torch.equal(terms['total'], expected)
    model.zero_grad(set_to_none=True)
    terms['total'].backward()
    assert sum(p.numel() for p in model.parameters() if p.grad is not None) == 59714
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return {'arm': ARM, 'parameters_total': total, 'parameters_active': active, 'parameters_inactive': total-active,
            'same_initial_tensors': True, 'finite_forward_backward': True,
            'finite_synthetic_study_objective_gradient': True,
            'tail_weight': .05 if ARM == 'tail' else 0.0,
            'scale_max_absolute_error': scale_error, 'cpu_threads': 2,
            'scope': 'synthetic engineering check on server; no optimizer updates and no DEV access'}


def preflight(base):
    require_server()
    from .train import local_sources, publish, choose_resume, FORMAT, DATA
    sources = local_sources()
    path = base / 'NUMERICAL_PREFLIGHT.json'
    if path.exists():
        saved = read(path)
        if saved['stage'] != 'NUMERICAL_PREFLIGHT_PASSED' or saved['source_sha256'] != sources:
            raise RuntimeError('Frozen preflight source identity mismatch')
        return saved
    geometry = geometry_check()
    torch.set_num_threads(4)
    data_hashes = {}
    for role in ('train', 'dev'):
        data_hashes[role] = sha(DATA / f'{role}.npz')
        if read(DATA / f'{role}.receipt.json')['sha256'] != data_hashes[role]:
            raise RuntimeError('Preflight data receipt mismatch')
    initialization = {}
    for seed in (17, 29, 41):
        seed_rng(seed)
        model = StudyModel()
        reference_binding(seed, data_hashes['train'])
        if ARM == 'tail':
            initial_cp = reference_folder(seed) / 'recovery/epoch_000.pt'
            digest = sha(initial_cp)
            assert digest == read(initial_cp.with_suffix('.json'))['checkpoint_sha256']
            original = torch.load(initial_cp, map_location='cpu', weights_only=False)
            assert original['seed'] == seed and original['epoch'] == 0
            assert original['identity']['source_sha256'] == sources
            assert all(torch.equal(v, original['model'][k]) for k, v in model.state_dict().items())
            initialization[str(seed)] = {'fresh_control_epoch0_sha256': digest, 'exact_match': True}
        else:
            seed_rng(seed)
            repeat = StudyModel()
            assert all(torch.equal(v, repeat.state_dict()[k]) for k,v in model.state_dict().items())
            initialization[str(seed)] = {'repeat_constructor_exact': True,
                'historical_initialization_equivalence_claimed': False}
    model = model.cuda()
    optimizer = optimizer_for(model)
    arrays = load_arrays(DATA / 'train.npz')
    batch = make_batch(arrays, np.arange(2), 'cuda')

    def update(net, opt):
        opt.zero_grad(set_to_none=True)
        correction = net(batch['y']).reshape(-1, 4, 2, 1800)
        losses = study_objective(correction, batch, 2)
        assert all(torch.isfinite(v).all() for v in losses.values())
        losses['total'].backward()
        assert sum(p.numel() for p in net.parameters() if p.grad is not None) == 59714
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
        opt.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    first = update(model, optimizer)
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='preflight_', dir=base) as scratch:
        folder = Path(scratch)
        identity = {'scope': 'ENGINEERING_PREFLIGHT_ONLY', 'source_sha256': sources}
        payload = {'format': FORMAT, 'identity': identity, 'seed': 41, 'epoch': 1,
                   'global_step': 48, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'rng': rng_snapshot(), 'epoch_record': first}
        digest = publish(folder, payload)
        saved = choose_resume(folder, identity)
        assert saved is not None and digest == sha(folder / 'recovery/epoch_001.pt')
        resumed = StudyModel().cuda()
        resumed.load_state_dict(saved['model'], strict=True)
        assert all(torch.equal(v.cpu(), saved['model'][k].cpu()) for k, v in resumed.state_dict().items())
        opt_resumed = optimizer_for(resumed)
        opt_resumed.load_state_dict(saved['optimizer'])
        restored_opt = opt_resumed.state_dict()
        assert restored_opt['param_groups'] == saved['optimizer']['param_groups']
        for parameter, state in saved['optimizer']['state'].items():
            for key, value in state.items():
                other = restored_opt['state'][parameter][key]
                assert torch.equal(value.cpu(), other.cpu()) if torch.is_tensor(value) else value == other
        restore_rng(saved['rng'])
        expected = torch.rand(4, device='cuda')
        second = update(model, optimizer)
        restore_rng(saved['rng'])
        assert torch.equal(expected, torch.rand(4, device='cuda'))
        second_resumed = update(resumed, opt_resumed)
        error = max(float((p - resumed.state_dict()[k]).abs().max()) for k, p in model.state_dict().items())
        loss_error = max(abs(second[k] - second_resumed[k]) for k in second)
        assert error <= 1e-6 and loss_error <= 1e-6, (error, loss_error)
    result = {'stage': 'NUMERICAL_PREFLIGHT_PASSED', 'arm': ARM,
              'source_sha256': sources, 'geometry': geometry, 'baseline_initialization': initialization,
              'recovery_model_max_absolute_error': error, 'recovery_next_loss_max_absolute_error': loss_error,
              'recovery_loaded_model_and_optimizer_exact': True, 'recovery_rng_exact': True,
              'recovery_numeric_tolerance_absolute': 1e-6, 'physical_gpu': int(GPU_INDEX),
              'torch': torch.__version__, 'created_utc': now(),
              'scope': 'engineering checks only; no performance evaluation or DEV tuning'}
    atomic_json(path, result)
    atomic_json(ROOT / 'evidence/outputs' / base.name / path.name, result)
    del model, resumed, optimizer, opt_resumed, batch, arrays
    torch.cuda.empty_cache()
    return result
