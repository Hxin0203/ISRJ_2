"""Server engineering checks before a queued factor receives a training budget."""
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
from tools.unet_ablation.models import PlainUNet, objective
from .model import FactorUNet
from .settings import FACTOR, GPU_INDEX
from .runtime import require_server, configure_precision


def geometry_check():
    """Server CPU forward/backward only; no optimization or DEV access."""
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Engineering checks also run on server B only')
    configure_precision()
    torch.set_num_threads(2)
    seed_rng(17)
    control = PlainUNet()
    seed_rng(17)
    model = FactorUNet()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items())
    x = torch.randn(2, 2, 1800)
    y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    active = sum(p.numel() for p in model.parameters() if p.grad is not None)
    assert active == sum(p.numel() for p in model.parameters()) == 59714
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        scale_error = float((model(x * 10) / 10 - y).abs().max())
        z = torch.complex(x[:, 0], x[:, 1])
        zf = torch.fft.fftshift(torch.fft.fft(z, norm='ortho'), dim=-1)
        roundtrip = torch.fft.ifft(torch.fft.ifftshift(zf, dim=-1), norm='ortho')
        fft_error = float((roundtrip - z).abs().max())
        parseval = float((zf.abs().square().sum() / z.abs().square().sum() - 1).abs())
    assert scale_error < 2e-5 and fft_error < 2e-6 and parseval < 2e-6
    return {'factor': FACTOR, 'parameters_total_and_active': active,
            'same_initial_tensors': True, 'finite_forward_backward': True,
            'scale_max_absolute_error': scale_error, 'fft_roundtrip_max_absolute_error': fft_error,
            'parseval_relative_error': parseval, 'cpu_threads': 2,
            'scope': 'synthetic engineering check on server; no optimizer updates and no DEV access'}


def preflight(base):
    require_server()
    from .train import local_sources, publish, choose_resume, FORMAT, DATA, ABLATION_BASE
    sources = local_sources()
    path = base / 'NUMERICAL_PREFLIGHT.json'
    if path.exists():
        saved = read(path)
        if saved['stage'] != 'NUMERICAL_PREFLIGHT_PASSED' or saved['source_sha256'] != sources:
            raise RuntimeError('Frozen preflight source identity mismatch')
        return saved
    geometry = geometry_check()
    torch.set_num_threads(4)
    initialization = {}
    for seed in (17, 29, 41):
        seed_rng(seed)
        model = FactorUNet()
        initial_cp = ABLATION_BASE / 'unet' / f'seed_{seed}' / 'recovery/epoch_000.pt'
        digest = sha(initial_cp)
        assert digest == read(initial_cp.with_suffix('.json'))['checkpoint_sha256']
        original = torch.load(initial_cp, map_location='cpu', weights_only=False)
        assert all(torch.equal(v, original['model'][k]) for k, v in model.state_dict().items())
        initialization[str(seed)] = {'baseline_epoch0_sha256': digest, 'exact_match': True}
    model = model.cuda()
    optimizer = optimizer_for(model)
    arrays = load_arrays(DATA / 'train.npz')
    batch = make_batch(arrays, np.arange(2), 'cuda')

    def update(net, opt):
        opt.zero_grad(set_to_none=True)
        correction = net(batch['y']).reshape(-1, 4, 2, 1800)
        losses = objective(correction, batch['s'], batch['j'], batch['e'], 'unet')
        losses['total'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
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
        resumed = FactorUNet().cuda()
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
    result = {'stage': 'NUMERICAL_PREFLIGHT_PASSED', 'factor': FACTOR,
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
