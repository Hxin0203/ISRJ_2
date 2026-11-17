"""Server CPU engineering evidence, independent of frozen training sources."""
import importlib
import os
import platform
from pathlib import Path

from paper2.io import ROOT, now, sha
from tools.durable_io import atomic_json


def main():
    if platform.node().casefold() != 'desktop-rfferm3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('Server B only; no workstation numerical execution')
    import psutil
    if psutil.cpu_percent(interval=1) >= 85 or psutil.virtual_memory().available < 8*2**30:
        raise RuntimeError('Preserve active GPU jobs: insufficient CPU resources')
    rows = []
    for factor in ('dilation', 'frequency'):
        os.environ['ISRJ2_FACTOR'] = factor
        from tools.unet_factor_screen import settings, model, runtime, preflight
        for module in (settings, model, runtime, preflight):
            importlib.reload(module)
        row = preflight.geometry_check()
        row['model_source_sha256'] = sha(ROOT/'tools/unet_factor_screen/model.py')
        rows.append(row)
    out = {
        'stage': 'CPU_GEOMETRY_CHECKS_PASSED', 'created_utc': now(),
        'host': platform.node(), 'rows': rows,
        'validator_sha256': sha(Path(__file__)),
        'limitations': 'Synthetic server CPU forward/backward only; no optimizer steps, GPU preflight, DEV evaluation or efficacy claim.'}
    atomic_json(ROOT/'evidence/outputs/IDEA_PIPELINE_ENGINEERING_CHECK_20261117.json', out)
    print(out)


if __name__ == '__main__':
    main()
