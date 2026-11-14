"""Incremental, explicitly partial convergence exports for the local queue."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper2.io import ROOT, now, read
from tools.durable_io import atomic_json


def refresh():
    run = ROOT / "outputs/paper2_256_epoch_8seed_20261114"
    out = ROOT / "evidence/outputs/paper2_256_epoch_8seed_20261114/progress"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    dev = []
    for seed in (17,29,41,53,67,79,97,109):
        by_epoch = {}
        if seed in (17,29,41):
            history = read(ROOT / f"outputs/paper2_1103_retry_20261114/seed_{seed}/history.json")
            for epoch in range(1,129):
                batch = [r for r in history if int(r['epoch']) == epoch]
                by_epoch[epoch] = dict(seed=seed, epoch=epoch, segment=0,
                    **{key:sum(r[key] for r in batch)/len(batch) for key in ('total','endpoint','pc','damage')})
        for path in sorted((run / f"seed_{seed}").glob("training_epoch_*_segment_*.json")):
            row = read(path)
            if 'total' not in row:
                continue
            epoch = int(row['epoch'])
            if epoch not in by_epoch or row['segment'] >= by_epoch[epoch]['segment']:
                by_epoch[epoch] = {key:row[key] for key in ('seed','epoch','segment','total','endpoint','pc','damage')}
        rows.extend(by_epoch[epoch] for epoch in sorted(by_epoch))
        for path in sorted((run / f"seed_{seed}").glob("dev_epoch_*_v4.json")):
            result = read(path)
            dev.append(dict(seed=seed, epoch=result['epoch'],
                checkpoint_sha256=result['checkpoint_sha256'], metrics=result['metrics']['model']))
    with (out / 'TRAINING_LOSS_PARTIAL.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','epoch','segment','total','endpoint','pc','damage'])
        writer.writeheader();writer.writerows(rows)
    fig, axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,key in zip(axes.flat,('total','endpoint','pc','damage')):
        for seed in (17,29,41,53,67,79,97,109):
            selected=[r for r in rows if r['seed']==seed]
            if selected:
                ax.plot([r['epoch'] for r in selected],[r[key] for r in selected],label=f'Seed {seed}')
        ax.set(xlabel='Epoch',ylabel=key,yscale='log',xlim=(1,256))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Partial training convergence: completed epochs only')
    fig.savefig(out/'TRAINING_CONVERGENCE_PARTIAL.png',dpi=170)
    plt.close(fig)
    atomic_json(out/'DEV_CHECKPOINTS_PARTIAL.json',dict(updated_utc=now(),scope='DEV_ONLY_PARTIAL',rows=dev))
    atomic_json(out/'STATUS.json',dict(updated_utc=now(),partial=True,training_epoch_rows=len(rows),
        dev_checkpoint_rows=len(dev),seed_max_epoch={str(seed):max(r['epoch'] for r in rows if r['seed']==seed)
        for seed in sorted({r['seed'] for r in rows})}))


if __name__=='__main__':
    refresh()
