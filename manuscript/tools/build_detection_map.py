"""Plot the complete frozen dense grid; no new inference or statistical test."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript/figures'
SKILL = Path('C:/Users/admin/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral'],'mathtext.fontset':'stix','font.size':7,'axes.titlesize':8,
    'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
source = ROOT / 'evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv'
df = pd.read_csv(source)
df = df[(df.suite == 'pd') & df.metric.isin(['pd','frame_pfa_jammer'])].copy()
assert len(df) == 620 and (df['count'] == 512).all()
assert not df.duplicated(['snr_db','jsr_db','metric']).any()
fig, axs = plt.subplots(2,3,figsize=(183/25.4,121/25.4))
fig.subplots_adjust(left=.075,right=.94,bottom=.12,top=.90,wspace=.49,hspace=.62)
for row, metric in enumerate(['pd','frame_pfa_jammer']):
    part = df[df.metric == metric]
    for col, key in enumerate(['paper2_mean','rsdr_mean','mean_difference']):
        ax = axs[row,col]
        p = part.pivot(index='jsr_db',columns='snr_db',values=key).sort_index().sort_index(axis=1)
        assert p.shape == (10,31) and p.notna().all().all()
        assert np.all(np.diff(p.columns) == 1) and np.all(np.diff(p.index) == 5)
        values = p.to_numpy()*100
        if col < 2:
            assert values.min() >= 0 and values.max() <= 100
            low, high, cmap = 0,100,'cividis'
        else:
            bound = max(5,5*np.ceil(np.abs(values).max()/5))
            low, high, cmap = -bound,bound,'RdBu_r'
        mesh = ax.pcolormesh(np.arange(-30.5,1,1),np.arange(-12.5,40,5),values,
            vmin=low,vmax=high,cmap=cmap,shading='flat',rasterized=False)
        label = ['CorrectionNet','RSDR-Net','CorrectionNet - RSDR-Net'][col]
        ax.set_title(f'({chr(97+row*3+col)}) {label}',loc='left',fontsize=7.5)
        ax.set(xlabel='SNR (dB)',ylabel='JSR (dB)',xticks=[-30,-20,-10,0],yticks=[-10,0,10,20,30])
        bar = fig.colorbar(mesh,ax=ax,fraction=.045,pad=.03)
        bar.ax.tick_params(labelsize=6)
        bar.set_label('Percentage points' if col==2 else '%',fontsize=6)
fig.text(.075,.967,'Detection probability',fontsize=9,weight='bold')
fig.text(.075,.515,'J+E frame false alarms',fontsize=9,weight='bold')
fig.text(.075,.025,'512 paired parents per condition; common nominal CFAR; no threshold calibration or significance mask.',fontsize=6.5)
stem = OUT / 'fig08_full_detection_grid'
require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True,
    axes=list(axs.flat),panel_ids=list('abcdef'),
    row_groups=[list('abc'),list('def')],column_groups=[list('ad'),list('be'),list('cf')])
fig.savefig(str(stem)+'.pdf')
fig.savefig(str(stem)+'.svg')
fig.savefig(str(stem)+'.png',dpi=600)
plt.close(fig)
df.to_csv(str(stem)+'.csv',index=False)
manifest={'source':source.relative_to(ROOT).as_posix(),
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'conditions':310,'independent_parents':512,'uncertainty':'Descriptive condition means and paired point differences; no significance claim',
    'source_transform':'Select dense grid, multiply probabilities by 100; no smoothing or interpolation',
    'color_scales':'Absolute panels share 0-100%; signed differences use separate symmetric limits per metric',
    'conclusion':'Detection and jammed-background false alarms must be inspected together across conditions'}
(OUT/'fig08_full_detection_grid.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(stem)
