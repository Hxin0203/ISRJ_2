"""Build paper figures from audited aggregate CSVs; does not run inference/training."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 8,
                     'legend.fontsize': 7, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.grid': True, 'grid.alpha': .2})
sources = [
    ROOT / 'evidence/outputs/TEST_CONDITION_TABLE_1103.csv',
    ROOT / 'evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv',
    ROOT / 'evidence/outputs/RSDR_UNIFIED_20261115/PAIRED_RESULTS.json',
    ROOT / 'evidence/outputs/CONVERGENCE_256_8SEED_20261114/TRAINING_LOSS_BY_EPOCH.csv',
    ROOT / 'evidence/outputs/CONVERGENCE_256_8SEED_20261114/DEV_METRICS_BY_CHECKPOINT.csv',
]

def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=str(OUT / f'{name}.alignment.json'),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    # Retain the declared physical page size; constrained layout reserves text space.
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.svg')
    fig.savefig(OUT / f'{name}.png', dpi=600)
    fig.savefig(OUT / f'{name}.review.png', dpi=300)
    plt.close(fig)

def architecture():
    fig, ax = plt.subplots(figsize=(13, 6.1))
    ax.set(xlim=(0, 13), ylim=(0, 6.1)); ax.axis('off')
    def box(x, y, w, h, text, color='#edf4fa', size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',
                     facecolor=color,edgecolor='#486276',linewidth=1.1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(a,b,**kw):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,
                                   linewidth=1.2,color='#486276',**kw))
    box(.2,4.5,1.15,.65,'Input y\n2 × 1800')
    box(1.7,4.5,1.5,.65,'RMS scale\ny / a(y)')
    box(3.55,4.5,1.7,.65,'Fixed dechirp\nD = diag(q*)')
    box(5.65,4.5,1.8,.65,'Residual U-Net\n16 / 32 / 64')
    box(7.85,4.5,1.75,.65,'Inverse dechirp\n× a(y)')
    box(10.05,4.5,1.25,.65,'Correction\nC(y)')
    box(11.7,4.5,1.05,.65,'Output\ny − C(y)', '#e7f1e9', 10)
    for a,b in [(1.35,1.7),(3.2,3.55),(5.25,5.65),(7.45,7.85),(9.6,10.05),(11.3,11.7)]:
        arrow((a,4.825),(b,4.825))
    ax.plot([.78,.78,12.2,12.2],[5.2,5.65,5.65,5.2],color='#486276',lw=1.1)
    ax.text(6.5,5.78,'Identity path',ha='center',fontsize=10)
    box(.4,2.7,2.5,.9,'Shared components\nS, J, E', '#fff4df', 10)
    box(3.55,2.55,4.0,1.2,'Four paired observations\nE  |  S + E  |  J + E  |  S + J + E\nCorrection labels: 0  |  0  |  J  |  J', '#fff4df', 10)
    box(8.25,2.55,4.2,1.2,'Shared single-pulse network\nEndpoint + 0.1 × pulse-compression loss\n+ target-increment damage loss', '#fff4df', 10)
    arrow((2.95,3.15),(3.5,3.15)); arrow((7.6,3.15),(8.2,3.15))
    ax.text(6.5,2.15,'Training only: component labels and paired states are unavailable at inference.',ha='center',fontsize=10)
    box(.5,.55,12,1.0,'U-Net: 2×1800 → 16×1800 → 32×900 → 64×450 → 32×900 → 16×1800 → 2×1800\nLinear interpolation + skip concatenation; residual Conv–GroupNorm–GELU blocks\n59,714 active parameters; pulses processed independently', '#f4f4f4', 11)
    save(fig,'fig01_architecture')

def convergence():
    train=pd.read_csv(sources[3]); dev=pd.read_csv(sources[4])
    assert len(train)==2048 and len(dev)==64
    fig,axs=plt.subplots(2,3,figsize=(183/25.4,125/25.4))
    fig.subplots_adjust(left=.075,right=.98,bottom=.12,top=.88,wspace=.39,hspace=.65)
    specs=[('total','Training objective',train,False),
           ('model__u_nmse__mean','DEV waveform NMSE',dev,False),
           ('model__damage__mean','DEV target-increment distortion',dev,False),
           ('model__sjrif_db__mean','DEV SJRIF (dB)',dev,False),
           ('model__pd__mean','DEV detection (%)',dev,True),
           ('model__frame_pfa_jammer__mean','DEV J+E frame alarms (%)',dev,True)]
    for i,(ax,(k,title,df,pct)) in enumerate(zip(axs.flat,specs)):
        p=df.pivot(index='epoch',columns='seed',values=k)*(100 if pct else 1)
        mean,sd=p.mean(axis=1),p.std(axis=1,ddof=1)
        ax.fill_between(p.index,mean-sd,mean+sd,color='#4477aa',alpha=.18)
        ax.plot(p.index,mean,color='#225588',lw=1.1)
        ax.axvline(128,color='.6',ls='--',lw=.9)
        ax.set(title=f'({chr(97+i)}) {title}',xlabel='Epoch',xlim=(1,256),xticks=[32,64,96,128,160,192,224,256])
        ax.tick_params(axis='x',labelsize=6)
        if i==0:
            assert (p.to_numpy()>0).all()
            ax.set_yscale('log')
    fig.suptitle('Eight seeds · Mean ± sample SD · Fixed development set',fontsize=7)
    save(fig,'fig02_convergence')

def main_comparison():
    df=pd.read_csv(sources[0]); d=df[(df.suite=='main')&(df.jam_type=='repeat')&(df.jsr_db==20)]
    methods=[('keep','Input','#999999'),('max_tf','Max-TF','#ee7733'),
             ('gmm_omp','GMM-OMP','#cc6677'),('bcs_oracle','BCS (oracle support)','#228833'),
             ('model','CorrectionNet (128)','#0077bb'),('reference','S+E reference','#000000')]
    assert len(d)==84, 'Fourteen SNR values and six methods; all selected rows retained.'
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,124/25.4),layout='constrained')
    for ax,k,title,pct in zip(axs.flat,['pd','frame_pfa_jammer','sjrif_db','damage'],
         ['(a) Detection probability (%)','(b) J+E frame false alarms (%)','(c) SJRIF (dB)','(d) Target-increment damage'],[True,True,False,False]):
        for method,label,color in methods:
            s=d[d.method==method].sort_values('snr_db')
            ax.plot(s.snr_db,s[k]*(100 if pct else 1),label=label,color=color,ls='--' if method in ('keep','reference') else '-',marker='o',markersize=2,lw=1.0)
        ax.set(title=title,xlabel='SNR (dB)',xlim=(-30,10))
        if pct: ax.set_ylim(0,100)
        if k=='damage':
            ax.set_yscale('symlog',linthresh=.01)
            ax.set_ylim(bottom=0)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=7)
    save(fig,'fig03_repeat_jsr20')

def old_new():
    df=pd.read_csv(sources[1]);d=df[(df.suite=='pd')&(df.jsr_db==20)]
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,116/25.4),layout='constrained')
    for ax,k,title,pct in zip(axs.flat,['pd','frame_pfa_jammer','u_nmse','damage'],
          ['(a) Detection probability (%)','(b) J+E frame false alarms (%)','(c) Waveform NMSE','(d) Target-increment damage'],[True,True,False,False]):
        s=d[d.metric==k].sort_values('snr_db')
        assert len(s)==31
        for col,label,color in [('paper2_mean','CorrectionNet (128)','#0077bb'),('rsdr_mean','RSDR-Net (frozen)','#bb5566')]:
            ax.plot(s.snr_db,s[col]*(100 if pct else 1),label=label,color=color,lw=1.1)
        ax.set(title=title,xlabel='SNR (dB)',xlim=(-30,0))
        if pct: ax.set_ylim(0,100)
        else: ax.set_ylim(bottom=0)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    save(fig,'fig04_dense_jsr20')

def width():
    df=pd.read_csv(sources[1]);d=df[(df.suite=='sliceB')&(df.snr_db==-20)&(df.jsr_db==35)]
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,116/25.4),layout='constrained')
    for col,m in enumerate((2,4)):
        for row,k in enumerate(('pd','frame_pfa_jammer')):
            ax=axs[row,col];s=d[(d.metric==k)&(d.repeats==m)].sort_values('width')
            assert len(s)==5
            for key,label,color in [('paper2_mean','CorrectionNet (128)','#0077bb'),('rsdr_mean','RSDR-Net (frozen)','#bb5566')]:
                ax.plot(s.width,s[key]*100,label=label,color=color,marker='o',markersize=3,lw=1.1)
            ax.set(title=f'({chr(97+row*2+col)}) M = {m}',xlabel='Sampling width (samples)',ylabel='Detection probability (%)' if row==0 else 'J+E frame false alarms (%)',xticks=[6,18,36,54,72],ylim=(0,100))
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    save(fig,'fig07_width_snr_minus20_jsre35')

if __name__=='__main__':
    # The manuscript uses the native editable Visio diagram; the legacy schematic
    # is retained only for explicitly requested historical reproduction.
    if '--legacy-architecture' in sys.argv:
        architecture()
    convergence();main_comparison();old_new();width()
    for source_name,dest in [('fixed_parent0_coherent_001.png','fig05_fixed_parent_pc_doppler.png'),
                             ('fixed_parent0_rd_001.png','fig06_fixed_parent_rd.png')]:
        p=ROOT/'evidence/outputs/RSDR_UNIFIED_20261115/figures'/source_name
        sources.append(p);shutil.copy2(p,OUT/dest)
    from index_and_audit_delivery import make_manifest
    manifest=make_manifest()
    print('Created',len(manifest['figures']),'figure files; source SHA recorded.')
