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
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
                     'mathtext.fontset': 'stix', 'font.size': 7,
                     'axes.spines.top': True, 'axes.spines.right': True,
                     'axes.linewidth': .65, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.grid': True, 'grid.alpha': .28,
                     'grid.color': '#b7c0c7', 'grid.linewidth': .38,
                     'xtick.major.width': .6, 'ytick.major.width': .6})
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
    fig,axs=plt.subplots(2,3,figsize=(183/25.4,116/25.4))
    fig.subplots_adjust(left=.075,right=.99,bottom=.17,top=.88,wspace=.34,hspace=.48)
    specs=[('total','Training objective',train,False),
           ('model__u_nmse__mean','DEV waveform NMSE',dev,False),
           ('model__damage__mean','DEV target-increment distortion',dev,False),
           ('model__sjrif_db__mean','DEV SJRIF (dB)',dev,False),
           ('model__pd__mean','DEV detection probability (%)',dev,True),
           ('model__frame_pfa_jammer__mean','DEV J+E frame alarms (%)',dev,True)]
    palette=['#4477aa','#228833','#cc6677','#eeaa33','#66ccee','#aa4499']
    markers=['o','s','^','D','P','X']
    for i,(ax,(k,title,df,pct)) in enumerate(zip(axs.flat,specs)):
        p=df.pivot(index='epoch',columns='seed',values=k)*(100 if pct else 1)
        mean,sd=p.mean(axis=1),p.std(axis=1,ddof=1)
        ax.fill_between(p.index,mean-sd,mean+sd,color=palette[i],alpha=.14,linewidth=0)
        ax.plot(p.index,mean,color=palette[i],lw=.95,ls='--',marker=markers[i],
                markevery=2,markersize=2.0,mec=palette[i],mfc='white',mew=.55)
        ax.axvline(128,color='.55',ls='--',lw=.7)
        ax.set(title=f'({chr(97+i)}) {title}',xlabel='Epoch',xlim=(1,256),xticks=[32,96,160,224,256])
        ax.tick_params(axis='x',labelsize=6)
        ax.grid(True,alpha=.22,linewidth=.35)
        if i==0:
            assert (p.to_numpy()>0).all()
            ax.set_yscale('log')
    fig.suptitle('Eight-seed convergence on the fixed development set',fontsize=7.5,y=.96)
    fig.text(.5,.035,'Lines: seed mean; bands: ±1 sample SD (n = 8); vertical line: epoch 128.',ha='center',fontsize=7)
    save(fig,'fig02_convergence')

def main_comparison():
    df=pd.read_csv(sources[0]); d=df[(df.suite=='main')&(df.jam_type=='repeat')&(df.jsr_db==20)]
    methods=[('keep','Input','#999999'),('max_tf','Max-TF','#ee7733'),
             ('gmm_omp','GMM-OMP','#cc6677'),('bcs_oracle','BCS (oracle support)','#228833'),
             ('model','CorrectionNet (128)','#0077bb'),('reference','S+E reference','#000000')]
    assert len(d)==84, 'Fourteen SNR values and six methods; all selected rows retained.'
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,126/25.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.88,wspace=.32,hspace=.52)
    for ax,k,title,pct in zip(axs.flat,['pd','frame_pfa_jammer','sjrif_db','damage'],
         ['(a) Detection probability (%)','(b) J+E frame false alarms (%)','(c) SJRIF (dB)','(d) Target-increment damage'],[True,True,False,False]):
        for mi,(method,label,color) in enumerate(methods):
            s=d[d.method==method].sort_values('snr_db')
            ax.plot(s.snr_db,s[k]*(100 if pct else 1),label=label,color=color,ls=['-', '--', '-.', ':', '-', '--'][mi],marker=['x','s','^','D','o','+'][mi],markevery=2,markersize=2.6,lw=.9,mfc='white')
        ax.set(title=title,xlabel='SNR (dB)',xlim=(-30,10))
        if pct: ax.set_ylim(0,100)
        if k=='damage':
            ax.set_yscale('symlog',linthresh=.01)
            ax.set_ylim(bottom=0)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,handlelength=2.0,columnspacing=1.2)
    save(fig,'fig03_repeat_jsr20')

def old_new():
    df=pd.read_csv(sources[1]);d=df[(df.suite=='pd')&(df.jsr_db==20)]
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,116/25.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.88,wspace=.32,hspace=.52)
    for ax,k,title,pct in zip(axs.flat,['pd','frame_pfa_jammer','u_nmse','damage'],
          ['(a) Detection probability (%)','(b) J+E frame false alarms (%)','(c) Waveform NMSE','(d) Target-increment damage'],[True,True,False,False]):
        s=d[d.metric==k].sort_values('snr_db')
        assert len(s)==31
        for col,label,color in [('paper2_mean','CorrectionNet (128)','#0077bb'),('rsdr_mean','RSDR-Net (frozen)','#bb5566')]:
            ax.plot(s.snr_db,s[col]*(100 if pct else 1),label=label,color=color,lw=1.1,ls='-' if col=='paper2_mean' else '--',marker='o' if col=='paper2_mean' else 's',markevery=3,markersize=2.5,mfc='white')
        ax.set(title=title,xlabel='SNR (dB)',xlim=(-30,0))
        if pct: ax.set_ylim(0,100)
        else: ax.set_ylim(bottom=0)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,handlelength=2.0,columnspacing=1.5)
    save(fig,'fig04_dense_jsr20')

def width():
    df=pd.read_csv(sources[1]);d=df[(df.suite=='sliceB')&(df.snr_db==-20)&(df.jsr_db==35)]
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,116/25.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.88,wspace=.32,hspace=.52)
    for col,m in enumerate((2,4)):
        for row,k in enumerate(('pd','frame_pfa_jammer')):
            ax=axs[row,col];s=d[(d.metric==k)&(d.repeats==m)].sort_values('width')
            assert len(s)==5
            for key,label,color in [('paper2_mean','CorrectionNet (128)','#0077bb'),('rsdr_mean','RSDR-Net (frozen)','#bb5566')]:
                ax.plot(s.width,s[key]*100,label=label,color=color,marker='o' if key=='paper2_mean' else 's',ls='-' if key=='paper2_mean' else '--',mfc='white',markersize=3,lw=1.1)
            ax.set(title=f'({chr(97+row*2+col)}) M = {m}',xlabel='Sampling width (samples)',ylabel='Detection probability (%)' if row==0 else 'J+E frame false alarms (%)',xticks=[6,18,36,54,72],ylim=(0,100))
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,handlelength=2.0,columnspacing=1.5)
    save(fig,'fig07_width_snr_minus20_jsre35')

def fixed_examples():
    """Replot saved predictions, never invoke a network or select a new parent."""
    sys.path.insert(0,str(ROOT))
    from paper2.radar_eval import pulse_compress, doppler_process, FS_HZ, PRI_S
    receipt=ROOT/'evidence/outputs/RSDR_UNIFIED_20261115/EXAMPLE_PROVENANCE.json'
    entry=json.loads(receipt.read_text(encoding='utf-8'))['examples'][1]
    path=ROOT/entry['local_raw_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    with np.load(path,allow_pickle=False) as a:
        waves=[a['s'][0],a['observed'][0,3],a['paper2'][0,3],a['rsdr'][0,3]]
    pcs=[pulse_compress(w)[0] for w in waves]
    lags=pulse_compress(waves[0])[1]
    powers=[np.mean(np.abs(p)**2,axis=0) for p in pcs]
    target=int(np.argmax(powers[0])); ref=float(np.max(np.abs(pcs[0])**2))
    # Use the true jammer in the saved paired observation only to choose one
    # diagnostic range cell; every method is then plotted at that same cell.
    with np.load(path,allow_pickle=False) as a:
        jammer=a['observed'][0,2]-a['observed'][0,0]
    jammer_power=np.mean(np.abs(pulse_compress(jammer)[0])**2,axis=0)
    jammer_bin=int(np.argmax(jammer_power))
    ranges=(lags-lags[target])*299792458/(2*FS_HZ)
    labels=['Target only','Input','CorrectionNet (128)','RSDR-Net (frozen)']
    colors=['#2266bb','#cc3311','#aa4499','#228833']
    styles=['--','-',':','-.']
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,126/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.84,wspace=.28,hspace=.48)
    fd=np.fft.fftshift(np.fft.fftfreq(4096,d=PRI_S))/1000
    for p,power,label,color,ls in zip(pcs,powers,labels,colors,styles):
        y=10*np.log10(np.maximum(power/ref,1e-14))
        for ax in (axs[0,0],axs[1,0]):ax.plot(ranges,y,label=label,color=color,ls=ls,lw=.9)
        for ax,idx in ((axs[0,1],target),(axs[1,1],jammer_bin)):
            spectrum=np.abs(np.fft.fftshift(np.fft.fft(p[:,idx],4096)))**2/p.shape[0]**2
            ax.plot(fd,10*np.log10(np.maximum(spectrum/ref,1e-14)),color=color,ls=ls,lw=.9)
    axs[0,0].set(title='(a) Pulse compression',xlim=(ranges[0],ranges[-1]))
    axs[1,0].set(title='(c) Target-region detail',xlim=(-50,200))
    axs[0,1].set(title='(b) Target-range Doppler spectrum')
    axs[1,1].set(title='(d) Jammer-peak-range Doppler spectrum')
    for row in range(2):
        axs[row,0].set(xlabel='Range offset (m)',ylabel='Relative power (dB)',ylim=(-70,25))
        axs[row,1].set(xlabel='Doppler frequency (kHz)',ylabel='Relative power (dB)',xlim=(-16.667,16.667),ylim=(-70,25))
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.97),ncol=4,frameon=False,handlelength=2.5)
    save(fig,'fig05_fixed_parent_pc_doppler')
    rds=[doppler_process(p) for p in pcs]
    rd_ref=max(float(np.max(np.abs(z)**2)) for z,_ in rds)
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,126/25.4))
    fig.subplots_adjust(left=.10,right=.84,bottom=.12,top=.91,wspace=.26,hspace=.40)
    for i,(ax,(z,f),label) in enumerate(zip(axs.flat,rds,labels)):
        ax.grid(False)
        mesh=ax.pcolormesh(ranges,f/1000,10*np.log10(np.maximum(np.abs(z)**2/rd_ref,1e-14)),vmin=-80,vmax=0,cmap='viridis',shading='nearest',rasterized=False)
        ax.set(title=f'({chr(97+i)}) {label}',xlabel='Range offset (m)',ylabel='Doppler frequency (kHz)')
    cax=fig.add_axes([.89,.18,.018,.66]);fig.colorbar(mesh,cax=cax,label='Relative power (dB)')
    save(fig,'fig06_fixed_parent_rd')
    provenance={'source_receipt':receipt.relative_to(ROOT).as_posix(),'raw_file':path.relative_to(ROOT).as_posix(),'raw_sha256':entry['sha256'],
        'condition':entry['condition'],'parent':0,'inference_run':False,'target_peak_lag_samples':int(lags[target]),'jammer_peak_lag_samples':int(lags[jammer_bin]),
        'pc_normalization':'Common maximum target-only PC power; pulse-mean power curves; floor 1e-14 before log10',
        'doppler':'8 pulses; 4096-point zero padding for line spectra only, divided by 8^2; no added resolution',
        'rd':'8 actual Doppler bins; shared global maximum across four panels; -80..0 dB; no interpolation',
        'range':'Offset from target-only compressed peak; c=299792458 m/s',
        'jammer_range_rule':'Strongest pulse-mean jammer-only PC bin, same bin for every displayed method; truth-assisted visualization only',
        'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'fixed_examples.manifest.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    # The manuscript uses the native editable Visio diagram; the legacy schematic
    # is retained only for explicitly requested historical reproduction.
    if '--legacy-architecture' in sys.argv:
        architecture()
    convergence();main_comparison();old_new();width()
    fixed_examples()
    from index_and_audit_delivery import make_manifest
    manifest=make_manifest()
    print('Created',len(manifest['figures']),'figure files; source SHA recorded.')
