"""Provisional review figures from audited paired physical-parent results.

Complete curve scopes only; dense-grid missing cells remain grey. Waveform
examples always use the pre-existing frozen parent zero, never best examples.
"""
from __future__ import annotations
import argparse
import csv
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paper2.io import ROOT, now, read, sha
from paper2.evaluation_1103_v4 import summarize, restore_model, states_from_components
from paper2.radar_eval import pulse_compress, doppler_process, target_peak_range, PRF_HZ
from paper2.network import CorrectionNet
from paper2.scan_1103 import build_batch
from tools.aggregate_rsdr_unified import load_records, paired_group
from tools.compare_rsdr_unified import LIGHT, DEST, FREEZE, OLD_FREEZE
from tools.durable_io import atomic_json, atomic_write

FIG=LIGHT/'figures'
COLORS={'model':'#0072B2','rsdr':'#A54E00'}
NAMES={'model':'Paper 2 (seed 29, epoch 128)','rsdr':'RSDR-Net (complete F-R7)'}
STYLE={'model':('o','-'),'rsdr':('s','--')}
ROWS=[]; EXPORTS=[]
WIDTH_ROWS=[]
LABELS={'u_nmse':'Restored-waveform NMSE (lower better)',
        'damage':'Target-response damage (lower better)', 'sjrif_db':'SJRIF proxy (dB; higher better)',
        'pd':'Detection probability', 'cell_pfa_jammer':'Jammer H0 cell false-alarm rate',
        'frame_pfa_jammer':'Jammer H0 frame false-alarm probability',
        'false_peak_suppression_db':'Outside-mainlobe peak suppression (dB)',
        'no_jammer_pooled_nmse':'No-jammer pooled NMSE', 'pd_no_jammer':'No-jammer detection probability',
        'cell_pfa_noise':'Noise H0 cell false-alarm rate','frame_pfa_noise':'Noise H0 frame false-alarm probability'}


def save(fig,name,caption):
    FIG.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):
        p=FIG/f'{name}.{suffix}';fig.savefig(p,dpi=180)
        EXPORTS.append(dict(file=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p),caption=caption))
    plt.close(fig)


def curves(records,suite,indexes,name,title,metrics):
    if not all((suite,i) in records for i in indexes):return
    snrs=sorted({records[suite,i]['job']['condition']['snr_db'] for i in indexes})
    # One audited parent summary per SNR, reused across panels and methods.
    groups={snr:paired_group(records,suite,[i for i in indexes
            if records[suite,i]['job']['condition']['snr_db']==snr]) for snr in snrs}
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    for ax,k in zip(axes.flat,metrics):
        for m in ('model','rsdr'):
            means=[]; lows=[]; highs=[]
            for snr in snrs:
                items=[i for i in indexes if records[suite,i]['job']['condition']['snr_db']==snr]
                group=groups[snr]
                stat=group['metrics'][m][k];lo,hi=stat['parent_bootstrap_ci95']
                means.append(stat['mean']);lows.append(lo);highs.append(hi)
                ROWS.append(dict(figure=name,metric=k,snr_db=snr,method=m,**stat,conditions=len(items)))
            marker,line=STYLE[m]
            ax.plot(snrs,means,color=COLORS[m],marker=marker,linestyle=line,ms=4,label=NAMES[m])
            ax.fill_between(snrs,lows,highs,color=COLORS[m],alpha=.12)
        ax.set(xlabel='SNR (dB)',ylabel=LABELS[k]);ax.grid(alpha=.2)
        if k.startswith(('pd','frame_pfa')):ax.set_ylim(-.025,1.025)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title+'\nEqual condition weights within parent; pointwise 95% parent bootstrap CI',fontsize=12)
    save(fig,name,'All listed conditions complete. No smoothing. Equal grid-cell weights within physical parent; 4000 parent bootstrap replicates, seed 20261114. Pointwise intervals, not simultaneous. Retention uses stratified parent bootstrap.')


def heatmaps(f,records):
    grid=f['grids']['pd'];snrs=sorted({c['snr_db'] for c in grid});jsrs=sorted({c['jsr_db'] for c in grid})
    if not any(s=='pd' for s,i in records):return
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,k in enumerate(('pd','frame_pfa_jammer')):
        matrices=[]
        for m in ('model','rsdr'):
            a=np.full((len(jsrs),len(snrs)),np.nan)
            for i,c in enumerate(grid):
                if ('pd',i) in records:a[jsrs.index(c['jsr_db']),snrs.index(c['snr_db'])]=records['pd',i]['data'][m][k].mean()
            matrices.append(a)
        for col,(a,label) in enumerate(zip(matrices+[matrices[0]-matrices[1]],['Paper 2','RSDR-Net','Paper 2 minus RSDR-Net'])):
            cmap=plt.get_cmap('viridis' if col<2 else 'RdBu_r').copy();cmap.set_bad('#BBBBBB')
            im=axes[row,col].imshow(a,origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,vmin=0 if col<2 else -1,vmax=1)
            ax=axes[row,col];ax.set_title(label+'\n'+LABELS[k],fontsize=10)
            ax.set_xticks(range(0,len(snrs),5),snrs[::5]);ax.set_yticks(range(len(jsrs)),jsrs)
            ax.set(xlabel='SNR (dB)',ylabel='JSR (dB)');fig.colorbar(im,ax=ax,shrink=.8)
    n=sum(('pd',i) in records for i in range(len(grid)))
    fig.suptitle(f'Dense repeat-jammer scan: {n}/{len(grid)} paired cells; grey = unfinished')
    save(fig,'dense_pd_and_false_alarm','Cell means on all frozen dense-grid coordinates. Grey denotes unfinished, not zero. Common 0..1 scales and signed difference -1..1. Paired n=512 physical parents per cell.')


def width_curves(f, records):
    for suite in ('sliceA', 'sliceB'):
        grid = f['grids'][suite]
        settings = sorted({(c['snr_db'], c['jsr_db'], c['repeats']) for c in grid})
        for snr, jsr, repeats in settings:
            indexes = [i for i, c in enumerate(grid)
                       if (c['snr_db'], c['jsr_db'], c['repeats']) == (snr, jsr, repeats)]
            if not all((suite, i) in records for i in indexes):
                continue
            indexes.sort(key=lambda i: grid[i]['width'])
            widths = [grid[i]['width'] for i in indexes]
            name = f'{suite}_snr{snr}_jsr{jsr}_m{repeats}'
            groups = {i: paired_group(records, suite, [i]) for i in indexes}
            metrics = ('u_nmse', 'damage', 'sjrif_db', 'pd',
                       'false_peak_suppression_db', 'frame_pfa_jammer')
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
            for ax, key in zip(axes.flat, metrics):
                for method in ('model', 'rsdr'):
                    values = [groups[i]['metrics'][method][key] for i in indexes]
                    means = [v['mean'] for v in values]
                    lower, upper = np.asarray([v['parent_bootstrap_ci95'] for v in values]).T
                    marker, line = STYLE[method]
                    ax.plot(widths, means, color=COLORS[method], marker=marker,
                            linestyle=line, label=NAMES[method], ms=4)
                    ax.fill_between(widths, lower, upper, color=COLORS[method], alpha=.12)
                    for i, value in zip(indexes, values):
                        WIDTH_ROWS.append(dict(figure=name, suite=suite, index=i,
                            snr_db=snr, jsr_db=jsr, repeats=repeats, width_samples=grid[i]['width'],
                            metric=key, method=method, **value))
                ax.set(xlabel='Sampling width (samples; 60 MHz)', ylabel=LABELS[key])
                ax.grid(alpha=.2)
                if key in ('pd', 'frame_pfa_jammer'):
                    ax.set_ylim(-.025, 1.025)
            axes.flat[0].legend(fontsize=8)
            period = 'fixed period 260 samples' if suite == 'sliceA' else 'period = (M+1) * width + 1'
            fig.suptitle(f'{suite}: SNR {snr} dB, group-energy JSR {jsr} dB, M={repeats}; {period}\n'
                         'All frozen widths; pointwise 95% physical-parent bootstrap CI', fontsize=12)
            save(fig, name, 'Every frozen width at this SNR/JSR/M is shown, with no smoothing or selection. '
                 'Shared physical parents across widths; 4000 pointwise parent bootstrap replicates, seed 20261114. '
                 'JSR is total pulse-group energy JSR, unlike the main scan active-support convention.')


def examples(f):
    import torch
    from tools.rsdr_adapter import RSDRNet
    torch.set_num_threads(1)
    model=CorrectionNet().to('cuda:0').eval()
    cp=torch.load(ROOT/f['selected_checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(cp['state'])
    rsdr=RSDRNet('cuda:0');raw=DEST/'fixed_parent_examples';raw.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for i,c in enumerate(f['grids']['coherent']):
        s,j,e,meta=build_batch('test','coherent',c,[f['typical_parent']]);obs=states_from_components(s,j,e)
        u=restore_model(model,obs,'cuda:0');r=rsdr.restore(obs)
        path=raw/f'condition_{i:03d}.npz'
        atomic_write(path,lambda stream:np.savez_compressed(stream,s=s,observed=obs,paper2=u,rsdr=r))
        manifest.append(dict(condition=c,metadata=meta,local_raw_file=str(path.relative_to(ROOT)),sha256=sha(path)))
        waves=[s[0],obs[0,3],u[0,3],r[0,3]]
        labels=['Target only (scoring reference)','Input','Paper 2','RSDR-Net']
        pcs=[pulse_compress(w)[0] for w in waves];lags=pulse_compress(waves[0])[1]
        tr=int(target_peak_range(s)[0]);norm=np.abs(pcs[0]).max()**2
        def db(x):return 10*np.log10(np.maximum(x/norm,1e-14))
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for n,(pc,label) in enumerate(zip(pcs,labels)):
            color=['#777777','#222222',COLORS['model'],COLORS['rsdr']][n]
            line=[':',':','-','--'][n]
            pcdb=db(np.mean(abs(pc)**2,axis=0))
            axes[0,0].plot(lags,pcdb,color=color,ls=line,label=label)
            axes[0,1].plot(lags,pcdb,color=color,ls=line,label=label)
            spectrum=np.abs(np.fft.fftshift(np.fft.fft(pc[:,tr],n=4096)))**2
            axes[1,0].plot(np.fft.fftshift(np.fft.fftfreq(4096,d=1/PRF_HZ)),db(spectrum/64),color=color,ls=line,label=label)
            phase=np.angle(pc[:,tr]*pcs[0][:,tr].conj(),deg=True)
            axes[1,1].plot(range(8),phase,color=color,ls=line,marker='o',ms=3,label=label)
        axes[0,0].set(title='Pulse-compression panorama',xlabel='Range lag (samples)',ylabel='Power (dB / target-only peak)')
        axes[0,1].set(title='Target-range detail',xlim=(tr-20,tr+20),xlabel='Range lag (samples)',ylabel='Power (dB / target-only peak)')
        axes[1,0].set(title='Target-bin Doppler spectrum (4096-point interpolation)',xlabel='Doppler frequency (Hz)',ylabel='Power (dB / common target reference)')
        axes[1,1].set(title='Phase relative to target-only response',xlabel='Pulse index',ylabel='Wrapped phase difference (degrees)',ylim=(-185,185))
        for ax in axes.flat:ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"Frozen parent 0; repeat jammer; SNR {c['snr_db']} dB, JSR {c['jsr_db']} dB")
        save(fig,f'fixed_parent0_coherent_{i:03d}','Frozen parent 0 at every coherent condition. Common target-only peak normalization, no per-model renormalization. Display floor -140 dB; zero padding interpolates and does not increase physical Doppler resolution. Example is illustrative, not independent statistical evidence.')
        rds=[doppler_process(pc)[0] for pc in pcs]
        fd=doppler_process(pcs[0])[1];df=PRF_HZ/8
        fig,axes=plt.subplots(1,4,figsize=(16,4.5),layout='constrained')
        # One common range-Doppler normalization and color scale across methods.
        rd_norm=max(float((abs(rd)**2).max()) for rd in rds)
        for ax,rd,label in zip(axes,rds,labels):
            im=ax.imshow(10*np.log10(np.maximum(abs(rd)**2/rd_norm,1e-14)),origin='lower',aspect='auto',interpolation='nearest',extent=[lags[0]-.5,lags[-1]+.5,fd[0]-df/2,fd[-1]+df/2],vmin=-80,vmax=0,cmap='cividis')
            ax.set(title=label,xlabel='Range lag (samples)',ylabel='Doppler frequency (Hz)')
        fig.colorbar(im,ax=axes,label='dB / common maximum',shrink=.85)
        fig.suptitle(f"Parent 0; SNR {c['snr_db']} dB, JSR {c['jsr_db']} dB; eight physical Doppler bins")
        save(fig,f'fixed_parent0_rd_{i:03d}','Common color scale -80..0 dB relative to the maximum over all four panels. Eight physical Doppler bins, no interpolation. Frozen parent 0 at every coherent condition.')
    atomic_json(LIGHT/'EXAMPLE_PROVENANCE.json',dict(created_utc=now(),freeze_sha256=sha(FREEZE),parent=f['typical_parent'],examples=manifest,raw_arrays_local_only=True))


def main(with_examples=False,audited=None):
    # The durable controller invokes this module at several milestones.
    ROWS.clear(); WIDTH_ROWS.clear(); EXPORTS.clear()
    if audited is None:
        _,records,_=load_records()
    else:
        records=audited
    f=read(OLD_FREEZE)
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        for jam in ('repeat','direct','cyclic','all'):
            ids=[i for i,c in enumerate(f['grids']['main']) if jam=='all' or c['jam_type']==jam]
            curves(records,'main',ids,f'main_{jam}',f'Main scan: {jam}; 256 paired physical parents',('u_nmse','damage','sjrif_db','pd','cell_pfa_jammer','frame_pfa_jammer'))
        ids=[i for i,c in enumerate(f['grids']['pd']) if c['jsr_db']==20]
        curves(records,'pd',ids,'pd_jsr20','Repeat jammer, JSR 20 dB; 512 paired physical parents',('u_nmse','damage','sjrif_db','pd','cell_pfa_jammer','frame_pfa_jammer'))
        curves(records,'retention',list(range(len(f['grids']['retention']))),'retention','No-jammer retention; three independent parent groups',('no_jammer_pooled_nmse','damage','pd_no_jammer','cell_pfa_noise','frame_pfa_noise','u_nmse'))
        heatmaps(f,records)
        width_curves(f,records)
        if with_examples:examples(f)
    if ROWS:
        stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
        atomic_write(LIGHT/'CURVE_SOURCE_SUMMARIES.csv',lambda dest:dest.write(stream.getvalue().encode('utf8')))
    if WIDTH_ROWS:
        stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(WIDTH_ROWS[0]));w.writeheader();w.writerows(WIDTH_ROWS)
        atomic_write(LIGHT/'WIDTH_SOURCE_SUMMARIES.csv',lambda dest:dest.write(stream.getvalue().encode('utf8')))
    atomic_json(LIGHT/'FIGURE_MANIFEST.json',dict(created_utc=now(),freeze_sha256=sha(FREEZE),script_sha256=sha(__file__),
        purpose='Provisional scientific review; journal requirements not specified.',formats=['PNG 180 dpi','PDF'],
        uncertainty='4000 paired physical-parent bootstrap replicates, seed 20261114. Pointwise 95%, not simultaneous.',
        raw_arrays_local_only=True,figures=EXPORTS))
    print(dict(figures=len(EXPORTS),paired_conditions=len(records)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--examples',action='store_true');a=p.parse_args();main(a.examples)
