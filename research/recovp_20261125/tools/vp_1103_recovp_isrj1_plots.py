"""Source-bound numerical tables and unsmoothed frozen-model figures."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vp_1103_common import WORK,read,sha,atomic_json,distribution
from vp_1103_recovp_radar_metrics import PRIMARY,DEFINITION
from paper2.radar_eval import pulse_compress,doppler_process,target_peak_range,mainlobe_guard,PRI_S
from tools.durable_io import atomic_write

OUT=WORK/'evidence/outputs/recovp_isrj1_suite'
PRIVATE=WORK/'outputs/recovp_isrj1_suite'
KEYS=('pd','frame_pfa_jammer',*PRIMARY,'sjrif_db','e_U','e_S','e_0_pool','cell_pfa_noise','frame_pfa_noise','pd_no_jammer')

def arrays(row):
    parts=[]
    for block in row['blocks']:
        p=WORK/block['file'];assert sha(p)==block['sha256']
        with np.load(p,allow_pickle=False) as z:parts.append({k:z[k] for k in z.files})
    return {k:np.concatenate([a[k] for a in parts]) for k in parts[0]}

def render():
    summary=read(OUT/'SUMMARY.json');cases=summary['conditions'];results=summary['results']
    figs=OUT/'figures';tables=OUT/'tables';figs.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    manifest=[]
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name,source):
        for ext in ('png','pdf'):
            p=figs/f'{name}.{ext}';fig.savefig(p,dpi=300)
            manifest.append({'file':p.relative_to(OUT).as_posix(),'sha256':sha(p),'source':source,
                'dimensions_inches':list(fig.get_size_inches()),'dpi':300,'note':'No smoothing or invented/interpolated grid cells.'})
        plt.close(fig)
    def table(name,rows):
        p=tables/f'{name}.csv'
        with p.open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return p.relative_to(OUT).as_posix()
    allrows=[]
    for i,case in enumerate(cases):
        for method,row in results[str(i)].items():
            for key,value in row['metrics'].items():
                allrows.append({'condition':i,'suite':case['suite'],'family':case['jam_type'],'snr_db':case['snr_db'],
                    'jsr_db':case['jsr_db'],'strength_definition':case['strength_definition'],'width_samples':case.get('width',''),
                    'method':method,'metric':key,'mean':value['mean'],'parent_count':value['parents'],
                    'ci95_lower':None if value['parent_bootstrap_ci95'] is None else value['parent_bootstrap_ci95'][0],
                    'ci95_upper':None if value['parent_bootstrap_ci95'] is None else value['parent_bootstrap_ci95'][1]})
    table('all_conditions',allrows)
    # Full 310-cell table and the two requested figure views share the same observations.
    pd_indices=[i for i,c in enumerate(cases) if c['suite']=='pd']
    heat_metrics=('pd','jcr_db','rd_residual_peak_db','abs_cpl_db')
    limits={key:(min(results[str(i)][method]['metrics'][key]['mean'] for i in pd_indices for method in ('ReCoVP-Winner','P0-NoDechirp-Tail')),
                 max(results[str(i)][method]['metrics'][key]['mean'] for i in pd_indices for method in ('ReCoVP-Winner','P0-NoDechirp-Tail'))) for key in heat_metrics}
    for method in ('ReCoVP-Winner','P0-NoDechirp-Tail'):
        for key in heat_metrics:
            matrix=np.array([[results[str(next(i for i in pd_indices if cases[i]['snr_db']==s and cases[i]['jsr_db']==r))][method]['metrics'][key]['mean']
                              for r in range(-10,36,5)] for s in range(-30,1)])
            views=(('full310',list(range(-30,1))),('overview',list(range(-30,1,5))),('transition',list(range(-30,-19))))
            for label,snrs in views:
                fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
                image=ax.imshow(matrix[np.array(snrs)+30],origin='lower',aspect='auto',interpolation='nearest',
                    vmin=0 if key=='pd' else limits[key][0],vmax=1 if key=='pd' else limits[key][1])
                ax.set_xticks(range(10),list(range(-10,36,5)));step=1 if len(snrs)<=11 else 5
                ax.set_yticks(list(range(0,len(snrs),step)),[snrs[k] for k in range(0,len(snrs),step)])
                ax.set(xlabel='Native JSR (dB)',ylabel='Native SNR (dB)',title=f'{method}: {key}; n=512 paired parents')
                fig.colorbar(image,ax=ax,label=key+(' (dB)' if key!='pd' else ''))
                save(fig,f'grid_{method}_{key}_{label}','tables/all_conditions.csv')
    # Main-grid comparisons and controlled width scan, all pointwise parent CIs.
    styles=[('o','-'),('s','--'),('^','-.'),('D',':'),('v','-'),('x','--'),('P','-.')]
    for family in ('repeat','direct','cyclic'):
        ids=[i for i,c in enumerate(cases) if c['suite']=='main' and c['jam_type']==family]
        names=list(results[str(ids[0])])
        for key in ('pd','jcr_db','rd_residual_peak_db','abs_cpl_db','bvr01_db'):
            fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for ax,jsr in zip(axs.flat,(0,10,20,35)):
                sel=sorted((i for i in ids if cases[i]['jsr_db']==jsr),key=lambda i:cases[i]['snr_db'])
                x=[cases[i]['snr_db'] for i in sel]
                for name,(marker,line) in zip(names,styles):
                    vals=[results[str(i)][name]['metrics'][key] for i in sel]
                    ax.plot(x,[v['mean'] for v in vals],marker=marker,linestyle=line,markersize=3,label=name)
                    ax.fill_between(x,[v['parent_bootstrap_ci95'][0] for v in vals],
                                    [v['parent_bootstrap_ci95'][1] for v in vals],alpha=.06)
                ax.set(xlabel='Native SNR (dB)',ylabel=key,title=f'{family}, JSR={jsr} dB, n=256')
                if key=='pd':ax.set_ylim(-.02,1.02)
            axs.flat[0].legend(fontsize=7,ncol=2);save(fig,f'methods_{family}_{key}','tables/all_conditions.csv')
    for key in ('pd','sjrif_db','jcr_db','abs_cpl_db'):
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,name in zip(axs,('P0-NoDechirp-Tail','ReCoVP-Winner')):
            for w,(marker,line) in zip((6,18,36,54,72),styles):
                sel=sorted((i for i,c in enumerate(cases) if c['suite']=='sliceB' and c['width']==w),key=lambda i:cases[i]['snr_db'])
                ax.plot([cases[i]['snr_db'] for i in sel],[results[str(i)][name]['metrics'][key]['mean'] for i in sel],
                    marker=marker,linestyle=line,label=f'{w/60:g} us')
            ax.set(xlabel='Native SNR (dB)',ylabel=key,title=f'{name}; group-energy JSR_E=20 dB; M=2')
            ax.legend(fontsize=8)
        save(fig,f'width_{key}','tables/all_conditions.csv')
    # Independent no-jammer groups; retain zero empirical false alarms as zeros.
    transparency=[]
    for family in ('direct','repeat','cyclic'):
        ids=sorted((i for i,c in enumerate(cases) if c['suite']=='retention' and c['jam_type']==family),key=lambda i:cases[i]['snr_db'])
        names=list(results[str(ids[0])]);fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for name,(marker,line) in zip(names,styles):
            for ax,key in zip(axs,('bvr00_db','cell_pfa_noise','pd_no_jammer')):
                y=[results[str(i)][name]['metrics'][key]['mean'] for i in ids]
                ax.plot([cases[i]['snr_db'] for i in ids],y,marker=marker,linestyle=line,label=name)
                ax.set(xlabel='Native SNR (dB)',ylabel=key,title=f'{family} no-jammer; n=512')
            difference=[results[str(i)][name]['metrics']['pd_no_jammer']['events']-results[str(i)]['Unprocessed']['metrics']['pd_no_jammer']['events'] for i in ids]
            transparency.append({'family':family,'method':name,'maximum_absolute_hit_count_difference':max(map(abs,difference)),
                'maximum_frame_false_alarm_count_increase':max(results[str(i)][name]['metrics']['frame_pfa_noise']['events']-results[str(i)]['Unprocessed']['metrics']['frame_pfa_noise']['events'] for i in ids)})
        axs[0].legend(fontsize=7);save(fig,f'transparency_{family}','tables/all_conditions.csv')
    table('transparency_maximum_differences',transparency)
    # Main-grid totals are descriptive repeated-condition counts, not independent n=3584.
    totals=[];paired={};worst={}
    for family in ('repeat','direct','cyclic'):
        for jsr in (0,10,20,35):
            ids=[i for i,c in enumerate(cases) if c['suite']=='main' and c['jam_type']==family and c['jsr_db']==jsr]
            names=list(results[str(ids[0])]);data={name:[arrays(results[str(i)][name]) for i in ids] for name in names}
            winner={k:np.mean([d[k] for d in data['ReCoVP-Winner']],0) for k in KEYS}
            for name in names:
                sums={k:sum(results[str(i)][name]['metrics'][k]['events'] for i in ids) for k in ('pd','frame_pfa_jammer')}
                totals.append({'family':family,'jsr_db':jsr,'method':name,'hits':sums['pd'],'target_absent_false_frames':sums['frame_pfa_jammer'],
                    'condition_trials':3584,'independent_parents':256})
                grouped={k:np.mean([d[k] for d in data[name]],0) for k in KEYS}
                paired[f'{family}/JSR={jsr}/{name}']={k:distribution(winner[k]-grouped[k]) for k in KEYS}
            for i,d in zip(ids,data['ReCoVP-Winner']):
                for p,pid in enumerate(results[str(i)]['ReCoVP-Winner']['parent_ids']):
                    rank=(d['frame_pfa_jammer'][p],d['rd_residual_peak_db'][p],d['abs_cpl_db'][p])
                    row={'parent_id':pid,'snr_db':cases[i]['snr_db'],'jsr_db':cases[i]['jsr_db'],'family':family,'condition':i,
                         'metrics':{k:float(d[k][p]) for k in KEYS},'rank':list(rank)}
                    if pid not in worst or rank>tuple(worst[pid]['rank']):worst[pid]=row
    table('main_hit_false_alarm_totals',totals)
    atomic_json(OUT/'PAIRED_MAIN_GRID.json',{'unit':'256 physical parents; average 14 SNR cells before paired resampling',
        'pointwise_no_multiplicity_claim':True,'comparisons':paired})
    failures=sorted(worst.values(),key=lambda v:tuple(v['rank']),reverse=True)[:5]
    atomic_json(OUT/'TOP5_FAILURE_PARENTS.json',{'distinct_physical_parents':True,'ranking':'frame Pfa, RD residual peak, absCPL; worst condition retained per parent','parents':failures})
    # Coherent figures use a fixed example and declared zero-padding for spectral interpolation.
    files=list((PRIVATE/'typical_waveforms').glob('*.npz'));waves={}
    for path in files:
        with np.load(path,allow_pickle=False) as z:waves[path.stem]={k:z[k] for k in z.files}
    base=waves['Unprocessed'];s,j,e=base['s'][0],base['j'][0],base['e'][0]
    pulse=int(np.flatnonzero((abs(j)**2).sum(-1)>0)[0]);target=int(target_peak_range(s));guard=mainlobe_guard()
    curves={'Observed':base['y'][0,3],'S+E reference':s+e,**{name:v['restored'][0,3] for name,v in waves.items() if name!='Unprocessed'}}
    t=np.arange(1800)/60;fig,axs=plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True)
    for name,v in curves.items():
        for ax,signal in zip(axs,(v[pulse].real,v[pulse].imag,abs(v[pulse]))):ax.plot(t,signal,label=name,linewidth=.8)
    active=abs(j[pulse])>0
    for ax,label in zip(axs,('I','Q','Envelope')):
        ax.fill_between(t,0,1,where=active,transform=ax.get_xaxis_transform(),alpha=.08,color='gray',label='Jammer support' if label=='I' else None)
        ax.set_ylabel(label)
    axs[0].legend(fontsize=7,ncol=3);axs[-1].set_xlabel('Fast time (us)');save(fig,'typical_time_IQ_envelope','Private fixed typical waveforms; first jammed pulse')
    support=np.flatnonzero(abs(s[pulse])>0);lo=max(0,int(support[0])-24);hi=min(1800,int(support[-1])+25)
    fig,axs=plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True)
    for name,v in curves.items():
        for ax,signal in zip(axs,(v[pulse].real,v[pulse].imag,abs(v[pulse]))):ax.plot(t[lo:hi],signal[lo:hi],label=name,linewidth=.8)
    for ax,label in zip(axs,('I','Q','Envelope')):
        ax.fill_between(t[lo:hi],0,1,where=active[lo:hi],transform=ax.get_xaxis_transform(),alpha=.08,color='gray')
        ax.set_ylabel(label)
    axs[0].legend(fontsize=7,ncol=3);axs[-1].set_xlabel('Fast time (us)')
    save(fig,'typical_local_time_IQ_envelope','Fixed crop: true target support plus 24 samples on each side; no outcome-based selection')
    compressed={name:pulse_compress(v)[0] for name,v in curves.items()}
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,v in compressed.items():
        power=10*np.log10(np.maximum(abs(v[pulse])**2,1e-30))
        axs[0].plot(np.arange(1201),power,label=name);axs[1].plot(np.arange(1201),power,label=name)
    axs[1].set_xlim(target-12,target+12)
    for ax in axs:ax.set(xlabel='Range lag (samples)',ylabel='Matched-filter power (dB)')
    axs[0].legend(fontsize=7);save(fig,'typical_pulse_compression','Private fixed typical waveforms; unnormalized common power scale')
    jp=pulse_compress(j)[0];jam_power=(abs(jp)**2).mean(0);jam_power[max(0,target-guard):target+guard+1]=0
    jammer=int(jam_power.argmax());bins=np.fft.fftshift(np.fft.fftfreq(4096,d=PRI_S))
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,v in compressed.items():
        for ax,r in zip(axs,(target,jammer)):
            value=np.fft.fftshift(np.fft.fft(v[:,r],n=4096));ax.plot(bins/1000,10*np.log10(np.maximum(abs(value)**2,1e-30)),label=name)
            ax.set(xlabel='Doppler (kHz)',ylabel='Power (dB)',title=f'Range lag={r}; 8 pulses, 4096-point zero padding')
    axs[0].legend(fontsize=7);save(fig,'typical_doppler_spectra','Private fixed typical waveforms; padding interpolates, does not improve resolution')
    before=np.max(abs(np.fft.fft(compressed['Observed'][:,jammer]))**2)
    reductions={name:float(10*np.log10(max(before,1e-300)/max(np.max(abs(np.fft.fft(v[:,jammer]))**2),1e-300)))
                for name,v in compressed.items()}
    atomic_json(OUT/'TYPICAL_DIAGNOSTICS.json',{'first_jammed_pulse':pulse,'target_range_lag':target,
        'strongest_jammer_range_lag_excluding_target_mainlobe':jammer,'native_8point_doppler_peak_suppression_db':reductions,
        'definition':'Observed versus restored maximum 8-point Doppler power at the same fixed jammer range cell',
        'local_time_crop_samples':[lo,hi],'example_selection':'fixed parent, no best-case search'})
    coherent_rows=[]
    for i,c in enumerate(cases):
        if c['suite']=='coherent':
            for name,row in results[str(i)].items():
                coherent_rows.append({'snr_db':c['snr_db'],'jsr_db':c['jsr_db'],'method':name,**{k:row['metrics'][k]['mean'] for k in
                    ('phase_rmse_deg','peak_error_hz','coherent_power_db','cpl_db','abs_cpl_db','sin2_theta_s','increment_phase_rmse_deg','increment_peak_error_hz')}})
    table('coherent_response',coherent_rows)
    strong=next(i for i,c in enumerate(cases) if c['suite']=='pd' and c['snr_db']==-20 and c['jsr_db']==35)
    raw=arrays(results[str(strong)]['Unprocessed'])['pd'];ref=arrays(results[str(strong)]['Reference'])['pd'];win=arrays(results[str(strong)]['ReCoVP-Winner'])['pd']
    denom=float((ref-raw).mean());recovered=None if denom==0 else float((win-raw).mean()/denom)
    atomic_json(OUT/'STRONG_CONDITION.json',{'snr_db':-20,'jsr_db':35,'parents':512,
        'lost_detection_recovery_fraction':recovered,'formula':'mean(Pd_winner-Pd_raw)/mean(Pd_reference-Pd_raw); not clipped',
        'methods':{name:row['metrics'] for name,row in results[str(strong)].items()},'paired_pd_delta':distribution(win-raw)})
    lines=['# ReCoVP 第一篇全对齐冻结扫描','',DEFINITION,'',
        '全部场景来自新的 scan/2026112407 命名空间；冻结 seed17 Winner。扫描结果不用于再选模型、epoch 或调参。',
        '共 579 个工况：典型 1、无模糊相参 3、能量网格 310、切片宽度 55、跨方法 168、无扰 42。',
        '第一篇正文与生成脚本的 310 格采用 SNR -30..0 步长1；指定的概览与过渡区图为其子集。',
        '所有逐工况指标、signed/absolute dB、兼容指标及 parent 区间见 SUMMARY.json 与 tables/all_conditions.csv。',
        'CSV 同时保留每个条件的实际 parent 数；跨网格累计数为重复工况描述计数，不扩张独立样本量。','',
        '## 跨方法累计命中与无目标虚警','',
        '| Family | JSR dB | Method | Hits / 3584 | False frames / 3584 |','|---|---:|---|---:|---:|']
    lines += [f"| {v['family']} | {v['jsr_db']} | {v['method']} | {v['hits']} | {v['target_absent_false_frames']} |" for v in totals]
    lines+=['','## 相参响应','',
        '| SNR | JSR | Method | Phase RMSE deg | Doppler error Hz | CPL dB | abs CPL dB | sin2 angle |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    lines += [f"| {v['snr_db']} | {v['jsr_db']} | {v['method']} | {v['phase_rmse_deg']:.4g} | {v['peak_error_hz']:.4g} | {v['cpl_db']:.4g} | {v['abs_cpl_db']:.4g} | {v['sin2_theta_s']:.4g} |" for v in coherent_rows]
    lines += ['','## 最差五个独立 parent','',*[json_line(v) for v in failures],
        '',f'强干扰 SNR=-20、JSR=35 的检测损失挽回比例：{recovered}；完整 Pd/Pfa 与噪声保真见 STRONG_CONDITION.json。',
        '无扰最大命中差与虚警增量见 tables/transparency_maximum_differences.csv。',
        '跨工况与方法的配对 parent 区间见 PAIRED_MAIN_GRID.json；区间为描述性、未作多重比较校正。',
        'RSDR 原训练预算与数据口径不同；名义相同 CFAR 不等同于实际虚警匹配。','',
        '**Gate：完成后暂停人工审阅；CONFIRM 与历史 TEST 未读取。SoftShare 和 CVaR 本轮未执行。**']
    atomic_write(OUT/'REPORT.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf8')))
    atomic_json(OUT/'FIGURE_MANIFEST.json',{'figures':manifest,'format':'PNG300dpi + vector PDF','matplotlib':matplotlib.__version__,
        'source_summary_sha256':sha(OUT/'SUMMARY.json'),'uncertainty':'Pointwise parent-cluster bootstrap95%; curves unsmoothed',
        'publisher_specific_requirements':'Not claimed; general research figures'})

def json_line(v):
    return f"- {v['parent_id']}：{v['family']}，SNR={v['snr_db']} dB、JSR={v['jsr_db']} dB；Pfa={v['metrics']['frame_pfa_jammer']:.4g}，RD峰值={v['metrics']['rd_residual_peak_db']:.4g} dB，absCPL={v['metrics']['abs_cpl_db']:.4g} dB。"

if __name__=='__main__':render()
