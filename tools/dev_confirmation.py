"""Frozen-model, frozen-gain new-parent and weak-target development checks."""
from dataclasses import asdict
import numpy as np
import torch
from paper2.vendor.generate_v2 import GeneratorConfig,generate_dataset,_draw_target
from paper2.science import four_states,metrics,parent_mean,bootstrap_difference
from paper2.network import CorrectionNet
from paper2.data import iq
from paper2.run import setup
from paper2.evaluate import summary
from paper2.io import ROOT,read,write,sha,source_bindings,now


def counterfactual_inputs(target,other,j,e):
    return four_states(target,j,other+e)[0]


def freeze():
    pilot=read(ROOT/'evidence'/'PILOT_RESULTS.json')
    rows=[r for r in pilot['rows'] if r['family']=='complex' and r['budget']==.05]
    methods={r['tag']:dict(gain_real=r['gain_real'],gain_imag=r['gain_imag'],
              checkpoint_sha256=sha(ROOT/'outputs'/r['tag']/'model.pt')) for r in rows}
    value=dict(source=source_bindings(),tool_sha256=sha(__file__),contract_sha256=sha(ROOT/'DEV_CONFIRMATION_CONTRACT.md'),
               pilot_results_sha256=sha(ROOT/'evidence'/'PILOT_RESULTS.json'),methods=methods,
               selected_controls={'17':'B1_seed17','29':'B2_seed29'},budget=.05,
               roles={'new_id':[192,2026102011,220050000],'two_target':[96,2026102012,220060000]})
    path=ROOT/'DEV_CONFIRMATION_FREEZE.json'
    if path.exists():
        assert read(path)['freeze']==value,'development confirmation freeze drift'
    else:
        write(path,dict(frozen_utc=now(),freeze=value))
    return value


def role_data(name,spec):
    path=ROOT/'data'/f'dev_confirmation_{name}.npz'
    receipt=path.with_suffix('.receipt.json')
    cfg=read(ROOT/'data'/'train.receipt.json')['config']
    cfg.update(num_scenes=spec[0],seed=spec[1],scene_id_start=spec[2],fixed_partition='validation')
    cfg['jam_types']=tuple(cfg['jam_types'])
    config=GeneratorConfig(**cfg)
    if not path.exists():
        generate_dataset(path,config)
        write(receipt,dict(created_utc=now(),sha256=sha(path),config=asdict(config),parents=spec[0],role='development only'))
    else:
        assert read(receipt)['sha256']==sha(path)
    with np.load(path,allow_pickle=False) as raw:
        a={k:raw[k] for k in ('s','j','epsilon','scene_id','target_amplitude')}
    if name=='two_target':
        rng=np.random.default_rng(2026102013)
        weak=np.zeros_like(a['s'])
        for parent in np.unique(a['scene_id']):
            idx=np.flatnonzero(a['scene_id']==parent)
            s2,_,meta=_draw_target(config,rng)
            s2*=.25*float(a['target_amplitude'][idx[0]])/meta['amplitude']
            weak[idx]=s2.astype(np.complex64)
        weak_path=ROOT/'data'/'dev_confirmation_weak_targets.npz'
        if weak_path.exists():
            with np.load(weak_path) as saved:
                assert np.array_equal(saved['weak'],weak)
        else:
            np.savez_compressed(weak_path,weak=weak)
        a['weak']=weak
    return a


def predict(tag,y):
    device=setup(int(tag.split('seed')[1]))
    net=CorrectionNet().to(device)
    net.load_state_dict(torch.load(ROOT/'outputs'/tag/'model.pt',weights_only=False,map_location=device)['state'])
    net.eval()
    observations=iq(y)
    result=[]
    with torch.inference_mode():
        for start in range(0,len(y),16):
            b=observations[start:start+16].to(device)
            c=net(b.flatten(0,1)).reshape_as(b).cpu().numpy()
            result.append(c[:,:,0]+1j*c[:,:,1])
    return np.concatenate(result)


def aggregate(s,other,j,e,c,gain,ids):
    m=metrics(s,j,other+e,c,gain)
    value=summary(m,ids)
    radians=np.deg2rad(m['phase_deg'])
    value['circular_phase_mean_deg']=float(np.rad2deg(np.angle(np.exp(1j*radians).mean())))
    value['median_abs_phase_deg']=float(np.median(np.abs(m['phase_deg'])))
    value['amplitude_parent_p05']=float(np.quantile(parent_mean(m['amplitude'],ids),.05))
    return value,parent_mean(m['u_nmse'],ids)


def main():
    frozen=freeze()
    final=ROOT/'evidence'/'DEV_CONFIRMATION_RESULTS.json'
    if final.exists():
        assert read(final)['freeze_sha256']==sha(ROOT/'DEV_CONFIRMATION_FREEZE.json')
        print('Completed development confirmation preserved.',flush=True)
        return
    rows=[];parent_errors={}
    for name,spec in frozen['roles'].items():
        a=role_data(name,spec)
        s,j,e=[a[k].astype(np.complex128) for k in ('s','j','epsilon')]
        ids=a['scene_id']
        targets=[('single',s,np.zeros_like(s))] if name=='new_id' else [('strong',s,a['weak'].astype(np.complex128)),('weak',a['weak'].astype(np.complex128),s)]
        for target_label,target,other in targets:
            y=counterfactual_inputs(target,other,j,e)
            assert np.max(np.abs(y[:,3]-y[:,2]-target))<1e-10
            for tag,control in frozen['methods'].items():
                assert sha(ROOT/'outputs'/tag/'model.pt')==control['checkpoint_sha256']
                c=predict(tag,y)
                gain=complex(control['gain_real'],control['gain_imag'])
                result,parents=aggregate(target,other,j,e,c,gain,ids)
                rows.append(dict(role=name,target=target_label,tag=tag,**result))
                parent_errors[(name,target_label,tag)]=parents
            print('Completed frozen inference:',name,target_label,flush=True)
    comparisons=[]
    for name,target in [('new_id','single'),('two_target','strong'),('two_target','weak')]:
        for seed in (17,29):
            baseline=frozen['selected_controls'][str(seed)]
            br=next(r for r in rows if (r['role'],r['target'],r['tag'])==(name,target,baseline))
            cr=next(r for r in rows if (r['role'],r['target'],r['tag'])==(name,target,f'B4_seed{seed}'))
            comparisons.append(dict(role=name,target=target,seed=seed,control=baseline,
                                    control_u_nmse=br['u_nmse'],candidate_u_nmse=cr['u_nmse'],
                                    relative_improvement=1-cr['u_nmse']/br['u_nmse'],damage=cr['damage'],
                                    severe_parent_rate=cr['severe_parent_rate'],
                                    empirical_budget_met=bool(cr['damage']<=.05),tail_met=bool(cr['severe_parent_rate']<=.1),
                                    paired_parent_bootstrap=bootstrap_difference(parent_errors[(name,target,f'B4_seed{seed}')],parent_errors[(name,target,baseline)])))
    repeated=all(r['relative_improvement']>0 and r['empirical_budget_met'] and r['tail_met'] for r in comparisons if r['role']=='new_id')
    weak_safe=all(r['empirical_budget_met'] and r['tail_met'] for r in comparisons if r['target']=='weak')
    value=dict(completed_utc=now(),freeze_sha256=sha(ROOT/'DEV_CONFIRMATION_FREEZE.json'),rows=rows,comparisons=comparisons,
               new_id_all_conditions_met=repeated,weak_target_conditions_met=weak_safe,
               no_training=True,no_retuning=True,final_test_opened=False,roles_are_development=True)
    write(final,value)
    report(value)
    print('Frozen development check complete; ID conditions:',repeated,'weak target conditions:',weak_safe,flush=True)


def report(value):
    lines=['# 冻结后的开发核验','', '模型、损失和增益全部冻结于原 DEV；无新增训练，无新角色上调参。以下均计入开发曝光，不是最终 TEST。',
           '', '| 新角色 | 目标 | seed | 对照 U-NMSE | B4 U-NMSE | 相对改善 | B4 实际损伤 | 严重损伤 parent 比例 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in value['comparisons']:
        lines.append(f"| {r['role']} | {r['target']} | {r['seed']} | {r['control_u_nmse']:.6f} | {r['candidate_u_nmse']:.6f} | {r['relative_improvement']:.2%} | {r['damage']:.6f} | {r['severe_parent_rate']:.2%} |")
    lines+=['',f"新 ID 全部预定经验条件通过：{value['new_id_all_conditions_met']}。",f"弱目标损伤与尾部条件通过：{value['weak_target_conditions_met']}。",'',
            '双目标的完整 U-NMSE 在 strong/weak 两行相同；变化的是移除哪个目标的差分与相应目标自己的归一化能量。强目标不能掩盖弱目标损伤。',
            '', '所有六个模型、幅度分布与圆相位诊断在 evidence/DEV_CONFIRMATION_RESULTS.json。禁止根据这里的结果重新缩放增益或宣称预算有统计保证。',
            '', '该检查只说明在已披露合成分布上的经验迁移表现；多目标 J 来源于主目标生成器，目标移除是固定 J/E 的干预诊断。全文查新与发表基线缺口不因此解决。']
    (ROOT/'PAPER2_DEV_CONFIRMATION.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
