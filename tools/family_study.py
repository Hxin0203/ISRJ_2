"""One bounded prospective eight-state study; no old results are overwritten."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from paper2.network import CorrectionNet
from paper2.data import iq
from paper2.run import setup,batches,fingerprint
from paper2.io import ROOT,read,write,sha,source_bindings,now
from paper2.science import bootstrap_difference
from .family_science import eight_states,family_loss,family_gain,family_metrics
from .family_data import load_family,components,training_tensors

OUT=ROOT/'outputs'/'family'
EVIDENCE=ROOT/'evidence'/'family'


def bindings():
    result=source_bindings()
    for file in ('family_science.py','family_data.py','family_study.py'):
        result['tools/'+file]=sha(Path(__file__).parent/file)
    return result


def freeze():
    value=dict(source=bindings(),contract_sha256=sha(ROOT/'FAMILY_STUDY_CONTRACT.md'),
               seeds=[17,29],methods=['B1','B2','B4'],steps=1200,batch_groups=16,lr=.001,
               frontier_budgets=[.01,.05,.1],operating_budget=.04,operating_tail=.05,
               confirmation_budget=.05,confirmation_tail=.1)
    path=ROOT/'FAMILY_STUDY_FREEZE.json'
    if path.exists():
        assert read(path)['freeze']==value,'family freeze changed'
    else:
        write(path,dict(frozen_utc=now(),freeze=value))
    return value


def step(model,opt,batch,method):
    opt.zero_grad(set_to_none=True)
    y=batch['y'];c=model(y.flatten(0,1)).reshape_as(y)
    losses=family_loss(c,batch['a'],batch['b'],batch['j'],batch['e'],method)
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    opt.step()
    return {k:float(v.detach()) for k,v in losses.items()}|dict(gradient_norm=float(norm))


def train(cfg):
    device=setup(991)
    eng=training_tensors(load_family('engineering'))
    if not (OUT/'ENGINEERING.json').exists():
        net=CorrectionNet().to(device);opt=torch.optim.Adam(net.parameters(),lr=.001)
        batch={k:v[:16].to(device) for k,v in eng.items()}
        start=time.perf_counter()
        for _ in range(21):
            loss=step(net,opt,batch,'B4')
        write(OUT/'ENGINEERING.json',dict(completed_utc=now(),updates=21,elapsed_seconds=time.perf_counter()-start,last_loss=loss))
    data=training_tensors(load_family('train'))
    dataset_sha=sha(ROOT/'data'/'family'/'train.npz')
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            tag=f'{method}_seed{seed}';folder=OUT/tag
            if (folder/'COMPLETE.json').exists():
                old=read(folder/'COMPLETE.json')
                assert old['source']==cfg['source'] and old['dataset_sha256']==dataset_sha
                assert old['checkpoint_sha256']==sha(folder/'model.pt')
                continue
            assert not (folder/'STARTED.json').exists(),'incomplete run requires investigation'
            device=setup(seed);model=CorrectionNet().to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            initial=fingerprint(model)
            write(folder/'STARTED.json',dict(started_utc=now(),tag=tag,source=cfg['source'],dataset_sha256=dataset_sha,
                                           freeze_sha256=sha(ROOT/'FAMILY_STUDY_FREEZE.json'),initial_parameters_sha256=initial))
            rng=np.random.default_rng(seed+10000);batch_hash=hashlib.sha256();history=[];updates=0;exposure=0
            start=time.perf_counter()
            while updates<1200:
                order=rng.permutation(len(data['a']))
                for offset in range(0,len(order),16):
                    if updates>=1200:break
                    idx=order[offset:offset+16];batch_hash.update(idx.tobytes())
                    loss=step(model,opt,{k:v[idx].to(device) for k,v in data.items()},method)
                    updates+=1;exposure+=len(idx)*8;history.append(dict(step=updates,**loss))
                    if updates==1 or updates%100==0:
                        status=dict(stage='FAMILY_TRAINING',tag=tag,step=updates,steps=1200,
                                    elapsed_seconds=time.perf_counter()-start,updated_utc=now(),pid=os.getpid(),loss=loss)
                        write(OUT/'STATUS.json',status);print(status,flush=True)
                    assert time.perf_counter()-start<900,'run time ceiling exceeded'
            payload=dict(state=model.state_dict(),tag=tag,source=cfg['source'],dataset_sha256=dataset_sha,
                         freeze_sha256=sha(ROOT/'FAMILY_STUDY_FREEZE.json'))
            temporary=folder/'model.tmp.pt';torch.save(payload,temporary);temporary.replace(folder/'model.pt')
            write(folder/'history.json',history)
            record=dict(completed_utc=now(),tag=tag,steps=updates,exposure=exposure,initial_parameters_sha256=initial,
                        batch_sequence_sha256=batch_hash.hexdigest(),source=cfg['source'],dataset_sha256=dataset_sha,
                        freeze_sha256=sha(ROOT/'FAMILY_STUDY_FREEZE.json'),checkpoint_sha256=sha(folder/'model.pt'),
                        elapsed_seconds=time.perf_counter()-start,final_loss=loss)
            write(folder/'COMPLETE.json',record)
    assert sha(ROOT/'data'/'family'/'train.npz')==dataset_sha
    for seed in cfg['seeds']:
        r=[read(OUT/f'{m}_seed{seed}'/'COMPLETE.json') for m in cfg['methods']]
        assert len({(x['initial_parameters_sha256'],x['batch_sequence_sha256'],x['exposure']) for x in r})==1
    write(OUT/'TRAINING_COMPLETE.json',dict(completed_utc=now(),runs=6,source=bindings(),dataset_sha256=dataset_sha))


def infer(tag,a,role,cfg):
    folder=OUT/tag;record=read(folder/'COMPLETE.json')
    assert record['source']==cfg['source'] and record['checkpoint_sha256']==sha(folder/'model.pt')
    device=setup(int(tag.split('seed')[1]));net=CorrectionNet().to(device)
    payload=torch.load(folder/'model.pt',map_location=device,weights_only=False)
    assert payload['source']==cfg['source'] and payload['dataset_sha256']==record['dataset_sha256']
    net.load_state_dict(payload['state']);net.eval()
    y,_=eight_states(*components(a));y=iq(y);chunks=[]
    with torch.inference_mode():
        for start in range(0,len(y),16):
            batch=y[start:start+16].to(device)
            c=net(batch.flatten(0,1)).reshape_as(batch).cpu().numpy()
            chunks.append(c[:,:,0]+1j*c[:,:,1])
    c=np.concatenate(chunks)
    path=folder/f'{role}_predictions.npz';np.savez_compressed(path,correction=c)
    write(folder/f'{role}_PREDICTIONS.json',dict(created_utc=now(),sha256=sha(path),source=bindings(),
          data_sha256=sha(ROOT/'data'/'family'/f'{role}.npz'),checkpoint_sha256=record['checkpoint_sha256']))
    return c


def evaluate(cfg):
    assert read(OUT/'TRAINING_COMPLETE.json')['source']==bindings()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    if (EVIDENCE/'RESULTS.json').exists():
        raise RuntimeError('completed family study is immutable; do not reselect')
    assert not (ROOT/'data'/'family'/'confirm.npz').exists(),'confirmation opened before DEV freeze'
    a=load_family('dev');aa,bb,j,e=components(a);ids=a['scene_id']
    frontiers=[];operating={}
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            tag=f'{method}_seed{seed}';c=infer(tag,a,'dev',cfg)
            for budget in cfg['frontier_budgets']:
                for complex_control in (False,True):
                    gain=family_gain(aa,bb,j,e,c,ids,budget,complex_control=complex_control)
                    result,_=family_metrics(aa,bb,j,e,c,gain,ids)
                    assert max(result[t]['damage'] for t in ('strong','weak'))<=budget+1e-10
                    frontiers.append(dict(tag=tag,budget=budget,family='complex' if complex_control else 'real',
                                          gain_real=float(np.real(gain)),gain_imag=float(np.imag(gain)),**result))
            gain=family_gain(aa,bb,j,e,c,ids,.04,tail_limit=.05,complex_control=True)
            result,_=family_metrics(aa,bb,j,e,c,gain,ids)
            assert all(result[t]['damage']<=.04+1e-10 and result[t]['severe_parent_rate']<=.05 for t in ('strong','weak'))
            operating[tag]=dict(gain_real=float(np.real(gain)),gain_imag=float(np.imag(gain)),dev=result,
                                checkpoint_sha256=sha(OUT/tag/'model.pt'))
            print('Family DEV complete',tag,flush=True)
    selected={str(seed):min([f'{m}_seed{seed}' for m in ('B1','B2')],key=lambda tag:operating[tag]['dev']['u_nmse']) for seed in cfg['seeds']}
    gain_freeze=dict(created_utc=now(),source=bindings(),dev_sha256=sha(ROOT/'data'/'family'/'dev.npz'),
                     methods=operating,selected_controls=selected,study_freeze_sha256=sha(ROOT/'FAMILY_STUDY_FREEZE.json'))
    write(EVIDENCE/'DEV_GAIN_FREEZE.json',gain_freeze)
    write(EVIDENCE/'DEV_FRONTIERS.json',frontiers)
    # Only now generate the prospective development confirmation role.
    a=load_family('confirm');aa,bb,j,e=components(a);ids=a['scene_id']
    rows={};parents={}
    keep,_=family_metrics(aa,bb,j,e,np.zeros((len(aa),8,aa.shape[-1]),np.complex64),0,ids)
    for tag,point in operating.items():
        c=infer(tag,a,'confirm',cfg)
        result,parent=family_metrics(aa,bb,j,e,c,complex(point['gain_real'],point['gain_imag']),ids)
        rows[tag]=result;parents[tag]=parent
        print('Family prospective confirmation complete',tag,flush=True)
    comparisons=[]
    for seed in cfg['seeds']:
        tag=f'B4_seed{seed}';baseline=selected[str(seed)];r=rows[tag]
        improvement=1-r['u_nmse']/rows[baseline]['u_nmse'];suppression=1-r['u_nmse']/keep['u_nmse']
        damage=max(r[t]['damage'] for t in ('strong','weak'));tail=max(r[t]['severe_parent_rate'] for t in ('strong','weak'))
        comparisons.append(dict(seed=seed,control=baseline,relative_improvement=improvement,suppression=suppression,
                                max_target_damage=damage,max_target_severe_parent_rate=tail,
                                paired_parent_bootstrap=bootstrap_difference(parents[tag],parents[baseline]),
                                passed=bool(improvement>=.05 and suppression>=.1 and damage<=.05 and tail<=.1)))
    passed=all(r['passed'] for r in comparisons)
    result=dict(completed_utc=now(),decision='PROCEED_TO_FORMAL_DESIGN' if passed else 'CLOSE_CANDIDATE',
                all_gates_passed=passed,comparisons=comparisons,confirmation=rows,keep=keep,source=bindings(),
                gain_freeze_sha256=sha(EVIDENCE/'DEV_GAIN_FREEZE.json'),final_test_opened=False,
                training_runs=6,prior_failed_confirmation_preserved=True,roles_are_development=True)
    write(EVIDENCE/'RESULTS.json',result)
    report(result)
    write(OUT/'COMPLETE.json',dict(completed_utc=now(),decision=result['decision'],results_sha256=sha(EVIDENCE/'RESULTS.json')))
    print('Family study decision:',result['decision'],flush=True)


def report(result):
    text=['# 双目标八状态：实际完成结果','',f"结论：**{result['decision']}**。六次新训练和冻结后的新开发确认已完成。",'',
          '骨干与第一轮完全相同。所有对照均使用八状态；候选只额外约束四条逐目标响应边。弱目标幅度比为 0.15—0.4，J/E 全程固定。',
          '', '| seed | 冻结对照 | 对照 U-NMSE | B4 U-NMSE | 相对改善 | 强目标损伤 | 弱目标损伤 | 最坏目标严重损伤 parent 比例 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['comparisons']:
        b=result['confirmation'][r['control']];c=result['confirmation'][f"B4_seed{r['seed']}"]
        text.append(f"| {r['seed']} | {r['control']} | {b['u_nmse']:.6f} | {c['u_nmse']:.6f} | {r['relative_improvement']:.2%} | {c['strong']['damage']:.6f} | {c['weak']['damage']:.6f} | {r['max_target_severe_parent_rate']:.2%} |")
    text+=['',f"KEEP U-NMSE：{result['keep']['u_nmse']:.6f}。",'',
           '工作点只在新 DEV 上选择：两目标分别平均损伤<=0.04、分别严重 parent<=5%，然后冻结并在新确认角色上按0.05/10%检查。所有对照具有完全相同规则。该余量与增益控制是共同实验控制，不作为新方法贡献。',
           '', '原单目标候选的冻结确认失败继续保留；本研究不是旧增益的事后重调。所有参数、数据角色、源码、checkpoint 和增益身份均有独立记录。',
           '', '该结果仍不是最终 TEST、风险保证或新颖性证明。两个 seed 只用于机制预研；正式研究需要至少三个 seed、发表基线、最近邻全文逐式差异和更真实的多目标/干扰模型。',
           '', '若本次门槛失败，按预注册关闭当前候选，不在本次会话继续增加结构、损失或分布版本。']
    (ROOT/'PAPER2_FAMILY_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cfg=freeze();lock=OUT/'PROCESS.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        train(cfg) if args.stage=='train' else evaluate(cfg)
    except BaseException as exc:
        write(OUT/'FAILURE.json',dict(time=now(),stage=args.stage,error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
