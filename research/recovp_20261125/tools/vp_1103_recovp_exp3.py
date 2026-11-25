"""Conditional parent-collated 2x2 risk experiment; never bypasses Exp1 admission."""
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,now,read,sha,provenance
from vp_1103_recovp import ReCoVPNet
from vp_1103_recovp_cvar import (parent_batches,record_risk,parent_losses,ru_cvar,two_pass_weights,
                               RISK_WEIGHT,ALPHA,TEMPERATURE,self_test)
from vp_1103_snr_jsr import frame_axes,summarize,DEFINITION,cases
from vp_1103_p0 import evaluate
from paper2.train_1103 import make_batch,load_arrays
from paper2.train_256_8seed import optimizer_for,learning_rate,seed_rng,rng_snapshot,restore_rng
from tools.correction_tail_study import objective
from tools.correction_tail_study.train import FORMAT,choose_resume,publish
from tools.unet_ablation.models import make_model
from tools.unet_ablation.runtime import configure_precision,user_control
from tools.durable_io import atomic_write,ProcessLock

OUT=WORK/"evidence/outputs/recovp_cvar_2x2"
PRIVATE=WORK/"outputs/recovp_cvar_2x2"
CORE=("frame_pfa_jammer","pd","damage","sjrif_db","u_nmse","no_jammer_pooled_nmse")

def one(backbone,risk,seed,config,dev,training,update):
    name=f"{backbone}-{'CVaR' if risk else 'Lnew'}";folder=PRIVATE/name/str(seed)
    with ProcessLock(folder/"PROCESS.lock"):
        complete=OUT/name/f"seed_{seed}.json"
        if complete.exists():
            row=read(complete)
            if sha(folder/"epoch_256.pt")!=row["checkpoint_sha256"]:raise ValueError("Risk checkpoint changed")
            return row
        configure_precision();torch.set_num_threads(2);seed_rng(seed)
        model=(make_model("no_dechirp") if backbone=="P0" else ReCoVPNet(config["extractor"],share_mode=config["best_share_mode"])).cuda()
        if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in model.modules()):
            raise RuntimeError("Exact two-pass CVaR requires deterministic, non-running-statistics forwards")
        optimizer=optimizer_for(model)
        threshold=torch.nn.Parameter(torch.tensor(0.,device="cuda")) if risk else None
        if risk:optimizer.add_param_group({"params":[threshold],"weight_decay":0.})
        identity={"run":"recovp_cvar_2x2","cell":name,"seed":seed,"config_sha256":sha(PRIVATE/"CONFIG.json")}
        saved=choose_resume(folder,identity);start=0
        if saved:
            model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"]);restore_rng(saved["rng"]);start=saved["epoch"]
            if risk:
                with torch.no_grad():threshold.copy_(saved["threshold"])
        def save(epoch,record):
            return publish(folder,{"format":FORMAT,"identity":identity,"seed":seed,"epoch":epoch,"global_step":epoch*48,
                "model":model.state_dict(),"optimizer":optimizer.state_dict(),"threshold":threshold.detach() if risk else None,
                "rng":rng_snapshot(),"epoch_record":record})
        if not saved:save(0,None)
        objective.ARM="tail"
        for epoch in range(start+1,257):
            user_control(PRIVATE);model.train();totals=[]
            for step,parents in enumerate(parent_batches(epoch,seed)):
                records=(parents[:,None]*3+np.arange(3)).reshape(-1)
                for group in optimizer.param_groups:group["lr"]=learning_rate(epoch,step)
                optimizer.zero_grad(set_to_none=True)
                # A complete three-family parent loss is measured before every update, in every cell.
                measured=[]
                with torch.no_grad():
                    for lo in range(0,len(records),8):
                        ids=records[lo:lo+8];batch=make_batch(training,ids,"cuda")
                        correction=model(batch["y"])
                        measured.append(record_risk(correction,batch,len(ids)))
                losses=parent_losses(torch.cat(measured))
                if risk:
                    weights,tgrad=two_pass_weights(losses,threshold.detach())
                    risk_value=float(ru_cvar(losses,threshold.detach()))
                    threshold.grad=RISK_WEIGHT*tgrad
                else:
                    weights=torch.zeros_like(losses);risk_value=0.
                base_value=0.
                for lo in range(0,len(records),8):
                    ids=records[lo:lo+8];batch=make_batch(training,ids,"cuda")
                    correction=model(batch["y"]).reshape(-1,4,2,1800)
                    terms=objective.study_objective(correction,batch,len(ids))
                    loss=terms["total"]*len(ids)/len(records)
                    if risk:
                        index=torch.arange(lo,lo+len(ids),device="cuda")//3
                        loss=loss+RISK_WEIGHT*(record_risk(correction,batch,len(ids))*weights[index]).sum()/len(records)
                    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite CVaR cell loss")
                    loss.backward();base_value+=float(terms["total"].detach())*len(ids)/len(records)
                parameters=[*model.parameters(),*([threshold] if risk else [])]
                torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);optimizer.step()
                totals.append({"base":base_value,"cvar":risk_value,"threshold":float(threshold.detach()) if risk else None,
                               "parents":len(parents),"records":len(records)})
            record={"epoch":epoch,"seed":seed,"cell":name,"base_loss":float(np.mean([r['base'] for r in totals])),
                    "cvar":float(np.mean([r['cvar'] for r in totals])),"threshold":totals[-1]["threshold"],"optimizer_updates":48}
            digest=save(epoch,record)
            update(stage="EXP3_TRAINING",cell=name,seed=seed,epoch=epoch,last_committed_epoch=epoch,checkpoint_sha256=digest)
        metrics=evaluate(model.eval(),dev)
        raw=folder/"dev_metrics.npz"
        atomic_write(raw,lambda f:np.savez_compressed(f,parent_id=dev["parent_id"],**metrics))
        row={"seed":seed,"cell":name,"epoch":256,"metrics":summarize(metrics,frame_axes(dev)),
             "worst_cases":cases(metrics,dev["parent_id"],frame_axes(dev)),"checkpoint_sha256":sha(folder/"epoch_256.pt"),
             "private_metrics_sha256":sha(raw),"threshold":float(threshold.detach()) if risk else None,
             "risk_auxiliary_parameters":1 if risk else 0}
        atomic_json(complete,row)
        del model,optimizer,training;torch.cuda.empty_cache()
        return row

def run(config,dev,formal_rows,update):
    status={**provenance(__file__,WORK/"tools/vp_1103_recovp_cvar.py"),"started_utc":now(),"stage":"RUNNING"}
    atomic_json(OUT/"STATUS.json",status)
    test=self_test();atomic_json(OUT/"MATH_TEST.json",test)
    if test["status"]!="PASS":raise ValueError("CVaR gradient identity failed")
    soft=formal_rows["ReCoVP-SoftShare"]
    def frontier(arm):
        return tuple(float(np.mean([r["metrics"]["global"][key]["mean"] for r in formal_rows[arm].values()]))
                     for key in ("frame_pfa_jammer","damage","u_nmse"))
    use_soft=(len(soft)==3 and all(r.get("engineering_ok",False) for r in soft.values()) and
              all(r["metrics"]["global"]["no_jammer_pooled_nmse"]["mean"]<=5e-4 for r in soft.values()) and
              frontier("ReCoVP-SoftShare")<frontier("ReCoVP-Base"))
    risk_config={**config,"best_share_mode":"soft_subspace" if use_soft else "none","parent_collation":"512 parents split into 48 batches, full 3 families",
                 "alpha":ALPHA,"temperature":TEMPERATURE,"risk_weight":RISK_WEIGHT,"initialization":"fresh, same seed in all paired cells",
                 "best_backbone_rule":"Complete 3-seed models only; no-jammer/engineering eligible; lexicographic mean Pfa, damage, NMSE"}
    cfg=PRIVATE/"CONFIG.json"
    if cfg.exists():
        if read(cfg)!=risk_config:raise ValueError("Risk protocol changed")
    else:atomic_json(cfg,risk_config)
    train_path=WORK/config["train"]["archive"]
    training=load_arrays(train_path)
    rows={}
    def execute(seed):
        for backbone in ("P0","ReCoVP"):
            for risk in (False,True):
                name=f"{backbone}-{'CVaR' if risk else 'Lnew'}"
                row=one(backbone,risk,seed,risk_config,dev,training,update)
                rows.setdefault(name,{})[str(seed)]=row
    execute(17)
    axes=frame_axes(dev);signals=[];safety=True;paired={}
    for backbone in ("P0","ReCoVP"):
        matrices=[]
        for risk in (False,True):
            name=f"{backbone}-{'CVaR' if risk else 'Lnew'}"
            with np.load(PRIVATE/name/"17/dev_metrics.npz",allow_pickle=False) as z:matrices.append({key:z[key] for key in CORE})
        delta=summarize({key:matrices[1][key]-matrices[0][key] for key in CORE},axes)
        paired[backbone]=delta
        signals.append(any(delta["global"][key]["parent_bootstrap_ci95"][1]<0 for key in ("frame_pfa_jammer","damage","u_nmse")))
        safety &= matrices[1]["frame_pfa_jammer"].mean()<=.125 and matrices[1]["damage"].mean()<=.065 and matrices[1]["no_jammer_pooled_nmse"].mean()<=5e-4
    if any(signals) and safety:
        execute(29);execute(41)
    aggregate={}
    for name,seeds in rows.items():
        matrices=[]
        for seed in seeds:
            with np.load(PRIVATE/name/seed/"dev_metrics.npz",allow_pickle=False) as z:matrices.append({key:z[key] for key in CORE})
        aggregate[name]={"seeds":seeds,"seed_average":summarize({key:np.mean([m[key] for m in matrices],0) for key in CORE},axes)}
    summary={"cells":aggregate,"paired_seed17":paired,"extended_seeds":bool(any(signals) and safety),"config":risk_config,
             "scope":"All 4 cells freshly trained under identical parent collation; old P0 row is not reused as an Exp3 control"}
    atomic_json(OUT/"SUMMARY.json",summary)
    lines=["# ReCoVP 2×2 parent CVaR 对照","",DEFINITION,"",
           "四格采用同一 parent 批次与固定 48 次更新/轮；每个 parent 的三类干扰先聚合，不当作独立样本。",
           "采用全 parent 第一遍无梯度评估及第二遍解析权重回传，计算的是平滑 RU 目标的精确一阶梯度。",
           "", "| 单元 | seed数 | 虚警 | Pd | 损伤 |","|---|---:|---:|---:|---:|"]
    for name,row in aggregate.items():
        m=row["seed_average"]["global"]
        lines.append(f"| {name} | {len(row['seeds'])} | {m['frame_pfa_jammer']['mean']:.5%} | {m['pd']['mean']:.5%} | {m['damage']['mean']:.7g} |")
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
    atomic_json(OUT/"STATUS.json",{**status,"stage":"COMPLETE","ended_utc":now(),"summary_sha256":sha(OUT/"SUMMARY.json")})
    from vp_1103_recovp_stress import run as stress
    stress(config,dev,formal_rows)
    from benchmark_latency_unified import run_recovp
    run_recovp()
    from vp_1103_recovp_score import report
    report(formal_rows,"AWAIT_GATE_RECOVP_REVIEW",{"risk_summary":"evidence/outputs/recovp_cvar_2x2/SUMMARY.json"})
