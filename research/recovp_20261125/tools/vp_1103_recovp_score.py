"""SNR/JSR-only fixed-epoch DEV scoring for the v2.1 formal screen."""
import numpy as np
from vp_1103_common import WORK, atomic_json, read, sha
from vp_1103_p0 import evaluate
from vp_1103_p2_evaluate import failure_details
from vp_1103_recovp_mechanisms import mechanism_metrics
from vp_1103_snr_jsr import DEFINITION, frame_axes, summarize, cases
from vp_1103_references import reference_metrics
from tools.durable_io import atomic_write
from vp_1103_recovp_drift import three_layers
from vp_1103_recovp_math import math_checks,selected_waves
from vp_1103_p2_preflight import subset
from vp_1103_recovp_failures import failure_heads

OUT=WORK/"evidence/outputs/recovp_formal"
PRIVATE=WORK/"outputs/recovp_formal"
CORE=("frame_pfa_jammer","pd","damage","sjrif_db","u_nmse","no_jammer_pooled_nmse")

def anchor_retention(model,data):
    from vp_1103_geometry import recover_geometry, INTEGER_FIELDS, FREQUENCY_FIELD
    from vp_1103_replay_truth import replay_components
    from vp_1103_recovp_extractors import extract_numpy
    cache=PRIVATE/"dev_geometry.npz"
    if cache.exists():
        binding=read(cache.with_suffix(".json"))
        if sha(cache)!=binding["sha256"]:raise ValueError("DEV geometry cache changed")
        with np.load(cache,allow_pickle=False) as z:
            if not np.array_equal(z["parent_id"],data["parent_id"]):raise ValueError("DEV geometry roster differs")
            geometry=z["geometry"]
    else:
        import vp_1103_geometry as geometry_module
        original_directory=geometry_module.DATA
        try:
            geometry_module.DATA=PRIVATE
            records=recover_geometry(WORK/"data/vp_1103/dev.npz",data)
        finally:
            geometry_module.DATA=original_directory
        geometry=np.full((256,3,8,7),np.nan)
        for index in np.ndindex(records.shape):
            if records[index] is not None:
                geometry[index]=[records[index][key] for key in (*INTEGER_FIELDS,FREQUENCY_FIELD)]
        atomic_write(cache,lambda f:np.savez_compressed(f,parent_id=data["parent_id"],geometry=geometry))
        atomic_json(cache.with_suffix(".json"),{"sha256":sha(cache),"scope":"DEV post-training diagnosis only"})
    result={key:np.zeros((256,3)) for key in ("anchor_energy_recall_2_samples","true_jammer_hit_retention","joint_hit_energy_recall")}
    for parent in range(256):
        for family in range(3):
            s,j,e=[data[key][parent,family].astype(np.complex128) for key in ("s","j","epsilon")]
            anchors=extract_numpy(np.concatenate((j+e,s+j+e)),model.extractor_config)
            hit_energy,retained_energy,total_energy=0.,0.,0.
            for pulse in range(8):
                truth=replay_components(float(data["target_delay_samples"][parent]),float(data["doppler_hz"][parent]),family,geometry[parent,family,pulse])
                if not truth:continue
                tau=np.array([c["phase_delay"] for c in truth]);energy=np.array([c["energy"] for c in truth])
                hit=abs(tau[:,None]-anchors[pulse]).min(-1)<=2
                retained=hit&(abs(tau[:,None]-anchors[pulse+8]).min(-1)<=2)
                hit_energy+=float(energy@hit);retained_energy+=float(energy@retained);total_energy+=float(energy.sum())
            result["anchor_energy_recall_2_samples"][parent,family]=hit_energy/total_energy
            result["joint_hit_energy_recall"][parent,family]=retained_energy/total_energy
            result["true_jammer_hit_retention"][parent,family]=retained_energy/hit_energy if hit_energy else np.nan
    return result

def references(data,seed):
    path=WORK/"outputs/p0_dechirp_tail_2x2/P0-NoDechirp-Tail"/str(seed)/"dev_parent_metrics.npz"
    row=read(WORK/"evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")["new_arms"]["P0-NoDechirp-Tail"][str(seed)]
    if sha(path)!=row["parent_metrics_sha256"]:raise ValueError("M2 binding failed")
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z["parent_id"],data["parent_id"]):raise ValueError("M2 roster failed")
        m2={key:z[key] for key in CORE}
    m1,binding=reference_metrics("unet_tail",data["parent_id"])
    return {"P0-NoDechirp-Tail":m2,"UNet-Tail":m1[seed]}, {"m2":row["checkpoint_sha256"],"m1":binding[str(seed)]}

def evaluate_seed(model,data,seed,arm,coordinates,checkpoint):
    folder=PRIVATE/arm/str(seed)
    # Full restoration metrics on GPU, then exact fixed-operator review on CPU.
    metrics=evaluate(model.eval(),data)
    model=model.cpu()
    mechanism,algebra,permutation=mechanism_metrics(model,data)
    metrics.update(mechanism)
    metrics.update(anchor_retention(model,data))
    layers,layer_errors=three_layers(model,data)
    math_result=math_checks(model,selected_waves(subset(data,np.linspace(0,255,16).astype(int))))
    axes=frame_axes(data)
    raw=folder/"dev_metrics.npz"
    atomic_write(raw,lambda f:np.savez_compressed(f,parent_id=data["parent_id"],**metrics))
    baselines,binding=references(data,seed)
    comparisons={name:summarize({key:metrics[key]-baseline[key] for key in CORE},axes) for name,baseline in baselines.items()}
    summary=summarize(metrics,axes)
    pfa,damage,keep=(summary["global"][key]["mean"] for key in ("frame_pfa_jammer","damage","no_jammer_pooled_nmse"))
    pfa_delta=comparisons["P0-NoDechirp-Tail"]["global"]["frame_pfa_jammer"]["mean"]
    high=comparisons["P0-NoDechirp-Tail"]["jsr/>=30"]["frame_pfa_jammer"]["mean"]
    damage_delta=comparisons["P0-NoDechirp-Tail"]["global"]["damage"]["parent_bootstrap_ci95"]
    breaker=pfa>.125 or damage>.065 or keep>5e-4
    positive=(pfa_delta<=-.01 and damage<=.048) or (pfa_delta<=.005 and
              ((damage<=.032 and damage_delta[1]<0) or (high is not None and high<=-.02)))
    failures=cases(metrics,data["parent_id"],axes)
    atoms=failure_details(model,data,failures,folder)
    extra_heads=failure_heads(model,data,failures,folder)
    expected_equivariant=getattr(model,"equivariant",True)
    engineering_ok=math_result["status"]=="PASS" if expected_equivariant else True
    if arm=="ReCoVP-SoftShare":
        base_path=PRIVATE/"ReCoVP-Base"/str(seed)/"dev_metrics.npz"
        if not base_path.exists():base_path=PRIVATE/"ReCoVP-Base/17/dev_metrics.npz"
        with np.load(base_path,allow_pickle=False) as z:
            base={key:z[key] for key in z.files if key!="parent_id"}
        changes={key:metrics[key]-base[key] for key in metrics if key in base}
        family_deltas=summarize(changes,axes)
        from vp_1103_common import distribution
        improvement={key:distribution((metrics[key][:,1:]-base[key][:,1:]).mean(1)) for key in
                     ("u_nmse","damage","frame_pfa_jammer","removed_noise_gain_frobenius_squared")}
        benefit=any(row["parent_bootstrap_ci95"][1]<0 for row in improvement.values())
        safe=all(family_deltas["direct/all"][key]["mean"]<=0 for key in ("frame_pfa_jammer","damage","u_nmse"))
        safe &= family_deltas["global"]["no_jammer_pooled_nmse"]["mean"]<=0
        positive=benefit and safe
        comparisons["ReCoVP-Base"]=family_deltas
    row={"seed":seed,"epoch":256,"arm":arm,"metrics":summary,"paired_comparisons":comparisons,
         "baseline_bindings":binding,"private_metrics_sha256":sha(raw),"checkpoint_sha256":sha(checkpoint),
         "worst_cases":failures,"diagnosis":atoms,"estimated_head_diagnosis":extra_heads,"operator_identity_error":algebra,"permutation_error":permutation,
         "continue_other_seeds":bool(positive and not breaker and engineering_ok),"circuit_breaker":bool(breaker),
         "math_checks":math_result,"equivariance_expected":expected_equivariant,"engineering_ok":engineering_ok,
         "three_layer_drift":{layer:summarize(values,axes) for layer,values in layers.items()},"three_layer_identity_error":layer_errors}
    atomic_json(OUT/arm/f"seed_{seed}.json",row)
    return row

def report(rows,route,extra=None):
    with np.load(WORK/"data/vp_1103/dev.npz",allow_pickle=False) as z:
        data={key:z[key] for key in ("parent_id","snr_db","jsr_db")}
    axes=frame_axes(data)
    base=read(WORK/"evidence/outputs/p0_snr_jsr_reference/SUMMARY.json")
    methods=dict(base["methods"])
    for arm,seeds in rows.items():
        if not seeds:continue
        matrices=[]
        for seed,row in seeds.items():
            path=PRIVATE/arm/seed/"dev_metrics.npz"
            if sha(path)!=row["private_metrics_sha256"]:raise ValueError("Formal metrics changed")
            with np.load(path,allow_pickle=False) as z:
                matrices.append({key:z[key] for key in z.files if key!="parent_id"})
        average={key:np.mean([m[key] for m in matrices],0) for key in matrices[0]}
        methods[arm]={"seeds":seeds,"seed_average":summarize(average,axes),"completed_seeds":list(seeds),
                      "worst_cases":cases(average,data["parent_id"],axes)}
    summary={"route":route,"coordinates":"RAW_IQ","methods":methods,"extra":extra or {},"confirmation_unlocked":False,
             "seed_status":{arm:{str(s):"COMPLETE" if str(s) in r else "NOT_STARTED_GATE" for s in (17,29,41)} for arm,r in rows.items()}}
    if extra and "failed_seed" in extra:
        failure=extra["failed_seed"]
        summary["seed_status"][failure["arm"]][str(failure["seed"]) ]="FAILED"
    if "ReCoVP-NoEquiv" in summary["seed_status"]:
        for seed in ("29","41"):summary["seed_status"]["ReCoVP-NoEquiv"][seed]="NOT_IN_PROTOCOL"
    atomic_json(OUT/"SUMMARY.json",summary)
    lines=["# SmoothVP-Net v2.1 DEV Gate-P2","",DEFINITION,"",f"路由 **{route}**；独立确认未解锁。",
           "","| 分层 | 模型 | 虚警 | Pd | 损伤 | SJRIF dB | 端点 NMSE | 保持 NMSE |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for label in methods["UNet-Tail"]["seed_average"]:
        for name,row in methods.items():
            m=row["seed_average"][label]
            values=["NA" if m[key]["mean"] is None else f"{m[key]['mean']:.6g}" for key in CORE]
            lines.append(f"| {label} | {name} | "+" | ".join(values)+" |")
    lines += ["","逐种子、parent 聚类区间与算子机制见 SUMMARY.json。未启动的种子明确标记，不计入均值。"]
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
