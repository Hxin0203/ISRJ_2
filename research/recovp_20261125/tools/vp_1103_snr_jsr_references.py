"""Re-bin cached fixed-epoch DEV metrics; no retraining or extractor selection."""
import numpy as np
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha, validate_role
from vp_1103_references import reference_metrics
from vp_1103_snr_jsr import DEFINITION, frame_axes, summarize, cases, protocol
from tools.durable_io import atomic_write

OUT=WORK/"evidence/outputs/snr_jsr_baselines"
def run():
    started=now()
    _,binding=validate_role("dev")
    with np.load(WORK/"data/vp_1103/dev.npz",allow_pickle=False) as z:
        data={key:z[key] for key in ("parent_id","snr_db","jsr_db")}
    axes=frame_axes(data)
    p0=read(WORK/"evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")
    unet,unet_binding=reference_metrics("unet_tail",data["parent_id"])
    methods={}
    for name in ("P0-NoDechirp-Tail","UNet-Tail"):
        matrices={}
        bindings={}
        for seed in (17,29,41):
            if name=="UNet-Tail":
                matrices[str(seed)]=unet[seed];bindings[str(seed)]=unet_binding[str(seed)]
            else:
                row=p0["new_arms"][name][str(seed)]
                path=WORK/"outputs/p0_dechirp_tail_2x2"/name/str(seed)/"dev_parent_metrics.npz"
                assert sha(path)==row["parent_metrics_sha256"]
                with np.load(path,allow_pickle=False) as z:
                    assert np.array_equal(z["parent_id"],data["parent_id"])
                    matrices[str(seed)]={key:z[key] for key in z.files if key!="parent_id"}
                bindings[str(seed)]={"checkpoint_sha256":row["checkpoint_sha256"],"metrics_sha256":sha(path)}
        average={key:np.mean([m[key] for m in matrices.values()],0) for key in matrices["17"]}
        methods[name]={"seeds":{seed:{"metrics":summarize(m,axes),"worst_cases":cases(m,data["parent_id"],axes),"binding":bindings[seed]} for seed,m in matrices.items()},
                       "seed_average":summarize(average,axes),"worst_cases":cases(average,data["parent_id"],axes)}
    summary={"methods":methods,"protocol":protocol(),"training_updates":0,"coordinates":"RAW_IQ","data":binding}
    atomic_json(OUT/"SUMMARY.json",summary)
    lines=["# SNR/JSR 同口径基线","",DEFINITION,"","不重训、不重新选 epoch，复用已核验的 256 DEV parent 固定末轮指标。",
           "","| 分层 | 模型 | 虚警 | Pd | 损伤 | parent |","|---|---|---:|---:|---:|---:|"]
    for stratum in methods["UNet-Tail"]["seed_average"]:
        for name,method in methods.items():
            m=method["seed_average"][stratum]
            show=lambda k:"NA" if m[k]["mean"] is None else f"{m[k]['mean']:.6g}"
            lines.append(f"| {stratum} | {name} | {show('frame_pfa_jammer')} | {show('pd')} | {show('damage')} | {m['pd']['parents']} |")
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
    atomic_json(OUT/"STATUS.json",{**provenance(__file__,WORK/"tools/vp_1103_snr_jsr.py"),"stage":"COMPLETE","started_utc":started,
             "ended_utc":now(),"training_updates":0,"data":binding,"summary_sha256":sha(OUT/"SUMMARY.json")})
if __name__=="__main__":run()
