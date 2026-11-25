"""Seal the completed v2 CPU review and P0 six-cell reference, retaining pause."""
import time
from pathlib import Path

from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_v2_preflight import OUT, finish
from tools.durable_io import atomic_write, process_alive


def run():
    started = now()
    while True:
        status = read(OUT / "STATUS.json")
        if status["stage"] == "FAILED":
            raise RuntimeError("v2 CPU preflight failed")
        if status["stage"] == "COMPLETE":
            break
        if not process_alive(status.get("pid")):
            raise RuntimeError("v2 CPU preflight exited without a receipt")
        time.sleep(30)
    summary = read(OUT / "SUMMARY.json")
    before = sha(OUT / "SUMMARY.json")
    expected = summary["substitution_summary_sha256"]
    snapshot = WORK / "evidence/outputs/p2_head_substitution_audit/admission_snapshot/SUMMARY.json"
    current = WORK / "evidence/outputs/p2_head_substitution_audit/SUMMARY.json"
    if expected == sha(current):
        admission = current
    elif snapshot.exists() and expected == sha(snapshot):
        admission = snapshot
    else:
        raise ValueError("S4 admission evidence is not bound to its original or refined receipt")
    summary["substitution_admission_snapshot"] = admission.relative_to(WORK).as_posix()
    summary["substitution_refinement_sha256"] = sha(current)
    finish(summary)
    p0 = read(WORK / "evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")
    if summary["G-v2-4"] == "PASS":
        methods = {}
        for method in ("unet_original", "unet_tail", "correction_original", "correction_tail"):
            methods[method] = p0["aligned_historical_four_cell"][method]["seed_averaged"]
        for method, values in p0["seed_averaged"].items():
            methods[method] = values
        summary["p0_six_cell"] = methods
        report = (OUT / "REPORT.md").read_text(encoding="utf8")
        lines = [report.rstrip(), "", "## P0 六格收尾", "",
                 f"锁定坐标 **{p0['coordinates']}**。",
                 "", "| 模型 | 虚警 | Pd | 损伤 | SJRIF dB | NMSE | 无扰 NMSE |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for method, values in methods.items():
            row = values["global"]
            keys = ("frame_pfa_jammer", "pd", "damage", "sjrif_db", "u_nmse", "no_jammer_pooled_nmse")
            lines.append(f"| {method} | " + " | ".join(f"{row[key]['mean']:.6g}" for key in keys) + " |")
        lines += ["", "全部六格按三种子平均和 parent 聚类区间归档在 SUMMARY.json。"]
        atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "SUMMARY.json", summary)
    receipt = {**provenance(__file__), "started_utc": started, "ended_utc": now(),
               "cpu_preflight_original_summary_sha256": before, "final_summary_sha256": sha(OUT / "SUMMARY.json"),
               "p0_coordinates": summary["p0_coordinates"], "G-v2-4": summary["G-v2-4"],
               "formal_gpu_training_paused": True}
    atomic_json(OUT / "FINAL_SEAL.json", receipt)
    atomic_json(OUT / "STATUS.json", {**status, "summary_sha256": sha(OUT / "SUMMARY.json"),
                                    "final_seal_sha256": sha(OUT / "FINAL_SEAL.json")})


if __name__ == "__main__":
    run()
