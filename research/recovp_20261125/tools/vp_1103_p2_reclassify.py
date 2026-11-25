"""Correct the extra percentage hurdle using the user's significant-fit criterion."""
import shutil
from pathlib import Path

import numpy as np

from vp_1103_common import WORK, atomic_json, frame_jnr, now, provenance, read, sha, write_cases
from vp_1103_p2_admission import fit_evidence
from vp_1103_p2_preflight import load, subset, OVERFIT_THRESHOLDS
from tools.durable_io import atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_preflight"


def run():
    # Preserve the previous decisions and the exact optimizer implementation.
    archive = OUT / "before_admission_correction"
    archive.mkdir(exist_ok=False)
    for name in ("STATUS.json", "SUMMARY.json", "REPORT.md"):
        shutil.copy2(OUT / name, archive / name)
    summary = read(OUT / "SUMMARY.json")
    status = read(OUT / "STATUS.json")
    script = WORK / "tools/vp_1103_p2_extend.py"
    original = script.read_text(encoding="utf8").replace(
        "from vp_1103_p2_admission import fit_evidence\n", "")
    original = original.replace('        fit = fit_evidence(before, after, jnr)\n        passed = fit["passes"]',
                                "        passed = residual_ratio <= .5 and pfa_gain >= .05")
    original = original.replace("                   fit_evidence=fit,\n", "")
    source_copy = PRIVATE / "extension_original_source.py"
    source_copy.write_text(original, encoding="utf8", newline="\n")
    if sha(source_copy) != status["source_sha256"]["tools/vp_1103_p2_extend.py"]:
        raise ValueError("Cannot bind the actual CPU optimizer source")
    training, _ = load("train")
    for coordinates, row in summary["coordinates"].items():
        with np.load(WORK / row["private_metrics_file"], allow_pickle=False) as z:
            indices = np.searchsorted(training["parent_id"], z["parent_id"])
            jnr = frame_jnr(subset(training, indices))
            before = {key.removeprefix("before__"): z[key] for key in z.files if key.startswith("before__")}
            after = {key.removeprefix("after__"): z[key] for key in z.files if key.startswith("after__")}
            fit = fit_evidence(before, after, jnr)
        row["fit_evidence"] = fit
        row["checks"]["train16_expression"] = "PASS" if fit["passes"] else "FAIL"
        row["status"] = "PASS" if all(v == "PASS" for v in row["checks"].values()) else "FAIL"
    correction = {"utc": now(), "reason": "The 50% residual-reduction hurdle was an assistant-added "
                  "engineering target. The user required significant reduction. Admission now uses "
                  "the parent-paired 95% bootstrap evidence on the same final fixed-budget predictions.",
                  "new_optimization_steps": 0, "previous_summary_sha256": sha(archive / "SUMMARY.json"),
                  "source": provenance(__file__, WORK / "tools/vp_1103_p2_admission.py"),
                  "cpu_optimizer_source_archive": source_copy.relative_to(WORK).as_posix(),
                  "cpu_optimizer_source_sha256": sha(source_copy)}
    summary.update(thresholds=OVERFIT_THRESHOLDS, admission_correction=correction)
    atomic_json(OUT / "SUMMARY.json", summary)
    lines = ["# SmoothVP-Net CPU preflight", "",
             "| Coordinates | Real parameters | Initial clean NMSE | High-JNR parent relative residual change | 95% CI | Frame Pfa decrease | 95% CI | Result |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for coordinates, row in summary["coordinates"].items():
        fit = row["fit_evidence"]
        residual = fit["high_jnr_parent_relative_residual_change"]
        pfa = fit["frame_pfa_parent_decrease"]
        rlo, rhi = residual["parent_bootstrap_ci95"]
        plo, phi = pfa["parent_bootstrap_ci95"]
        lines.append(f"| {coordinates} | {row['health']['effective_real_parameters']} | "
                     f"{row['health']['clean_pooled_nmse']:.6g} | {residual['mean']:.2%} | "
                     f"{rlo:.2%}–{rhi:.2%} | {pfa['mean']:.2%} | {plo:.2%}–{phi:.2%} | {row['status']} |")
    lines += ["", "The high-JNR comparison contains eight parents. The false-alarm comparison contains all 16 TRAIN parents.",
              "All eight pulses and three interference families are retained within each parent.",
              "The numerical, initial-clean and connected-gradient checks pass in both coordinates.",
              "The fixed 3840-update CPU fit provides significant improvements but retains substantial residual error.",
              "The conditional oracle dictionary audit and this small training check do not establish a blind DEV win.",
              "", "## Admission correction", "",
              correction["reason"],
              "The original 192-step, 768-step and percentage-hurdle decisions remain archived.",
              "", "## TRAIN16 failure analysis", "",
              "Per-coordinate FIT_DIAGNOSIS reports and private dictionaries retain the localization and drift evidence."]
    for coordinates, row in summary["coordinates"].items():
        lines += ["", f"## {coordinates} worst parents", ""]
        write_cases(lines, row["worst_cases"])
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "STATUS.json", {**status, "summary_sha256": sha(OUT / "SUMMARY.json"),
                                    "admission_correction": correction, "updated_utc": now()})


if __name__ == "__main__":
    run()
