"""Review existing receipts and reports; never load arrays, weights, or models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wt, prior, out = (p.resolve() for p in (args.source_root, args.prior, args.out))
    out.mkdir(parents=True, exist_ok=False)
    prior_status = read_json(prior / "STATUS.json")
    manifest = read_json(prior / "INPUT_MANIFEST.json")["files"]
    results = read_json(prior / "EXISTING_RESULTS.json")
    checks, binary_records, source_records = [], [], []

    def check(label, actual, expected):
        checks.append({"check": label, "pass": actual == expected,
                       "actual": actual, "expected": expected})

    # Binary identities remain historical evidence: no binary file is opened here.
    allowed = {".py", ".json", ".md", ".txt", ".csv", ""}
    for relative, entry in manifest.items():
        path = Path(relative)
        if path.suffix.lower() not in allowed:
            binary_records.append({"path": relative, **entry,
                                   "verification": "PRIOR_RECEIPT_ONLY_NOT_REHASHED"})
            continue
        path = path if path.is_absolute() else wt / path
        path = path.resolve()
        if not (path.is_relative_to(wt) or path.is_relative_to(Path.cwd().resolve())):
            raise ValueError(f"Unexpected source outside workspace: {path}")
        actual = digest(path) if path.is_file() else None
        check(f"input_sha256:{relative}", actual, entry["sha256"])
        source_records.append({"path": relative, "sha256": actual,
                               "bytes": path.stat().st_size if path.is_file() else None})

    for name, entry in prior_status["outputs"].items():
        check(f"prior_output_sha256:{name}", digest(prior / name), entry["sha256"])

    gap = wt / "evidence/outputs/recovp_base_dev_gap_audit"
    gap_status, verification = (read_json(gap / n) for n in ("STATUS.json", "VERIFICATION.json"))
    for key, name in (("summary_sha256", "SUMMARY.json"),
                      ("core_summary_sha256", "CORE_SUMMARY.json"),
                      ("report_sha256", "REPORT.md")):
        check(f"gap_status_reference:{name}", digest(gap / name), gap_status[key])
        if key in verification:
            check(f"gap_verification_reference:{name}", digest(gap / name), verification[key])
    check("gap_status_byte_identity", digest(gap / "STATUS.json"),
          prior_status["original_gap_status_sha256"])
    check("data_receipt_identity", gap_status["dev"], verification["dev"])
    check("checkpoint_receipt_identity", gap_status["checkpoint_sha256"],
          verification["checkpoint_sha256"])
    queue = read_json(gap / "QUEUE_CLOSED.json")
    formal = wt / "evidence/outputs/recovp_formal_dev/SUMMARY.json"
    check("closed_queue_formal_summary", digest(formal), queue["source_summary_sha256"])
    check("completed_noequiv_seeds", queue["completed_noequiv_dev_seeds"], [17, 29, 41])
    check("recorded_scan_conditions", queue["scan_completed_conditions"], 0)
    check("recorded_research_backbone", queue["backbone_override"], "ReCoVP-Base")
    check("last_control_progress_field", gap_status["model"], "RSDR-Net")
    for relative in ("outputs/recovp_isrj1_suite/STOP", "outputs/recovp_isrj1_suite/PAUSE",
                     "outputs/unet_matched58943/STOP"):
        check(f"pause_marker:{relative}", (wt / relative).is_file(), True)
    semantics = (wt / "tools/vp_1103_recovp_dev_gap_audit.py").read_text(encoding="utf-8-sig")
    check("control_loop_contains_progress_update", "update(phase='CFAR_CONTROL',model=name)" in semantics, True)

    for row in results["single_seed_rows"]:
        score = read_json(wt / row["source"])
        check(f"archived_row:{row['model']}:{row['native_seed']}:identity",
              row["identity"], score["identity"]["method"])
        check(f"archived_row:{row['model']}:{row['native_seed']}:metrics",
              row["metrics"], score["metrics"]["global"])
    for row in results["existing_paired_interval_review"]:
        lo, hi = row["ci95"]
        check(f"interval_zero_classification:{row['comparison']}:{row['metric']}",
              row["interval_excludes_zero"], lo > 0 or hi < 0)

    # Recheck the text inputs after the review without executing any research entry point.
    for record in source_records:
        p = Path(record["path"])
        p = p if p.is_absolute() else wt / p
        check(f"input_unchanged:{record['path']}", digest(p), record["sha256"])
    failed = [x for x in checks if not x["pass"]]
    details = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "document and source-byte integrity only; no performance validation",
        "prior_archive_status_sha256": digest(prior / "STATUS.json"),
        "prior_reported_input_count": prior_status["protected_input_files"],
        "prior_reported_check_count": prior_status["integrity_checks"],
        "current_text_source_count": len(source_records),
        "binary_records_not_rehashed": binary_records,
        "current_check_count": len(checks), "failed_check_count": len(failed),
        "checks": checks,
    }
    # Large metric objects are already byte-checked against the prior manifest.
    # Keep only their equality outcome in this lightweight review receipt.
    for item in checks:
        if isinstance(item["actual"], dict) and len(str(item["actual"])) > 1000:
            item["actual"] = "IDENTICAL_TO_SOURCE" if item["pass"] else "MISMATCH"
            item["expected"] = "SOURCE_JSON_OBJECT"
    (out / "DOCUMENT_RECHECK.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("RESEARCH_STAGE_SUMMARY.md", "CLAIM_EVIDENCE_MATRIX.md", "UNRESOLVED_ISSUES.md", "STATISTICAL_REVIEW.md"):
        original = (prior / name).read_text(encoding="utf-8")
        header = ("> 本文件保留既有收口报告正文。当前复核只检查文档、源码字节身份及已有统计记录，"
                  "没有重跑实验。原归档时间：2026-11-25 06:31 UTC。\n"
                  f"> 原文件 SHA-256：`{digest(prior / name)}`。当前复核详见 `DOCUMENT_RECHECK.json`。\n\n")
        (out / name).write_text(header + original, encoding="utf-8")
    for name in ("EXISTING_RESULTS.json", "INPUT_MANIFEST.json", "STATUS_SEMANTICS.json"):
        shutil.copyfile(prior / name, out / name)
    print(json.dumps({"out": str(out), "checks": len(checks), "failed": len(failed),
                      "text_sources": len(source_records), "binary_records_only": len(binary_records)}, ensure_ascii=False))
    if failed:
        print(json.dumps([x["check"] for x in failed], ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
