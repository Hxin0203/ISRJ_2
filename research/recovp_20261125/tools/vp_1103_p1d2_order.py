"""Isolate replay selection order while holding the historical replay fitter fixed."""
import time

import numpy as np

from vp_1103_common import (WORK, atomic_json, frame_jnr, now, parent_roster,
                           provenance, read, sha, validate_role)
from vp_1103_p1 import Projection, complement, observations, orthogonal_basis
from vp_1103_p1b import estimate_protection, estimate_replay
from vp_1103_p1d_replay import EMPTY, operator_scores, result
from tools.durable_io import ProcessLock, atomic_write, process_alive


NAME = "p1d2_replay_first_drift_audit"
OUT = WORK / "evidence" / "outputs" / NAME
PRIVATE = WORK / "outputs" / NAME
KEYS = ("frame_pfa_jammer", "pd", "damage", "drift_over_target_energy",
        "leak_over_target_energy", "no_jammer_pooled_nmse")


def fixed_proxy(projected):
    def getter(pick, state, y):
        operators = []
        for record in y:
            row = []
            for x in record:
                if projected:
                    a, _ = estimate_protection(x)
                    x = complement(orthogonal_basis(a), x)
                b = estimate_replay(x)
                row.append(Projection(EMPTY, b, "adaptive"))
            operators.append(row)
        return operators, None
    return getter


def run():
    path, binding = validate_role("dev")
    with np.load(path, allow_pickle=False) as archive:
        dev = {key: archive[key] for key in archive.files}
    ids, jnr = parent_roster(dev, 256), frame_jnr(dev)
    y = observations(dev)
    started = now()
    rows = {}
    for name, projected in (("unchanged_proxy_raw_y", False), ("unchanged_proxy_projected_y", True)):
        metrics, error = operator_scores(dev, y, fixed_proxy(projected))
        rows[name] = result(PRIVATE, name, metrics, jnr, ids, error, {})
    payload = {"started_utc": started, "ended_utc": now(), "stage": "COMPLETE", "data": {"dev": binding},
               "micro_batch": None, "training_seeds": [],
               **provenance(__file__, WORK / "tools/vp_1103_p1d_replay.py",
                            WORK / "tools/vp_1103_p1b.py", WORK / "tools/vp_1103_p1.py"),
               "methods": rows}
    # D2 has one report writer while its main audit is active.
    while True:
        try:
            status = read(OUT / "STATUS.json")
        except PermissionError:
            time.sleep(.3)
            continue
        if status["stage"] == "COMPLETE":
            break
        if status["stage"] == "FAILED" or not process_alive(status["pid"]):
            raise RuntimeError("D2 report did not finish")
        time.sleep(10)
    atomic_json(OUT / "ORDER_CONTROL_SUMMARY.json", payload)
    report = OUT / "ORDER_CONTROL_REPORT.md"
    lines = ["# P1-D2 · 固定旧候选器的顺序对照", "",
             "两组均使用 Gate 2 的能量窗口重放候选器、无目标保护闭式求解和同一正则。"
             "唯一差别是拟合候选字典时读取原始观测还是目标正交补。", "",
             "| 候选输入 | 虚警 | Pd | 损伤 | 漂移比 | 泄漏比 | 无扰保持 NMSE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        values = " | ".join(f"{row['metrics']['global'][key]['mean']:.6g}" for key in KEYS)
        lines.append(f"| {name} | {values} |")
    lines += ["", "逐 parent 聚类区间和分层见同目录的 `ORDER_CONTROL_SUMMARY.json`。"]
    atomic_write(report, lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


if __name__ == "__main__":
    with ProcessLock(PRIVATE / "ORDER_CONTROL.lock"):
        run()
