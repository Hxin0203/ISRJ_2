"""Keep P0 running, admit P2 automatically, and stop at the DEV decision."""
import os
from pathlib import Path
import time
import traceback

import numpy as np

from vp_1103_common import WORK, atomic_json, now, provenance, read, sha, stratified, validate_role
from tools.durable_io import ProcessLock, atomic_write, process_alive

OUT = WORK / "evidence/outputs/p2_smoothvp_screen"
PRIVATE = WORK / "outputs/p2_smoothvp_screen"
P0 = WORK / "evidence/outputs/p0_dechirp_tail_2x2"
PREFLIGHT = WORK / "evidence/outputs/p2_smoothvp_preflight"
GATE_D = WORK / "evidence/outputs/vp_1103_gate_d"


def safe_read(path):
    for attempt in range(20):
        try:
            return read(path)
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.2)


def checkpoint_seal(p0):
    for arm, rows in p0["new_arms"].items():
        for seed, row in rows.items():
            if row["status"] != "COMPLETE" or row["epoch"] != 256:
                raise ValueError("P0 is incomplete")
            folder = WORK / "outputs/p0_dechirp_tail_2x2" / arm / seed
            if sha(folder / "epoch_256.pt") != row["checkpoint_sha256"]:
                raise ValueError("P0 checkpoint binding mismatch")
            if sha(folder / "dev_parent_metrics.npz") != row["parent_metrics_sha256"]:
                raise ValueError("P0 DEV metrics binding mismatch")


def report(rows, route, coordinates, extra=None):
    from vp_1103_p2_evaluate import CORE, m2_metrics
    from vp_1103_common import DATA, frame_jnr, parent_roster
    from vp_1103_references import reference_metrics
    with np.load(DATA / "dev.npz", allow_pickle=False) as z:
        ids, jnr = parent_roster(z, 256), frame_jnr(z)
    unet, unet_binding = reference_metrics("unet_tail", ids)
    methods = {"M1-UNet-Tail": {
        "seeds": {str(seed): stratified(row, jnr) for seed, row in unet.items()},
        "seed_average": stratified({key: np.mean([r[key] for r in unet.values()], axis=0)
                                    for key in CORE}, jnr)}}
    if coordinates in ("RAW_IQ", "DECHIRP"):
        m2 = {seed: m2_metrics(seed, coordinates)[0] for seed in (17, 29, 41)}
        methods["M2-P0-Best-Tail"] = {"seeds": {str(seed): stratified(row, jnr) for seed, row in m2.items()},
             "seed_average": stratified({key: np.mean([r[key] for r in m2.values()], axis=0) for key in CORE}, jnr)}
    for label, folder, filename in (
        ("P1-D2-Blind-Smooth", "p1d2_replay_first_drift_audit", "replay_smooth"),
        ("P1-D1-Anchored-K8", "p1d1_replay_dict_sensitivity", "neighborhood_1_1_0_k8_smooth")):
        summary = read(WORK / "evidence/outputs" / folder / "SUMMARY.json")["methods"][filename]
        methods[label] = {"training_seeds": [], "metrics": summary["metrics"],
                          "private_metrics_sha256": summary["private_metrics_sha256"],
                          "scope": "Zero-training diagnostic with a single shared DEV roster"}
    for arm in ("M3-SmoothVP", "M3-FixedLambda"):
        seeds = rows.get(arm, {})
        if seeds:
            matrices = []
            for seed, row in seeds.items():
                path = PRIVATE / arm / seed / "dev_metrics.npz"
                if sha(path) != row["private_metrics_sha256"]:
                    raise ValueError("P2 DEV metrics altered")
                with np.load(path, allow_pickle=False) as z:
                    matrices.append({key: z[key] for key in z.files if key != "parent_id"})
            methods[arm] = {"seeds": seeds, "seed_average": stratified({
                key: np.mean([r[key] for r in matrices], axis=0) for key in matrices[0]}, jnr),
                "completed_seeds": list(seeds)}
    seed_status = {arm: {str(seed): ("COMPLETE" if str(seed) in rows.get(arm, {}) else
                                   "NOT_STARTED_GATE") for seed in (17, 29, 41)}
                   for arm in ("M3-SmoothVP", "M3-FixedLambda")}
    if extra and isinstance(extra.get("failed_seed"), dict):
        seed_status["M3-SmoothVP"][str(extra["failed_seed"]["seed"])] = "FAILED"
    summary = {"route": route, "coordinates": coordinates, "methods": methods,
               "seed_status": seed_status,
               "unet_binding": unet_binding, "extra": extra or {},
               "confirmation_unlocked": False, "selection": "All evaluated models fixed at epoch256"}
    lines = ["# SmoothVP-Net DEV decision", "", f"**{route}**. Coordinates **{coordinates}**.", "",
             "| Method | Seeds completed | Frame Pfa | Pd | Damage | SJRIF dB | Endpoint NMSE | No-jammer NMSE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, method in methods.items():
        metrics = method.get("seed_average", method.get("metrics"))["global"]
        values = " | ".join(f"{metrics[key]['mean']:.6g}" for key in CORE)
        seeds = method.get("completed_seeds", list(method.get("seeds", {})))
        lines.append(f"| {name} | {','.join(seeds) or 'zero training'} | {values} |")
    if extra:
        lines += ["", "## Admission or stop evidence", ""]
        for key, value in extra.items():
            if isinstance(value, str):
                lines.append(f"- {key.replace('_', ' ')} {value}")
            elif isinstance(value, bool):
                lines.append(f"- {key.replace('_', ' ')} {value}")
            else:
                lines.append(f"- {key.replace('_', ' ')} is recorded in SUMMARY.json.")
    lines += ["", "Parent bootstrap intervals and all available seed results are in SUMMARY.json.",
              "Private failure dictionaries and operator decompositions are bound in each seed receipt.",
              "An incomplete three-seed run is reported with its actual completed seed count."]
    atomic_json(OUT / "SUMMARY.json", summary)
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


def run():
    source_paths = [Path(__file__).with_name(name) for name in (
        "vp_1103_p2_run.py", "vp_1103_p2_smoothvp.py", "vp_1103_p2_train.py",
        "vp_1103_p2_evaluate.py", "benchmark_latency_unified.py")]
    source = provenance(*source_paths)
    status = {**source, "stage": "WAITING_P0_AND_PREFLIGHT", "pid": os.getpid(),
              "started_utc": now(), "ended_utc": None, "seeds": [17, 29, 41], "micro_batch": 8,
              "authorization": "User 2026-11-23 authorized automatic P2-v2 after sealed P0 and passed CPU preflight."}
    pause_path = OUT / "USER_PAUSE.json"
    if pause_path.exists() or (PRIVATE / "PAUSE").exists():
        atomic_json(OUT / "STATUS.json", {**status, "stage": "PAUSED_BY_USER",
                                         "updated_utc": now(), "pause": safe_read(pause_path) if pause_path.exists() else {}})
        return
    atomic_json(OUT / "AUTHORIZATION.json", {
        "recorded_utc": now(), "automatic_p2_authorized": True, "confirmation_authorized": False,
        "conditions": ["P0 six runs complete and coordinates locked", "Gate D1 and D2 PASS",
                       "Selected-coordinate CPU preflight PASS"],
        "fixed_scope": {"model": "M3-SmoothVP", "k": 8, "seeds": [17, 29, 41],
                        "micro_batch": 8, "epochs": 256, "m4": False},
        "stop": "Seed17 DEV screen decides continuation. Independent confirmation requires later review."})

    def update(**fields):
        status.update(fields, updated_utc=now())
        atomic_json(OUT / "STATUS.json", status)

    update()
    while True:
        if pause_path.exists() or (PRIVATE / "PAUSE").exists():
            update(stage="PAUSED_BY_USER", pause=safe_read(pause_path) if pause_path.exists() else {})
            return
        p0_state = safe_read(P0 / "STATUS.json")
        pre_state = safe_read(PREFLIGHT / "STATUS.json")
        p0 = safe_read(P0 / "SUMMARY.json")
        coord = p0["coordinates"]
        if p0_state["stage"] == "FAILED" or (p0_state["stage"] == "TRAINING" and
                                             not process_alive(p0_state["pid"])):
            raise RuntimeError("P0 requires recovery before P2 admission")
        if pre_state["stage"] == "FAILED":
            raise RuntimeError("CPU preflight execution failed")
        if pre_state["stage"] == "RUNNING" and not process_alive(pre_state["pid"]):
            raise RuntimeError("CPU preflight stopped without a completion receipt")
        update(p0_progress={key: p0_state.get(key) for key in ("arm", "seed", "last_committed_epoch")},
               preflight_stage=pre_state["stage"], coordinates=coord)
        if p0_state["stage"] == "COMPLETE" and pre_state["stage"] == "COMPLETE":
            checkpoint_seal(p0)
            if coord not in ("RAW_IQ", "DECHIRP"):
                raise RuntimeError("Completed P0 did not lock coordinates")
            # The original read-only monitors seal the paired six-cell table.
            if p0.get("six_cell_status") != "COMPLETE":
                time.sleep(30)
                continue
            break
        time.sleep(45)
    gate = safe_read(GATE_D / "SUMMARY.json")
    if pause_path.exists() or (PRIVATE / "PAUSE").exists():
        update(stage="PAUSED_BY_USER", pause=safe_read(pause_path) if pause_path.exists() else {})
        return
    for name in ("D1", "D2"):
        if gate["gates"][name]["status"] != "PASS":
            raise RuntimeError(f"Bridge {name} has not passed")
    preflight = safe_read(PREFLIGHT / "SUMMARY.json")["coordinates"][coord]
    if safe_read(PREFLIGHT / "STATUS.json")["source_sha256"]["tools/vp_1103_p2_smoothvp.py"] != sha(
            Path(__file__).with_name("vp_1103_p2_smoothvp.py")):
        raise RuntimeError("CPU-preflight model source no longer matches the admitted model")
    if preflight["status"] != "PASS":
        update(stage="STOPPED_PREFLIGHT", ended_utc=now(), reason="CPU TRAIN16 expression check failed")
        report({}, "STOPPED_PREFLIGHT", coord, {"cpu_preflight": preflight,
                "receipt": "evidence/outputs/p2_smoothvp_preflight/SUMMARY.json"})
        return
    from vp_1103_p2_train import train
    from tools.unet_ablation.runtime import configure_precision
    import torch
    train_path, train_id = validate_role("train")
    dev_path, dev_id = validate_role("dev")
    with np.load(dev_path, allow_pickle=False) as z:
        dev = {key: z[key] for key in z.files}
    config = {**source, "coordinates": coord, "seeds": [17, 29, 41], "epochs": 256, "micro_batch": 8,
              "updates_per_epoch": 48, "data": {"train": train_id, "dev": dev_id},
              "precision": configure_precision(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "p0_summary_sha256": sha(P0 / "SUMMARY.json"),
              "preflight_summary_sha256": sha(PREFLIGHT / "SUMMARY.json"),
              "loss": "endpoint + .1 pc + damage + .05 tail", "additional_regularizers": [],
              "solver_dtype": "complex128", "seed17_stop_pfa": .14, "seed17_stop_keep": 5e-4}
    target = PRIVATE / "CONFIG.json"
    if target.exists() and read(target) != config:
        raise RuntimeError("P2 identity changed across a restart")
    atomic_json(target, config)
    rows = {"M3-SmoothVP": {}, "M3-FixedLambda": {}}
    for seed in (17, 29, 41):
        try:
            result = train(seed, coord, train_path, dev, config, update)
        except Exception as exc:
            from vp_1103_p2_smoothvp import SmoothVPNet
            from vp_1103_p0 import evaluate
            from vp_1103_p2_evaluate import failure_details
            from vp_1103_common import frame_jnr, worst_cases
            folder = PRIVATE / "M3-SmoothVP" / str(seed)
            latest = folder / "latest.pt"
            failure = {"seed": seed, "error": repr(exc), "utc": now()}
            if latest.exists():
                receipt = read(latest.with_suffix(".json"))
                if sha(latest) != receipt["checkpoint_sha256"]:
                    raise RuntimeError("Failure-recovery checkpoint has an invalid hash") from exc
                payload = torch.load(latest, map_location="cpu", weights_only=False)
                diagnostic_model = SmoothVPNet(coord).cpu().eval()
                diagnostic_model.load_state_dict(payload["model"])
                metrics = evaluate(diagnostic_model, dev)
                cases = worst_cases(metrics, dev["parent_id"], frame_jnr(dev))
                failure.update(last_safe_epoch=payload["epoch"], checkpoint_sha256=sha(latest),
                               worst_cases=cases, diagnosis=failure_details(diagnostic_model, dev, cases, folder))
            atomic_json(OUT / f"FAILED_SEED_{seed}.json", failure)
            report(rows, "STOPPED_NUMERICAL_FAILURE", coord, {"failed_seed": failure})
            update(stage="STOPPED_NUMERICAL_FAILURE", ended_utc=now(), failed_seed=seed, error=repr(exc))
            return
        rows["M3-SmoothVP"][str(seed)] = result
        report(rows, "DEV_SCREEN_IN_PROGRESS", coord)
        if seed == 17 and not result["continue_other_seeds"]:
            report(rows, "STOPPED_SEED17", coord, {"circuit_breaker": result["circuit_breaker"]})
            break
    if len(rows["M3-SmoothVP"]) == 3:
        result = train(17, coord, train_path, dev, config, update, arm="M3-FixedLambda")
        rows["M3-FixedLambda"]["17"] = result
        smooth = rows["M3-SmoothVP"]["17"]["metrics"]["global"]
        fixed = result["metrics"]["global"]
        # Predeclare a clear ablation signal to avoid subjective seed extension.
        clear = (fixed["no_jammer_pooled_nmse"]["mean"] >= 2 * smooth["no_jammer_pooled_nmse"]["mean"] or
                 fixed["drift_over_target_energy"]["mean"] >= 2 * smooth["drift_over_target_energy"]["mean"])
        if clear:
            for seed in (29, 41):
                rows["M3-FixedLambda"][str(seed)] = train(seed, coord, train_path, dev, config, update, arm="M3-FixedLambda")
    update(stage="LATENCY_BENCHMARK")
    from benchmark_latency_unified import run as benchmark
    benchmark()
    route = "STOPPED_SEED17" if len(rows["M3-SmoothVP"]) == 1 else "AWAIT_GATE_P2_REVIEW"
    report(rows, route, coord, {"latency_summary": "evidence/outputs/p2_smoothvp_latency/SUMMARY.json"})
    update(stage="COMPLETE", route=route, ended_utc=now(), summary_sha256=sha(OUT / "SUMMARY.json"))


if __name__ == "__main__":
    with ProcessLock(PRIVATE / "CONTROLLER.lock"):
        try:
            run()
        except Exception as exc:
            path = OUT / "STATUS.json"
            previous = safe_read(path) if path.exists() else {}
            atomic_json(OUT / f"FAILURE_{time.time_ns()}.json",
                        {"error": repr(exc), "traceback": traceback.format_exc(), "utc": now()})
            atomic_json(path, {**previous, "stage": "FAILED", "ended_utc": now(), "error": repr(exc)})
            raise
