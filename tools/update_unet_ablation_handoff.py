"""Read server evidence and refresh the handoff; never start training or edit a freeze."""
from __future__ import annotations

import json
import statistics
import subprocess

from paper2.io import ROOT, now, read, sha
from tools.deploy_unet_ablation import fetch, remote, SERVER_ROOT
from tools.durable_io import atomic_json
from tools.unet_ablation.protocol import CONFIG, EVIDENCE, RUN_ID, source_hashes


def audit_server():
    program = f'''import os,sys,json,platform
from pathlib import Path
os.chdir({SERVER_ROOT!r});sys.path.insert(0,str(Path.cwd()))
from paper2.io import now,read,sha
from tools.unet_ablation.protocol import BASE,CONFIG,validate_config,baseline_bindings,source_hashes
from tools.unet_ablation.train import valid_checkpoint
cfg=validate_config()
assert baseline_bindings()==cfg['original_epoch256']
rows=[]
for folder in sorted(BASE.glob('*/seed_*')):
 paths=sorted(folder.glob('training_epoch_*.json'))
 if not paths:continue
 expected={{'variant':folder.parent.name,'seed':int(folder.name.split('_')[1]),'config_sha256':sha(CONFIG),
            'data_sha256':cfg['data_sha256']['train.npz'],'source_sha256':cfg['source_sha256']}}
 verified=[]
 for p in paths:
  r=read(p);cp=folder/'recovery'/('epoch_%03d.pt'%r['epoch'])
  saved=valid_checkpoint(cp,expected)
  assert saved is not None and saved['global_step']==r['epoch']*48
  assert r['checkpoint_sha256']==sha(cp)
  assert saved['epoch_record']=={{k:v for k,v in r.items() if k!='checkpoint_sha256'}}
  assert saved['optimizer']['state'] and saved['rng'] and saved['architecture']
  verified.append(r)
 assert [r['epoch'] for r in verified]==list(range(1,len(verified)+1))
 rows.append({{'variant':folder.parent.name,'seed':expected['seed'],'saved_epochs_verified':len(verified),
              'last_global_step':saved['global_step'],'last_checkpoint_sha256':sha(cp),
              'optimizer_state_entries':len(saved['optimizer']['state']),
              'epoch_seconds':[r['epoch_elapsed_seconds'] for r in verified],
              'first_epoch_sha256':verified[0]['checkpoint_sha256']}})
print(json.dumps({{'stage':'SERVER_RECOVERY_AND_IDENTITY_VERIFIED','utc':now(),'host':platform.node(),
                 'config_sha256':sha(CONFIG),'source_sha256':source_hashes(),
                 'data_sha256':cfg['data_sha256'],'original_epoch256_verified':True,'runs':rows}}))
'''
    result = json.loads(remote(program, timeout=120))
    assert result["config_sha256"] == sha(CONFIG)
    assert result["source_sha256"] == source_hashes()
    initial = read(EVIDENCE / "FIRST_EPOCH_AUDIT.json")
    unet = next(r for r in result["runs"] if r["variant"] == "unet" and r["seed"] == 17)
    assert unet["first_epoch_sha256"] == initial["checkpoint_sha256"]
    result["local_source_matches_server"] = True
    result["first_epoch_preserved_across_launcher_restart"] = True
    atomic_json(EVIDENCE / "RECOVERY_IDENTITY_AUDIT.json", result)
    return result


def timing(audit):
    times = [t for r in audit["runs"] if r["variant"] == "unet" for t in r["epoch_seconds"][1:]]
    if len(times) < 3:
        return {"status": "INSUFFICIENT_FORMAL_EPOCHS"}
    median = statistics.median(times)
    saved = sum(r["saved_epochs_verified"] for r in audit["runs"] if r["variant"] == "unet")
    unet_seconds = (3 * 256 - saved) * median
    smoke = read(EVIDENCE.with_name(RUN_ID + "_engineering") / "SMOKE.json")
    other_seconds = sum(r["seconds"] * 48 * 256 * 3 for r in smoke["rows"] if r["variant"] != "unet")
    result = {
        "utc": now(), "unet_formal_epochs_used": len(times), "unet_epoch_seconds_median": median,
        "unet_epoch_seconds_min_max": [min(times), max(times)],
        "unet_remaining_compute_hours": unet_seconds / 3600,
        "unet_remaining_wall_hours_with_15_to_50_percent_allowance": [unet_seconds * k / 3600 for k in (1.15, 1.5)],
        "four_ablations_compute_hours_from_single_update_smoke_only": other_seconds / 3600,
        "whole_queue_rough_hours_with_20_to_100_percent_allowance": [(unet_seconds + other_seconds) * k / 3600 for k in (1.2, 2)],
        "limitations": "Only U-Net has formal multi-epoch timing. Ablations use one engineering update each; their estimate is provisional. Allowances are planning margins, not confidence intervals. Resource waits, failures and unusually expensive evaluation can exceed them.",
    }
    atomic_json(EVIDENCE / "TIMING_ESTIMATE.json", result)
    return result


def refresh_handoff(audit, eta):
    status = read(EVIDENCE / "SERVER_STATUS.json")
    cfg = read(CONFIG)
    stamp = status["retrieved_utc"]
    live = status["files"]["STATUS.json"]
    queue = status["files"]["QUEUE_STATUS.json"]
    saved = {f"{r['variant']}/seed_{r['seed']}": r["saved_epochs_verified"] for r in audit["runs"]}
    current = (f"Server snapshot {stamp}: {queue['stage']}; job {queue.get('job_index', '?')}/15, "
               f"{live.get('variant', '?')} seed {live.get('seed', '?')}, epoch {live.get('epoch', '?')}. "
               f"Independently SHA-verified saved epochs at {audit['utc']}: {saved}.")
    history = (
        "Original training completed for all eight seeds (17,29,41,53,67,79,97,109) at epoch 256: "
        "64 immutable milestones, 64 DEV v4 evaluations and 2048 epoch metric rows. "
        "Original independent TEST used frozen seed 29/epoch 128: 2237 method-condition receipts and "
        "103826 audited parent blocks. The additive F-R7 RSDR-Net comparison completed 564/564 conditions "
        "on the same second-paper TEST observations; it compares one frozen seed per model with different "
        "training budgets/distributions. The original epoch-256 curves remain DEV evidence. "
        "Reports: `PAPER2_PROGRESS_CHECK_20261115.md`, `PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md`."
    )
    safety = (
        "Long training is server-only: rsdr-b / DESKTOP-RFFERM3, `D:/code/Xin/ISRJ_222`, physical GPU 1. "
        "No local formal training or fallback. First-paper sources, environments and unrelated jobs are read-only. "
        "Frozen `paper2/`, `tools/unet_ablation/` and `tools/durable_io.py` must match config SHA bindings. "
        "Only the `paper2` remote (`Hxin0203/ISRJ_2`) may be pushed. Do not upload datasets, weights, raw arrays, "
        "caches, credentials or full logs. No old Gate access, MCA-ADMM/FNet or ChatGPT collaboration. "
        "The user's 2026-11-15 request explicitly authorizes these ablations and supersedes older pause statements."
    )
    plan = (
        "U-Net seeds 17/29/41 run first, each 256 epochs; then no_dechirp, no_rms, no_pc and no_damage, "
        "each with the same three seeds and budget. 15 new runs in total. U-Net has exactly 59,714 active/registered "
        "parameters; the full model has 59,714 active plus 53,042 unused registered parameters. "
        "The three full epoch-256 checkpoints are reused and their DEV evaluations were rerun on the server. "
        "Every new epoch retains a complete recovery archive; every 32 epochs retains a milestone and DEV result. "
        "After all training, generate 512 fresh confirmation parents and compare all six models at fixed epoch 256 "
        "on the same observations. No best-seed/epoch selection. Report u_nmse, damage, SJRIF, Pd and empirical Pfa. "
        "Parent bootstrap intervals are conditional on the three trained seeds; seed SD is separate. "
        "Historical full training environment differs from the new server training; do not claim identical trajectories."
    )
    event = (
        "A planned saved-boundary launcher restart preserved epoch 1 and moved execution to a hidden WMI-owned "
        "launcher independent of SSH lifetime. The retained CONTROLLER_FAILURE.json for old PID 30264 records that "
        "planned stop; assess the current queue/PIDs and saved epochs. Initial TF32 numerical-preflight failure is "
        "retained; explicit TF32-off FP32 passed the unchanged tolerances. No U-Net performance conclusion is available yet."
    )
    handoff = ROOT / "handoff"
    (handoff / "CURRENT_STATE.md").write_text(
        f"# Current state\n\n{current}\n\n{plan}\n\n{event}\n\n{history}\n\n{safety}\n", encoding="utf8")
    running = read(handoff / "RUNNING_EXPERIMENTS.json")
    running.update(checked_utc=stamp, active_experiment_processes=status["processes"],
                   note="Original training/TEST and RSDR comparison complete. New server-only U-Net/ablation study is active.")
    running["unet_ablation"] = {
        "run_id": RUN_ID, "server": "rsdr-b", "root": SERVER_ROOT, "physical_gpu": 1,
        "config_sha256": sha(CONFIG), "queue": queue, "training_heartbeat": live,
        "saved_epochs_independently_verified": saved, "recovery_audit_utc": audit["utc"],
        "local_formal_training": False, "eta": eta, "new_performance_conclusions_available": False,
        "confirmation": "NOT_YET_GENERATED: held back until all 15 runs finish", "plan": cfg["queue"],
    }
    running["uncompleted_scientific_work"] = [
        "Server U-Net and four ablations: 15 runs x 256 epochs with 3 seeds",
        "Fresh fixed-distribution epoch-256 confirmation after all new training",
        "Baseline bibliography/discrete implementation verification", "Final manuscript",
    ]
    atomic_json(handoff / "RUNNING_EXPERIMENTS.json", running)
    (handoff / "LAST_CODEX_EXECUTION.md").write_text(
        f"# Last execution\n\nUpdated {now()}. {current}\n\n"
        f"{plan}\n\nVerified post-restart archives, optimizer/RNG/architecture, metric bindings, frozen sources "
        f"and train/dev/original checkpoint hashes. Local regression tests: see `VALIDATION.json` in the study evidence. "
        f"{event}\n\n{history}\n\n{safety}\n", encoding="utf8")
    (handoff / "OUTPUT_EVIDENCE_INDEX.md").write_text(
        f"# Output evidence index\n\nUpdated {stamp}.\n\n"
        f"New study: `PAPER2_UNET_ABLATION_PLAN_20261115.md`; freeze: `{CONFIG.relative_to(ROOT).as_posix()}` "
        f"(SHA `{sha(CONFIG)}`). Evidence directory: `{EVIDENCE.relative_to(ROOT).as_posix()}`. "
        "Read SERVER_STATUS.json, RECOVERY_IDENTITY_AUDIT.json, FIRST_EPOCH_AUDIT.json, QUEUE_STATUS.json, "
        "NUMERICAL_PREFLIGHT.json, DIAGNOSTIC_BEFORE.json, DIAGNOSTIC_TF32_OFF.json, LAUNCHER_RESTART_EVENT.json, "
        "TIMING_ESTIMATE.json, VALIDATION.json and TEXT_AUDIT.json. Full epoch-256 DEV receipts are under full/seed_*/. "
        f"Engineering-only update/memory evidence is in `{EVIDENCE.relative_to(ROOT).as_posix()}_engineering/`. "
        "Absent U-Net milestone/confirmation files mean not yet produced, never a scientific zero. "
        "Per-epoch metric JSON is fetched to ignored outputs/; weights and raw arrays remain on the server.\n\n"
        f"{history}\n\nOriginal audits and curves: `evidence/outputs/PROGRESS_CHECK_20261115/`, "
        "`evidence/outputs/CONVERGENCE_256_8SEED_20261114/`, `evidence/outputs/RSDR_UNIFIED_20261115/`, "
        "`evidence/outputs/paper2_long_pipeline_20261114/`, `evidence/outputs/TEST_COVERAGE_1103.json` "
        "and `TEST_CONDITION_TABLE_1103.csv`. Preserve original failure history.\n\n"
        f"{safety}\n", encoding="utf8")
    git_state = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True)
    (handoff / "GIT_STATUS.txt").write_text(
        f"Checked {now()}\nBranch codex/1103-experiment; tracked changes only. Exact new study paths are staged separately.\n"
        f"{git_state}\nOnly paper2 remote may be pushed. No data/weights/raw arrays/full logs.\n", encoding="utf8")


def main():
    fetch()
    audit = audit_server()
    eta = timing(audit)
    refresh_handoff(audit, eta)
    print(json.dumps({"audit_utc": audit["utc"], "saved": audit["runs"], "eta": eta}, indent=2))


if __name__ == "__main__":
    main()
