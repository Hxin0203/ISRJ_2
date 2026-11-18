"""Teacher headroom and data consistency on public 2x flower-photo enlargement."""

import argparse
import copy
import hashlib
import io
import itertools
import json
import platform
import re
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
import PIL
import scipy
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "image_distillation_pilot"
sys.path.insert(0, str(PREVIOUS))
from run import DigitUNet, text_hash

CFG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
DEV = ("A", "B")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    tmp.replace(path)


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_groups(archive):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CFG["archive_sha256"]:
            raise ValueError("Unexpected TensorFlow Flowers archive")
    with tarfile.open(archive) as tar:
        credit = tar.extractfile("flower_photos/LICENSE.txt").read().decode("utf-8")
    by_account, id_owner = {}, {}
    for line in credit.splitlines():
        if ".jpg CC-BY" not in line:
            continue
        path = line.split(" CC-BY", 1)[0]
        match = re.search(r"flickr\.com/photos/([^/]+)/(?:(\d+)/)?", line, re.I)
        account = match[1].casefold()
        photo_id = PurePosixPath(path).name.split("_")[0]
        if (match[2] and match[2] != photo_id) or id_owner.setdefault(photo_id, account) != account:
            raise ValueError(f"Inconsistent original photo ID for {path}")
        by_account.setdefault(account, []).append(path)
    salt = CFG["split_salt"]
    accounts = sorted(by_account, key=lambda name: stable_hash(f"{salt}/account/{name}"))
    splits, start = {}, 0
    for split, count in CFG["split_sizes"].items():
        rows = []
        for account in accounts[start:start + count]:
            path = min(by_account[account], key=lambda p: stable_hash(f"{salt}/image/{p}"))
            rows.append({"account": account, "path": path})
        if len(rows) != count:
            raise ValueError(f"Insufficient independent account slugs for {split}")
        splits[split] = rows
        start += count
    return splits, {"source_files": sum(map(len, by_account.values())), "source_ids": len(id_owner),
                    "source_accounts": len(accounts), "unused_accounts": len(accounts) - start}


def load_photos(archive, selected_splits):
    groups, provenance = source_groups(archive)
    wanted = {"flower_photos/" + row["path"] for split in selected_splits for row in groups[split]}
    images = {}
    side = CFG["image_side"]
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if member.name in wanted:
                with Image.open(io.BytesIO(tar.extractfile(member).read())) as im:
                    view = ImageOps.fit(im.convert("L"), (side, side), method=Image.Resampling.LANCZOS)
                    images[member.name] = np.array(view, dtype=np.uint8)
    if set(images) != wanted:
        raise ValueError("Missing selected photographs")
    if len({hashlib.sha256(x.tobytes()).digest() for x in images.values()}) != len(images):
        raise ValueError("Duplicate decoded photographs among selected account slugs")
    data, identity = {}, {}
    for split, rows in groups.items():
        identity[split] = {"accounts": len(rows), "account_path_sha256": stable_hash(
            "\n".join(f"{r['account']}\t{r['path']}" for r in rows))}
        if split in selected_splits:
            pixels = np.stack([images["flower_photos/" + r["path"]] for r in rows])
            identity[split]["ordered_pixels_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
            data[split] = torch.from_numpy(pixels).unsqueeze(1).float().div_(255)
    provenance.update(archive_sha256=CFG["archive_sha256"], splits=identity,
                      decoded_splits=list(selected_splits))
    return data, groups, provenance


def lift(low):
    return F.interpolate(low, scale_factor=2, mode="nearest")


def block_part(img):
    return lift(F.avg_pool2d(img, 2))


def consistent(pred, measured_up):
    return pred + (measured_up - block_part(pred))


def crop_batch(images, rng):
    patch = CFG["patch_side"]
    batch = CFG["batch_size"]
    idx = torch.randint(len(images), (batch,), generator=rng)
    starts = torch.randint(images.shape[-1] - patch + 1, (2, batch, 1), generator=rng)
    yy, xx = starts + torch.arange(patch)
    return images[idx[:, None, None], 0, yy[:, :, None], xx[:, None, :]].unsqueeze(1)


def setup_model(width, init_seed):
    torch.manual_seed(init_seed)
    return DigitUNet(width)


def update(model, opt, clean, train_detail):
    measured = block_part(clean)
    pred = model(measured)
    if train_detail:
        pred = consistent(pred, measured)
    loss = F.mse_loss(pred, clean)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def fit(images, seed, arm, match_steps=None, replay_step=None):
    wide = arm == "wide"
    width = CFG["teacher_width" if wide else "student_width"]
    init_seed = seed + (CFG["init_teacher_seed_offset"] if wide else 0)
    warm = setup_model(width, init_seed)
    warm_opt = torch.optim.Adam(warm.parameters(), lr=CFG["learning_rate"])
    warm_rng = torch.Generator().manual_seed(190000 + seed)
    for _ in range(CFG["warmup_steps"]):
        update(warm, warm_opt, crop_batch(images, warm_rng), arm == "detail")
    model = setup_model(width, init_seed)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    rng = torch.Generator().manual_seed(CFG["patch_seed_offset"] + seed)
    required = {} if wide else {"short": CFG["short_steps"]}
    if match_steps is not None:
        required["matched"] = match_steps
    snapshots, times = {}, {}
    start, paused = time.perf_counter(), 0.0
    for step in itertools.count(1):
        update(model, opt, crop_batch(images, rng), arm == "detail")
        now = time.perf_counter()
        elapsed = now - start - paused
        keys = [k for k, v in required.items() if step == v]
        deadline = elapsed >= CFG["seconds_per_fit"] if replay_step is None else step == replay_step
        if deadline and "timed" not in snapshots:
            keys.append("timed")
        if keys:
            checkpoint = copy.deepcopy(model).eval()
            for key in keys:
                snapshots[key] = checkpoint
                times[key] = {"steps": step, "seconds": elapsed}
            paused += time.perf_counter() - now
        if "timed" in snapshots and required.keys() <= snapshots.keys():
            break
    if not wide and times["short"]["steps"] >= times["timed"]["steps"]:
        raise RuntimeError("The small model did not reach its short checkpoint before the deadline")
    return snapshots, times


def cpu_setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def gradient_magnitude(img):
    return (img.diff(dim=2).abs().mean((1, 2, 3)) + img.diff(dim=3).abs().mean((1, 2, 3))) / 2


def photo_scores(pred, clean):
    err = pred - clean
    mse = err.square().mean((1, 2, 3))
    display_mse = (pred.clamp(0, 1) - clean).square().mean((1, 2, 3))
    return {"psnr": (-10 * mse.clamp_min(1e-16).log10()).numpy(),
            "edge_mae": gradient_magnitude(err).numpy(),
            "clipped_psnr": (-10 * display_mse.clamp_min(1e-16).log10()).numpy(),
            "detail_mse": (err - block_part(err)).square().mean((1, 2, 3)).numpy()}


@torch.inference_mode()
def predict(model, measured_up):
    return torch.cat([model(batch.float()).double() for batch in measured_up.split(16)])


def average(scores):
    return {key: float(np.mean(values)) for key, values in scores.items()}


def ambiguity_check(clean):
    integer = (255 * clean).round().long()
    swapped = integer.reshape(*integer.shape[:-1], -1, 2).flip((-1,)).reshape_as(integer)

    def block_sums(img):
        n, _, h, w = img.shape
        return img.reshape(n, 1, h // 2, 2, w // 2, 2).sum((3, 5))

    same = torch.all(block_sums(integer) == block_sums(swapped), dim=(1, 2, 3))
    bound = ((integer - swapped).double() / 255).square().mean((1, 2, 3)) / 4
    if not same.all():
        raise ArithmeticError("Column swap changed a block sum")
    return {"identical_integer_measurement_pairs": int(same.sum()), "distinct_label_pairs": int((bound > 0).sum()),
            "mean_two_label_minimum_mse": float(bound.mean())}


def projection_check(pred, up, clean):
    adjusted = consistent(pred, up)
    dims = (1, 2, 3)
    err = pred - clean
    changed = adjusted - clean
    gain = err.square().mean(dims) - changed.square().mean(dims)
    expected = block_part(err).square().mean(dims) - (up - block_part(clean)).square().mean(dims)
    identity_error = (gain - expected).abs().max().item()
    detail_change = ((err - block_part(err)) - (changed - block_part(changed))).abs().max().item()
    if identity_error > 1e-12 or detail_change > 1e-12:
        raise ArithmeticError("Failed the orthogonal error decomposition")
    psnr_gain = photo_scores(adjusted, clean)["psnr"] - photo_scores(pred, clean)["psnr"]
    return {"mean_psnr_gain_db": float(psnr_gain.mean()), "photos_worse": int((gain < -1e-12).sum()),
            "max_abs_mse_identity_error": identity_error, "max_abs_detail_change": detail_change}


def teacher_gradient_check(small, teacher, clean):
    model = copy.deepcopy(small).double()
    up = block_part(clean[:4].double())
    with torch.no_grad():
        target = copy.deepcopy(teacher).double()(up)
    output = consistent(model(up), up)
    delta = output - target
    full = torch.autograd.grad(delta.square().mean(), tuple(model.parameters()), retain_graph=True)
    detail = torch.autograd.grad((delta - block_part(delta)).square().mean(), tuple(model.parameters()))
    abs_max = max((a - b).abs().max().item() for a, b in zip(full, detail))
    if abs_max > 1e-12:
        raise ArithmeticError("Full-teacher and detail-teacher gradients disagree")
    return abs_max


def paired(scores, candidate, control):
    gains = np.stack([s[candidate]["psnr"] - s[control]["psnr"] for s in scores])
    edge_improvements = np.stack([s[control]["edge_mae"] - s[candidate]["edge_mae"] for s in scores])
    per_photo = gains.mean(axis=0)
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    draws = rng.integers(len(per_photo), size=(CFG["bootstrap_draws"], len(per_photo)))
    ci = np.quantile(per_photo[draws].mean(axis=1), [0.025, 0.975])
    by_seed = gains.mean(axis=1)
    edge_change = -edge_improvements.mean() / np.mean([s[control]["edge_mae"] for s in scores])
    criteria = {"mean_gain": bool(per_photo.mean() >= CFG["min_psnr_gain_db"]),
                "every_seed_positive": bool(np.all(by_seed > 0)),
                "edge_mae": bool(edge_change <= CFG["max_relative_gradient_mae_increase"])}
    return {"gain_by_seed_db": dict(zip(map(str, CFG["seeds"]), by_seed.tolist())),
            "mean_gain_db": float(per_photo.mean()), "ci95_photo_conditional_on_fits_db": ci.tolist(),
            "mean_display_clipped_gain_db": float(np.mean([s[candidate]["clipped_psnr"] - s[control]["clipped_psnr"] for s in scores])),
            "relative_edge_mae_change": float(edge_change), "criteria": criteria, "passes": all(criteria.values())}


def profile_headroom(scores, strata):
    gains = np.mean([s["wide_dc"]["psnr"] - s["small_dc"]["psnr"] for s in scores], axis=0)
    edges = np.mean([s["small_dc"]["edge_mae"] - s["wide_dc"]["edge_mae"] for s in scores], axis=0)
    bins = {name: {"photos": int(np.sum(strata == i)), "psnr_gain_db": float(gains[strata == i].mean()),
                   "edge_mae_reduction": float(edges[strata == i].mean())}
            for i, name in enumerate(("low", "middle", "high"))}
    return {"spearman_psnr_vs_edge_reduction": float(spearmanr(gains, edges).statistic),
            "fraction_both_improve": float(np.mean((gains > 0) & (edges > 0))),
            "bins": bins, "highest_gain_bin": max(bins, key=lambda b: bins[b]["psnr_gain_db"])}


@torch.inference_mode()
def latency(small, wide, low):
    methods = {"small_raw": (small, False), "small_dc": (small, True), "wide_dc": (wide, True)}
    samples = {k: [] for k in methods}
    names = list(methods)
    for i in range(69):
        for key in names[i % 3:] + names[:i % 3]:
            model, dc = methods[key]
            start = time.perf_counter_ns()
            up = lift(low)
            pred = model(up)
            if dc:
                consistent(pred, up)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if i >= 9:
                samples[key].append(elapsed)
    return {k: {"median_ms": float(np.median(v)), "p95_ms": float(np.quantile(v, 0.95))}
            for k, v in samples.items()}


def run(args):
    cpu_setup()
    data, groups, provenance = load_photos(args.archive, ("train", *DEV))
    args.scratch.mkdir(parents=True, exist_ok=True)
    save_json(args.scratch / "account_manifest.json", groups)
    replay = json.loads(args.replay.read_text(encoding="utf-8"))["timing"] if args.replay else {}
    report = {"protocol_sha256": text_hash(HERE / "protocol.json"),
              "code_sha256_utf8_lf": {p.name: text_hash(p) for p in (Path(__file__), PREVIOUS / "run.py")},
              "shared_model_protocol_sha256": text_hash(PREVIOUS / "protocol.json"),
              "timing_mode": "recorded steps" if replay else "live wall time", "dataset": provenance,
              "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
                              "pillow": PIL.__version__, "python": platform.python_version(), "platform": platform.platform(),
                              "cpu": platform.processor(), "threads": torch.get_num_threads()},
              "timing": {}, "development": {}, "per_seed_means": {}, "numeric_checks_by_seed": {}}
    scores = {split: [] for split in DEV}
    clean = {s: data[s].double() for s in DEV}
    measured = {s: block_part(clean[s]) for s in DEV}
    stress = {}
    thresholds = np.quantile(gradient_magnitude(data["train"].double()).numpy(), [1 / 3, 2 / 3])
    report["training_texture_tertiles"] = thresholds.tolist()
    report["parameters"] = {key: sum(p.numel() for p in DigitUNet(CFG[key + "_width"]).parameters())
                            for key in ("student", "teacher")}
    rng = torch.Generator().manual_seed(CFG["stress_noise_seed"])
    for s in DEV:
        low = F.avg_pool2d(clean[s], 2)
        stress[s] = lift(low + CFG["stress_sigma"] * torch.randn(low.shape, generator=rng, dtype=torch.float64))
        report["development"][s] = {"ambiguity": ambiguity_check(clean[s])}

    for index, seed in enumerate(CFG["seeds"]):
        key = str(seed)
        order = ("wide", "small") if index % 2 == 0 else ("small", "wide")
        fits = {}
        times = {"order": list(order)}
        for arm in order:
            reference = replay[key][arm]["timed"]["steps"] if replay else None
            fits[arm], times[arm] = fit(data["train"], seed, arm, replay_step=reference)
            torch.save(fits[arm]["timed"].state_dict(), args.scratch / f"{arm}-{seed}.pt")
        report["timing"][key] = times
        small, wide = fits["small"]["timed"], fits["wide"]["timed"]
        checks = {"max_parameter_gradient_difference": teacher_gradient_check(small, wide, data["train"])}
        if index == 0:
            report["inference_batch1_cpu"] = latency(small, wide, F.avg_pool2d(data["train"][:1], 2))
        for s in DEV:
            up = measured[s]
            bicubic = F.interpolate(F.avg_pool2d(clean[s], 2), scale_factor=2, mode="bicubic", align_corners=False)
            row = {"nearest": photo_scores(up, clean[s]), "bicubic": photo_scores(bicubic, clean[s]),
                   "bicubic_dc": photo_scores(consistent(bicubic, up), clean[s])}
            for name, model in (("small_short", fits["small"]["short"]), ("small", small), ("wide", wide)):
                pred = predict(model, up)
                row[name] = photo_scores(pred, clean[s])
                row[name + "_dc"] = photo_scores(consistent(pred, up), clean[s])
            checks[s] = {"exact": projection_check(predict(small, up), up, clean[s]),
                         "noisy": projection_check(predict(small, stress[s]), stress[s], clean[s])}
            scores[s].append(row)
        report["numeric_checks_by_seed"][key] = checks
        print(f"Teacher comparison complete: seed {seed}, small {times['small']['timed']['steps']} and wide {times['wide']['timed']['steps']} steps", flush=True)

    def timed_ok(arm):
        ratios = [abs(v[arm]["timed"]["seconds"] / v["small"]["timed"]["seconds"] - 1)
                  for v in report["timing"].values()]
        return max(ratios) <= CFG["max_wall_relative_difference"]

    teacher_ok = timed_ok("wide")
    for s in DEV:
        dev = report["development"][s]
        strata = np.digitize(gradient_magnitude(clean[s]).numpy(), thresholds)
        dev["headroom_by_photo"] = profile_headroom(scores[s], strata)
        dev["teacher_vs_wall"] = paired(scores[s], "wide_dc", "small_dc")
        dev["teacher_vs_short"] = paired(scores[s], "wide_dc", "small_short_dc")
        dev["postprojection_vs_same_small"] = paired(scores[s], "small_dc", "small")
        qualified_here = (
            dev["teacher_vs_wall"]["passes"]
            and dev["teacher_vs_short"]["criteria"]["every_seed_positive"]
            and dev["headroom_by_photo"]["spearman_psnr_vs_edge_reduction"] >= CFG["min_photo_gain_spearman"]
        )
        teacher_ok = teacher_ok and qualified_here
    report["teacher_qualifies"] = teacher_ok
    print(f"Teacher qualifies: {teacher_ok}. Starting the prespecified direct-supervision comparison.", flush=True)

    for index, seed in enumerate(CFG["seeds"]):
        key = str(seed)
        steps = report["timing"][key]["small"]["timed"]["steps"]
        reference = replay[key]["detail"]["timed"]["steps"] if replay else None
        snaps, times = fit(data["train"], seed, "detail", match_steps=steps, replay_step=reference)
        report["timing"][key]["detail"] = times
        for snap in ("matched", "timed"):
            torch.save(snaps[snap].state_dict(), args.scratch / f"detail-{snap}-{seed}.pt")
        for s in DEV:
            for snap, model in snaps.items():
                up = measured[s]
                scores[s][index]["detail_" + snap] = photo_scores(consistent(predict(model, up), up), clean[s])
        print(f"Detail training complete: seed {seed}", flush=True)
    detail_ok = timed_ok("detail")
    for s in DEV:
        dev = report["development"][s]
        for snap, baseline in (("short", "small_short_dc"), ("matched", "small_dc"), ("timed", "small_dc")):
            dev["detail_vs_" + snap] = paired(scores[s], "detail_" + snap, baseline)
        detail_ok = detail_ok and dev["detail_vs_matched"]["passes"] and dev["detail_vs_timed"]["passes"]
        dev["means"] = {arm: {metric: float(np.mean([row[arm][metric] for row in scores[s]]))
                              for metric in scores[s][0][arm]} for arm in scores[s][0]}
    report["detail_training_qualifies"] = detail_ok
    report["timing_within_tolerance"] = {arm: timed_ok(arm) for arm in ("wide", "detail")}
    for index, seed in enumerate(CFG["seeds"]):
        report["per_seed_means"][str(seed)] = {
            s: {arm: average(metrics) for arm, metrics in scores[s][index].items()} for s in DEV}
    report["checkpoint_sha256"] = {}
    for seed in CFG["seeds"]:
        for arm in ("wide", "small", "detail-matched", "detail-timed"):
            path = args.scratch / f"{arm}-{seed}.pt"
            with path.open("rb") as stream:
                report["checkpoint_sha256"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    save_json(args.output, report)
    print(f"Decision: teacher={teacher_ok}, detail training={detail_ok}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--scratch", type=Path, required=True, help="Local-only manifest and checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Use the stopping steps in a previous result, for numerical reproduction")
    run(parser.parse_args())
