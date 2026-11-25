"""Parent-paired inference for the requested TRAIN16 fit preflight."""
import numpy as np

from vp_1103_common import distribution, parent_mean


def fit_evidence(before, after, jnr):
    mask = jnr >= 35
    parent_before = parent_mean(before["background_nmse"], mask)
    parent_after = parent_mean(after["background_nmse"], mask)
    residual = distribution((parent_after - parent_before) / parent_before)
    pfa = distribution((before["frame_pfa_jammer"] - after["frame_pfa_jammer"]).mean(axis=1))
    passed = residual["parent_bootstrap_ci95"][1] < 0 and pfa["parent_bootstrap_ci95"][0] > 0
    return {"passes": bool(passed), "high_jnr_parent_relative_residual_change": residual,
            "high_jnr_parent_ratio_of_means": float(parent_after.mean() / parent_before.mean()),
            "frame_pfa_parent_decrease": pfa,
            "criterion": "The 95% parent-paired bootstrap interval supports a decrease in both "
                         "high-JNR residual and training frame false alarms."}
