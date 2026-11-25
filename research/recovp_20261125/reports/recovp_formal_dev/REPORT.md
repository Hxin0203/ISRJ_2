# ReCoVP 正式 DEV 对决

SNR/JSR read native active-support metadata. JCR uses T(Y01)-E; RD peak uses the same residual and the 9032 valid CFAR cells, normalized by mean RD noise power. CPL and angle use T(Y11)-T(Y01); signed and absolute dB are reported separately. BVR is an energy ratio, not a test of whiteness or independence. RD peak is a risk descriptor, not a universal CFAR false-alarm bound. Actual CFAR is always measured. Matched filtering is not generally unitary, and correlated CUTs do not justify an independent-cell product formula. A global phase rotation can leave CPL/angle zero; signed complex gain and pulse-phase errors are retained.

Frozen winner: **ReCoVP-NoEquiv**. Scan evaluation runs regardless of competitiveness.

| Method | Seed | pd | frame_pfa_jammer | jcr_db | rd_residual_peak_db | cpl_db | abs_cpl_db | sin2_theta_s | bvr00_db | abs_bvr00_db | bvr01_db | abs_bvr01_db | no_jammer_pooled_nmse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReCoVP-Base | 17 | 0.848958 | 0.0690104 | 12.2564 | 2.03761 | 0.17483 | 0.175693 | 0.0547157 | -0.00137331 | 0.00137331 | 0.944176 | 0.944422 | 1.94346e-06 |
| ReCoVP-NoEquiv | 17 | 0.838542 | 0.0664062 | 12.0974 | 2.22651 | 0.200701 | 0.202035 | 0.0528406 | -0.000820325 | 0.000820325 | 0.973122 | 0.973241 | 8.3517e-06 |
| P0-NoDechirp-Tail | 17 | 0.876302 | 0.102865 | 16.3554 | -0.628795 | 0.556283 | 0.55635 | 0.0442665 | -0.00379176 | 0.00523009 | 0.0106017 | 0.0822527 | 9.18982e-05 |
| UNet-Tail | 17 | 0.864583 | 0.108073 | 15.7554 | -0.0536165 | 0.684608 | 0.684608 | 0.0499131 | -0.0469725 | 0.0469725 | -0.0213594 | 0.114732 | 0.000117402 |
| RSDR-Net | 40312102 | 0.869792 | 0.0507812 | 17.7605 | -2.85102 | 1.05369 | 1.05369 | 0.0936762 | -0.0100759 | 0.0100759 | -0.0693743 | 0.102563 | 1.34618e-05 |

完整原生 SNR/JSR 分层、逐 parent 配对区间、复缩放与三层漂移见 SUMMARY.json。
RSDR-Net 保留原论文训练口径，不宣称同训练预算。扫描不用于重选模型、epoch 或超参数。
