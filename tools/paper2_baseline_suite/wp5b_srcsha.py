import hashlib
for p in ('tools/unet_norm_screen/model.py','tools/unet_factor_screen/model.py','tools/unet_factor_screen/settings.py','tools/unet_ablation/models.py','paper2/network.py','tools/paper2_baseline_suite/wp5b_suite.py','tools/paper2_baseline_suite/equal_pfa_suite.py','tools/paper2_baseline_suite/classical_suite.py','tools/paper2_baseline_suite/rsdr_suite.py','paper2/data_1103.py'):
    print(hashlib.sha256(open(p,'rb').read()).hexdigest()[:16], p)
