"""Strict numeric equivariance, permutation and exit-node tests."""
import copy
import math
import numpy as np
import torch
from vp_1103_recovp import soft_subspace

def relative(a,b):
    return float((torch.linalg.vector_norm(a-b,dim=-1)/torch.linalg.vector_norm(b,dim=-1).clamp_min(torch.finfo(torch.float64).tiny)).max())

def boundary_checks(k):
    generator=torch.Generator().manual_seed(20261124)
    a=torch.rand(4,k,generator=generator,dtype=torch.float64)
    ell=torch.complex(torch.randn(4,k,generator=generator,dtype=torch.float64),torch.randn(4,k,generator=generator,dtype=torch.float64))
    eta=torch.full((4,),.7,dtype=torch.float64)
    rows=[]
    for value in (1e-2,1e-6,1e-10,1e-14,0.):
        changed=a.clone();changed[:,-1]=value
        gamma=soft_subspace(changed,ell,eta)
        reduced=soft_subspace(changed[:,:-1],ell[:,:-1],eta)
        rows.append({"activity":value,"exit_row_column_max":float(gamma[:,-1].abs().max()),
                     "active_block_difference":float((gamma[:,:-1,:-1]-reduced).abs().max()),
                     "minimum_eigenvalue":float(torch.linalg.eigvalsh(gamma).min()),
                     "hermitian_error":float((gamma-gamma.mH).abs().max())})
    passed=all(row["minimum_eigenvalue"]>=-1e-12 for row in rows) and all(
        row["exit_row_column_max"]<1e-12 and row["active_block_difference"]<1e-12 for row in rows[-2:])
    return {"status":"PASS" if passed else "FAIL","curves":rows}

def math_checks(model,wave):
    model.eval()
    wave=torch.as_tensor(wave,dtype=torch.complex128,device=next(model.parameters()).device)
    result={"samples":len(wave),"internal_arithmetic":"real float64 features/CNN/heads and complex128 solve; stored parameters float32",
            "cases":{},"boundary":boundary_checks(model.k)}
    with torch.inference_mode():
        for policy in ("initialized_damping","open_damping_stress"):
            current=model if policy=="initialized_damping" else copy.deepcopy(model)
            if policy=="open_damping_stress":
                current.shared_precision_head[-1].weight.zero_();current.shared_precision_head[-1].bias.fill_(-12.)
            base=current.factor(wave)
            permutation=torch.tensor(np.random.default_rng(20261124).permutation(model.k),device=wave.device)
            permuted=current.factor(wave,anchors_override=base["anchors"][:,permutation])
            p_error=relative(permuted["restored"],base["restored"])
            scales=[]
            for amplitude,angle in ((.1,.7),(10.,-2.1)):
                c=amplitude*complex(math.cos(angle),math.sin(angle))
                changed=current.factor(c*wave)
                core=relative(changed["restored"],c*base["restored"])
                invariant_features=float((changed["global_features"]-base["global_features"]).abs().max())
                anchors_equal=bool(torch.equal(changed["anchors"],base["anchors"]))
                y32=wave.to(torch.complex64)
                r32=y32-current.factor(y32)["correction"].to(torch.complex64)
                cy32=(c*wave).to(torch.complex64)
                cr32=cy32-current.factor(cy32)["correction"].to(torch.complex64)
                fp32=relative(cr32.to(torch.complex128),c*r32.to(torch.complex128))
                scales.append({"amplitude":amplitude,"phase":angle,"complex128_relative_error":core,
                               "float32_relative_error":fp32,"invariant_feature_max_error":invariant_features,
                               "anchors_exactly_equal":anchors_equal})
            mask_peak=base["mask_peak"][...,0]
            mask_error=float((base["mask"].amax(-1)-mask_peak/mask_peak.clamp_min(.05)).abs().max())
            fp32_permutation=relative(permuted["restored"].to(torch.complex64).to(torch.complex128),base["restored"].to(torch.complex64).to(torch.complex128))
            result["cases"][policy]={"complex128_permutation_relative_error":p_error,"float32_permutation_relative_error":fp32_permutation,
                "scales":scales,"peak_anchor_error":mask_error,"fraction_peak_below_guard":float((mask_peak<.05).double().mean()),
                "gamma_minimum_eigenvalue":float(torch.linalg.eigvalsh(base["gamma"]).min()),
                "pulse_energy":float(current.unit_spectrum.abs().square().sum()/4096)}
    passed=result["boundary"]["status"]=="PASS"
    for row in result["cases"].values():
        passed &= row["complex128_permutation_relative_error"]<1e-11 and row["float32_permutation_relative_error"]<1e-5
        passed &= row["peak_anchor_error"]<1e-12 and row["gamma_minimum_eigenvalue"]>=-1e-12
        passed &= all(item["complex128_relative_error"]<1e-11 and item["float32_relative_error"]<1e-5 and item["anchors_exactly_equal"] for item in row["scales"])
    result["status"]="PASS" if passed else "FAIL"
    return result

def selected_waves(data):
    waves=[]
    for parent in range(len(data["parent_id"])):
        family=parent%3
        pulse=int(abs(data["j"][parent,family]).sum(-1).argmax())
        waves.append(sum(data[key][parent,family,pulse].astype(np.complex128) for key in ("s","j","epsilon")))
    return np.stack(waves)
