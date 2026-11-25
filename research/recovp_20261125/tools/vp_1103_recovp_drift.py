"""Paired operator review with all, anchor-only, and no operator re-estimation."""
import numpy as np
import torch
from vp_1103_p2_evaluate import iq_tensor
from vp_1103_p2_smoothvp import operator_apply,noise_gain

def three_layers(model,data):
    model=model.cpu().eval();size=len(data["parent_id"])
    rows={layer:{key:np.zeros((size,3)) for key in ("leak_over_target_energy","drift_over_target_energy",
          "cross_term_over_target_energy","damage","noise_gain_H0","noise_gain_H1")} for layer in ("frozen_B_omega","frozen_anchors","full_reestimate")}
    errors={layer:0. for layer in rows}
    with torch.inference_mode():
        for parent in range(size):
            for family in range(3):
                s,j,e=[data[key][parent,family].astype(np.complex128) for key in ("s","j","epsilon")]
                y0,y1=j+e,s+j+e
                p0=model.factor(iq_tensor(y0,"cpu"));full=model.factor(iq_tensor(y1,"cpu"))
                anchored=model.factor(iq_tensor(y1,"cpu"),anchors_override=p0["anchors"])
                st,x0,x1=[torch.from_numpy(v) for v in (s,y0,y1)]
                c0=operator_apply(p0,x0);energy=st.abs().square().sum()
                for layer,p1 in (("frozen_B_omega",p0),("frozen_anchors",anchored),("full_reestimate",full)):
                    c1=operator_apply(p1,x1)
                    leak=-operator_apply(p1,st)
                    drift=c0-operator_apply(p1,x0)
                    lhs=x1-c1-(x0-c0)-st
                    error=float(torch.linalg.vector_norm(lhs-leak-drift)/torch.linalg.vector_norm(x1))
                    errors[layer]=max(errors[layer],error)
                    values={"leak_over_target_energy":leak.abs().square().sum()/energy,
                            "drift_over_target_energy":drift.abs().square().sum()/energy,
                            "cross_term_over_target_energy":2*(leak.conj()*drift).real.sum()/energy,
                            "damage":lhs.abs().square().sum()/energy,
                            "noise_gain_H0":noise_gain(p0).sum(),"noise_gain_H1":noise_gain(p1).sum()}
                    for key,value in values.items():rows[layer][key][parent,family]=float(value)
    if max(errors.values())>=1e-10:raise ValueError("Three-layer identity failed")
    return rows,errors
