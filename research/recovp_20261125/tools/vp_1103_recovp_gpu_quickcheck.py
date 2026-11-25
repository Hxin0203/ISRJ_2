"""Deployment precision admission; does not reopen the historical performance gate."""
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,now,provenance,validate_role,read
from vp_1103_recovp_fp32_ablation import make_recovp
from vp_1103_p2_smoothvp import real_parameters
from tools.unet_ablation.runtime import configure_precision

OUT=WORK/'evidence/outputs/recovp_formal_dev'

def run():
    started=now();configure_precision();torch.set_num_threads(2);torch.manual_seed(17)
    path,binding=validate_role('train')
    with np.load(path,allow_pickle=False) as z:
        # Same 16 fixed TRAIN parent IDs, all families and paired states represented.
        ids=read(WORK/'evidence/outputs/exp1_recovp_base_preflight/STATUS.json')['parent_ids']
        index=np.searchsorted(z['parent_id'],ids)
        s,j,e=(z[k][index].reshape(-1,1800) for k in ('s','j','epsilon'))
        waves=np.concatenate((e[::24],(s+e)[::24],(j+e)[::24],(s+j+e)[::24]))
    x=torch.from_numpy(waves.astype(np.complex64)).cuda()
    config=read(WORK/'evidence/outputs/exp0_anchor_audit/SUMMARY.json')['locked_config']
    rows={}
    for arm in ('ReCoVP-Base','ReCoVP-NoEquiv'):
        torch.manual_seed(17);m=make_recovp(config,arm).cuda().eval()
        with torch.no_grad():
            out=m.factor(x);base=out['restored'];scale_errors=[]
            for amplitude,phase in ((.1,.7),(10.,-2.1)):
                c=torch.tensor(amplitude*np.exp(1j*phase),dtype=torch.complex64,device='cuda')
                scaled=m.factor(c*x)['restored']
                scale_errors.append(float(torch.linalg.vector_norm(scaled-c*base)/torch.linalg.vector_norm(c*base)))
            perm=torch.randperm(m.k,device='cuda')
            shuffled=m.factor(x,anchors_override=out['anchors'][:,perm])['restored']
            permutation=float(torch.linalg.vector_norm(shuffled-base)/torch.linalg.vector_norm(base))
            finite=bool(torch.isfinite(base).all())
        # Backpropagate a nonzero correction objective to every learned head.
        m.zero_grad(set_to_none=True);m.factor(x[:8])['correction'].abs().square().mean().backward()
        gradients={name:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0)
                   for name,p in m.named_parameters()}
        rows[arm]={'parameters':real_parameters(m),'scale_relative_errors':scale_errors,
            'equivariance_required':arm=='ReCoVP-Base','permutation_relative_error':permutation,
            'finite':finite,'all_parameters_nonzero_finite_gradient':all(gradients.values()),'gradient_tensors':gradients,
            'feature_dtype':str(out['global_features'].dtype),'dictionary_dtype':str(out['b'].dtype)}
        assert 56728<=real_parameters(m)<=62700 and finite and all(gradients.values())
        assert permutation<1e-5
        if arm=='ReCoVP-Base':assert max(scale_errors)<1e-5
        del m,out
    result={**provenance(__file__,WORK/'tools/vp_1103_recovp_fp32.py',WORK/'tools/vp_1103_recovp_fp32_ablation.py',WORK/'tools/vp_1103_recovp_fp32_anchors.py'),
        'started_utc':started,'ended_utc':now(),'status':'PASS','data':binding,'models':rows,
        'note':'NoEquiv scale invariance is not required; same precision, gradients and permutation requirements.'}
    atomic_json(OUT/'FP32_QUICKCHECK.json',result);print({k:{x:v for x,v in row.items() if x!='gradient_tensors'} for k,row in rows.items()},flush=True)
    return result

if __name__=='__main__':run()
