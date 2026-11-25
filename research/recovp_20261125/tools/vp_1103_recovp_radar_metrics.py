"""Model-independent paired radar metrics; no model or dataset inputs."""
import numpy as np
from vp_1103_common import WORK
from paper2.evaluation_1103_v4 import group_metrics,radar_metrics,coherent_metrics,energy
from paper2.radar_eval import pulse_compress,doppler_process

PRIMARY=('jcr_db','rd_residual_peak_db','cpl_db','abs_cpl_db','sin2_theta_s',
         'bvr00_db','abs_bvr00_db','bvr01_db','abs_bvr01_db')
DEFINITION=('SNR/JSR read native active-support metadata. JCR uses T(Y01)-E; RD peak uses '
 'the same residual and the 9032 valid CFAR cells, normalized by mean RD noise power. '
 'CPL and angle use T(Y11)-T(Y01); signed and absolute dB are reported separately. '
 'BVR is an energy ratio, not a test of whiteness or independence. RD peak is a risk '
 'descriptor, not a universal CFAR false-alarm bound. Actual CFAR is always measured. '
 'Matched filtering is not generally unitary, and correlated CUTs do not justify an '
 'independent-cell product formula. A global phase rotation can leave CPL/angle zero; '
 'signed complex gain and pulse-phase errors are retained.')

def metrics(s,j,e,observed,u,fd,coherent=False):
    s,j,e,u=(np.asarray(v,np.complex128) for v in (s,j,e,u))
    residual=u[:,2]-e;increment=u[:,3]-u[:,2]
    es=energy(s);ed=energy(increment);inner=(s.conj()*increment).sum((-2,-1))
    gain=inner/np.maximum(es,1e-300)
    def db(v):return 10*np.log10(np.maximum(v,1e-300))
    ej=energy(j)
    jcr=db(np.maximum(ej,1e-300)/(energy(residual)+1e-12))
    jcr=np.where(ej>0,jcr,np.nan)  # Undefined when no jammer exists.
    cpl=-20*np.log10(np.abs(gain)+1e-12)
    angle=np.clip(1-np.abs(inner)**2/(es*ed+1e-12),0,1)
    b00=db(energy(u[:,0])/(energy(e)+1e-12))
    b01=db(energy(u[:,2])/(energy(e)+1e-12))
    compressed,_=pulse_compress(np.stack((residual,e),axis=1))
    rd,_=doppler_process(compressed)
    valid=rd[...,36:-36]  # Same CFAR detection domain: 8*(1201-72)=9032.
    peak=abs(valid[:,0]).max((-2,-1))**2
    mean_noise=(abs(valid[:,1])**2).mean((-2,-1))
    out=group_metrics(s,j,e,u)
    out.update(radar_metrics(s,observed,u,fd))
    out.update(jcr_db=jcr,rd_residual_peak_db=db(peak/(mean_noise+1e-12)),
        cpl_db=cpl,abs_cpl_db=abs(cpl),sin2_theta_s=angle,bvr00_db=b00,abs_bvr00_db=abs(b00),
        bvr01_db=b01,abs_bvr01_db=abs(b01),c_s_real=gain.real,c_s_imag=gain.imag,
        e_U=out['u_nmse'],e_S=out['damage'],e_0_pool=out['no_jammer_pooled_nmse'])
    if coherent:
        out.update(coherent_metrics(s,u[:,3]))
        out.update({'increment_'+k:v for k,v in coherent_metrics(s,increment).items()})
    return out

def unit_tests():
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import lfm_reference
    rng=np.random.default_rng(20261124)
    s=np.zeros((2,8,1800),complex);s[:,:,300:900]=lfm_reference()
    j=rng.normal(size=s.shape)+1j*rng.normal(size=s.shape)
    e=.05*(rng.normal(size=s.shape)+1j*rng.normal(size=s.shape))
    y=states_from_components(s,j,e);m=metrics(s,j,e,y,y,np.zeros(2))
    assert np.max(abs(m['jcr_db']))<1e-10 and np.max(abs(m['cpl_db']))<1e-10
    assert np.max(abs(m['bvr00_db']))<1e-10 and np.max(m['sin2_theta_s'])<1e-10
    reference=states_from_components(s,np.zeros_like(j),e)
    r=metrics(s,j,e,y,reference,np.zeros(2))
    assert np.max(r['damage'])<1e-25 and np.min(r['jcr_db'])>100
    assert np.max(abs(r['bvr01_db']))<1e-10
    # Explicit gain/phase counterexamples: signed losses, magnitude-only limitation.
    doubled=reference.copy();doubled[:,3]=e+2*s
    d=metrics(s,j,e,y,doubled,np.zeros(2))
    assert np.max(abs(d['cpl_db']+20*np.log10(2)))<1e-9
    rotated=reference.copy();rotated[:,3]=e+1j*s
    p=metrics(s,j,e,y,rotated,np.zeros(2))
    assert np.max(abs(p['cpl_db']))<1e-9 and np.min(p['c_s_imag'])>.999
    return {'status':'PASS','identity_jcr_max_db':float(abs(m['jcr_db']).max()),
        'ideal_removal_min_jcr_db':float(r['jcr_db'].min()),'gain2_cpl_db':d['cpl_db'].tolist(),
        'quarter_turn_cpl_db':p['cpl_db'].tolist(),'quarter_turn_gain_imag':p['c_s_imag'].tolist(),
        'cfar_valid_cells':9032,'mathematical_interpretation':DEFINITION}

if __name__=='__main__':
    from vp_1103_common import atomic_json
    value=unit_tests();atomic_json(WORK/'evidence/outputs/recovp_formal_dev/METRIC_UNIT_TESTS.json',value);print(value)
