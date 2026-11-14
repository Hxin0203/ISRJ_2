import numpy as np
import pytest

from paper2.scan_1103 import binomial_interval, digest, evaluate_condition, process_batch


def test_exact_binomial_interval_handles_zero_and_all_events():
    lo,hi = binomial_interval(0,512)
    assert lo==0 and 0<hi<.01
    lo,hi = binomial_interval(512,512)
    assert .99<lo<1 and hi==1


def test_parent_pair_differences_and_reference_are_correct():
    c=dict(jam_type='repeat',snr_db=-20,jsr_db=35,count=1)
    for method in ('keep','reference'):
        values,meta,_=process_batch('engineering','main',c,[0],method)
        assert len(meta)==1
        assert not any('paired_reference__paired_' in k for k in values)
        for name in ('u_nmse','damage','pd','frame_pfa_noise'):
            np.testing.assert_array_equal(values[f'paired_{method}__{name}'],0)
        if method=='reference':
            assert values['u_nmse'][0]==0
            assert values['damage'][0]<1e-26


def test_oracle_receives_support_only_in_jammer_states(monkeypatch):
    seen=[]
    def oracle(y,j):
        seen.append(j.copy())
        return y,dict(converged_pulses=0,pulse_count=len(y))
    monkeypatch.setattr('paper2.scan_1103.run_bcs_oracle',oracle)
    c=dict(jam_type='repeat',snr_db=-20,jsr_db=35,count=1)
    process_batch('engineering','main',c,[0],'bcs_oracle')
    assert len(seen)==1
    states=seen[0].reshape(4,8,1800)
    assert not states[0].any() and not states[1].any()
    np.testing.assert_array_equal(states[2],states[3])
    assert states[2].any()


def test_digest_ignores_mapping_order():
    assert digest(dict(a=1,b=2))==digest(dict(b=2,a=1))


def test_resume_rejects_changed_identity_and_corrupt_parent_block(tmp_path):
    c=dict(jam_type='repeat',snr_db=-20,jsr_db=35,count=1)
    args=(tmp_path/'local',tmp_path/'light',{'source':'engineering'},
          'engineering','main',0,c,'keep',None,'cpu',None)
    result=evaluate_condition(*args)
    assert evaluate_condition(*args)['identity_sha256']==result['identity_sha256']
    changed=list(args)
    changed[2]={'source':'changed'}
    with pytest.raises(RuntimeError,match='identity mismatch'):
        evaluate_condition(*changed)
    path=tmp_path/'local'/result['blocks'][0]['file']
    path.write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='SHA mismatch'):
        evaluate_condition(*args)
