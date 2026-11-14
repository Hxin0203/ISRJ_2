import numpy as np
from paper2.scenes_1103 import conditions, scene


def test_1103_grid_counts_and_steps_are_exact():
    assert len(conditions('main')) == 168
    assert len(conditions('pd')) == 310
    assert len(conditions('sliceA')) == 21
    assert len(conditions('sliceB')) == 20
    assert len(conditions('retention')) == 42
    assert all(c['count']==512 for c in conditions('pd'))


def test_common_parents_change_only_snr_jsr_scaling():
    a = scene('engineering','main',3,'repeat',-20,10)
    b = scene('engineering','main',3,'repeat',0,30)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[2], 10*b[2], atol=1e-12)
    np.testing.assert_allclose(10*a[1], b[1], atol=1e-12)
    assert a[3]['paired_invariant_sha256']==b[3]['paired_invariant_sha256']


def test_width_intervention_fixes_nuisance_and_total_energy():
    a = scene('engineering','sliceA',2,'repeat',-10,20,width=3,repeats=2)
    b = scene('engineering','sliceA',2,'repeat',-10,20,width=84,repeats=2)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])
    assert a[3]['paired_invariant_sha256']==b[3]['paired_invariant_sha256']
    for s,j,e,m in (a,b):
        assert np.isclose(10*np.log10(np.sum(abs(j)**2)/np.sum(abs(s)**2)),20)
        assert m['sampling_phase_samples']==0
        assert m['forward_delay_samples']==1
        assert m['replay_gap_samples']==0


def test_retention_groups_are_independent_but_pair_across_snr():
    a = scene('engineering','retention',0,'direct',-20,0)
    b = scene('engineering','retention',0,'repeat',-20,0)
    c = scene('engineering','retention',0,'direct',0,0)
    assert a[3]['parent_id'] != b[3]['parent_id']
    assert not np.array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[0], c[0])
    np.testing.assert_allclose(a[2], 10*c[2], atol=1e-12)
