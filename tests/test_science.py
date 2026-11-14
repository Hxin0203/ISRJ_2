import numpy as np
import pytest
import torch

from paper2.science import four_states, metrics, optimal_gain, parent_mean
from paper2.network import CorrectionNet, training_loss, matched_filter


def components():
    rng = np.random.default_rng(4)
    return [rng.normal(size=(6, 1800)) + 1j * rng.normal(size=(6, 1800)) for _ in range(3)]


def test_four_states_preserve_components_without_rescaling():
    s, j, e = components()
    y, c = four_states(s, j, e)
    np.testing.assert_allclose(y[:, 3] - y[:, 2], s, atol=1e-12)
    np.testing.assert_allclose(y[:, 1] - y[:, 0], s, atol=1e-12)
    np.testing.assert_array_equal(c[:, 0], np.zeros_like(s))
    np.testing.assert_array_equal(c[:, 2], j)
    np.testing.assert_array_equal(c[:, 3], j)


def test_keep_and_perfect_cancellation_and_cross_term():
    s, j, e = components()
    correction = np.stack((0*j, 0*j, j, j), axis=1)
    perfect = metrics(s, j, e, correction)
    np.testing.assert_allclose(perfect['u_nmse'], 0, atol=1e-25)
    np.testing.assert_allclose(perfect['damage'], 0, atol=1e-25)
    rng = np.random.default_rng(5)
    c = correction + rng.normal(size=correction.shape)
    m = metrics(s, j, e, c, 0.7+0.2j)
    np.testing.assert_allclose(m['u_nmse'], m['r0_u'] + m['rs_u'] + m['cross_u'], atol=1e-12)
    m2 = metrics(s, j, e, c, 2)
    m1 = metrics(s, j, e, c, 1)
    np.testing.assert_allclose(m2['damage'], 4*m1['damage'])


def test_gain_optimizer_dominates_dense_grid_and_respects_budget():
    s, j, e = components()
    c = np.stack((0*j, 0*j, (0.8+0.2j)*j, (0.8+0.2j)*j+0.1*s), axis=1)
    for complex_control in (False, True):
        gain = optimal_gain(s, j, e, c, 0.01, complex_control=complex_control)
        assert abs(gain) <= 2 + 1e-12
        assert metrics(s,j,e,c,gain)['damage'].mean() <= 0.01 + 1e-12
        chosen = metrics(s,j,e,c,gain)['u_nmse'].mean()
        for alpha in np.linspace(0,2,81):
            m = metrics(s,j,e,c,alpha)
            if m['damage'].mean() <= 0.01:
                assert chosen <= m['u_nmse'].mean()+1e-12


def test_parent_reduction_does_not_count_variants_as_independent():
    ids = np.array([5,5,5,7,7,7])
    np.testing.assert_array_equal(parent_mean(np.arange(6), ids), [1,4])


def test_complex_gain_removes_constant_amplitude_phase_error():
    s,j,e=components()
    k=.8+.2j
    c=np.stack((0*j,0*j,k*j,k*j),axis=1)
    gain=optimal_gain(s,j,e,c,.01,complex_control=True)
    assert gain==pytest.approx(1/k)
    assert metrics(s,j,e,c,gain)['u_nmse'].mean()<1e-28


def test_pair_difference_uses_final_correction_and_not_internal_mask():
    s,j,e=components()
    y,_=four_states(s,j,e)
    mask=.7
    c=(1-mask)*y
    expected=(1-mask)**2
    np.testing.assert_allclose(metrics(s,j,e,c)['damage'],expected)


def test_model_inference_is_observation_only_and_physical_scale_equivariant():
    torch.manual_seed(2)
    net = CorrectionNet()
    x = torch.randn(2,2,1800)
    assert net(x).shape == x.shape
    torch.testing.assert_close(net(3*x), 3*net(x), atol=1e-5, rtol=1e-4)
    assert torch.isfinite(net(torch.zeros_like(x))).all()


def test_losses_are_nested_and_target_response_backpropagates():
    y = torch.randn(3,4,2,1800)
    j = torch.randn(3,2,1800)
    s = torch.randn(3,2,1800)
    e = torch.randn(3,2,1800)
    pred = torch.randn_like(y, requires_grad=True)
    base = training_loss(pred,s,j,e,'B1')
    fidelity = training_loss(pred,s,j,e,'B2')
    candidate = training_loss(pred,s,j,e,'B4')
    torch.testing.assert_close(fidelity['total'],base['endpoint']+0.1*fidelity['pc'])
    torch.testing.assert_close(candidate['total'],fidelity['total']+candidate['damage'])
    candidate['damage'].backward()
    assert pred.grad[:,2:].abs().sum() > 0
    assert pred.grad[:,:2].abs().sum() == 0


def test_matched_filter_equals_direct_linear_correlation_energy():
    x = torch.zeros(1,2,1800)
    x[0,0,0] = 1
    mf = matched_filter(x)
    assert mf.shape[-1] >= 1800+600-1
    assert mf.abs().square().sum().item() == pytest.approx(600, rel=1e-5)
