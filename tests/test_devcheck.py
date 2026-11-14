import numpy as np
from tools.dev_confirmation import counterfactual_inputs


def test_weak_target_intervention_keeps_strong_jammer_noise_fixed():
    rng=np.random.default_rng(10)
    s1,s2,j,e=[rng.normal(size=(3,32))+1j*rng.normal(size=(3,32)) for _ in range(4)]
    y=counterfactual_inputs(s2,s1,j,e)
    np.testing.assert_allclose(y[:,3]-y[:,2],s2,atol=1e-14)
    np.testing.assert_allclose(y[:,2],s1+j+e,atol=1e-14)
    np.testing.assert_allclose(y[:,1],s1+s2+e,atol=1e-14)
