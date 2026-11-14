import numpy as np
import torch
from tools.family_study import eight_states,family_loss,family_gain


def test_eight_states_and_per_target_losses_use_their_own_energy():
    rng=np.random.default_rng(20)
    a,b,j,e=[rng.normal(size=(2,32))+1j*rng.normal(size=(2,32)) for _ in range(4)]
    y,label=eight_states(a,b,j,e)
    np.testing.assert_allclose(y[:,7]-y[:,3],a,atol=1e-14)
    np.testing.assert_allclose(y[:,7]-y[:,5],b,atol=1e-14)
    np.testing.assert_array_equal(label[:,1],j)
    np.testing.assert_array_equal(label[:,6],0*j)
    def iq(x):
        return torch.from_numpy(np.stack([x.real,x.imag],axis=-2)).float()
    c=iq(label).requires_grad_()
    losses=family_loss(c,iq(a),iq(b),iq(j),iq(e),'B4')
    assert losses['total'].item()==0
    # Deleting a fraction of every target gives that fraction squared per edge.
    c=iq(.3*y).requires_grad_()
    losses=family_loss(c,iq(a),iq(b),iq(j),iq(e),'B4')
    torch.testing.assert_close(losses['response'],torch.tensor(.09))


def test_family_gain_respects_each_target_and_tail():
    rng=np.random.default_rng(21)
    a,b,j,e=[rng.normal(size=(12,32))+1j*rng.normal(size=(12,32)) for _ in range(4)]
    y,label=eight_states(a,b,j,e)
    c=label+.2*y
    ids=np.repeat(np.arange(4),3)
    gain=family_gain(a,b,j,e,c,ids,budget=.04,tail_limit=.05,complex_control=True)
    for delta,s in [(c[:,7]-c[:,3],a),(c[:,7]-c[:,5],b)]:
        d=(np.abs(gain*delta)**2).sum(-1)/(np.abs(s)**2).sum(-1)
        assert d.mean()<=.04+1e-12
        assert np.mean([np.any(d[ids==p]>.25+1e-12) for p in np.unique(ids)])<=.05


def test_family_tail_constraint_can_bind_before_mean_budget():
    n=60
    a=b=j=np.ones((n,32),dtype=np.complex128)
    e=np.zeros_like(a);ids=np.repeat(np.arange(20),3)
    c=np.zeros((n,8,32),dtype=np.complex128)
    c[:6,7]=1
    mean_only=family_gain(a,b,j,e,c,ids,.04,complex_control=True)
    with_tail=family_gain(a,b,j,e,c,ids,.04,tail_limit=.05,complex_control=True)
    assert abs(mean_only)>.6
    assert .49<abs(with_tail)<=.5
