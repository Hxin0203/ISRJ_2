import numpy as np
import torch
import pytest
from paper2.network import CorrectionNet
from tools.restore_waveform import restore_array


def test_restore_uses_observed_complex_array_and_global_correction_gain(tmp_path):
    torch.manual_seed(22)
    model=CorrectionNet().eval()
    path=tmp_path/'model.pt';torch.save({'state':model.state_dict()},path)
    rng=np.random.default_rng(22)
    y=(rng.normal(size=(2,1800))+1j*rng.normal(size=(2,1800))).astype(np.complex64)
    restored=restore_array(y,path,0,device='cpu')
    np.testing.assert_array_equal(restored,y)
    gain=.7+.1j
    restored=restore_array(y,path,gain,device='cpu')
    iq=torch.from_numpy(np.stack([y.real,y.imag],axis=1))
    with torch.no_grad():
        c=model(iq).numpy()
    expected=y-gain*(c[:,0]+1j*c[:,1])
    np.testing.assert_allclose(restored,expected,atol=2e-6,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA batch-shape regression')
def test_cuda_inference_is_stable_for_small_vs_full_observation_batches(tmp_path):
    torch.manual_seed(22)
    path=tmp_path/'model.pt';torch.save({'state':CorrectionNet().state_dict()},path)
    rng=np.random.default_rng(22)
    y=(rng.normal(size=(2,1800))+1j*rng.normal(size=(2,1800))).astype(np.complex64)
    full=np.concatenate([y,np.zeros((126,1800),np.complex64)])
    small=restore_array(y,path,1,device='cuda')
    large=restore_array(full,path,1,device='cuda')[:2]
    np.testing.assert_allclose(small,large,atol=1e-6,rtol=1e-6)
