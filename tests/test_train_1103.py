import numpy as np
import torch

from paper2.network import CorrectionNet, training_loss
from paper2.train_1103 import make_batch


def test_1103_batch_keeps_state_and_iq_axes_for_model_update():
    rng = np.random.default_rng(12)
    shape = (1, 3, 8, 1800)
    arrays = {
        "s": (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64),
        "j": (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64),
        "epsilon": (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64),
    }
    batch = make_batch(arrays, np.array([1]), torch.device("cpu"))
    assert batch["y"].shape == (32, 2, 1800)

    net = CorrectionNet()
    correction = net(batch["y"]).reshape(8, 4, 2, 1800)
    loss = training_loss(correction, batch["s"], batch["j"], batch["e"], "B4")["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in net.parameters())
