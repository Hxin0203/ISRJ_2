"""Same U-Net tensors and RMS; change only dilation or processing coordinates."""
import torch

from tools.unet_ablation.models import PlainUNet
from .settings import FACTOR


class FactorUNet(PlainUNet):
    def __init__(self):
        super().__init__()
        if FACTOR == "dilation":
            # Keep weight shapes, count, RNG draws and initialization unchanged.
            for block_name, dilation in (("enc0", 1), ("enc1", 2), ("enc2", 4),
                                         ("dec1", 2), ("dec0", 1)):
                for layer in getattr(self, block_name):
                    if isinstance(layer, torch.nn.Conv1d):
                        layer.dilation = (dilation,)
                        layer.padding = (dilation,)

    def forward(self, y):
        if FACTOR == "dilation":
            return super().forward(y)
        # A unitary change of coordinates, no new learned spectral module.
        z = torch.complex(y[:, 0], y[:, 1])
        zf = torch.fft.fftshift(torch.fft.fft(z, norm="ortho"), dim=-1)
        correction = super().forward(torch.stack((zf.real, zf.imag), dim=1))
        cf = torch.complex(correction[:, 0], correction[:, 1])
        c = torch.fft.ifft(torch.fft.ifftshift(cf, dim=-1), norm="ortho")
        return torch.stack((c.real, c.imag), dim=1)
