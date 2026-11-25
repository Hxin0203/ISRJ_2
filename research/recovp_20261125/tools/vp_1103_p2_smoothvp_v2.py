"""Peak-anchored, permutation-equivariant CPU prototype for K=8 replay VP."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from vp_1103_p2_smoothvp import SmoothVPNet, N, K, MIN_PRECISION


class SmoothVPNetV2(SmoothVPNet):
    def __init__(self, coordinates="RAW_IQ"):
        super().__init__(coordinates)
        del self.spatial_head
        del self.precision_head
        self.refine_head = nn.Sequential(nn.Linear(88, 64), nn.SiLU(), nn.Linear(64, 1))
        self.window_head = nn.Sequential(nn.Conv1d(26, 12, 3, padding=1), nn.SiLU(),
                                          nn.Conv1d(12, 1, 3, padding=1))
        self.shared_precision_head = nn.Sequential(nn.Linear(90, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.normal_(self.refine_head[-1].weight, std=.005)
        nn.init.zeros_(self.refine_head[-1].bias)
        nn.init.normal_(self.shared_precision_head[-1].weight, std=.01)
        nn.init.constant_(self.shared_precision_head[-1].bias, 22.)

    @staticmethod
    def anchors(matched):
        """Nonlearned physical candidates with six-sample minimum separation."""
        with torch.no_grad():
            c = matched.detach()
            local = c >= F.max_pool1d(c[:, None], 13, stride=1, padding=6)[:, 0]
            position = torch.arange(N, device=c.device)[None]
            legal = (position >= 2) & (position <= N - 3)
            available = legal.expand_as(c).clone()
            anchors = []
            fallback = torch.zeros(len(c), dtype=torch.int64, device=c.device)
            for _ in range(K):
                scores = c.masked_fill(~(available & local), -torch.inf)
                missing = ~torch.isfinite(scores).any(-1)
                fallback += missing
                scores = torch.where(missing[:, None], c.masked_fill(~available, -torch.inf), scores)
                index = scores.argmax(-1)
                anchors.append(index)
                available &= (position - index[:, None]).abs() >= 6
            return torch.stack(anchors, -1).to(c.dtype), fallback

    @staticmethod
    def sample_features(features, positions):
        x = positions / (N - 1) * 2 - 1
        grid = torch.stack((x, torch.zeros_like(x)), -1)[:, None]
        sampled = F.grid_sample(features[:, :, None], grid, align_corners=True, mode="bilinear")
        return sampled[:, :, 0].transpose(1, 2)

    def factor(self, iq, anchors_override=None):
        y, scale, f2, features, matched = self.encode(iq)
        anchors, fallback = self.anchors(matched)
        if anchors_override is not None:
            if anchors_override.shape != anchors.shape:
                raise ValueError("Expected one set of eight anchors for each pulse")
            anchors = anchors_override
        local = self.sample_features(features, anchors)
        peak = self.sample_features(matched[:, None], anchors)
        shared = f2.mean(-1)[:, None].expand(-1, K, -1)
        anchor_position = (anchors / N).unsqueeze(-1)
        refine_input = torch.cat((local, shared, anchor_position, torch.log1p(peak)), -1)
        delta = 2. * torch.tanh(self.refine_head(refine_input)[..., 0])
        delay = anchors + delta
        phase = -2j * math.pi * self.frequency[None, None] * delay.double()[..., None]
        template = torch.fft.ifft(self.pulse_spectrum[None, None] * torch.exp(phase))[..., :N]
        z = y[:, None].to(torch.complex128) * template.conj()
        channels = []
        for length in (5, 11):
            smooth = torch.complex(
                F.avg_pool1d(z.real, length, stride=1, padding=length // 2, count_include_pad=False),
                F.avg_pool1d(z.imag, length, stride=1, padding=length // 2, count_include_pad=False))
            channels.extend((smooth.real, smooth.imag, smooth.abs()))
        channels.extend((template.abs(), y.abs()[:, None].expand(-1, K, -1)))
        physical = torch.stack(channels, 2).float()
        broadcast = features[:, None].expand(-1, K, -1, -1)
        window_input = torch.cat((broadcast, physical), 2).flatten(0, 1)
        masks = torch.sigmoid(self.window_head(window_input)).reshape(-1, K, N)
        masks = F.avg_pool1d(masks, 3, stride=1, padding=1, count_include_pad=False)
        b = masks.double() * template
        b = (b / torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min(1e-12)).transpose(1, 2)
        yd = y.to(torch.complex128)
        rho = (b.mH @ yd[..., None])[..., 0].abs().square()
        noise = (y.abs().square().median(-1).values / math.log(2)).double().clamp_min(1e-12)
        rho = rho / noise[:, None]
        # Both learned heads operate on candidate attributes with shared weights.
        precision_input = torch.cat((shared, local, torch.log1p(rho).float()[..., None],
                                     masks.mean(-1)[..., None], anchor_position, torch.log1p(peak)), -1)
        log_precision = self.shared_precision_head(precision_input)[..., 0].double()
        exponent = log_precision + 2. * (math.log(9.) - torch.log1p(rho))
        precision = MIN_PRECISION + torch.exp(30. * torch.tanh(exponent / 30.))
        gram = b.mH @ b
        normal = .5 * (gram + gram.mH) + torch.diag_embed(precision).to(torch.complex128)
        right = torch.linalg.solve(normal, b.mH)
        beta = (right @ yd[..., None])[..., 0]
        correction = (b @ beta[..., None])[..., 0] * scale[:, 0]
        return {"correction": correction, "b": b, "right": right, "delay": delay,
                "mask": masks, "log_precision": log_precision, "precision": precision,
                "rho": rho, "scale": scale, "gram": gram, "anchors": anchors,
                "refinement": delta, "anchor_fallback_count": fallback}
