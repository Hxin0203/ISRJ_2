"""Exactly 58,943 active real parameters; widths selected by count only."""
import torch
from torch import nn
import torch.nn.functional as F
from vp_1103_common import WORK
from tools.unet_ablation.models import DoubleConv

TARGET_PARAMETERS=58943
WIDTHS=(21,40,74)

def count_formula(a,b,c):
    return 9*a*a+9*b*b+3*c*c+6*a*b+6*b*c+20*a+12*b+6*c+2

class MatchedUNet(nn.Module):
    def __init__(self):
        super().__init__();a,b,c=WIDTHS
        self.enc0=DoubleConv(2,a);self.enc1=DoubleConv(a,b);self.enc2=DoubleConv(b,c)
        self.dec1=DoubleConv(c+b,b);self.dec0=DoubleConv(b+a,a);self.out=nn.Conv1d(a,2,1)
    def forward(self,y):
        scale=y.square().sum(1,keepdim=True).mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        f0=self.enc0(y/scale);f1=self.enc1(F.max_pool1d(f0,2));f2=self.enc2(F.max_pool1d(f1,2))
        d1=self.dec1(torch.cat((F.interpolate(f2,size=f1.shape[-1],mode='linear',align_corners=False),f1),1))
        d0=self.dec0(torch.cat((F.interpolate(d1,size=f0.shape[-1],mode='linear',align_corners=False),f0),1))
        return self.out(d0)*scale

def make_unet(config=None,arm='UNet-Matched58943'):
    if arm!='UNet-Matched58943':raise ValueError(arm)
    return MatchedUNet()
