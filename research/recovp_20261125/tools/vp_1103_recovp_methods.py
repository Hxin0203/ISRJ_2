"""Observed-only, hash-bound adapters. Never loads historical dataset arrays."""
from pathlib import Path
import numpy as np
import torch
from vp_1103_common import WORK,read,sha
from vp_1103_recovp_fp32_ablation import make_recovp
from tools.unet_ablation.models import make_model
from paper2.evaluation_1103_v4 import restore_model

ASSETS=Path('G:/Project Code/ISRJ_222/outputs/rsdr_unified_20261115/assets')
CONFIG={'method':'E0','window':'rect','rounds':1,'k':12}

class Method:
    def __init__(self,name,seed=17,device='cuda'):
        self.name,self.device=name,device;self.model=None;self.adapter=None
        self.binding={'name':name,'seed':seed,'inputs':'observations only'}
        if name=='Unprocessed':return
        if name in ('Max-TF','GMM-OMP'):
            from paper2.baselines_1103 import run_max_tf,run_gmm_omp
            self.adapter=run_max_tf if name=='Max-TF' else run_gmm_omp
            self.binding.update(training='none',implementation='frozen local formula reproduction')
            return
        if name=='RSDR-Net':
            import tools.rsdr_adapter as rsdr
            rsdr.ASSETS=ASSETS
            self.adapter=rsdr.RSDRNet(device=device,batch_pulses=64)
            count=sum(p.numel()*(2 if p.is_complex() else 1) for m in
                [*self.adapter.experts,*self.adapter.routers,self.adapter.gate] for p in m.parameters())
            self.binding.update(seed=40312102,epoch=32,assets=rsdr.EXPECTED,effective_real_parameters=count,
                training='first-paper F-R7; seed40312102; epoch32; 16384 updates/component; unequal training budget')
            return
        if name=='ReCoVP-Winner':
            winner=read(WORK/'evidence/outputs/recovp_formal_dev/WINNER.json')
            actual=winner['arm'];self.model=make_recovp(CONFIG,actual)
            path=WORK/'outputs/recovp_formal_dev'/actual/'17/epoch_256.pt'
            assert sha(path)==winner['seed17_checkpoint_sha256']
        elif name.startswith('ReCoVP-'):
            self.model=make_recovp(CONFIG,name)
            path=WORK/'outputs/recovp_formal_dev'/name/str(seed)/'epoch_256.pt'
        elif name=='P0-NoDechirp-Tail':
            self.model=make_model('no_dechirp')
            path=WORK/'outputs/p0_dechirp_tail_2x2/P0-NoDechirp-Tail'/str(seed)/'epoch_256.pt'
            expected=read(WORK/'evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json')['new_arms']['P0-NoDechirp-Tail'][str(seed)]['checkpoint_sha256']
            assert sha(path)==expected
        elif name in ('UNet-Tail','CorrectionNet-Tail'):
            self.model=make_model('unet' if name=='UNet-Tail' else 'full')
            path=WORK/'outputs/vp_1103_checkpoint_cache'/f'{name}.pt'
            expected=read(WORK/'outputs/vp_1103_checkpoint_cache/VERIFIED_IMPORT.json')['models'][name]['local_checkpoint_sha256']
            assert sha(path)==expected
        else:raise ValueError(name)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload.get('seed',seed)==seed and payload['epoch']==256
        self.model.load_state_dict(payload['model']);self.model.eval()
        total=sum(p.numel()*(2 if p.is_complex() else 1) for p in self.model.parameters())
        if name in ('P0-NoDechirp-Tail','CorrectionNet-Tail'):
            with torch.enable_grad():self.model(torch.ones(1,2,1800)).square().mean().backward()
            effective=sum(p.numel() for p in self.model.parameters() if p.grad is not None)
            self.model.zero_grad(set_to_none=True)
        else:effective=total
        self.model.to(device)
        self.binding.update(checkpoint=str(path.relative_to(WORK)),checkpoint_sha256=sha(path),epoch=256,
            registered_real_parameters=total,effective_real_parameters=effective)

    def restore(self,states):
        if self.name=='Unprocessed':return np.asarray(states).copy()
        if self.model is not None:return restore_model(self.model,states,self.device,batch_pulses=64)
        if self.name=='RSDR-Net':return self.adapter.restore(states)
        return np.stack([self.adapter(x.reshape(32,1800)).reshape(4,8,1800) for x in states])

    def close(self):
        self.model=None;self.adapter=None
        if torch.cuda.is_available():torch.cuda.empty_cache()
