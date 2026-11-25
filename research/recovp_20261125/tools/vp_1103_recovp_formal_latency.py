"""Current precision suite using the established synchronized benchmark kernel."""
import numpy as np
import torch
from torch import nn
from vp_1103_common import WORK,atomic_json,now,validate_role,sha
from vp_1103_recovp_methods import Method
from benchmark_latency_unified import measure,macs_count,Restore
from tools.unet_ablation.runtime import configure_precision

class RSDRGroup(nn.Module):
    """One native eight-pulse observed row; other feature rows are independent."""
    def __init__(self,adapter):
        super().__init__();self.experts=nn.ModuleList(adapter.experts);self.routers=nn.ModuleList(adapter.routers)
        self.gate=adapter.gate;self.weights=adapter.weights
        self.register_buffer('mean',adapter.mean);self.register_buffer('std',adapter.std)
    def forward(self,iq):
        from tools.rsdr_vendor.gate_core import features
        from tools.rsdr_vendor.m12_model import mix_actions
        if iq.shape!=(8,2,1800):raise ValueError('RSDR has a native eight-pulse group gate')
        outputs=[m(iq) for m in self.experts];a,b=[v['u_hat'] for v in outputs]
        mixed=torch.stack([mix_actions(torch.softmax(r(iq,a,b,*[v['reliability'] for v in outputs])/.25,1),iq,a,b) for r in self.routers]).mean(0)
        def group(x):
            value=torch.zeros((1,4,8,2,1800),device=x.device,dtype=x.dtype)
            value[:,0]=x;return value
        y,aa,bb,mm=[group(v) for v in (iq,a,b,mixed)]
        w=self.weights;anchor=w[0]*y+w[1]*aa+w[2]*bb
        rho=self.gate((features(y,anchor,mm)-self.mean)/self.std)
        return (anchor+rho[...,None,None,None]*(mm-anchor))[0,0]

def run(warmup=50,repeats=200):
    out=WORK/'evidence/outputs/recovp_isrj1_suite';started=now()
    path,binding=validate_role('train')
    with np.load(path,allow_pickle=False) as z:wave=(z['s'][0,0]+z['j'][0,0]+z['epsilon'][0,0]).astype(np.complex64)
    pulse=torch.from_numpy(np.stack((wave.real,wave.imag),1));rows={}
    names=('RSDR-Net','UNet-Tail','CorrectionNet-Tail','P0-NoDechirp-Tail','ReCoVP-Winner')
    for device in ('cpu','cuda'):
        rows[device]={}
        for name in names:
            method=Method(name,device=device);configure_precision();torch.set_num_threads(2)
            model=RSDRGroup(method.adapter) if name=='RSDR-Net' else Restore(method.model)
            model.eval();timings={}
            if name=='RSDR-Net':
                states=np.broadcast_to(wave,(1,4,8,1800)).copy()
                with torch.inference_mode():actual=model(pulse.to(device)).cpu().numpy()
                expected=method.restore(states)[0,0]
                error=float(np.max(abs(actual[:,0]+1j*actual[:,1]-expected)))
                relative=float(np.linalg.norm(actual[:,0]+1j*actual[:,1]-expected)/np.linalg.norm(expected))
                assert relative<1e-5,('RSDR tensor wrapper parity',relative,error)
                timings['1']={'status':'NOT_APPLICABLE_NATIVE_GROUP_GATE','note':'Native RSDR requires eight pulses; do not fabricate single-pulse timing.'}
            for count in ((8,) if name=='RSDR-Net' else (1,8)):
                x=pulse[:count].to(device)
                timings[str(count)]={**measure(model,x,warmup,repeats),'conv1d_linear_macs':macs_count(model,x)}
            row={'binding':method.binding,'timings':timings}
            if name=='RSDR-Net':
                row.update(native_group_parity_max_error=error,native_group_parity_relative_error=relative,amortized_per_pulse_p50_ms=timings['8']['p50_ms']/8,
                    note='Amortized per-pulse value is not native single-pulse latency.')
            rows[device][name]=row
            atomic_json(out/'LATENCY.json',{'started_utc':started,'updated_utc':now(),'devices':rows,
                'warmup':warmup,'repeats':repeats,'batch_groups':1,'cpu_threads':2,'precision':configure_precision(),
                'timing_kernel':'tools/benchmark_latency_unified.py::measure','kernel_sha256':sha(WORK/'tools/benchmark_latency_unified.py'),
                'coverage':'Full restored output; includes all model preprocessing, FP32 real CNN/FFT and complex128 VP. Resident-device inputs; transfer excluded uniformly.',
                'mac_scope':'Conv1d/Linear only; excludes FFT, complex solve, normalization and elementwise operations. Timing includes them.',
                'data':binding,'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__})
            del model;method.close()
    return rows

if __name__=='__main__':run()
