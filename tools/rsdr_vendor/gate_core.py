"""Scoped canonical waveform Gate math. No runtime, data generation or dispatch."""
import copy
import os
import random
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

SEED=40312102

def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False

class BoundedGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(10,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    def forward(self,x):return .25*torch.sigmoid(self.net(x)[...,0])

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def fresh(device):
    seed_all(SEED*10+6)
    model=BoundedGate().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda _: .5)
    return model,optimizer,scheduler

def scene_order(epoch):
    if not 1<=epoch<=32:raise PermissionError('Fixed 32 epoch protocol')
    return np.random.default_rng(SEED*1000+epoch).permutation(512).tolist()

def features(y,a,m):
    if y.shape!=a.shape or y.shape!=m.shape or y.ndim!=5 or y.shape[1:4]!=(4,8,2) or not all(torch.isfinite(v).all() for v in (y,a,m)):
        raise ValueError('finite scene,state,pulse,IQ,time features')
    dims=(2,3,4);power=lambda v:v.square().mean(dims).clamp_min(1e-12)
    p=[power(v) for v in (y,a,m)];x=[v.log() for v in p]
    for v,w in ((y,a),(y,m),(a,m)):x.append(power(v-w).log())
    for (v,w),pv,pw in zip(((y,a),(y,m),(a,m)),(p[0],p[0],p[1]),(p[1],p[2],p[2])):
        x.append((v*w).mean(dims)/(pv*pw).sqrt())
    x.append((y[...,1:]-y[...,:-1]).square().mean(dims)/p[0])
    return torch.stack(x,-1)

def fit_normalization(x):
    if x.shape!=(512,4,10) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('TRAIN512 float32 features only')
    return dict(mean=x.mean((0,1)).tolist(),std=x.std((0,1)).clip(1e-6).tolist(),dtype='float32',ddof=0,states=2048,DEV_fit=False)

def paired_loss(pred,w):
    if pred.shape!=w.shape or w.ndim!=4 or w.shape[:3]!=(4,8,2) or not torch.isfinite(w).all() or not torch.isfinite(pred).all():
        raise ValueError('finite fourstate full8pulse')
    p=pred.double();z=w.double();u=z[[0,0,2,2]];dims=(1,2,3)
    v=(p-u).square().sum(dims)/u.square().sum(dims).clamp_min(1e-12)
    s=z[2]-z[0];energy=s.square().sum()
    if energy<=1e-12:raise ValueError('zero target forbidden for normalized Pair')
    return dict(iq=v.mean(),pair=(p[3]-p[1]-s).square().sum()/energy,j0=v[[0,2]].mean())

def predict(model,data,norm,weights):
    y,u1,u2,m,x=data
    x=(x-torch.tensor(norm['mean'],dtype=x.dtype,device=x.device))/torch.tensor(norm['std'],dtype=x.dtype,device=x.device)
    a=weights[0]*y+weights[1]*u1+weights[2]*u2
    rho=model(x)
    return a+rho[:,None,None,None]*(m-a),rho

def update(model,optimizer,data,norm,weights,role='TRAIN'):
    if role!='TRAIN':raise PermissionError('Optimizer may read TRAIN only')
    model.train();optimizer.zero_grad(set_to_none=True)
    pred,rho=predict(model,data,norm,weights);terms=paired_loss(pred,data[0])
    total=terms['iq']+.1*terms['pair']+.1*terms['j0'];total.backward()
    grads={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
    if len(grads)!=4 or not torch.isfinite(total) or not all(torch.isfinite(v).all() for v in grads.values()):raise ValueError('All four active finite Gate gradients required')
    optimizer.step()
    return dict(output=pred.detach().cpu(),rho=rho.detach().cpu(),loss={**{k:v.detach().cpu() for k,v in terms.items()},'total':total.detach().cpu()},gradients=grads)

def cpu_copy(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return copy.deepcopy(x)

def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[v.cpu().clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])

def capture(model,optimizer,scheduler,**metadata):
    return cpu_copy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng(),
        gradients={k:p.grad for k,p in model.named_parameters()},training=model.training,seed=SEED,**metadata))

def restore(state,model,optimizer,scheduler):
    if state['seed']!=SEED:raise PermissionError('Wrong Gate seed')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(copy.deepcopy(state['optimizer']));scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
    for k,p in model.named_parameters():p.grad=None if state['gradients'][k] is None else state['gradients'][k].to(p.device).clone()
    model.train(state['training']);r=state['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])

def exact(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,torch.Tensor):return torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
