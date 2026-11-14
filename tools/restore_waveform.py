"""Standalone observed-only inference for the unchanged residual backbone."""
import argparse
from pathlib import Path
import numpy as np
import torch
from paper2.network import CorrectionNet
from paper2.data import iq


def restore_array(observed,checkpoint,gain=1,device=None):
    y=np.asarray(observed)
    if not np.iscomplexobj(y) or y.ndim not in (1,2) or y.shape[-1]!=1800 or not np.isfinite(y).all():
        raise ValueError('expected finite complex waveform [1800] or [N,1800]')
    chosen=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload=torch.load(checkpoint,map_location=chosen,weights_only=True)
    net=CorrectionNet().to(chosen).eval()
    net.load_state_dict(payload['state'])
    batch=y.reshape(-1,1800);result=[]
    width=128 if chosen.type=='cuda' else 16
    with torch.inference_mode():
        for start in range(0,len(batch),width):
            observed_batch=batch[start:start+width]
            count=len(observed_batch)
            # Match the scientific 128-observation CUDA shape. GroupNorm is
            # per-observation, so zero padding cannot add contextual evidence.
            if chosen.type=='cuda' and count<width:
                padded=np.zeros((width,1800),dtype=batch.dtype)
                padded[:count]=observed_batch
            else:
                padded=observed_batch
            c=net(iq(padded).to(chosen)).cpu().numpy()[:count]
            correction=c[:,0]+1j*c[:,1]
            result.append(observed_batch-gain*correction)
    return np.concatenate(result).reshape(y.shape)


def main():
    parser=argparse.ArgumentParser(description='Restore observed I/Q only; no target, jammer or paired background input.')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--key',default='y')
    parser.add_argument('--gain-real',type=float,default=1)
    parser.add_argument('--gain-imag',type=float,default=0)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('output already exists')
    with np.load(args.input,allow_pickle=False) as raw:
        y=raw[args.key]
    gain=complex(args.gain_real,args.gain_imag)
    restored=restore_array(y,args.checkpoint,gain)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,u_hat=restored)
    print('Restored',restored.shape,'observed complex samples; output',args.output)


if __name__=='__main__':main()
