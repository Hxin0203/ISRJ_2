"""Add crossed native SNR x JSR strata without modifying historical statistics."""
import numpy as np
from vp_1103_snr_jsr import summarize as marginal,BINS,LABELS
from vp_1103_common import FAMILIES,parent_mean,distribution

def summarize(values,axes):
    out=marginal(values,axes)
    for slo,shi,slab in zip(BINS['snr'][:-1],BINS['snr'][1:],LABELS['snr']):
        for jlo,jhi,jlab in zip(BINS['jsr'][:-1],BINS['jsr'][1:],LABELS['jsr']):
            mask=(axes['snr']>=slo)&(axes['snr']<shi)&(axes['jsr']>=jlo)&(axes['jsr']<jhi)
            label=f'snr/{slab}/jsr/{jlab}'
            out[label]={k:distribution(parent_mean(v,mask)) for k,v in values.items()}
            for i,family in enumerate(FAMILIES):
                fm=mask&(np.arange(3)[None]==i)
                out[f'{family}/{label}']={k:distribution(parent_mean(v,fm)) for k,v in values.items()}
    return out
