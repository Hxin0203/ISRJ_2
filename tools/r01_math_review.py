"""Isolated transform boundary checks; no data or model access."""
import cmath
import math
import json

mu = 0.37
errors=[]
for t in range(-8,9):
    for tau in (-3, 0, 2):
        lhs=cmath.exp(-1j*math.pi*mu*t*t)
        rhs=cmath.exp(1j*math.pi*mu*tau*tau)*cmath.exp(-2j*math.pi*mu*tau*t)*cmath.exp(-1j*math.pi*mu*(t-tau)**2)
        errors.append(abs(lhs-rhs))
assert max(errors)<1e-12
# A legitimate nonlinear normalized core F(z)=z^2 disproves unconditional homogeneity.
def c(x):
    scale=max(abs(x),1e-8)
    return scale*(x/scale)**2
assert abs(c(6)-3*c(2))<1e-12
assert abs(c(2e-10)-2*c(1e-10))>1e-13
# Cropping a finite-window shift can destroy energy, unlike a unitary phase rotation.
x=[0,0,1]
shift=[0]+x[:-1]
assert sum(v*v for v in shift)!=sum(v*v for v in x)
print(json.dumps({'status':'PASS','phase_identity_cases':len(errors),'max_error':max(errors),'checks':['dechirp_shift','positive_scale_above_floor','floor_counterexample','cropping_counterexample'],'scope':'abstract CPU only'}))
