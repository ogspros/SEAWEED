# Source library: airgun signatures with real diversity of shape (primary
# width, bubble, phase), mean correlation about 0.71 between signatures.  The
# generator draws a different one for each line.
import numpy as np
from airgun_generator import to_minimum_phase, bubble_period, GUN_VOLS, DT as AG_DT

def _ricker_like(t,t0,w):
    return -(t-t0)/w**2*np.exp(-((t-t0)/w)**2)

def diverse_wavelet(rng, nt=400, dt=AG_DT, z_src=None):
    """Far-field signature of a tuned airgun array, with diversity of shape.

    In a tuned array the primaries are synchronised (same t0, coherent sum)
    and the bubbles of the gun groups have different periods (incoherent sum,
    partial cancellation); the primary-to-bubble ratio comes out at 4-9.

    z_src: if given, the dominant bubble period T0 is coupled to the source
    depth by Rayleigh-Willis (a deeper gun has a shorter bubble,
    T_b ~ 1/(d+10.1)^(5/6)), with scatter between arrays; this avoids
    unphysical combinations such as a gun at 15 m with a 100 ms bubble.
    z_src=None leaves T0 free in [55, 105] ms.
    """
    t=np.arange(nt)*dt
    prim_w=rng.uniform(0.0024,0.0075)  # primary width (wide range)
    t0=0.006
    n_grp=int(rng.integers(2,6))  # gun groups (different volumes)
    if z_src is None:
        T0=rng.uniform(0.055,0.105)  # T0 independent of the depth
    else:
        V_dom=rng.uniform(GUN_VOLS.min(),GUN_VOLS.max())  # dominant volume (diversity)
        T0=bubble_period(V_dom, z_src)*(1+rng.normal(0,0.06))  # coupled to z_src
        T0=float(np.clip(T0,0.040,0.120))  # within a plausible physical range
    Tbs=T0*rng.uniform(0.85,1.15,n_grp)  # groups near T0: partial cancellation
    amps=rng.uniform(0.5,1.0,n_grp)  # group weight (number of guns, volume)
    decay=rng.uniform(0.42,0.62)  # bubble decay per group
    n_bub=int(rng.integers(3,7))
    sync_jit=rng.normal(0,0.0012,n_grp)  # synchronisation jitter (about 1 ms on archive arrays)
    s=np.zeros_like(t)
    for A,Tb,dj in zip(amps,Tbs,sync_jit):
        s+=A*_ricker_like(t,t0+dj,prim_w)  # nearly synchronous primaries
        for n in range(1,n_bub+1):
            an=A*decay**n*(1+rng.normal(0,0.08))
            s+=an*_ricker_like(t,t0+dj+n*Tb,prim_w*rng.uniform(1.4,2.0))  # broad bubbles
    s=s/np.abs(s).max()
    mp=to_minimum_phase(s); alpha=rng.uniform(0,1)  # degree of mixed phase
    s=alpha*mp+(1-alpha)*s; s/=np.abs(s).max()
    return s.astype(np.float32)
