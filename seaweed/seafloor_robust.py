# Seafloor picking by semblance along the water moveout.
#
# semblance_bottom() is the stage-2 velocity and anchor estimate used by
# run_real.  It searches only t0 above t0_min, which excludes the direct
# arrival, and equalises amplitudes with an AGC before measuring coherence.
import numpy as np
from seaweed.wavelet_extractor_qc import sta_lta_onset

NEAR=125.; DX=25.  # used only by the self-test below

def agc(trace, dt, win_ms=250):
    """AGC: normalise the amplitude over a moving window.

    Needed before the semblance, where strong events such as the direct
    arrival would otherwise dominate.
    """
    w=max(3,int(win_ms/1e3/dt)); a=np.abs(trace)
    env=np.convolve(a,np.ones(w)/w,'same')  # local mean amplitude
    return trace/(env+1e-9*np.max(a+1e-12))

def semblance_bottom(gather, dt, offsets, V_range=(1430,1560), nV=27,
                     t0_min_ms=150, t0_max_ms=1400, n_near=None, n_smooth=7,
                     sem_win_ms=28, use_agc=True, agc_ms=250,
                     x_max=None, stretch_max=0.45, return_width=False):
    """Neidell-Taner semblance with a time gate and AGC.

    The gate (sem_win_ms) sums energy in a short window along the hyperbola
    rather than at a single sample, which is far more stable, and the AGC
    equalises amplitudes before coherence is measured.  The direct arrival is
    excluded by searching only t0 > t0_min.

    The offsets used are set by the NMO stretch limit.  Velocity is resolved
    only where the moveout difference exceeds the gate: at t0 = 940 ms, 1450
    and 1500 m/s differ by about 13 ms at 1100 m, less than the 28 ms gate, and
    the semblance surface is then flat in velocity.  The useful offset grows
    with water depth, and the stretch s = (t - t0)/t0 follows it: the limit
    x <= 2 D sqrt((1+s)^2 - 1) admits, for s = 0.45, offsets to about 2D.
    Beyond it the moveout distorts the waveform enough that the traces no
    longer stack coherently, so further offsets add smearing, not resolution.
    On IT90-65 the peak then sits at 1462-1467 m/s with t0 = 921-941 ms, in
    agreement with the direct arrival and the tracked seafloor; without a
    limit it is sharper but locks onto another event at t0 = 860 ms.

    x_max (metres) and n_near (channels) override the stretch limit when given.

    With return_width=True the half-width of the velocity peak is also
    returned.  It is the uncertainty of the estimate, tens of m/s on real data,
    and a 3 per cent error in velocity is a 3 per cent error in water depth.
    """
    off_all = np.asarray(offsets, float)
    if n_near is None and x_max is not None:
        n_near = int(np.searchsorted(off_all, x_max)) or len(off_all)
        n_near = max(8, min(n_near, len(off_all)))
    if n_near is None:
        n_near = len(off_all)          # the stretch limit is applied per (t0, V)
    g=gather[:n_near].copy(); off=off_all[:n_near]
    valid=np.std(g,axis=1)>1e-9
    g=g[valid]; off=off[valid]; nc=g.shape[0]
    if nc<3: return None
    nt=g.shape[1]
    if use_agc:
        g=np.array([agc(g[c],dt,agc_ms) for c in range(nc)])
    hw=max(1,int(sem_win_ms/1e3/dt/2))  # half-length of the semblance gate, in samples
    t0s=np.arange(int(t0_min_ms/1e3/dt), int(t0_max_ms/1e3/dt))
    Vs=np.linspace(*V_range,nV); S=np.zeros((len(t0s),len(Vs)))
    chan=np.arange(nc)
    for iv,V in enumerate(Vs):
        for it,t0i in enumerate(t0s):
            t0=t0i*dt; t=np.sqrt(t0**2+(off/V)**2); tx=t/dt
            kc=np.round(tx).astype(int)
            ok=(kc-hw>=0)&(kc+hw<nt)
            if stretch_max is not None:
                # NMO stretch (t - t0)/t0: beyond the limit the correction
                # distorts the waveform and those traces smear the peak
                ok &= (t - t0) / max(t0, 1e-9) <= stretch_max
            if ok.sum()<3: continue
            num=0.0; den=0.0
            # sum over the time gate around the hyperbola (Neidell-Taner)
            for j in range(-hw,hw+1):
                amps=g[chan[ok], kc[ok]+j]
                num+=amps.sum()**2; den+=(amps**2).sum()
            S[it,iv]=num/(ok.sum()*den+1e-12)
    if n_smooth>1:
        k=np.ones(n_smooth)/n_smooth; S=np.apply_along_axis(lambda m:np.convolve(m,k,'same'),0,S)
    it,iv=np.unravel_index(np.argmax(S),S.shape)
    # offsets actually admitted at the winning (t0, V), i.e. what the stretch
    # limit allowed rather than what was available
    t_w=np.sqrt((t0s[it]*dt)**2+(off/Vs[iv])**2)
    adm=off[(t_w-t0s[it]*dt)/max(t0s[it]*dt,1e-9)<=(stretch_max if stretch_max else 9e9)]
    out=dict(t0=t0s[it]*dt, V=Vs[iv], peak=S[it,iv], n_ch_used=int(nc),
             x_max_used=float(adm[-1]) if len(adm) else float(off[-1]),
             stretch_max=stretch_max)
    # half-width of the velocity peak: the resolution actually achieved
    prof=S[it,:]; half=0.5*prof.max()
    above=np.where(prof>=half)[0]
    out['V_halfwidth']=float(Vs[above[-1]]-Vs[above[0]]) if len(above)>1 else np.nan
    out['V_at_edge']=bool(iv==0 or iv==len(Vs)-1)
    return (out, S) if return_width else out

def detect_line(shots_data, dt, offsets, D_hint=None, Vw=1500.):
    """Seafloor t0 per shot, with every constraint and shot-to-shot continuity."""
    # search window: around the depth hint when there is one
    if D_hint is not None:
        t0_nom=2*D_hint/Vw*1e3
        t0_min=max(150, t0_nom-300); t0_max=t0_nom+300
    else:
        t0_min=150; t0_max=1400
    t0_raw=[]; Vs=[]
    for g in shots_data:
        r=semblance_bottom(g, dt, offsets, t0_min_ms=t0_min, t0_max_ms=t0_max)
        if r is None: t0_raw.append(np.nan); Vs.append(np.nan); continue
        # refine with STA/LTA around the semblance t0, on the near trace
        c0=g[0]; guess=int(r['t0']/dt); SR=int(0.08/dt)
        lo=max(0,guess-SR); hi=min(len(c0),guess+SR)
        onset=lo+sta_lta_onset(c0[lo:hi], dt) if hi>lo+10 else guess
        t0_raw.append(onset*dt); Vs.append(r['V'])
    t0_raw=np.array(t0_raw)
    # continuity: replace jumps of more than 100 ms by the running median
    t0_s=t0_raw.copy()
    for i in range(len(t0_raw)):
        lo=max(0,i-3); hi=min(len(t0_raw),i+4); med=np.nanmedian(t0_raw[lo:hi])
        if not np.isnan(t0_raw[i]) and abs(t0_raw[i]-med)>0.10: t0_s[i]=med
    return t0_s, np.array(Vs)

if __name__=="__main__":
    import glob
    files=sorted(glob.glob("dataset2/line_*.npz"))
    shots=[]; D_true=[]; dt=None
    for f in files:
        d=np.load(f,allow_pickle=True); g=d['gathers']; dt=float(d['dt'])
        for s in range(g.shape[0]): shots.append(g[s].T); D_true.append(float(d['D_water']))
    offs=NEAR+DX*np.arange(shots[0].shape[0])
    # self-test on synthetic lines, without a depth hint (the harder case)
    t0_det,Vdet=detect_line(shots, dt, offs)
    t0_true=np.array([2*D/1500. for D in D_true])
    err=np.nanmean(np.abs(t0_det-t0_true))*1e3
    print(f"no depth hint: mean t0 error = {err:.0f} ms, mean V {np.nanmean(Vdet):.0f}")
