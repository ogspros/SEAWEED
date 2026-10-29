# First version of the extractor, with quality checks.
#
# Only sta_lta_onset() is used by the workflow, through seafloor_robust.
# extract_wavelet_qc() stacks the near channel of each shot after a physical
# pre-alignment on the tracked seafloor and a cross-correlation refinement,
# with coherence weights, and returns the checks below in its qc dict:
#   n_shots_used / n_shots_tot  shots that entered the stack (dead traces out)
#   align_residual_ms           shift the cross-correlation still had to apply
#                               after the pre-alignment (small = good tracking)
#   stack_coherence             mean trace-to-stack correlation, 0 to 1
#   snr_gain                    coherent over incoherent energy of the stack
#   per_shot_corr               correlation of each shot with the stack
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt

VW_DEFAULT=1500.; NEAR_OFF_DEFAULT=125.; DX_R=25.

def sta_lta_onset(tr, dt, sta_ms=32., lta_ms=160., thr=3.0):
    """STA/LTA onset with windows in time, so that it behaves alike at any dt.

    Falls back on the envelope peak: the primary is the strong event in the
    window.
    """
    from scipy.ndimage import uniform_filter1d
    n_sta=max(3,int(round(sta_ms*1e-3/dt)))
    n_lta=max(10,int(round(lta_ms*1e-3/dt)))
    a=np.abs(tr); N=len(a)
    env=uniform_filter1d(a, max(3,n_sta//2))  # smoothed envelope
    if N < n_lta+n_sta+1: return int(np.argmax(env))
    r=np.zeros(N)
    for i in range(n_lta, N-n_sta):
        lta=a[i-n_lta:i].mean(); sta=a[i:i+n_sta].mean()
        r[i]=sta/(lta+1e-9)
    above=np.where(r>thr)[0]
    return int(above[0]) if len(above) else int(np.argmax(env))

def extract_wavelet_qc(gathers, dt, D_guess, Vw=VW_DEFAULT,
                       win_ms=0.18, lead_ms=0.03, search_ms=0.30, up=4,
                       band=(6,90), n_iter=4, w_len_ms=0.20,
                       t0_seafloor=None, near_off=NEAR_OFF_DEFAULT,
                       fmax_out=None, shot_start=0, n_shots_stack=None):
    """Extract the wavelet from the near channel of each shot.

    gathers: (S, NT, NCH) common-shot gathers of one line (shared source).
    Only the first (near) channel of each shot is used.
    t0_seafloor: (S,) tracked seafloor time per shot (s), for the
                 pre-alignment; if None, the estimate from D_guess is used.
    fmax_out: if given, a final zero-phase low-pass.
    Returns (w, qc): w the normalised wavelet (phase preserved), qc a dict.
    """
    S_tot,NT,NCH=gathers.shape
    # --- the run of shots to stack (the source is assumed stable over it) ---
    s0=max(0,shot_start); s1=min(S_tot, s0+n_shots_stack) if n_shots_stack else S_tot
    gathers=gathers[s0:s1]
    S=gathers.shape[0]
    nyq_g=0.5/dt
    # zero-phase high-pass on the long traces (removes swell and drift)
    if band[0]/nyq_g < 0.95:
        bh,ah=butter(4, band[0]/nyq_g, 'high')
        gathers=filtfilt(bh,ah,gathers,axis=1).astype(np.float32)

    WLEN=int(win_ms/dt); lead=int(lead_ms/dt); SR=int(search_ms/dt)
    t0_guess=2*D_guess/Vw

    # --- extraction: channel 0 (near) of each shot only ---
    segs=[]; onset_used=[]; shots_ok=[]
    c=0; x_near=near_off
    for s in range(S):
        # physical pre-alignment: centre the search on the tracked seafloor of this shot
        if t0_seafloor is not None:
            t0_s=float(t0_seafloor[s0+s])  # index in the original run
            t_pri_s=np.sqrt(t0_s**2 + (x_near/Vw)**2)  # near-offset moveout from the tracked seafloor
        else:
            t_pri_s=np.sqrt((2*D_guess)**2 + x_near**2)/Vw
        guess=int(t_pri_s/dt); lo=max(0,guess-SR); hi=min(NT,guess+SR)
        tr_search=gathers[s,lo:hi,c]
        if len(tr_search)<WLEN or np.all(np.abs(tr_search)<1e-12) or np.any(np.isnan(tr_search)):
            continue  # shot skipped (dead trace or NaN)
        # onset on the band-limited trace: high-frequency noise at fine dt makes STA/LTA erratic
        nyq_s=0.5/dt
        if band[1]/nyq_s<0.95:
            bb,ab=butter(4,[band[0]/nyq_s,band[1]/nyq_s],'band')
            tr_onset=filtfilt(bb,ab,tr_search)
        else:
            tr_onset=tr_search
        onset=lo+sta_lta_onset(tr_onset, dt)
        k0=max(0,onset-lead)
        seg=gathers[s,k0:k0+WLEN,c]
        if len(seg)==WLEN and np.std(seg)>1e-9 and not np.any(np.isnan(seg)):
            segs.append(seg); onset_used.append(onset*dt); shots_ok.append(s)
    n_used=len(segs)
    if n_used<3:
        return None, dict(n_shots_used=n_used, n_shots_tot=S, error="too few valid traces")
    segs=np.array(segs)

    # --- upsampling and band for the cross-correlation ---
    su=resample_poly(segs,up,1,axis=1)
    nyq=0.5/(dt/up); b,a=butter(4,[band[0]/nyq,band[1]/nyq],'band'); su=filtfilt(b,a,su,axis=1)

    # --- alignment: fine refinement after the physical pre-alignment ---
    ref=su.mean(0); lags_refine=[]
    for it in range(n_iter):
        al=[]; lags_it=[]
        for tr in su:
            pilot=ref-tr/len(su); pilot=pilot/(np.std(pilot)+1e-9)
            cc=np.correlate(tr/(np.std(tr)+1e-9),pilot,'full')
            lag=np.argmax(cc)-(len(tr)-1)
            lags_it.append(lag); al.append(np.roll(tr,-lag))
        al=np.array(al); ref=al.mean(0)
        if it==n_iter-1: lags_refine=np.array(lags_it)

    # --- coherence: how far each trace agrees with the final stack ---
    per_shot_corr=np.array([np.corrcoef(tr,ref)[0,1] if np.std(tr)>1e-9 else 0.0 for tr in al])
    per_shot_corr=np.nan_to_num(per_shot_corr,nan=0.0)
    wts=np.clip(per_shot_corr,0,None)

    # --- coherence-weighted stack ---
    st=(wts[:,None]*al).sum(0)/(np.sum(wts)+1e-9)

    # --- quality checks ---
    # alignment residual: the cross-correlation shift after the physical
    # pre-alignment, in ms at the upsampled interval
    dt_up=dt/up
    align_res_ms=np.abs(lags_refine)*dt_up*1e3 if len(lags_refine) else np.array([0.])
    # stack SNR gain: coherent signal survives the sum while incoherent noise
    # falls as sqrt(N).  Measured as the energy of the stack over the residual
    # trace-to-stack variance; about 1 when the traces already equal the stack,
    # larger the more incoherent noise the stack removes.
    stack_mean=al.mean(0)
    var_residua=np.mean([np.var(tr-stack_mean) for tr in al])  # incoherent, per trace
    var_segnale=np.var(stack_mean)  # coherent
    snr_gain=float(var_segnale/(var_residua+1e-12))  # coherent over incoherent SNR of the stack
    coh=float(np.mean(per_shot_corr))

    qc=dict(
        n_shots_used=int(n_used), n_shots_tot=int(S),
        align_residual_ms_mean=float(np.mean(align_res_ms)),
        align_residual_ms_max=float(np.max(align_res_ms)),
        stack_coherence=coh,
        snr_gain=snr_gain,
        per_shot_corr=per_shot_corr,
        onset_times_s=np.array(onset_used),
        shots_ok=np.array(shots_ok),
        aligned_segments=al, stack_upsampled=ref,
    )

    # --- downsample and output band ---
    st=resample_poly(st,1,up); nyq2=0.5/dt
    b,a=butter(4,[band[0]/nyq2,band[1]/nyq2],'band'); st=filtfilt(b,a,st)
    st=st-st.mean()
    # optional final low-pass
    if fmax_out is not None and fmax_out<nyq2*0.95:
        bl,al_=butter(6,fmax_out/nyq2,'low'); st=filtfilt(bl,al_,st)
    Lw=int(w_len_ms/dt)
    st=st[:Lw] if len(st)>Lw else np.pad(st,(0,Lw-len(st)))
    w=(st/(np.linalg.norm(st)+1e-9)).astype(np.float32)
    return w, qc


def plot_qc(w, qc, dt, out="extract_qc.png"):
    """Plot the quality checks of the extraction."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    al=qc.get('aligned_segments'); pc=qc.get('per_shot_corr')
    fig,ax=plt.subplots(2,2,figsize=(14,8))
    fig.suptitle("Wavelet extraction QC (near channel, seafloor pre-alignment + cross-correlation)",fontsize=12)

    # (1) aligned near traces as wiggles, filled
    if al is not None:
        nshow=min(len(al), 40)  # readable up to about 40 traces
        step=max(1,len(al)//nshow)
        sub=al[::step][:nshow]
        sc=np.percentile(np.abs(sub),98)+1e-9
        tax=np.arange(sub.shape[1])
        for i,tr in enumerate(sub):
            y=i + tr/sc*0.45  # vertical offset per trace, normalised amplitude
            ax[0,0].plot(tax, y, 'k', lw=0.5)
            ax[0,0].fill_between(tax, i, y, where=(y>i), color='k', alpha=0.7, lw=0)
        ax[0,0].set_title(f'aligned near traces ({len(al)} shots, {len(sub)} shown)')
        ax[0,0].set_xlabel('time (upsampled samples)'); ax[0,0].set_ylabel('shot'); ax[0,0].invert_yaxis()

    # (2) coherence per shot: where along the line the extraction is good or poor
    if pc is not None:
        ax[0,1].plot(qc.get('shots_ok',np.arange(len(pc))), pc,'b.-',ms=3,lw=0.7)
        ax[0,1].axhline(0.7,color='g',ls=':',label='0.7')
        ax[0,1].set_title(f'coherence per shot (mean {qc["stack_coherence"]:.2f})')
        ax[0,1].set_xlabel('shot'); ax[0,1].set_ylabel('correlation with the stack'); ax[0,1].legend(fontsize=8)
        ax[0,1].set_ylim(-0.1,1.05)

    # (3) alignment residual: how much the cross-correlation corrected after the seafloor
    ax[1,0].axvline(qc['align_residual_ms_mean'],color='r',label=f'mean {qc["align_residual_ms_mean"]:.0f}ms')
    ax[1,0].axvline(qc['align_residual_ms_max'],color='orange',ls='--',label=f'max {qc["align_residual_ms_max"]:.0f}ms')
    ax[1,0].set_title('alignment residual (small = accurate seafloor tracking)')
    ax[1,0].set_xlabel('ms'); ax[1,0].legend(fontsize=8); ax[1,0].set_xlim(0,None)

    # (4) extracted wavelet
    tt=np.arange(len(w))*dt*1e3
    ax[1,1].plot(tt,w,'k',lw=1.8)
    ax[1,1].set_title(f'extracted wavelet (coh {qc["stack_coherence"]:.2f}, gain {qc["snr_gain"]:.1f})')
    ax[1,1].set_xlabel('ms')
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(out,dpi=135)
    return out


if __name__=="__main__":
    # test on synthetic files with known ground truth
    import glob
    files=sorted(glob.glob("dataset_deep/test/line_*.npz"))[:5]
    if not files: files=sorted(glob.glob("dataset_deep/test/line_*.npz"))[:5]
    print(f"{'D[m]':>5} | {'n_used':>6} | {'coh':>5} | {'gain':>5} | {'res_ms':>6} | {'NCC vs w_bare':>13}")
    def nrm(x): return x/(np.linalg.norm(x)+1e-9)
    def ncc_a(a,b,R=40): return max(nrm(a)@nrm(np.roll(b,l)) for l in range(-R,R+1))
    for f in files:
        d=np.load(f,allow_pickle=True); g=d["gathers"]; dt=float(d["dt"]); D=float(d["D_water"]); w=d["w_bare"]
        w_ext,qc=extract_wavelet_qc(g,dt,D*np.random.uniform(0.95,1.05))
        if w_ext is None: print(f"{D:5.0f} | extraction failed: {qc.get('error')}"); continue
        w200=w[:len(w_ext)]; wt=w200 if len(w200)==len(w_ext) else np.pad(w200,(0,len(w_ext)-len(w200)))
        nf=ncc_a(w_ext,wt)
        print(f"{D:5.0f} | {qc['n_shots_used']:6d} | {qc['stack_coherence']:5.2f} | {qc['snr_gain']:5.1f} | {qc['align_residual_ms_mean']:6.0f} | {nf:13.3f}")
        # visual comparison (aligned traces and QC) for the first line
        if f==files[0]:
            plot_qc(w_ext,qc,dt,out="extract_qc.png")
            print("  -> visual comparison: extract_qc.png")
