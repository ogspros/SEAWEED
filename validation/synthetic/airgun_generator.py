# Source signatures for the synthetic generator: a tuned airgun array.
#
# Each gun gives a primary pulse and a train of bubbles with the
# Rayleigh-Willis period T_b ~ P^(1/3) V^(1/3) / (d + 10.1)^(5/6); guns of
# different volume have different bubble periods, which the array sums out of
# phase and partly cancels.  Shot-to-shot variability (pressure, bubble period
# and amplitude) is optional.  The signature returned is the far-field one,
# near-vertical and without ghosts, which the forward modelling adds.
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq

# ---- source constants -------------------------------------------------------
# The acquisition geometry lives in generate_line2_hf.py; only what builds the
# signature and its ghosts is here.
DT         = 0.002  # native sampling of the signature (s)
Z_SRC      = 6.0  # source depth (as in generate_line2_hf)
Z_REC      = 6.0  # streamer depth
VW         = 1500.0  # water velocity
RS         = -1.0  # free-surface reflection coefficient (ghost)

# ---- airgun array: 16 guns, 1373 cu.in in total ---------------------------
# a plausible tuned array: clusters of different volumes
GUN_VOLS = np.array([30,40,50,60,70,80,90,100,110,120,90,100,70,150,113,100],float)
GUN_VOLS *= 1373.0/GUN_VOLS.sum()  # renormalised to exactly 1373 cu.in
FIRING_PSI = 2000.0
K_BUBBLE   = 0.0138  # Rayleigh-Willis constant

def bubble_period(V_cuin, depth_m, P_psi=FIRING_PSI):
    """Rayleigh-Willis bubble period T_b ~ P^(1/3) V^(1/3) / (d+10.1)^(5/6), in s."""
    return K_BUBBLE * P_psi**(1/3) * V_cuin**(1/3) / (depth_m+10.1)**(5/6)

def gun_notional(V_cuin, depth_m, dt=DT, nt=400, P_psi=FIRING_PSI,
                 n_bub=5, decay=0.30, prim_w=0.004, rng=None):
    """Notional signature of one gun: a primary pulse and bubbles at T_b, 2 T_b...

    The n-th bubble has amplitude decay**n.
    """
    t = np.arange(nt)*dt
    Tb = bubble_period(V_cuin, depth_m, P_psi)
    if rng is not None:  # shot-to-shot variability
        Tb *= (1 + rng.normal(0,0.03))  # 3% jitter of the bubble period
    s = np.zeros(nt)
    # primary: derivative of a Gaussian, amplitude ~ V^(1/3)
    amp0 = V_cuin**(1/3)
    def pulse(t0, w):
        return -(t-t0)/w**2*np.exp(-((t-t0)/w)**2)
    s += amp0*pulse(0.006, prim_w)
    # bubbles: same polarity as the primary, decaying amplitude
    for n in range(1, n_bub+1):
        an = amp0*(decay**n)
        if rng is not None: an *= (1+rng.normal(0,0.05))
        s += an*pulse(0.006 + n*Tb, prim_w*1.8)  # broad bubbles: partial cancellation between guns
    return s

def array_farfield(depth_m=Z_SRC, dt=DT, nt=400, rng=None, pressure_jitter=True):
    """Far-field signature of the array (sum of the 16 guns), near-vertical,
    without ghosts.

    The bubble decay varies by line, giving a primary-to-bubble ratio of
    about 4-8.
    """
    P = FIRING_PSI*(1+rng.normal(0,0.02)) if (rng is not None and pressure_jitter) else FIRING_PSI
    dec = rng.uniform(0.45, 0.65) if rng is not None else 0.55  # tuning varies by line: primary-to-bubble ratio about 4-8
    s = np.zeros(nt)
    for V in GUN_VOLS:
        s += gun_notional(V, depth_m, dt, nt, P_psi=P, decay=dec, rng=rng)
    return s/np.max(np.abs(s))

def to_minimum_phase(w):
    """Minimum-phase wavelet with the same amplitude spectrum."""
    W=np.fft.fft(w,4096); c=np.fft.ifft(np.log(np.abs(W)+1e-9)).real
    n=len(c); c[1:n//2]*=2; c[n//2:]=0
    mp=np.fft.ifft(np.exp(np.fft.fft(c))).real[:len(w)]
    return mp/np.max(np.abs(mp))

# =============================== self-test and figure =========================
if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng=np.random.default_rng(0)

    print("=== IT90-65 geometry ===")
    print(f"  {NCH} channels, offsets {OFFSETS[0]:.0f}..{OFFSETS[-1]:.0f} m, dt {DT*1e3:.0f} ms, record {REC_LEN} s")
    print(f"  source {Z_SRC} m, streamer {Z_REC} m")
    print(f"\n=== Airgun array 1x16, {GUN_VOLS.sum():.0f} cu.in ({GUN_VOLS.sum()/61.02:.1f} L) ===")
    Tbs=[bubble_period(V,Z_SRC) for V in GUN_VOLS]
    print(f"  gun volumes: {GUN_VOLS.round(0).astype(int)}")
    print(f"  bubble periods: {np.array(Tbs).round(3)} s  (spread {min(Tbs)*1e3:.0f}-{max(Tbs)*1e3:.0f} ms)")
    print(f"  -> different bubble periods: the array puts them out of phase and suppresses them")

    # signature and spectrum
    nt=400; tt=np.arange(nt)*DT
    w_mix=array_farfield(rng=np.random.default_rng(1))
    w_min=to_minimum_phase(w_mix)
    f=rfftfreq(nt,DT); Wm=rfft(w_mix); 
    dist=np.linalg.norm(w_mix-w_min)/np.linalg.norm(w_mix)
    print(f"\n  distance of the mixed-phase from the minimum-phase wavelet: {dist:.2f}  (>0.1 = genuinely mixed)")

    # shot-to-shot variability
    shots=[array_farfield(rng=np.random.default_rng(100+i)) for i in range(8)]

    fig,ax=plt.subplots(2,2,figsize=(12,8))
    fig.suptitle("Synthetic generator, IT90-65 - airgun array source (1x16, 1373 cu.in)",fontsize=12)
    ax[0,0].plot(tt*1e3,w_mix,'k',lw=1.8,label='mixed phase (with bubble)')
    ax[0,0].plot(tt*1e3,w_min,'C3--',lw=1.4,label='minimum phase')
    ax[0,0].set_xlim(0,250); ax[0,0].set_xlabel('time [ms]')
    ax[0,0].set_title('array far-field signature (near-vertical, no ghost)'); ax[0,0].legend(fontsize=8)
    ax[0,1].plot(f,np.abs(Wm),'k'); ax[0,1].set_xlim(0,125)
    ax[0,1].set_xlabel('frequency [Hz]'); ax[0,1].set_title('array amplitude spectrum')
    for i,s in enumerate(shots):
        ax[1,0].plot(tt*1e3,s+i*0.4,lw=1)
    ax[1,0].set_xlim(0,200); ax[1,0].set_xlabel('time [ms]'); ax[1,0].set_ylabel('shot (vertical offset)')
    ax[1,0].set_title('shot-to-shot variability (8 shots, same nominal source)')
    # single guns: different bubble periods
    for V in [GUN_VOLS.min(),np.median(GUN_VOLS),GUN_VOLS.max()]:
        gn=gun_notional(V,Z_SRC); gn/=np.abs(gn).max()
        ax[1,1].plot(tt*1e3,gn,label=f'{V:.0f} cu.in (Tb={bubble_period(V,Z_SRC)*1e3:.0f}ms)')
    ax[1,1].set_xlim(0,250); ax[1,1].set_xlabel('time [ms]')
    ax[1,1].set_title('single guns: the bubble scales with volume'); ax[1,1].legend(fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig("airgun_signatures.png",dpi=140)

    print("\nfigure saved: airgun_signatures.png")
    print("ok - source signature generator ready (geometry in generate_line2_hf.py)")
