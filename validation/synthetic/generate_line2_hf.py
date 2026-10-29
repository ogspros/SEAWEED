# Forward modelling of one synthetic line with Devito: single-sided geometry
# and an airgun-array source.  The vessel moves towards high x and the
# streamer trails behind it, so the receivers are at x_s - offset.  The source
# is the mixed-phase far-field signature of the array, injected in place of a
# Ricker and resampled to the Devito time step.
import os
os.environ['DEVITO_LOGGING']='WARNING'; os.environ['DEVITO_ARCH']='gcc'
import numpy as np, time
from scipy.signal import resample
from examples.seismic import Model, RickerSource, Receiver, TimeAxis
from devito import TimeFunction, Eq, Operator, solve
from velocity_generator import generate_velocity_model
from airgun_generator import array_farfield, to_minimum_phase, DT as AG_DT

# ---- geometry ----
Z_SRC, Z_REC = 6.0, 6.0
NEAR_OFF, DX_R, NCH = 125.0, 25.0, 120  # default: the 3000 m geometry
OFFSETS = NEAR_OFF + DX_R*np.arange(NCH)
FAR = OFFSETS[-1]; SHOT_DX = 25.0
DH = 3.0; NB = 200; MARGIN = NB*DH  # 3 m grid for an 80 Hz band; the absorbing layer keeps the same physical margin

# Canonical geometries (near offset, receiver spacing, channels), as in the
# OGS field data:
#   1500 m: 120 channels at 12.5 m (short streamer, small near offset)
#   3000 m: 120 channels at 25 m   (IT90-65)
#   4500 m: 180 channels at 25 m   (long streamer)
GEOMS = {
    "g1500": dict(near=12.5,  dx=12.5, nch=120),   # FAR = 12.5 + 12.5*119 = 1500
    "g3000": dict(near=125.0, dx=25.0, nch=120),   # FAR = 125 + 25*119 = 3100 (~3000)
    "g4500": dict(near=125.0, dx=25.0, nch=180),   # FAR = 125 + 25*179 = 4600 (~4500)
}

def geom_offsets(geom):
    """Native offsets (m) of a geometry dict(near, dx, nch)."""
    return geom["near"] + geom["dx"] * np.arange(geom["nch"])

def geom_far(geom):
    return geom["near"] + geom["dx"] * (geom["nch"] - 1)

def domain_for_shots_geom(n, geom):
    """Physical domain for n shots with the given geometry (a longer streamer needs a wider domain)."""
    return MARGIN + geom_far(geom) + MARGIN + (n-1)*SHOT_DX

def shot_positions_geom(XMAX, geom):
    xs_min = MARGIN + geom_far(geom)
    xs_max = XMAX - MARGIN
    return np.arange(xs_min, xs_max+1e-6, SHOT_DX)

def domain_for_shots(n): return MARGIN + FAR + MARGIN + (n-1)*SHOT_DX

def shot_positions_singlesided(XMAX):
    """Source positions for a vessel moving towards high x with the streamer
    behind it: receivers at x_s - offset, the streamer tail inside the domain
    on the low-x side, the source inside it on the high-x side.
    """
    xs_min = MARGIN + FAR
    xs_max = XMAX - MARGIN
    return np.arange(xs_min, xs_max+1e-6, SHOT_DX)

def make_source_wavelet(dt_devito_ms, nt_devito, phase='mixed', rng=None, fmax=80.0):
    """Airgun signature resampled to the Devito step (ms); returns an array of length nt_devito."""
    from scipy.signal import butter, filtfilt
    w = array_farfield(rng=rng)  # at dt = 2 ms
    if phase=='min': w = to_minimum_phase(w)
    nyq=0.5/AG_DT
    if fmax<nyq*0.95:
        b,a=butter(6,fmax/nyq,'low'); w=filtfilt(b,a,w)
    # resample from AG_DT (s) to the Devito step (ms -> s)
    dt_dev = dt_devito_ms/1000.0
    n_new = int(len(w)*AG_DT/dt_dev)
    w_rs = resample(w, n_new)
    # place it in a vector of length nt_devito (the source is a transient at the start)
    out = np.zeros(nt_devito, np.float32)
    L = min(len(w_rs), nt_devito); out[:L] = w_rs[:L]
    return out/np.max(np.abs(out))

def build_model(XMAX, ZMAX=2800.0, rng=None, D_water_range=(150,1500), flat_bottom=False):
    vp,rho,kwb,meta=generate_velocity_model(rng,XMAX=XMAX,ZMAX=ZMAX,DH=DH,
                                            D_water_range=D_water_range,flat_bottom=flat_bottom)
    model=Model(vp=(vp/1000.),b=1.0/rho,origin=(0.,0.),shape=vp.shape,
                spacing=(DH,DH),space_order=8,nbl=NB,bcs="damp",fs=True)
    return model,vp,rho,kwb,meta

def shoot(model, xs, src_wavelet_fn, tn=4500.0, offsets=None, z_src=None, z_rec=None):
    """Common-shot gather, single-sided, with the airgun source; receivers at
    x_s - offset.

    offsets: native offsets (m); default: the global geometry.
    z_src, z_rec: source and receiver depths (m).  They must be the depths used
    to build the reference wavelet with its ghosts: the free surface creates
    the ghost in the data, and different depths would give the data and the
    reference different ghosts.
    """
    if offsets is None:
        offsets = OFFSETS
    if z_src is None: z_src = Z_SRC
    if z_rec is None: z_rec = Z_REC
    nch = len(offsets)
    dt=model.critical_dt; ta=TimeAxis(start=0.,stop=tn,step=dt); nt=ta.num
    src=RickerSource(name='src',grid=model.grid,f0=0.025,time_range=ta,npoint=1)  # structure only
    src.coordinates.data[0,:]=[xs,z_src]
    src.data[:,0]=src_wavelet_fn(dt, nt)  # overwritten with the airgun signature
    rec=Receiver(name='rec',grid=model.grid,npoint=nch,time_range=ta)
    rec.coordinates.data[:,0]=xs-offsets  # streamer behind the source (low x)
    rec.coordinates.data[:,1]=z_rec
    u=TimeFunction(name='u',grid=model.grid,time_order=2,space_order=4)
    pde=model.m*u.dt2-u.laplace+model.damp*u.dt; st=solve(pde,u.forward)
    Operator([Eq(u.forward,st)]+src.inject(field=u.forward,expr=src*dt**2/model.m)
             +rec.interpolate(expr=u)).apply(dt=dt)
    return np.array(rec.data),dt

if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    N_TARGET=80; XMAX=domain_for_shots(N_TARGET)
    print(f"domain {XMAX:.0f} m for {N_TARGET} shots")
    rng=np.random.default_rng(11)
    model,vp,rho,kwb,meta=build_model(XMAX,rng=rng)
    xs_all=shot_positions_singlesided(XMAX)
    print(f"vessel moving towards high x, streamer behind (low x)")
    print(f"source from x_s={xs_all[0]:.0f} to {xs_all[-1]:.0f} m, {len(xs_all)} shots")
    print(f"  example, shot 1: source {xs_all[0]:.0f} m, receivers {xs_all[0]-OFFSETS[-1]:.0f}..{xs_all[0]-OFFSETS[0]:.0f} m")

    # airgun signature shared by the line (the same source), mixed phase
    src_rng=np.random.default_rng(500)
    def src_fn(dt,nt): 
        # the same signature for every shot of the line
        return make_source_wavelet(dt,nt,phase='mixed',rng=np.random.default_rng(500))

    idx=np.linspace(0,len(xs_all)-1,6).astype(int); times=[]; gathers=[]
    for k,i in enumerate(idx):
        ts=time.time(); data,dt=shoot(model,xs_all[i],src_fn); el=time.time()-ts
        times.append(el); gathers.append(data)
        print(f"  shot {k+1}/6 at x_s={xs_all[i]:.0f} m: {el:.1f}s")
    print(f"mean time per shot: {np.mean(times[1:]):.1f}s -> {len(xs_all)} shots in ~{np.mean(times[1:])*len(xs_all)/60:.1f} min")

    # signature for the panel
    wav=make_source_wavelet(dt,400,phase='mixed',rng=np.random.default_rng(500))
    fig,ax=plt.subplots(1,4,figsize=(17,5))
    fig.suptitle(f"Single-sided line (vessel towards high x, streamer behind), airgun source",fontsize=12)
    tw=np.arange(len(wav))*dt/1000.
    ax[0].plot(tw*1e3,wav,'k'); ax[0].set_xlim(0,150); ax[0].set_xlabel('time [ms]')
    ax[0].set_title('injected airgun signature (mixed phase)')
    for c,k in enumerate([0,3,5]):
        nt=gathers[k].shape[0]; t=np.arange(nt)*dt/1000.; gd=gathers[k]*((t+0.05)[:,None])
        cl=np.percentile(np.abs(gd),98)
        # receivers at x_s - offset, plotted against the (positive) offset
        ax[c+1].imshow(gd,aspect='auto',cmap='gray_r',vmin=-cl,vmax=cl,extent=[OFFSETS[0],OFFSETS[-1],t[-1],0])
        ax[c+1].set_title(f'shot at x_s={xs_all[idx[k]]:.0f} m'); ax[c+1].set_xlabel('offset[m]'); ax[c+1].set_ylabel('time [s]')
    plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig("line_airgun.png",dpi=135); print("ok")
