# Synthetic lines for the matrices: make_line() builds one line (velocity
# model, source, Devito modelling, resampling) and is what run_matrix_*.py
# call.  Run as a script it generates a whole set of lines in parallel: each
# process takes a disjoint range of indices and writes one file per line,
# skipping those already written, so an interrupted run resumes.  Seeds are
# derived from the index (seed = idx, or 1_000_000 + idx for the test split),
# so disjoint ranges never collide.
#
#     python validation/synthetic/generate_parallel_hf.py --split train --idx_start 0 --idx_end 250
import os, time, argparse, numpy as np
os.environ.setdefault('DEVITO_LOGGING','ERROR')
os.environ.setdefault('DEVITO_ARCH','gcc')
# CPU only: no GPU platform
os.environ.pop('DEVITO_PLATFORM', None)
os.environ.setdefault('DEVITO_LANGUAGE','C')
# one Devito thread per process: the parallelism is between processes
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('DEVITO_OPT','advanced')

from scipy.signal import resample_poly
import generate_line2_hf as G
from source_library import diverse_wavelet


def apply_ghost(w, dt, z_src, z_rec, Vw=1500., R=-1.0):
    """Signature with source and receiver ghosts, near-vertical.

    W_ghost = W * (1 + R e^{-i w 2 z_src/V}) * (1 + R e^{-i w 2 z_rec/V});
    z_src, z_rec: source and receiver depths below the surface (m).
    """
    import numpy as np
    n=len(w); W=np.fft.rfft(w, n*2)
    f=np.fft.rfftfreq(n*2, dt); om=2*np.pi*f
    Gs=1+R*np.exp(-1j*om*2*z_src/Vw); Gr=1+R*np.exp(-1j*om*2*z_rec/Vw)
    return np.fft.irfft(W*Gs*Gr)[:n].astype(np.float32)

def domain_for(n_shots): return G.domain_for_shots(n_shots+4)

def make_source_fn(src_seed, fmax=80.0, source_file=None, source_type="airgun"):
    from airgun_generator import DT as AG_DT
    from scipy.signal import butter, filtfilt
    if source_file is not None:
        w = np.load(source_file).astype(np.float32)  # a fixed source read from a file
    elif source_type == "minphase":
        # minimum-phase source (explosive-like), band set by fmax per line:
        # a zero-phase FIR made minimum phase, with its peak fc below fmax
        from scipy.signal import firwin2, minimum_phase
        rng = np.random.default_rng(src_seed)
        nyq = 0.5 / AG_DT
        # the peak fc scales with fmax: low-band sources have a low peak
        fc = rng.uniform(0.55, 0.75) * fmax
        fhi = min(fmax, 0.9 * nyq)
        numtaps = 401
        freqs = [0, 5/nyq, 10/nyq, fc/nyq, fhi/nyq, min((fhi+5)/nyq, 1.0), 1.0]
        freqs = sorted(set(freqs))  # increasing, no duplicates
        gains = [0, 0, 0.7, 1.0, 0.5, 0, 0][:len(freqs)]
        while len(gains) < len(freqs): gains.append(0.0)
        fir = firwin2(numtaps, freqs, gains)
        w = minimum_phase(fir, method='homomorphic').astype(np.float32)
    else:
        w=diverse_wavelet(np.random.default_rng(src_seed))
    nyq=0.5/AG_DT
    if fmax<nyq*0.95:
        b,a=butter(6,fmax/nyq,'low'); w=filtfilt(b,a,w)
    def fn(dt, nt):
        dt_s=dt/1000.0; n_new=int(len(w)*AG_DT/dt_s)
        wr=resample_poly(w,n_new,len(w)) if n_new!=len(w) else w.copy()
        out=np.zeros(nt,np.float32); L=min(len(wr),nt); out[:L]=wr[:L]
        return out/np.max(np.abs(out))
    return fn, w

def make_line(seed, n_shots, tn=None, dt_out=0.002, geom_name="g3000",
              source_file=None, z_ghost=(4.0, 5.0), source_type="airgun", fmax=80.0):
    rng=np.random.default_rng(seed)
    geom=G.GEOMS[geom_name]; offsets=G.geom_offsets(geom)
    XMAX=G.domain_for_shots_geom(n_shots+4, geom)
    model,vp,rho,kwb,meta=G.build_model(XMAX,rng=rng)
    xs_all=G.shot_positions_geom(XMAX, geom)
    n_shots=min(n_shots,len(xs_all))
    src_fn,w_bare=make_source_fn(10_000+seed, source_file=source_file, source_type=source_type, fmax=fmax)
    # signature with ghosts, the reference; the ghost depths are parameters
    from airgun_generator import DT as AG_DT
    Z_SRC_REAL, Z_REC_REAL = z_ghost
    w_ghost=apply_ghost(w_bare, AG_DT, Z_SRC_REAL, Z_REC_REAL)
    # fixed tn and a fixed number of samples: the Devito step depends on the
    # model (CFL), so after resampling to dt_out the length would vary by a few
    # samples between lines; it is forced to NT_FIXED exactly (cut or padded).
    if tn is None:
        tn=4500.0
    NT_FIXED=int(round(tn/1000./dt_out))  # 4500 ms / 4 ms = 1125 samples
    gathers=[]; dt_dev=None
    for i in range(n_shots):
        # the same depths as the reference: the free surface makes the ghost in the data
        d,dt=G.shoot(model,xs_all[i],src_fn,tn=tn,offsets=offsets,
                     z_src=Z_SRC_REAL,z_rec=Z_REC_REAL); dt_dev=dt; gathers.append(d)
    g=np.stack(gathers)
    # rational resampling to dt_out (Fraction): the Devito step is not an exact
    # submultiple of dt_out, and integer decimation stretched the time axis by
    # a factor that varied by line (an apparent water velocity of 1433 instead
    # of 1500 m/s)
    from fractions import Fraction
    fr=Fraction(float(dt_dev)/1000./float(dt_out)).limit_denominator(2000)
    up,down=fr.numerator,fr.denominator
    g=resample_poly(g,up,down,axis=1).astype(np.float32)
    # force NT exactly: cut if longer, zero-pad if shorter
    if g.shape[1]>NT_FIXED:   g=g[:,:NT_FIXED,:]
    elif g.shape[1]<NT_FIXED: g=np.pad(g,((0,0),(0,NT_FIXED-g.shape[1]),(0,0)))
    return g, dt_out, w_bare, w_ghost, meta, offsets, geom_name

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",choices=["train","test"],required=True)
    ap.add_argument("--idx_start",type=int,required=True)
    ap.add_argument("--idx_end",type=int,required=True)  # exclusive
    ap.add_argument("--n_shots",type=int,default=12)
    ap.add_argument("--out",default="dataset_gpu")
    ap.add_argument("--geoms",default="g1500,g3000,g4500",
                    help="geometries assigned to the lines in turn, by index")
    ap.add_argument("--source_file",default=None,
                    help="if given, use this fixed source from a file instead of diverse_wavelet")
    ap.add_argument("--z_ghost",default="4,5",help="ghost depths z_src,z_rec in m (default 4,5; MS-29: 15,25)")
    ap.add_argument("--vary",action="store_true",
                    help="draw per line the depths (z_src 4-18, z_rec 4-28 m) and the source type (70%% airgun, 30%% minimum phase)")
    ap.add_argument("--p_minphase",type=float,default=0.30,help="fraction of minimum-phase sources (with --vary)")
    args=ap.parse_args()
    geom_list=args.geoms.split(",")
    zg=tuple(float(x) for x in args.z_ghost.split(","))
    outdir=os.path.join(args.out,args.split); os.makedirs(outdir,exist_ok=True)
    seed_base=0 if args.split=="train" else 1_000_000
    t0=time.time(); done=0
    for idx in range(args.idx_start,args.idx_end):
        fpath=os.path.join(outdir,f"line_{idx:05d}.npz")
        if os.path.exists(fpath): continue  # resume
        geom_name=geom_list[idx % len(geom_list)]  # geometry chosen round-robin by line
        # --- --vary: depths, source type and band drawn per line (seed = idx, reproducible) ---
        if args.vary:
            rng_line=np.random.default_rng(seed_base+idx+777)  # a seed of its own, independent of the geology
            zs=float(rng_line.uniform(4.0,18.0)); zr=float(rng_line.uniform(4.0,28.0))
            zg_line=(zs,zr)
            stype="minphase" if rng_line.uniform()<args.p_minphase else "airgun"
            # variable fmax, to cover the bands of the real surveys:
            # about 40% low band 35-55 Hz, 30% 55-70 Hz, 30% 70-90 Hz
            u=rng_line.uniform()
            if u<0.40:   fmax_line=float(rng_line.uniform(35,55))
            elif u<0.70: fmax_line=float(rng_line.uniform(55,70))
            else:        fmax_line=float(rng_line.uniform(70,90))
        else:
            zg_line=zg; stype="airgun"; fmax_line=80.0
        try:
            gr,dt,w,w_g,meta,offsets,gname=make_line(seed_base+idx,args.n_shots,geom_name=geom_name,
                                                     source_file=args.source_file, z_ghost=zg_line,
                                                     source_type=stype, fmax=fmax_line)
            np.savez(fpath,gathers=gr,dt=dt,w_bare=w,w_ghost=w_g,D_water=meta['D_water'],
                     kwb_depth=meta.get('kwb_depth'),D_nominal=meta.get('D_nominal'),
                     z_src=zg_line[0],z_rec=zg_line[1],source_type=stype,fmax=fmax_line,
                     offsets=offsets.astype(np.float32), geom=gname,  # native geometry
                     meta=str({k:v for k,v in meta.items() if k!='kwb_depth'}),
                     seed=seed_base+idx)
            done+=1
            if done%5==0 or done==1:
                el=time.time()-t0
                print(f"[pid {os.getpid()} {args.split} {args.idx_start}-{args.idx_end}] "
                      f"{done} done, {el/done:.0f}s per line",flush=True)
        except Exception as e:
            print(f"[pid {os.getpid()}] ERR idx {idx}: {e}",flush=True)
    print(f"[pid {os.getpid()} {args.split} {args.idx_start}-{args.idx_end}] FINISHED {done} lines in {(time.time()-t0)/60:.1f}min",flush=True)
