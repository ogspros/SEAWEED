# Procedural velocity models for the synthetic matrices, generated on the fly.
#
# Geological ingredients, each optional and parametrised:
#   - a water column of variable depth over a seafloor that need not be flat
#   - a sequence of sedimentary layers of increasing velocity, with a
#     compaction gradient
#   - folds (undulating interfaces), a fault (normal or reverse, variable
#     throw), lateral velocity variation, an anomalous body (lens)
# Density: 1000 kg/m3 in water, Gardner's relation below.
# The domain covers the IT90-65 offsets (about 3.8 km).
import numpy as np

VP_WATER, RHO_WATER = 1500.0, 1000.0

def _smooth_profile(rng, n, lo, hi):
    """Smooth random profile in [lo, hi] (filtered random walk)."""
    x=np.cumsum(rng.normal(0,1,n)); x-=x.min(); x/=(x.max()+1e-9)
    return lo+(hi-lo)*x

def generate_velocity_model(rng=None, XMAX=3825.0, ZMAX=2800.0, DH=5.0,
                            D_water_range=(150,1500), n_layers_range=(6,14),
                            p_fault=0.7, p_folds=0.6, p_lens=0.3, p_lateral=0.6,
                            flat_bottom=False,
                            slope_deg=None, roughness_m=None, first_layer_m=None):
    """Returns (vp[nx,nz], rho[nx,nz], kwb_profile[nx], meta), in m/s and kg/m3.

    kwb_profile: index of the seafloor in each column.
    """
    if rng is None: rng=np.random.default_rng()
    nx=int(XMAX/DH)+1; nz=int(ZMAX/DH)+1
    vp=np.zeros((nx,nz),np.float32)

    # --- 1) bathymetry ---
    D0=rng.uniform(*D_water_range)
    if flat_bottom:  # exactly flat seafloor, for controlled tests
        bath=np.full(nx,D0)
    elif rng.random()<0.7 or slope_deg is not None or roughness_m is not None:
        n_b=rng.integers(2,5)
        bx=np.linspace(0,1,nx)
        # amplitude of the undulations, controllable
        if roughness_m is None:
            # default: kept bit for bit as it was, so that the models and the
            # extractions of the matrices are reproduced
            bath=D0+sum(rng.uniform(-40,40)*np.sin(2*np.pi*rng.uniform(0.3,1.5)*bx+rng.uniform(0,6.28))
                        for _ in range(n_b))
        else:
            # the random draws are always consumed, even at zero roughness, so that
            # between cells of different roughness only the seafloor changes and not
            # the geology below it (one variable at a time)
            osc=sum(rng.uniform(-1,1)*np.sin(2*np.pi*rng.uniform(0.3,1.5)*bx+rng.uniform(0,6.28))
                    for _ in range(n_b))
            sd=float(np.std(osc))
            # roughness_m is the RMS of the bathymetry, not the amplitude of the sinusoids
            osc=osc/sd*float(roughness_m) if (sd>1e-12 and roughness_m>0) else osc*0.0
            bath=D0+osc
        # slope: in degrees when given, otherwise random;
        # tan(slope) * XMAX is the total drop across the domain
        if slope_deg is None:
            bath=bath+rng.uniform(-0.06,0.06)*XMAX*bx
        else:
            bath=bath+np.tan(np.radians(float(slope_deg)))*XMAX*bx
    else:
        bath=np.full(nx,D0)
    bath=np.clip(bath,80,ZMAX*0.55)
    kwb=(bath/DH).astype(int)

    # --- 2) sedimentary layers (velocity increasing with depth) ---
    n_lay=rng.integers(*n_layers_range)
    # base thicknesses and velocities of the layers
    v_top=rng.uniform(1550,1750); v_bot=rng.uniform(3200,4200)
    v_interfaces=np.sort(rng.uniform(v_top,v_bot,n_lay))
    # depths of the interfaces below the seafloor, as fractions
    fr=np.sort(rng.uniform(0,1,n_lay))
    # first reflector at a controlled depth below the seafloor.  Otherwise the
    # interfaces sit at random fractions of (ZMAX - D), the contamination is
    # coupled to the water depth, and the two axes of the matrix cannot be
    # separated.  The other layers stay distributed below it.
    if first_layer_m is not None:
        depth_avail_ref = nz - int(np.median(bath)/DH)
        f1 = float(first_layer_m)/DH/max(depth_avail_ref,1)
        f1 = float(np.clip(f1, 0.01, 0.9))
        fr = np.concatenate([[f1], f1 + (1.0-f1)*np.sort(rng.uniform(0,1,n_lay-1))])
    # --- 3) folds: undulation of the interfaces, shared but with different amplitudes ---
    folds = rng.random()<p_folds
    if folds:
        n_f=rng.integers(1,4)
        phases=rng.uniform(0,6.28,n_f); freqs=rng.uniform(0.4,1.6,n_f); amps=rng.uniform(10,60,n_f)
    bx=np.linspace(0,1,nx)
    # --- 4) vertical velocity gradient (compaction) ---
    grad=rng.uniform(0.3,0.9)  # m/s per m, added with depth

    for j in range(nx):
        kw=kwb[j]; vp[j,:kw]=VP_WATER
        depth_avail=nz-kw
        # interfaces of this column, with the folds
        z_int=(fr*depth_avail).astype(float)
        if folds:
            for a,f,p in zip(amps,freqs,phases):
                z_int+=a/DH*np.sin(2*np.pi*f*bx[j]+p)
        z_int=np.clip(z_int,0,depth_avail-1).astype(int)
        # fill the layers
        prev=0
        for li in range(n_lay):
            zi=z_int[li]
            v=v_interfaces[li]
            vp[j,kw+prev:kw+zi]=v
            prev=zi
        vp[j,kw+prev:]=v_interfaces[-1]
        # compaction gradient
        zz=np.arange(depth_avail)
        vp[j,kw:]+=grad*zz*DH/1000.0*rng.uniform(0.8,1.2)  # gradient
    # --- 5) smooth lateral variation ---
    if rng.random()<p_lateral:
        lat=_smooth_profile(rng,nx,0.92,1.08)
        vp[:,:]=np.where(np.arange(nz)[None,:]>=kwb[:,None], vp*lat[:,None], vp)

    # --- 6) fault ---
    D_true_mean=float(np.mean(kwb)*DH)
    meta=dict(D_water=D_true_mean, D_nominal=float(D0),
              slope_deg=slope_deg, roughness_m=roughness_m,
              first_layer_m=first_layer_m,
              kwb_depth=(kwb*DH).astype(np.float32),
              DH=float(DH), n_layers=int(n_lay), folded=bool(folds))
    if rng.random()<p_fault:
        fx=int(rng.uniform(0.3,0.7)*nx)
        throw=int(rng.uniform(15,60)/DH)*(1 if rng.random()<0.5 else -1)  # throw in cells
        dip=rng.uniform(-0.3,0.3)  # dip
        for j in range(fx,nx):
            shift=throw+int(dip*(j-fx))
            kw=kwb[j]
            col=vp[j,kw:].copy()
            vp[j,kw:]=np.roll(col,shift)
            if shift>0: vp[j,kw:kw+shift]=col[0]
        meta['fault']=dict(x=fx*DH, throw_m=throw*DH)

    # --- 7) optional anomalous body (lens) ---
    if rng.random()<p_lens:
        cx=int(rng.uniform(0.2,0.8)*nx); cz=int(rng.uniform(0.4,0.8)*nz)
        rx=int(rng.uniform(150,400)/DH); rz=int(rng.uniform(40,120)/DH)
        dv=rng.uniform(-400,500)
        X,Z=np.meshgrid(np.arange(nx),np.arange(nz),indexing='ij')
        mask=((X-cx)/rx)**2+((Z-cz)/rz)**2<1
        mask&=(Z>=kwb[:,None])
        vp[mask]+=dv
        meta['lens']=True

    # physical clip and density
    vp=np.clip(vp,VP_WATER,5000).astype(np.float32)
    rho=np.empty_like(vp)
    for j in range(nx):
        kw=kwb[j]; rho[j,:kw]=RHO_WATER
        rho[j,kw:]=0.31*(vp[j,kw:])**0.25*1000.0
    return vp,rho,kwb,meta

if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    XMAX,ZMAX,DH=3825.,1500.,5.
    fig,axes=plt.subplots(2,3,figsize=(16,7))
    fig.suptitle("Procedural velocity models (6 realisations)",fontsize=13)
    for i,ax in enumerate(axes.ravel()):
        rng=np.random.default_rng(10+i)
        vp,rho,kwb,meta=generate_velocity_model(rng,XMAX,ZMAX,DH)
        im=ax.imshow(vp.T,aspect='auto',cmap='viridis',extent=[0,XMAX,ZMAX,0],vmin=1500,vmax=4500)
        ax.plot(np.arange(len(kwb))*DH,kwb*DH,'w:',lw=0.8)
        tag=[]; 
        if meta.get('folded'): tag.append('folds')
        if 'fault' in meta: tag.append(f"fault {meta['fault']['throw_m']:+.0f}m")
        if meta.get('lens'): tag.append('lens')
        ax.set_title(f"D={meta['D_water']:.0f}m, {meta['n_layers']} layers"+(" | "+", ".join(tag) if tag else ""),fontsize=8)
        ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig("velocity_models.png",dpi=130)
    print("figure saved: velocity_models.png")
    # diversity statistics
    print("\ndiversity over 50 models:")
    Ds=[]; nls=[]; nf=0
    for i in range(50):
        _,_,_,m=generate_velocity_model(np.random.default_rng(1000+i),XMAX,ZMAX,DH)
        Ds.append(m['D_water']); nls.append(m['n_layers']); nf+=('fault' in m)
    print(f"  water depth: {min(Ds):.0f}-{max(Ds):.0f} m")
    print(f"  layers: {min(nls)}-{max(nls)}")
    print(f"  with a fault: {nf}/50")
