#!/usr/bin/env python3
"""
qc_synth_case.py -- what fails in one synthetic record, and what works in its
twin.

The worst record of Fig. 3 (grid_D260_L300_r011, NCC 0.406) and the best one
(grid_D260_L300_r015, NCC 0.944) are the same cell: 260 m of water, first
reflector 300 m below, one channel, 60 Hz, 30 dB.  Only the seed of the
procedural geology differs.  The extracted segment of the worst one carries
little of the energy of its own extraction window (peak 0.05 after the long
window is normalised), and its spectrum is combed with a regular ripple, which
is what two events a fixed time apart produce.  Both point to the window
holding something other than, or besides, the primary.

This script regenerates each model exactly as the matrix did (same seeds, same
code path as make_fig_synthetic.py, from which it imports everything), repeats
the extraction, and draws four things per record:

  (a) the impedance section under the shots, with the seafloor and the depth
      range the extraction window reaches
  (b) the IDEAL primary: for every shot, the normal-incidence reflectivity of
      its column, in two-way time from the seafloor, convolved with the true
      wavelet.  No multiples, no noise, no processing.  The image shows which
      sub-bottom reflectors fall inside the window and whether they are
      laterally uniform (they then stack like the primary) or not
  (c) the stack of (b) over the shots, against the long extracted window, both
      on the same time axis: if the extraction looks like the ideal stack, the
      workflow did its job and the geology made the answer wrong
  (d) the true wavelet and the compared segment, as in Fig. 3

It also prints: tracked against true seafloor time, where the origin landed,
the slide offset, how much of the long window's energy the compared segment
holds, and the strongest sub-bottom event within the window relative to the
seafloor.

    python3 validation/synthetic/qc_synth_case.py --config validation/synthetic/matrix_v2.yaml --wavelets wavelet_matrix_v2 \\
        --runs grid_D260_L300_r011 grid_D260_L300_r015 --snr 30

Requires Devito, like make_fig_synthetic.py; two models take a few minutes.
"""
import argparse
import json
import os

import numpy as np
import yaml
from scipy.signal import butter, filtfilt

import make_fig_synthetic as F

# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

BAND = F.BAND


def model_arrays(run, cfg):
    """The velocity and density grids of the record, rebuilt with the same
    seed and arguments as make_fig_synthetic.regenerate()."""
    p = run["params"]; der = run["derived"]; m = cfg["meta"]
    geom = F.G.GEOMS[p["geom"]]
    offs = F.G.geom_offsets(geom)
    XMAX = F.G.domain_for_shots_geom(m["n_shots"] + 4, geom)
    rng = np.random.default_rng(run["seed"])
    vp, rho, kwb, meta = F.generate_velocity_model(
        rng, XMAX=XMAX, ZMAX=der["ZMAX_m"], DH=F.G.DH,
        D_water_range=(p["D_m"], p["D_m"]),
        p_fault=p["p_fault"], p_folds=p["p_folds"], p_lens=p["p_lens"],
        slope_deg=p["slope_deg"], roughness_m=p["roughness_m"],
        first_layer_m=p["first_layer_m"])
    xs = F.G.shot_positions_geom(XMAX, geom)[:m["n_shots"]]
    ix = np.clip(((xs - offs[0] / 2.) / F.G.DH).astype(int), 0, vp.shape[0] - 1)
    return vp, rho, kwb, ix, offs


def ideal_primary(vp, rho, kwb, ix, w_ref, dt, tmax):
    """Normal-incidence reflectivity of each shot column, in two-way time
    measured from the seafloor, convolved with the true wavelet.  Row s is what
    shot s would record from the seafloor down if nothing but primaries
    existed."""
    nt = int(round(tmax / dt))
    out = np.zeros((ix.size, nt))
    for s, i in enumerate(ix):
        v = vp[i].astype(float); r_ = rho[i].astype(float)
        I = v * r_
        k0 = int(kwb[i])
        R = (I[1:] - I[:-1]) / (I[1:] + I[:-1] + 1e-30)
        # two-way time from the seafloor interface (between k0-1 and k0)
        tt = np.concatenate([[0.0], np.cumsum(2 * F.G.DH / v[k0:-1])])
        spikes = np.zeros(nt)
        for k in range(k0 - 1, R.size):
            j = k - (k0 - 1)
            if j >= tt.size:
                break
            n = int(round(tt[j] / dt))
            if n >= nt:
                break
            spikes[n] += R[k]
        out[s] = np.convolve(spikes, w_ref)[:nt]
    return out


def run_case(run, rec, cfg, snr, ax_row, archive=None):
    m = cfg["meta"]
    p = run["params"]
    print("\n== %s  (stored NCC %.3f)" % (run["run_id"], rec["ncc_prim_true"]))

    g, dt, offs, w_bare, D = F.regenerate(run, cfg)
    wl = int(round(m["w_len_ms"] * 1e-3 / dt))
    D_nom = float(p["D_m"])
    zs_ = p.get("z_src_m", F.Z_DEV)
    zr_ = p.get("z_rec_m", F.Z_DEV)
    w_ref = F.reference(w_bare, dt, wl, zs_, zr_, two_d=True)

    # ---- the matrix pipeline, as in make_fig_synthetic.main ----------------
    vw, _, _ = F.estimate_vw_direct(g, dt, offs, D_guess=D_nom)
    r0 = F.semblance_bottom(g[0].T, dt, offs,
                            t0_min_ms=max(120, 2 * D_nom / vw * 1e3 - 200),
                            t0_max_ms=2 * D_nom / vw * 1e3 + 200)
    # n_group=5 was the default of track_seafloor_xcorr when the synthetic results
    # were computed; it is written out so that the default of 20 does not change them.
    t0 = F.track_seafloor_xcorr(g, dt, offs,
                                t0_init_s=(r0["t0"] if r0 else 2 * D_nom / vw),
                                vw=vw, n_group=5)
    gg = g
    if snr < 60:
        rng_n = np.random.default_rng(run["seed"] + int(snr))
        ns = np.sqrt(np.mean(g ** 2)) / (10 ** (snr / 20.))
        gg = g + ns * rng_n.normal(0, 1, g.shape).astype(np.float32)
    t_arr_min = float(np.min(np.sqrt(np.asarray(t0) ** 2 + (offs[0] / vw) ** 2)))
    lead_max = max(20., 0.90 * t_arr_min * 1e3)
    pad_ms = float(np.clip(lead_max - 80., 0., 120.))
    lead_ms = min(80. + pad_ms, lead_max)
    w, qc = F.extract_wavelet(gg, dt, t0, offs,
                              w_len_ms=m["w_len_ms"] + 2 * pad_ms,
                              lead_ms=lead_ms, n_ch=run["derived"]["n_ch_used"],
                              order=0, band=BAND, vw=vw)
    if w is None:
        print("   extraction failed: %s" % qc.get("error"))
        return
    wlong = F.nrm(np.asarray(w, float))
    npad = int(round(pad_ms * 1e-3 / dt))
    sep_s = (np.hypot(2 * D_nom, offs[0]) - offs[0]) / vw
    o_lim = max(2, min(npad, int(round(0.5 * sep_s / dt))))
    o_lo, o_hi = max(0, npad - o_lim), min(len(wlong) - wl, npad + o_lim)
    best, bo = -2.0, npad
    for o in range(o_lo, o_hi + 1):
        seg = wlong[o:o + wl]
        # the last offset can fall short of a full window when the long
        # extraction was padded: a short segment is not comparable and the
        # correlation would fail on the length rather than report anything
        if seg.size != wl:
            continue
        x = seg - seg.mean(); y = w_ref - w_ref.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        c = float(x @ y / d) if d > 1e-12 else 0.
        if c > best:
            best, bo = c, o
    oml = float(qc["origin_minus_lead_ms"])
    # The matrix slides the truth over [npad - o_lim, npad + o_lim], i.e. it
    # looks for the wavelet up to 2*npad samples AFTER the origin and never
    # before it, since wlong starts at the origin.  Score the wavelet as the
    # workflow returns it -- starting at the origin -- against the truth with a
    # small shift either way, zero-filled before the origin.  If this is high
    # while the matrix score is low, the matrix score measures the comparison
    # and not the extraction.
    k15 = int(round(0.030 / dt))
    wr = np.concatenate([np.zeros(k15), wlong])
    best_o, bo_o = -2.0, 0
    for o in range(0, 2 * k15 + 1):
        seg = wr[o:o + wl]
        if seg.size != wl:            # same guard as the slide above
            continue
        x = seg - seg.mean(); y = w_ref - w_ref.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        c = float(x @ y / d) if d > 1e-12 else 0.
        if c > best_o:
            best_o, bo_o = c, o - k15
    print("   score of the returned wavelet at the origin, +-30 ms: NCC %.3f "
          "at %+d ms  (matrix: %.3f at %+.0f ms after the origin)"
          % (best_o, bo_o * dt * 1e3, rec["ncc_prim_true"], (bo - 0) * dt * 1e3))
    # make_fig_synthetic warns that rebuilding a model has not always
    # reproduced the matrix.  The archived segment settles whether it did here:
    # if the rebuilt compared segment matches it, everything below describes
    # the record of the paper; if not, the geology panels still do (the model
    # is fixed by the seed) but the extraction panels describe a rerun.
    k = "%s|%s|%s" % (run["run_id"], rec.get("snr_db"), rec.get("t0_bias_ms"))
    if archive is not None and k in archive:
        a_ = np.asarray(archive[k], float)
        s_ = wlong[bo:bo + wl]
        print("   rebuilt segment against the archived one: NCC %.4f%s"
              % (F.ncc_pair(s_, a_), "" if F.ncc_pair(s_, a_) > 0.99 else
                 "   <-- the rebuild is NOT the record of the paper"))

    # ---- what Stage 1 picks on these gathers, which is the v6 seafloor ----
    # qc_synth_case rebuilds the v2 chain, so nothing above describes the v6
    # run; this line does, and it is the only way to see from here why a v6
    # pick landed where it did.
    try:
        from seaweed import pick_consensus as PC
        t0_pick, _rep = PC.pick_line(np.asarray(gg, float), dt, offs, vw,
                                     group=10, tol_ms=120.0)
        print("   stage 1 pick on these gathers: %+.0f ms from the true "
              "seafloor" % ((np.median(t0_pick) - np.median(np.asarray(D, float)
                             * 2.0 / m["v_water"])) * 1e3))
    except Exception as exc:
        print("   stage 1 pick not computed: %s" % str(exc)[:60])

    # ---- where things are, in time from the TRUE seafloor arrival ----------
    vp, rho, kwb, ix, _ = model_arrays(run, cfg)
    v_water = float(m["v_water"])
    t0_true = 2 * kwb[ix] * F.G.DH / v_water
    x0 = offs[0]
    ta_true = np.sqrt(t0_true ** 2 + (x0 / v_water) ** 2)
    ta_track = np.sqrt(np.asarray(t0) ** 2 + (x0 / vw) ** 2)
    # wlong[0] sits at the tracked arrival + origin offset; in time from the
    # true arrival that is:
    tau0 = float(np.mean(ta_track - ta_true)) + oml * 1e-3
    tau_seg = tau0 + bo * dt

    print("   seafloor time, tracked minus true: mean %+.1f ms, max |.| %.1f ms"
          % (1e3 * np.mean(ta_track - ta_true),
             1e3 * np.max(np.abs(ta_track - ta_true))))
    print("   origin past the lead %+.0f ms | slide offset %+d samples "
          "(limit +-%d) | NCC %.3f (stored %.3f)"
          % (oml, bo - npad, o_lim, best, rec["ncc_prim_true"]))
    e = wlong ** 2
    print("   energy of the long window inside the compared segment: %.0f%%"
          % (100 * e[bo:bo + wl].sum() / e.sum()))
    print("   compared segment starts %+.0f ms from the true seafloor arrival"
          % (tau_seg * 1e3))

    tmax = tau0 + len(wlong) * dt + 0.05
    ip = ideal_primary(vp, rho, kwb, ix, w_ref, dt, max(tmax, 0.6))
    st = ip.mean(0)
    k200 = int(round(0.2 / dt))
    pk_sf = np.abs(st[:k200 // 4]).max()
    j = np.arange(st.size) * dt
    later = (j > 0.03) & (j < 0.2)
    print("   ideal stack: strongest event 30-200 ms below the seafloor is "
          "%.2f of the seafloor peak, at %.0f ms"
          % (np.abs(st[later]).max() / pk_sf,
             1e3 * j[later][np.argmax(np.abs(st[later]))]))
    lat = np.corrcoef(ip[:, later])
    print("   lateral uniformity of the sub-bottom part (mean correlation "
          "between shots): %.2f" % np.mean(lat[np.triu_indices_from(lat, 1)]))

    # ---- figure row --------------------------------------------------------
    a0, a1, a2, a3 = ax_row
    z0 = int(max(0, kwb[ix].min() - 50 / F.G.DH))
    z1 = int(min(vp.shape[1], kwb[ix].max() + 500 / F.G.DH))
    imp = (vp * rho)[ix.min():ix.max() + 1, z0:z1].T
    a0.imshow(imp, aspect="auto", cmap="viridis",
              extent=[ix.min() * F.G.DH, ix.max() * F.G.DH,
                      z1 * F.G.DH, z0 * F.G.DH])
    a0.plot(ix * F.G.DH, kwb[ix] * F.G.DH, "w", lw=0.8)
    a0.set_title("%s: impedance" % run["run_id"], fontsize=8)
    a0.set_xlabel("x (m)"); a0.set_ylabel("depth (m)")

    lim = np.percentile(np.abs(ip), 99)
    a1.imshow(ip.T, aspect="auto", cmap="gray_r", vmin=-lim, vmax=lim,
              extent=[0, ip.shape[0], ip.shape[1] * dt * 1e3, 0])
    a1.axhspan(tau_seg * 1e3, tau_seg * 1e3 + wl * dt * 1e3, color="C3",
               alpha=0.15)
    a1.set_title("ideal primaries per shot; compared segment shaded",
                 fontsize=8)
    a1.set_xlabel("shot"); a1.set_ylabel("time from seafloor (ms)")

    tl = tau0 * 1e3 + np.arange(len(wlong)) * dt * 1e3
    ts = np.arange(st.size) * dt * 1e3
    a2.plot(ts, st / np.abs(st).max(), "k", lw=1.0, label="ideal stack")
    a2.plot(tl, wlong / np.abs(wlong).max(), "C3", lw=0.9,
            label="extracted, long window")
    a2.axvspan(tau_seg * 1e3, tau_seg * 1e3 + wl * dt * 1e3, color="C3",
               alpha=0.10)
    a2.set_xlim(min(tl[0], -20), max(tl[-1], ts[-1]))
    a2.set_title("same time axis, from the true seafloor", fontsize=8)
    a2.set_xlabel("time (ms)"); a2.legend(fontsize=6)

    t = np.arange(wl) * dt * 1e3
    a3.plot(t, w_ref, "k", lw=1.4, label="true")
    a3.plot(t, F.nrm(wlong[bo:bo + wl]), "C3", lw=1.0, label="compared segment")
    a3.set_title("NCC %.3f" % best, fontsize=8)
    a3.set_xlabel("time (ms)"); a3.legend(fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--runs", nargs="+",
                    default=["grid_D260_L300_r011", "grid_D260_L300_r015"])
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--out", default="qc_synth_case.png")
    ap.add_argument("--wavelets", default=None,
                    help="matrix directory with the archived segments, to "
                         "check that the rebuild reproduces the record")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    root = cfg["paths"]["out_root"]
    runs = {r["run_id"]: r for r in
            (json.loads(l) for l in open(os.path.join(root, "manifest.jsonl")))}
    recs = {}
    for l in open(os.path.join(root, "results.jsonl")):
        r = json.loads(l)
        if r.get("snr_db") == a.snr and r.get("t0_bias_ms", 0.0) == 0.0:
            recs[r["run_id"]] = r

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(a.runs), 4, figsize=(16, 3.6 * len(a.runs)),
                           squeeze=False)
    archive = F.load_stored(a.wavelets) if a.wavelets else None
    for k, rid in enumerate(a.runs):
        if rid not in runs or rid not in recs:
            raise SystemExit("%s not found at %g dB" % (rid, a.snr))
        run_case(runs[rid], recs[rid], cfg, a.snr, ax[k], archive)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
