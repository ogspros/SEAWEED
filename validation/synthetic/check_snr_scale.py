# =============================================================================
#  What is "SNR = 0 dB" really worth?
#
#  The matrices calibrate the noise on the RMS of the whole gather,
#      ns = sqrt(mean(g**2)) / 10**(snr/20),
#  but the gather is mostly zero (3-4 s of record, an event of a few tens of
#  ms), so the global RMS is far below the amplitude of the event, and "0 dB"
#  is noise well below the signal in the window.  This measures the factor,
#  without regenerating anything; it is what turns the nominal levels into
#  the effective ones quoted in the paper.
#
#      python validation/synthetic/check_snr_scale.py --root dataset_geomvar --line test/line_00204.npz
#      python validation/synthetic/check_snr_scale.py --root dataset_geomvar --n 20
# =============================================================================
import os, glob, argparse, numpy as np

VW = 1500.


def measure(path, win_ms=200., lead_ms=80., n_ch=1):
    d = np.load(path, allow_pickle=True)
    g = d["gathers"]; dt = float(d["dt"]); D = float(d["D_water"])
    S, T, C = g.shape
    offs = (d["offsets"].astype(np.float32) if "offsets" in d.files
            else 125. + 25. * np.arange(C))
    W = int(round(win_ms * 1e-3 / dt)); lead = int(round(lead_ms * 1e-3 / dt))

    rms_glob = float(np.sqrt(np.mean(g ** 2)))
    seg, pre = [], []
    for s in range(S):
        for c in range(min(n_ch, C)):
            t_arr = np.sqrt((2 * D / VW) ** 2 + (offs[c] / VW) ** 2)
            k0 = int(round(t_arr / dt)) - lead
            if k0 < 0 or k0 + W > T:
                continue
            seg.append(g[s, k0:k0 + W, c])
            if k0 - W > 0:
                pre.append(g[s, k0 - W:k0, c])  # pre-arrival window
    if not seg:
        return None
    rms_win = float(np.sqrt(np.mean(np.asarray(seg) ** 2)))
    rms_pre = float(np.sqrt(np.mean(np.asarray(pre) ** 2))) if pre else np.nan
    return dict(rms_glob=rms_glob, rms_win=rms_win, rms_pre=rms_pre,
                ratio=rms_win / (rms_glob + 1e-30),
                D=D, T_ms=T * dt * 1e3, fmax=float(d["fmax"]) if "fmax" in d.files else 80.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_geomvar")
    ap.add_argument("--line", default=None)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    files = ([os.path.join(args.root, args.line)] if args.line else
             sorted(glob.glob(os.path.join(args.root, "test", "line_*.npz"))))
    if not args.line and args.n < len(files):
        files = [files[i] for i in
                 np.random.default_rng(0).permutation(len(files))[:args.n]]

    rows = []
    for f in files:
        r = measure(f)
        if r:
            r["file"] = os.path.basename(f); rows.append(r)
    if not rows:
        print("no line measured"); return

    ratio = np.array([r["ratio"] for r in rows])
    print(f"lines measured: {len(rows)}")
    print(f"\nRMS(primary window) / RMS(whole gather):")
    print(f"  median {np.median(ratio):6.2f}   [min {ratio.min():.2f}, "
          f"max {ratio.max():.2f}]  ->  {20*np.log10(np.median(ratio)):+.1f} dB")
    print(f"\nNOMINAL SNR (calibrated on the gather) -> EFFECTIVE SNR in the window:")
    for snr in (0., 10., 20., 30.):
        print(f"  {snr:5.0f} dB nominal  ->  {snr + 20*np.log10(np.median(ratio)):+6.1f} dB "
              f"effective on the event")
    print("\n  A difference of 15-25 dB puts all four levels of the matrix at")
    print("  high SNR, and the noise axis is not exercised.")

    if len(rows) > 3:
        print(f"\n--- does the factor depend on the record? ---")
        D = np.array([r["D"] for r in rows]); Tms = np.array([r["T_ms"] for r in rows])
        print(f"  Pearson(ratio, D_water)    = {np.corrcoef(D, ratio)[0,1]:+.3f}")
        print(f"  Pearson(ratio, record len) = {np.corrcoef(Tms, ratio)[0,1]:+.3f}")
        print("  (if it depends on D or on the record length, the nominal SNR is not")
        print("   even comparable between cells of the matrix)")

    if args.line:
        r = rows[0]
        print(f"\n{r['file']}: D {r['D']:.0f} m, record {r['T_ms']:.0f} ms, fmax {r['fmax']:.0f} Hz")
        print(f"  RMS global {r['rms_glob']:.3e} | window {r['rms_win']:.3e} | "
              f"pre-arrival {r['rms_pre']:.3e}")


if __name__ == "__main__":
    main()
