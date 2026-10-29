"""
count_padding.py — how many wavelets in the matrix have a zero-filled tail?

THE DEFECT
----------
extract_wavelet ends with

    w = st[i0:i0 + Lw]
    if len(w) < Lw:
        w = np.pad(w, (0, Lw - len(w)))

so when the estimated origin i0 falls late enough that the window runs past the
end of the stack, the shortfall is filled with zeros instead of being reported.
Correlating against a zero tail is EASIER than against a wrong one, because the
product there is null rather than adverse, so the padding inflates the score of
exactly the records where the origin was hardest to place.

It is data-dependent, not geometric: grid_D600_L150_r015 has 82 padded samples
at 0 dB and none at 10, 20 and 30 dB, the same model with a different noise
realisation.

WHAT THIS COUNTS
----------------
The wavelets themselves were stored for only a handful of runs, but the padding
is reconstructible for every record, because

    i0 = lead + origin_minus_lead_ms / dt        (lead = min(80 + pad, lead_max))
    len(st) = lead + (w_len + 2*pad) + tail      (tail = 60 ms by default)
    padded  = max(0, i0 + Lw - len(st))          (Lw = (w_len + 2*pad)/dt)

so the count needs nothing but the records.  The formula is stated here rather
than imported because it must be checked against the six stored wavelets before
being trusted: the script does that first and refuses if it does not reproduce
them.

    python3 validation/synthetic/count_padding.py wavelet_matrix_v2 --wavelets wavelet_matrix_fig
"""
import argparse
import glob
import json
import os

import numpy as np

W_LEN_MS = 200.0      # meta.w_len_ms
TAIL_MS = 60.0        # extract_wavelet default
LEAD_CAP_MS = 80.0    # lead = min(80 + pad, lead_max)
KEPT = 100            # samples the runner keeps: wp = wlong[off_p:off_p+100]


def padded_samples(rec, dt_ms):
    """Zero-filled samples, in the long wavelet and in the slice that is kept.

    Two quantities, and they are not the same.  extract_wavelet returns a
    wavelet of w_len + 2*pad and fills its tail with zeros whenever the origin
    falls more than `tail` past the nominal lead:

        P = max(0, origin_minus_lead / dt - tail)

    The runner then keeps a slice of it, wp = wlong[off_p : off_p + 100], and
    the padding is only VISIBLE where the slide reaches into it:

        visible = max(0, (off_p + 100) - (Lw - P)),  off_p = (match_pos - oml)/dt

    That distinction is the whole point.  On grid_D600_L150_r015 the long
    wavelet is padded by about 95 samples at every SNR, yet the stored slice
    shows 82 zeros at 0 dB and none at 10, 20 and 30 -- because there the slide
    happens to stop short of the filled region.  Counting only what is visible
    would therefore understate the defect by a wide margin.
    """
    pad = float(rec.get("pad_ms", 120.0))
    Lw = int(round((W_LEN_MS + 2 * pad) / dt_ms))
    tail = int(round(TAIL_MS / dt_ms))
    oml = float(rec["origin_minus_lead_ms"])
    P = max(0, int(round(oml / dt_ms)) - tail)
    mp = rec.get("match_pos_ms")
    if mp is None:
        return P, None, Lw
    off_p = int(round((float(mp) - oml) / dt_ms))
    visible = max(0, min(KEPT, (off_p + KEPT) - (Lw - P)))
    return P, visible, Lw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--wavelets", default=None,
                    help="directory with stored wavelets, used to VERIFY the "
                         "formula before any number is quoted from it")
    ap.add_argument("--dt_ms", type=float, default=2.0)
    ap.add_argument("--d_min", type=float, default=220.0)
    a = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(os.path.join(a.root, "results_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("ok") and r.get("ncc_prim_true") is not None \
                    and r.get("origin_minus_lead_ms") is not None:
                recs.append(r)
    if not recs:
        raise SystemExit("no usable records under %s" % a.root)
    print("%d records with a score and an origin" % len(recs))

    # ---- verify the formula against the stored wavelets ---------------------
    if a.wavelets:
        z = {}
        for f in glob.glob(os.path.join(a.wavelets, "*_wavelets.npz")):
            with np.load(f) as arch:
                z.update({k: arch[k] for k in arch.files})
        by_key = {f"{r['run_id']}|{r.get('snr_db')}|{r.get('t0_bias_ms')}": r
                  for r in recs}
        checked = bad = 0
        print("\nverifying the formula against the stored wavelets:")
        for k in sorted(x for x in z if not x.endswith("|ref")):
            if k not in by_key:
                continue
            w = z[k]
            nz = w.size if not np.any(w) else w.size - int(np.max(np.nonzero(w))) - 1
            _, pred, _ = padded_samples(by_key[k], a.dt_ms)
            ok = abs(nz - pred) <= 2
            checked += 1
            bad += (not ok)
            print("  %-34s stored %3d | predicted %3d  %s"
                  % (k, nz, pred, "" if ok else "<-- MISMATCH"))
        if checked == 0:
            print("  no stored wavelet matches a record; cannot verify")
        elif bad:
            raise SystemExit(
                "\nThe formula does not reproduce %d of %d stored wavelets, so the\n"
                "counts below would be guesses.  Read the lead and tail actually\n"
                "used by the runner before trusting them." % (bad, checked))
        else:
            print("  formula reproduces all %d stored wavelets" % checked)

    # ---- the count ----------------------------------------------------------
    long_n, vis_n, Lw = [], [], None
    for r in recs:
        P, vis, Lw = padded_samples(r, a.dt_ms)
        long_n.append(P); vis_n.append(vis if vis is not None else 0)
    p = np.array(long_n, float)          # the defect, whether or not it shows
    pv = np.array(vis_n, float)          # what a stored slice would reveal
    v = np.array([r["ncc_prim_true"] for r in recs])
    D = np.array([r.get("D_m", np.nan) for r in recs])
    frac = p / max(Lw, 1)

    print("\nvisible in the kept slice: %d of %d records (%.1f%%) -- this is"
          % (int((pv > 0).sum()), len(pv), 100 * np.mean(pv > 0)))
    print("what an inspection of the stored wavelets would have found.\n")
    print("zero-filled fraction of the extracted (long) wavelet:")
    print("  none            : %5d records (%.1f%%)" % ((p == 0).sum(), 100 * np.mean(p == 0)))
    for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, 1.01)]:
        m = (frac > lo) & (frac <= hi)
        if m.sum():
            print("  %3.0f-%-3.0f%% padded : %5d records (%.1f%%)  mean NCC %.3f"
                  % (100 * lo, 100 * hi, m.sum(), 100 * m.mean(), v[m].mean()))

    ap_ = D >= a.d_min
    clean = ap_ & (p == 0)
    print("\ninside the applicable domain (D >= %.0f m), %d records:" % (a.d_min, ap_.sum()))
    print("  all of them           : median %.3f  mean %.3f +- %.3f"
          % (np.median(v[ap_]), v[ap_].mean(), v[ap_].std()))
    if clean.sum():
        print("  unpadded only (%4d)  : median %.3f  mean %.3f +- %.3f"
              % (clean.sum(), np.median(v[clean]), v[clean].mean(), v[clean].std()))
        print("  difference            : median %+.3f  mean %+.3f"
              % (np.median(v[ap_]) - np.median(v[clean]),
                 v[ap_].mean() - v[clean].mean()))
    if (ap_ & (p > 0)).sum() > 10:
        print("  correlation between padded fraction and NCC, applicable domain: "
              "%+.2f" % np.corrcoef(frac[ap_], v[ap_])[0, 1])

    print("\nReading: if the unpadded subset scores the same, the padding is")
    print("cosmetic and only the figure's selection needs fixing.  If it scores")
    print("lower, the headline numbers are inflated by the records where the")
    print("origin fell latest, and the extractor should refuse those rather than")
    print("fill them.")


if __name__ == "__main__":
    main()
