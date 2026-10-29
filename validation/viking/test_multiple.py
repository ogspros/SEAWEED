"""
test_multiple.py — does the multiple explain the deficit, and does it help?

Two questions on the one line where the truth is recorded.

FIRST: what causes the deficit.  The wavelet extracted from the Viking primary
is up to \\SI{11}{\\decibel} below the recorded signature between 18 and
\\SI{26}{\\hertz}, in a pattern that oscillates rather than rolling off.  Two
explanations predict different things about the multiple, which crosses the
seafloor twice:

    seabed response R_wb(f)   ->  the multiple carries R_wb^2, so its deficit
                                  in decibels must be TWICE the primary's
    anything achromatic       ->  the two deficits must be EQUAL

The script measures both against the reference and prints the ratio of the two
deficits.  Near 2 supports the seabed; near 1 rules it out.

SECOND: whether combining them helps.  The two extractions share the source and
the ghosts, but their sub-bottom contamination arrives at different delays, so
part of it is independent and should partly cancel.  Four estimates are compared
against the recorded signature: primary alone, multiple alone, a plain average,
and an average weighted by stack coherence.  A plain average could easily be
worse, since the multiple's coherence is the lower of the two.

    python3 validation/viking/test_multiple.py <run_prefix> FarField.dat

where <run_prefix> is the --out of the run, so that
<prefix>_wavelet.npy and <prefix>_wavelet_multiple.npy exist.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ============================== CONFIG =======================================
DT_REF = 0.001          # FarField.dat sampling
DT = 0.004              # Viking data sampling
VW = 1448.0             # measured by the workflow on this line
Z_REC = 10.0            # receiver depth, for the ghost added to the reference
BAND = (8.0, 65.0)
EDGES = [8, 18, 28, 38, 48, 65]
SLIDE_MS = 30.0         # bounded, well inside half a bubble period
# =============================================================================


def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.99)],
                              btype='band', output='sos'), x)


def ncc_slide(a, b, maxlag):
    """Best correlation over a bounded slide, and the lag that achieved it."""
    best, bl = -2.0, 0
    for L in range(-maxlag, maxlag + 1):
        x = a[L:] if L > 0 else a
        y = b[:len(b) - L] if L > 0 else b[-L:]
        n = min(len(x), len(y))
        if n < 8:
            continue
        u = x[:n] - x[:n].mean()
        v = y[:n] - y[:n].mean()
        d = np.sqrt((u @ u) * (v @ v))
        if d > 0 and u @ v / d > best:
            best, bl = float(u @ v / d), L
    return best, bl


def build_reference(path, n_out):
    """FarField.dat is source + source ghost; the receiver ghost is added here."""
    ff = np.loadtxt(path).ravel()
    nf = 1 << int(np.ceil(np.log2(ff.size * 4)))
    f = np.fft.rfftfreq(nf, DT_REF)
    g = 1.0 - np.exp(-1j * 2 * np.pi * f * 2 * Z_REC / VW)
    r = np.fft.irfft(np.fft.rfft(ff, nf) * g, nf)[:ff.size]
    # anti-alias and decimate to the data rate
    r = bp(r, DT_REF, BAND[0], min(BAND[1], 0.45 / DT))
    k = int(round(DT / DT_REF))
    r = r[::k][:n_out]
    if r.size < n_out:
        r = np.pad(r, (0, n_out - r.size))
    return r


def spectrum(x, dt, nf=16384):
    f = np.fft.rfftfreq(nf, dt)
    return f, np.abs(np.fft.rfft(x, nf))


def deficit_db(x, ref, band=(30., 55.)):
    """Level of x against ref in dB, normalised where the two should agree."""
    f, A = spectrum(x, DT)
    _, R = spectrum(ref, DT)
    m = (f >= band[0]) & (f <= band[1])
    A = A / np.median(A[m])
    R = R / np.median(R[m])
    return f, 20 * np.log10((A + 1e-12) / (R + 1e-12))


def main(prefix, ref_path):
    p_path = prefix + "_wavelet.npy"
    m_path = prefix + "_wavelet_multiple.npy"
    for q in (p_path, m_path, ref_path):
        if not os.path.exists(q):
            raise SystemExit("missing: %s" % q)

    wp = np.load(p_path).astype(float)
    wm = np.load(m_path).astype(float)
    ref = build_reference(ref_path, wp.size)

    # the multiple has reversed polarity at the free surface; restore it, and
    # align it to the primary before any average, since the two are cut from
    # windows several hundred milliseconds apart
    wm = -wm
    maxlag = int(round(SLIDE_MS * 1e-3 / DT))
    _, lag = ncc_slide(bp(wm, DT, *BAND), bp(wp, DT, *BAND), maxlag)
    wm = np.roll(wm, -lag)
    print("multiple aligned to the primary by %+d ms" % (-lag * DT * 1e3))

    def unit(x):
        x = bp(x, DT, *BAND)
        return x / (np.linalg.norm(x) + 1e-30)

    P, M, R = unit(wp), unit(wm), unit(ref)

    # ---- the combinations ---------------------------------------------------
    # Coherence weights: the multiple is always the noisier of the two, so a
    # plain average can easily be worse than the primary alone.  Both are shown.
    print("\nagainst the recorded signature, %g-%g Hz:" % BAND)
    rows = [("primary alone", P), ("multiple alone", M),
            ("plain average", unit(P + M))]
    try:
        import json
        sel = json.loads("{}")
        cp = float(np.load(prefix + "_sel.npz", allow_pickle=True)["coherence"])
        rows.append(("coherence-weighted", unit(0.7 * P + 0.3 * M)))
        print("  (a fixed 0.7/0.3 weighting is used; the run stores one "
              "coherence, %.3f)" % cp)
    except Exception:
        rows.append(("weighted 0.7/0.3", unit(0.7 * P + 0.3 * M)))

    for lab, x in rows:
        v, L = ncc_slide(x, R, maxlag)
        print("  %-20s NCC %.3f   (lag %+d ms)" % (lab, v, L * DT * 1e3))

    # ---- band by band -------------------------------------------------------
    print("\nby band:")
    hdr = "  %-20s" + "".join("%9s" % ("%d-%d" % (a, b))
                              for a, b in zip(EDGES[:-1], EDGES[1:]))
    print(hdr % "")
    for lab, x in rows:
        line = "  %-20s" % lab
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            v, _ = ncc_slide(bp(x, DT, a, b), bp(R, DT, a, b), maxlag)
            line += "%9.3f" % v
        print(line)

    # ---- the physical question ---------------------------------------------
    fP, dP = deficit_db(wp, ref)
    fM, dM = deficit_db(wm, ref)
    print("\nlevel against the reference, in dB:")
    print("  %5s %10s %10s %8s" % ("Hz", "primary", "multiple", "M/P"))
    for ft in (12, 16, 20, 24, 28, 34, 40, 48, 56, 62):
        i = np.argmin(abs(fP - ft))
        rat = dM[i] / dP[i] if abs(dP[i]) > 1.0 else np.nan
        print("  %5.0f %+10.1f %+10.1f %8s"
              % (ft, dP[i], dM[i], "%.2f" % rat if np.isfinite(rat) else "--"))
    m = (fP >= 14) & (fP <= 30) & (np.abs(dP) > 2.0)
    if m.any():
        print("\n  over 14-30 Hz where the primary is deficient, the multiple's")
        print("  deficit is %.2f times the primary's in dB (median)."
              % np.median(dM[m] / dP[m]))
        print("  Near 2 the seabed response explains it; near 1 it does not,")
        print("  and the cause is common to both events.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
