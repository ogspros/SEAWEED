#!/usr/bin/env python3
"""
phase_synth_test.py -- the Viking phase test, repeated where the phase is known.

On Viking, replacing the phase of the extracted wavelet with that of its
minimum-phase equivalent RAISED the agreement with the recorded signature by
0.030 +- 0.003 on ten combs of ten.  One line cannot say whether that is a
general limit of the extraction or a property of that line and that source.
The synthetic matrix can: every record carries its true reference, and the
aligned extracted segment is archived next to it (results_shard*_wavelets.npz,
keys "run_id|snr_db|t0_bias_ms" and the same with "|ref").

For each record three candidates are scored against the true reference:

    extracted   the aligned segment as the workflow returned it
    minimum     its minimum-phase equivalent (same modulus, exactly)

The zero-phase candidate of viking_phase_test.py is left out: centred in the
window it sits 100 ms from a causal reference, beyond any shift limit that
keeps the comparison honest, and it would score near zero for that reason
alone.

Per record there is also one bound that needs no extraction at all:

    ceiling     the minimum-phase equivalent of the TRUE reference, scored
                against the reference: the best any amplitude-only method can
                do on that record, whatever the wavelet

PROTOCOL.  All candidates go through the same comparison: the reference is
treated as zero outside its 200 ms (it is causal, so before t = 0 that is
exact), and each candidate of the same length slides over it by at most
+-MAXLAG samples, the score being the signed NCC over the full candidate
length.  No candidate is band-passed differently from another, none gets a
polarity flip, and all get the same shift limit, because a comparison in which
the candidates are treated differently measures the treatment.

MINIMUM-PHASE CONSTRUCTION.  Cepstral: the phase is the Hilbert transform of
the log amplitude, on nfft = 8192.  Where the amplitude is zero the log is
undefined, so it is floored at EPS times the peak.  That floor is a choice and
the answer may depend on it, which is why the test is repeated at three floors.

Reading it
  extracted above minimum on most records   -> the extraction does recover
      phase on synthetics, and the Viking result is a property of that line
  minimum above extracted on most records   -> the phase is a general weakness
      of the extraction, and the recommendation can be made more broadly
  no consistent sign                        -> the phase is not resolved either
      way, and the paper should say only what Viking shows

VALIDATION (run before trusting it on the matrix):
    python3 validation/synthetic/phase_synth_test.py --selftest
builds a mixed-phase airgun-like wavelet with a ghost, and checks that
(a) a perfect extraction scores extracted > minimum by exactly 1 - ceiling,
(b) an extraction that IS the minimum-phase equivalent scores the two equal,
(c) the minimum-phase construction preserves the modulus in band.

On the matrix:
    python3 validation/synthetic/phase_synth_test.py wavelet_matrix_v2 --wlen_ms 200
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

EPS_LIST = (1e-8, 1e-4, 1e-2)          # log-amplitude floors, relative to peak
NFFT = 8192


# ---------------------------------------------------------------- phases
def minimum_phase(x, eps=1e-8, nfft=NFFT):
    """Same construction as viking_phase_test.py, with the floor exposed."""
    n = x.size
    A = np.abs(np.fft.rfft(x, nfft))
    logA = np.log(np.maximum(A, eps * A.max()))
    full = np.concatenate([logA, logA[-2:0:-1]])
    c = np.fft.ifft(full).real
    h = np.zeros(nfft)
    h[0] = 1.0
    h[1:nfft // 2] = 2.0
    h[nfft // 2] = 1.0
    return np.fft.ifft(np.exp(np.fft.fft(c * h))).real[:n]


def modulus_error(a, b, eps=1e-8, nfft=NFFT):
    """Largest difference of the two amplitude spectra, relative to the peak,
    where a stands well above the floor.  Below the floor the construction
    changes the modulus by design, so that part is excluded, not hidden: at
    eps = 1e-2 the excluded region is everything under 10 per cent of the
    peak, and the floor itself is what the three-floor repeat tests."""
    A = np.abs(np.fft.rfft(a, nfft))
    B = np.abs(np.fft.rfft(b, nfft))
    m = A > max(0.01, 10 * eps) * A.max()
    return float(np.abs(A - B)[m].max() / A.max())


# ---------------------------------------------------------------- comparison
def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 1e-30 else 0.0


def score(cand, ref, maxlag):
    """Best signed NCC of a candidate against a reference zero-padded on both
    sides, over +-maxlag samples.  Full candidate length at every lag."""
    n = ref.size
    rp = np.concatenate([np.zeros(maxlag), ref, np.zeros(maxlag)])
    best, bl = -2.0, 0
    for L in range(-maxlag, maxlag + 1):
        v = _ncc(cand, rp[maxlag + L:maxlag + L + n])
        if v > best:
            best, bl = v, L
    return best, bl


def evaluate(w, ref, maxlag, eps):
    wmin = minimum_phase(w, eps)
    rmin = minimum_phase(ref, eps)
    return dict(
        ext=score(w, ref, maxlag)[0],
        ext0=_ncc(w, ref),
        mn=score(wmin, ref, maxlag)[0],
        ceil=score(rmin, ref, maxlag)[0],
        merr=modulus_error(w, wmin, eps))


# ---------------------------------------------------------------- self test
def selftest():
    dt = 0.002
    n = 100
    t = np.arange(n) * dt
    # mixed phase by construction: a primary pulse, a precursor-free bubble
    # train, and a ghost of the wrong sign 8 ms later
    rng = np.random.default_rng(0)
    # the main pulse is the SECOND lobe, which puts zeros outside the unit
    # circle: a minimum-phase equivalent front-loads the energy and differs
    s = (0.45 * np.exp(-((t - 0.010) / 0.004) ** 2)
         - 1.0 * np.exp(-((t - 0.020) / 0.004) ** 2)
         + 0.5 * np.exp(-((t - 0.030) / 0.005) ** 2)
         + 0.35 * np.exp(-((t - 0.090) / 0.012) ** 2)
         - 0.2 * np.exp(-((t - 0.100) / 0.012) ** 2))
    g = np.zeros(5); g[0] = 1; g[4] = -1
    ref = np.convolve(s, g)[:n]
    ref /= np.linalg.norm(ref)
    ok = True
    for eps in EPS_LIST:
        # (a) perfect extraction
        r = evaluate(ref, ref, 10, eps)
        a_ok = abs(r["ext"] - 1.0) < 1e-9 and abs((r["ext"] - r["mn"])
                                                  - (1 - r["ceil"])) < 1e-9
        # (b) extraction that already IS the minimum-phase equivalent
        r2 = evaluate(minimum_phase(ref, eps), ref, 10, eps)
        b_ok = abs(r2["ext"] - r2["mn"]) < 2e-3
        # (c) modulus preserved
        c_ok = r["merr"] < 5e-3 and r2["merr"] < 5e-3
        # a noisy extraction should sit between the two
        wn = ref + 0.15 * rng.normal(0, 1, n) / np.sqrt(n)
        r3 = evaluate(wn, ref, 10, eps)
        print("eps %.0e  ceiling %.4f | perfect: ext %.4f min %.4f (%s) | "
              "already-min: ext %.4f min %.4f (%s) | modulus err %.1e (%s) | "
              "noisy: ext %.4f min %.4f"
              % (eps, r["ceil"], r["ext"], r["mn"], "ok" if a_ok else "FAIL",
                 r2["ext"], r2["mn"], "ok" if b_ok else "FAIL",
                 max(r["merr"], r2["merr"]), "ok" if c_ok else "FAIL",
                 r3["ext"], r3["mn"]))
        ok = ok and a_ok and b_ok and c_ok
    if r["ceil"] > 0.95:
        print("NOTE: the test wavelet is too close to minimum phase to "
              "discriminate; the self test is not informative")
        ok = False
    print("self test", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- matrix
def load(root):
    recs = []
    rp = os.path.join(root, "results.jsonl")
    files = [rp] if os.path.exists(rp) else sorted(
        glob.glob(os.path.join(root, "results_shard*.jsonl")))
    for f in files:
        recs.extend(json.loads(l) for l in open(f))
    waves = {}
    for f in sorted(glob.glob(os.path.join(root, "results_shard*_wavelets.npz"))):
        with np.load(f) as z:
            for k in z.files:
                waves[k] = z[k]
    return recs, waves


def summarise(name, R, key_a, key_b):
    d = np.array([r[key_a] - r[key_b] for r in R])
    if d.size == 0:
        return
    print("  %-34s %+.4f +- %.4f  median %+.4f  positive %d/%d (%.0f%%)"
          % (name, d.mean(), d.std(ddof=1) if d.size > 1 else 0.0,
             np.median(d), (d > 0).sum(), d.size, 100 * (d > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="wavelet_matrix_v2")
    ap.add_argument("--wlen_ms", type=float, default=200.0)
    ap.add_argument("--maxlag_ms", type=float, default=40.0,
                    help="shift limit, the same for every candidate")
    ap.add_argument("--dmin", type=float, default=220.0,
                    help="applicable domain, on the NOMINAL depth D_m as in the "
                         "paper.  D_true is the depth the 3 m grid actually "
                         "holds, 219 m in the 220 m cell, and filtering on it "
                         "silently drops that whole cell")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    recs, waves = load(a.root)
    print("%d records, %d archived arrays in %s" % (len(recs), len(waves), a.root))
    if not waves:
        raise SystemExit("no *_wavelets.npz in %s: the matrix was run before "
                         "the wavelets were archived, and this test needs them"
                         % a.root)
    pool, missing = [], 0
    for r in recs:
        if not r.get("ok") or r.get("D_m", r.get("D_true", 0)) < a.dmin:
            continue
        if str(r.get("run_id", "")).startswith("ref_"):
            continue                          # controls, as in n = 2480
        k = "%s|%s|%s" % (r["run_id"], r.get("snr_db"), r.get("t0_bias_ms"))
        if k not in waves or k + "|ref" not in waves:
            missing += 1
            continue
        pool.append((r, waves[k].astype(float), waves[k + "|ref"].astype(float)))
    print("applicable domain: %d records with arrays, %d without" % (len(pool), missing))
    if not pool:
        raise SystemExit("nothing to test")

    n = pool[0][1].size
    dt = a.wlen_ms * 1e-3 / n
    maxlag = int(round(a.maxlag_ms * 1e-3 / dt))
    print("wavelet %d samples, dt %.1f ms, shift limit +-%d samples (%.0f ms)"
          % (n, dt * 1e3, maxlag, maxlag * dt * 1e3))

    # consistency with the matrix: the archived segment scored at zero lag
    # must reproduce ncc_prim_true, or the arrays are not the ones scored
    for eps in EPS_LIST:
        R = []
        for r, w, ref in pool:
            e = evaluate(w, ref, maxlag, eps)
            e.update(snr=r.get("snr_db"), axis=r.get("axis"),
                     pad=r.get("n_invented_kept"), ncc=r.get("ncc_prim_true"))
            R.append(e)
        dev = np.nanmax([abs(e["ext0"] - e["ncc"]) for e in R
                         if e["ncc"] is not None])
        print("\n=== log-amplitude floor %.0e of the peak ===" % eps)
        print("  zero-lag score vs ncc_prim_true in the file: max |diff| %.1e%s"
              % (dev, "" if dev < 1e-3 else "   <-- arrays and file disagree"))
        print("  modulus error of the minimum-phase candidate: max %.1e"
              % max(e["merr"] for e in R))
        for lab, k in (("extracted", "ext"), ("minimum phase", "mn"),
                       ("ceiling, min-phase of truth", "ceil")):
            v = np.array([e[k] for e in R])
            print("  %-30s mean %.4f  median %.4f" % (lab, v.mean(), np.median(v)))
        summarise("extracted minus minimum", R, "ext", "mn")
        summarise("extracted minus ceiling", R, "ext", "ceil")
        # an absent field is not a zero padding: report it as absent
        if all(e["pad"] is None for e in R):
            print("    (n_invented_kept not recorded in this file: no "
                  "unpadded subset)")
        else:
            clean = [e for e in R if e["pad"] == 0]
            summarise("  same, unpadded records only", clean, "ext", "mn")
        print("  by noise level:")
        for s in sorted({e["snr"] for e in R}, key=lambda x: float(x)):
            summarise("    SNR %s dB" % s, [e for e in R if e["snr"] == s],
                      "ext", "mn")
        print("  by axis:")
        for ax in sorted({str(e["axis"]) for e in R}):
            summarise("    %s" % ax, [e for e in R if str(e["axis"]) == ax],
                      "ext", "mn")


if __name__ == "__main__":
    main()
