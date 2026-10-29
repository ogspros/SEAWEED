#!/usr/bin/env python3
"""Compare extracted wavelets against the frozen reference.

The smoke test of the repository is an md5 comparison: a refactoring that
changes one byte of one wavelet has changed a number of the paper, and md5
says so without needing an opinion.  md5 alone, however, cannot say *how* a
file changed, and it also fires when nothing changed in the code and only the
numerical library underneath did.  This script answers both questions.

    compare_wavelets.py REF NEW              compare two directories, or two
                                             .npy files, wavelet by wavelet
    compare_wavelets.py --stats DIR          write the reference statistics
                                             table (one line per wavelet)
    compare_wavelets.py --against-stats T D  compare a directory with such a
                                             table, when the reference .npy
                                             files are not at hand
    compare_wavelets.py --npz A B            compare two .npz, array by array;
                                             a tracking file is a .npz and its
                                             md5 is not reproducible, because
                                             numpy stamps the entries with the
                                             time they were written

Verdicts, one per file:

    identical   the arrays are bit-for-bit equal; md5 agrees as well
    numerical   they differ below --rtol of the reference peak amplitude;
                this is what a different BLAS or numpy version produces, and
                it needs an explanation in the environment, not in the code
    CHANGED     they differ above that; the modification is not cosmetic

The exit code is 0 when every file is identical, 2 when the worst verdict is
`numerical`, and 1 as soon as one file is CHANGED or missing.  The 2 is what
separates "the machine moved" from "the result moved": both deserve attention,
only one of them invalidates the paper's numbers.

The companion .npz of each run is compared as well when it is present.  A
wavelet can be identical while the stack coherence or the alignment residual
quoted in Tables 3 and 4 has moved, because those are computed alongside it.
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np

# Scalars of <run>_sel.npz that reach the paper, with the absolute tolerance
# applied to each.  Coherence is quoted to three decimals and the residuals to
# one, so the tolerances sit one digit below what is printed.
SEL_KEYS = (("coherence", 1e-4),
            ("origin_minus_lead_ms", 1e-3),
            ("align_residual_ms", 1e-3),
            ("align_scatter_ms", 1e-3),
            ("n_use", 0.0),
            ("sp_first", 0.0),
            ("sp_last", 0.0),
            ("sp_step", 0.0),
            ("D_med", 1e-3),
            ("vw", 1e-3))

MAX_LAG = 5          # samples searched when reporting the shift between two
                     # versions of the same wavelet; a genuine bug usually
                     # shows up as a shift, not only as a shape change


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def ncc_best_lag(a, b, max_lag=MAX_LAG):
    """Signed NCC maximised over a small shift, and the shift that gives it."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan"), 0
    best, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:a.size - lag]
        else:
            x, y = a[:a.size + lag], b[-lag:]
        if x.size < 4:
            continue
        d = np.linalg.norm(x) * np.linalg.norm(y)
        if d == 0.0:
            continue
        c = float(np.dot(x, y) / d)
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


def compare_arrays(ref, new, rtol):
    """Return (verdict, metrics) for one pair of wavelets."""
    m = {}
    if ref.shape != new.shape:
        m["note"] = "shape %s against %s" % (ref.shape, new.shape)
        return "CHANGED", m
    scale = float(np.max(np.abs(ref))) or 1.0
    d = np.abs(new.astype(np.float64) - ref.astype(np.float64))
    m["n"] = int(ref.size)
    m["max_abs"] = float(np.max(d))
    m["rel"] = float(np.max(d)) / scale
    m["rms_rel"] = float(np.sqrt(np.mean(d ** 2))) / scale
    ncc, lag = ncc_best_lag(ref, new)
    m["ncc"] = ncc
    m["lag"] = lag
    if m["max_abs"] == 0.0:
        return "identical", m
    if m["rel"] <= rtol:
        return "numerical", m
    return "CHANGED", m


def load_sel(path):
    """Read the scalars of a <run>_sel.npz, or None when it is absent."""
    if not os.path.exists(path):
        return None
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for key, _tol in SEL_KEYS:
            if key in z.files:
                try:
                    out[key] = float(np.asarray(z[key]).reshape(-1)[0])
                except (ValueError, IndexError):
                    pass
    return out


def compare_sel(ref_npy, new_npy):
    """Compare the companion .npz of two runs.  Returns a list of differences."""
    ref_sel = load_sel(ref_npy.replace("_wavelet.npy", "_sel.npz"))
    new_sel = load_sel(new_npy.replace("_wavelet.npy", "_sel.npz"))
    if ref_sel is None or new_sel is None:
        return []
    bad = []
    for key, tol in SEL_KEYS:
        if key not in ref_sel or key not in new_sel:
            continue
        a, b = ref_sel[key], new_sel[key]
        if np.isnan(a) and np.isnan(b):
            continue
        if abs(b - a) > tol:
            bad.append("%s %.6g -> %.6g" % (key, a, b))
    return bad


def wavelets(d, pattern):
    if os.path.isfile(d):
        return [d]
    return sorted(glob.glob(os.path.join(d, pattern)))


def cmd_compare(args):
    ref_files = wavelets(args.ref, args.glob)
    if not ref_files:
        sys.exit("no reference wavelet in %s" % args.ref)
    worst = 0
    print("%-34s %-9s %10s %10s %8s %5s" %
          ("wavelet", "verdict", "max|d|", "rel", "ncc", "lag"))
    for ref_path in ref_files:
        name = os.path.basename(ref_path)
        new_path = args.new if os.path.isfile(args.new) else os.path.join(args.new, name)
        if not os.path.exists(new_path):
            print("%-34s %-9s  not produced by this run" % (name, "MISSING"))
            worst = max(worst, 1)
            continue
        ref, new = np.load(ref_path), np.load(new_path)
        verdict, m = compare_arrays(ref, new, args.rtol)
        if "note" in m:
            print("%-34s %-9s  %s" % (name, verdict, m["note"]))
        else:
            print("%-34s %-9s %10.3e %10.3e %8.5f %5d" %
                  (name, verdict, m["max_abs"], m["rel"], m["ncc"], m["lag"]))
        if verdict == "CHANGED":
            worst = max(worst, 1)
        elif verdict == "numerical":
            worst = max(worst, 2) if worst != 1 else worst
        for line in compare_sel(ref_path, new_path):
            print("%-34s %-9s  %s" % ("", "qc", line))
            worst = 1
    print()
    if worst == 0:
        print("every wavelet is bit-for-bit the one that produced the paper")
    elif worst == 2:
        print("no wavelet moved beyond %g of its peak, but not every file is"
              % args.rtol)
        print("bit-identical: look for a changed numpy, BLAS or thread count")
        print("before reading this as a result of the modification")
    else:
        print("at least one wavelet or one quality-control scalar has moved:")
        print("this modification is not cosmetic and the numbers of the paper")
        print("have to be re-read before it is kept")
    return worst


def cmd_npz(args):
    """Compare two .npz files array by array, exactly."""
    ref_path, new_path = args.npz
    bad = 0
    with np.load(ref_path, allow_pickle=False) as a, \
            np.load(new_path, allow_pickle=False) as b:
        keys = sorted(set(a.files) | set(b.files))
        for key in keys:
            # params records how a file was made (code version included), not
            # what it holds, and the files of the paper predate it
            if key == "params":
                print("%-24s %s" % (key, "not compared (provenance)"))
                continue
            if key not in a.files or key not in b.files:
                print("%-24s %s" % (key, "only in one of the two files"))
                bad = 1
                continue
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if x.shape != y.shape:
                print("%-24s CHANGED  shape %s -> %s" % (key, x.shape, y.shape))
                bad = 1
            elif x.dtype.kind in "fc":
                d = float(np.max(np.abs(y.astype(np.float64)
                                        - x.astype(np.float64)))) if x.size else 0.0
                print("%-24s %-9s max|d| %.3e" %
                      (key, "identical" if d == 0.0 else "CHANGED", d))
                bad = bad or (d != 0.0)
            else:
                same = bool(np.array_equal(x, y))
                print("%-24s %s" % (key, "identical" if same else "CHANGED"))
                bad = bad or (not same)
    print()
    print("the two files agree" if not bad else
          "the tracking is not the one the paper was written on")
    return 1 if bad else 0


def cmd_stats(args):
    files = wavelets(args.stats, args.glob)
    if not files:
        sys.exit("no wavelet in %s" % args.stats)
    rows = ["\t".join(("file", "n", "md5", "rms", "peak_abs", "i_peak",
                       "i_min", "sum") +
                      tuple(k for k, _ in SEL_KEYS))]
    for path in files:
        w = np.load(path).astype(np.float64)
        sel = load_sel(path.replace("_wavelet.npy", "_sel.npz")) or {}
        row = [os.path.basename(path), str(w.size), md5(path),
               "%.9e" % float(np.sqrt(np.mean(w ** 2))),
               "%.9e" % float(np.max(np.abs(w))),
               str(int(np.argmax(np.abs(w)))),
               str(int(np.argmin(w))),
               "%.9e" % float(np.sum(w))]
        for key, _tol in SEL_KEYS:
            row.append("%.9g" % sel[key] if key in sel else "-")
        rows.append("\t".join(row))
    text = "\n".join(rows) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print("%d wavelets summarised in %s" % (len(files), args.out))
    else:
        sys.stdout.write(text)
    return 0


def cmd_against_stats(args):
    table, directory = args.against_stats
    with open(table) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        ref_rows = {}
        for line in fh:
            if not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            ref_rows[cells[0]] = dict(zip(head, cells))
    if not ref_rows:
        sys.exit("%s holds no reference row" % table)
    worst = 0
    print("%-34s %-9s %s" % ("wavelet", "verdict", "first disagreement"))
    for name in sorted(ref_rows):
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            print("%-34s %-9s  not produced by this run" % (name, "MISSING"))
            worst = 1
            continue
        row = ref_rows[name]
        if md5(path) == row["md5"]:
            print("%-34s %-9s" % (name, "identical"))
            continue
        w = np.load(path).astype(np.float64)
        got = {"n": w.size,
               "rms": float(np.sqrt(np.mean(w ** 2))),
               "peak_abs": float(np.max(np.abs(w))),
               "i_peak": int(np.argmax(np.abs(w))),
               "i_min": int(np.argmin(w)),
               "sum": float(np.sum(w))}
        bad = ""
        for key in ("n", "i_peak", "i_min", "peak_abs", "rms", "sum"):
            want = float(row[key])
            tol = 0.0 if key in ("n", "i_peak", "i_min") else abs(want) * args.rtol
            if abs(got[key] - want) > tol:
                bad = "%s %.9g -> %.9g" % (key, want, got[key])
                break
        if bad:
            print("%-34s %-9s  %s" % (name, "CHANGED", bad))
            worst = 1
        else:
            print("%-34s %-9s  md5 differs, statistics agree" % (name, "numerical"))
            worst = max(worst, 2) if worst != 1 else worst
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?", help="reference directory or .npy file")
    ap.add_argument("new", nargs="?", help="directory or .npy file to check")
    ap.add_argument("--stats", metavar="DIR",
                    help="write the reference statistics of DIR instead of comparing")
    ap.add_argument("--against-stats", nargs=2, metavar=("TABLE", "DIR"),
                    help="compare DIR with a statistics table written by --stats")
    ap.add_argument("--npz", nargs=2, metavar=("REF", "NEW"),
                    help="compare two .npz files array by array")
    ap.add_argument("--out", default=None, help="destination of --stats")
    ap.add_argument("--glob", default="*_wavelet.npy",
                    help="wavelets to compare (default %(default)s)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="difference tolerated as numerical, as a fraction of "
                         "the reference peak (default %(default)s)")
    args = ap.parse_args()

    if args.stats:
        sys.exit(cmd_stats(args))
    if args.against_stats:
        sys.exit(cmd_against_stats(args))
    if args.npz:
        sys.exit(cmd_npz(args))
    if not args.ref or not args.new:
        ap.error("give two directories, or --stats, or --against-stats")
    sys.exit(cmd_compare(args))


if __name__ == "__main__":
    main()
