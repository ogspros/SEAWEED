"""
check_weight_spread.py — is there anything for a weighting scheme to do?

On the Viking line, replacing coherence weights with uniform ones moved the
agreement with the recorded signature from 0.831 to 0.829.  The reason is that
the per-window coherences are narrowly distributed, so the weights are already
nearly uniform and no scheme can differ much from any other.

Before proposing a different weighting -- local rather than global coherence,
signal-to-noise, or anything else -- it is worth asking whether the other
surveys leave more room.  A weighting can only matter where the weights are
dispersed.  Two numbers say it:

    the coefficient of variation of the weights, sd/mean.  At 0 every window
    counts the same whatever the scheme; the effective number of windows is
    then the actual number.

    the effective count, (sum w)^2 / sum w^2, which is how many windows the
    weighted stack is really averaging.  Divided by the actual count it gives
    the fraction of the data the weighting is using.

The per-window coherences are stored by the workflow as coh_per_window, so this
needs no new extraction.  Where they are not stored, the script says so rather
than substituting the mean.

    python3 validation/viking/check_weight_spread.py <sel.npz> [<sel.npz> ...]
"""
import os
import sys

import numpy as np


def summarise(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)
    if "coh_per_window" not in keys:
        return None, keys
    c = np.asarray(z["coh_per_window"], float).ravel()
    c = c[np.isfinite(c)]
    if c.size == 0:
        return None, keys
    w = np.clip(c, 0, None)                    # the weights the stack uses
    n_eff = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-30)
    return dict(n=c.size, mean=float(w.mean()), sd=float(w.std()),
                cv=float(w.std() / (w.mean() + 1e-30)),
                p05=float(np.percentile(w, 5)), p95=float(np.percentile(w, 95)),
                zero=float(np.mean(w <= 0)), n_eff=float(n_eff),
                frac=float(n_eff / c.size)), keys


def main(paths):
    print("%-34s %6s %7s %7s %7s %8s %8s"
          % ("run", "n", "mean", "sd", "cv", "n_eff", "n_eff/n"))
    missing = []
    for p in paths:
        s, keys = summarise(p)
        name = os.path.basename(p).replace("_sel.npz", "")
        if s is None:
            missing.append((name, keys))
            print("  %-32s  coh_per_window not stored" % name)
            continue
        print("%-34s %6d %7.3f %7.3f %7.3f %8.0f %8.3f"
              % (name, s["n"], s["mean"], s["sd"], s["cv"],
                 s["n_eff"], s["frac"]))
        print("%-34s   5th-95th percentile %.3f-%.3f, %.1f%% at zero weight"
              % ("", s["p05"], s["p95"], 100 * s["zero"]))

    if missing:
        print("\ncoh_per_window is in the qc dictionary but was not written to "
              "these files.\nIt can be added where run_real saves the selection, "
              "or read from a fresh run.")
    print("""
Reading: cv near 0 and n_eff/n near 1 mean the weights are already uniform and
no scheme can differ from another by much -- which is what Viking showed, where
uniform and coherence weights gave 0.829 against 0.831.  A cv of a few tenths,
or n_eff/n well below 1, means the weighting is discarding a real fraction of
the data, and there the choice of scheme is worth testing against ground truth
-- which exists only on Viking, so it would have to be tested there and assumed
to carry over.""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
