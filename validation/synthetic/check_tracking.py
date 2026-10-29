"""
check_tracking.py — how much of the synthetic matrix is a tracking failure?

On grid_D260_L300_r015 the matrix tracked the seafloor at
\\SI{256.9}{\\milli\\second} where the truth is \\SI{346.7}{}: an error of
\\SI{90}{\\milli\\second}, or \\SI{68}{\\metre} of water.  That record is the
worst of the 619 applicable ones and enters the paper as a limit of the method.
It is not: it is a limit of the tracking on that cell.

The question this answers is how often that happens, and whether the
correlation follows it.  It needs nothing new: every record already stores
t0_track_ms and t0_true_ms.  If the correlation collapses where the tracking
error is large, then part of what the synthetic section attributes to local
contamination is really a tracking failure, and the matrix should be
regenerated with the anchors taken from the local autocorrelation rather than
from the semblance.

    python3 validation/synthetic/check_tracking.py wavelet_matrix_v2
"""
import glob
import json
import os
import sys

import numpy as np


def main(root):
    files = sorted(glob.glob(os.path.join(root, 'results_shard*.jsonl')))
    if not files:
        raise SystemExit('no results_shard*.jsonl under %s' % root)

    err, ncc, snr, dnom, rid = [], [], [], [], []
    n_all = n_bad = 0
    for f in files:
        for line in open(f):
            n_all += 1
            r = json.loads(line)
            if not r.get('ok') or r.get('ncc_prim_true') is None:
                n_bad += 1
                continue
            if 't0_track_ms' not in r or 't0_true_ms' not in r:
                n_bad += 1
                continue
            err.append(r['t0_track_ms'] - r['t0_true_ms'])
            ncc.append(r['ncc_prim_true'])
            snr.append(r.get('snr_db', np.nan))
            dnom.append(r.get('D_true', np.nan))
            rid.append(r['run_id'])

    # Completeness first: a silent loss has biased an average on this project
    # before, and a number quoted without its denominator is not a measurement.
    print('%d records read from %d shards, %d unusable (%.1f%%)'
          % (n_all, len(files), n_bad, 100.0 * n_bad / max(n_all, 1)))
    if not err:
        raise SystemExit('nothing usable')

    e = np.array(err); v = np.array(ncc)
    s = np.array(snr); d = np.array(dnom)
    print('\ntracking error (tracked minus true):')
    print('  median %+.1f ms, mean %+.1f, s.d. %.1f' % (np.median(e), e.mean(), e.std()))
    for th in (10, 25, 50, 100):
        print('  |error| > %3d ms : %5d records (%.1f%%)'
              % (th, int((np.abs(e) > th).sum()), 100.0 * np.mean(np.abs(e) > th)))

    print('\ncorrelation against tracking error:')
    print('  %-18s %7s %9s %9s' % ('|error|', 'records', 'mean NCC', 'median'))
    edges = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 1e9)]
    for lo, hi in edges:
        m = (np.abs(e) >= lo) & (np.abs(e) < hi)
        if m.sum():
            print('  %3.0f-%-5.0f ms      %7d %9.3f %9.3f'
                  % (lo, hi, m.sum(), v[m].mean(), np.median(v[m])))
    good = np.abs(e) <= 25
    if good.any() and (~good).any():
        print('\n  well tracked (<=25 ms) : %.3f over %d records'
              % (v[good].mean(), int(good.sum())))
        print('  badly tracked (>25 ms) : %.3f over %d records'
              % (v[~good].mean(), int((~good).sum())))
        print('  difference             : %+.3f' % (v[~good].mean() - v[good].mean()))

    # Where does it fail?  If the failures cluster in shallow water the cause is
    # the search window, which is the same failure seen on the real long lines.
    if np.isfinite(d).any():
        print('\ntracking failure against nominal depth:')
        for lo, hi in [(0, 250), (250, 400), (400, 700), (700, 1e9)]:
            m = (d >= lo) & (d < hi)
            if m.sum():
                print('  D %4.0f-%-5.0f m : %5d records, |error|>25 ms in %5.1f%%'
                      % (lo, hi, m.sum(), 100.0 * np.mean(np.abs(e[m]) > 25)))

    worst = np.argsort(-np.abs(e))[:10]
    print('\nten largest tracking errors:')
    for i in worst:
        print('  %-28s error %+7.1f ms | NCC %.3f | D %5.0f m | SNR %4.0f'
              % (rid[i], e[i], v[i], d[i], s[i]))

    print('\nReading: if the two means above differ by little, the tracking is not')
    print('what limits the matrix and only the wavelets need regenerating.  If')
    print('they differ by a lot, part of what the paper attributes to local')
    print('contamination is a tracking failure, and the anchors should come from')
    print('the local autocorrelation before the matrix is run again.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'wavelet_matrix_v2')
