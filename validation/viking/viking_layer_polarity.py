#!/usr/bin/env python3
"""
Polarity and delay of a sub-seafloor reflector from two extracted wavelets.

Model: the wavelet extracted where the reflector is present is the wavelet
extracted in the clear zone plus a delayed, scaled copy of itself,

    w_layer(t) = a * w_clear(t - s) + b * w_clear(t - s - tau),   a > 0.

The clear wavelet carries the same source signature, bubble and ghosts, so
these cancel and only the reflector is left to explain.  The sign of b is the
polarity of the reflector relative to the seafloor.  A mixed-phase wavelet
whose first two lobes have similar amplitude makes a reversed copy at tau look
much like a same-sign copy at a different tau, so the fit is run with the sign
of b constrained both ways and the two best correlations are compared.

On the multiple, the two first-order peg-legs arrive at the same delay tau
with twice the relative amplitude.  Running the same fit on the multiples is
therefore an independent check on tau and on the sign.

usage:
  python3 validation/viking/viking_layer_polarity.py --clear viking_zones/clear_wavelet.npy \
      --layer viking_zones/layer_wavelet.npy \
      [--clear_mult viking_zones/clear_wavelet_multiple.npy \
       --layer_mult viking_zones/layer_wavelet_multiple.npy] \
      [--dt 0.004] [--out viking_zones/polarity]
  python3 validation/viking/viking_layer_polarity.py --selftest
"""
import argparse
import numpy as np


def bandpass(x, dt, f1, f2, tap=4.0):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, dt)
    w = np.zeros_like(f)
    w[(f >= f1) & (f <= f2)] = 1.0
    lo = (f > f1 - tap) & (f < f1)
    hi = (f > f2) & (f < f2 + tap)
    w[lo] = 0.5 - 0.5 * np.cos(np.pi * (f[lo] - (f1 - tap)) / tap)
    w[hi] = 0.5 + 0.5 * np.cos(np.pi * (f[hi] - f2) / tap)
    return np.fft.irfft(X * w, n)


def fshift(x, sh, npad):
    """Delay x by sh samples (fractional) on a zero-padded support, crop back."""
    xp = np.pad(x, (npad, npad))
    N = len(xp)
    X = np.fft.rfft(xp)
    f = np.fft.rfftfreq(N)
    y = np.fft.irfft(X * np.exp(-2j * np.pi * f * sh), N)
    return y[npad:npad + len(x)]


def fit_two_term(wc, wl, dt, s_max=30., tau_min=15., tau_max=90., step=0.5):
    """Grid over s and tau (ms). Returns single-term NCC, and for each sign of b
    the NCC profile over tau (best over s) and the overall best (tau, s, b/a)."""
    step_n = int(round(tau_min / step))
    s_grid = np.arange(-s_max, s_max + 1e-9, step)
    tau_grid = tau_min + step * np.arange(int(round((tau_max - tau_min) / step)) + 1)
    sh_grid = np.arange(-s_max, s_max + tau_max + 1e-9, step)
    npad = int(np.ceil((s_max + tau_max) / (dt * 1e3))) + 8
    C = np.array([fshift(wc, sh / (dt * 1e3), npad) for sh in sh_grid])
    y = wl
    yy = y @ y
    P = C @ C.T
    q = C @ y
    ns, nt = len(s_grid), len(tau_grid)

    # single term: a * wc(t - s), a > 0
    c11 = np.diag(P)[:ns]
    ncc1 = np.where(q[:ns] > 0, q[:ns] / np.sqrt(c11 * yy + 1e-30), 0.)
    i1 = int(np.argmax(ncc1))
    single = dict(ncc=float(ncc1[i1]), s=float(s_grid[i1]))

    I, J = np.meshgrid(np.arange(ns), np.arange(nt), indexing='ij')
    M = I + step_n + J
    g11, g22, g12 = P[I, I], P[M, M], P[I, M]
    r1, r2 = q[I], q[M]
    det = g11 * g22 - g12 ** 2 + 1e-30
    a = (g22 * r1 - g12 * r2) / det
    b = (g11 * r2 - g12 * r1) / det
    ncc2 = np.sqrt(np.clip((a * r1 + b * r2) / yy, 0, None))
    out = dict(single=single, tau=tau_grid)
    for name, ok in (('same', b > 0), ('opposite', b < 0)):
        valid = ok & (a > 0) & (np.abs(b) <= 2.5 * a)   # peg-legs: |b/a| <= 2
        # where the unconstrained optimum has the wrong sign, the constrained
        # optimum sits on b = 0, i.e. the single-term fit at that s
        prof = np.where(valid, ncc2, ncc1[I])
        k = np.unravel_index(int(np.argmax(prof)), prof.shape)
        out[name] = dict(ncc=float(prof[k]), s=float(s_grid[k[0]]),
                         tau=float(tau_grid[k[1]]),
                         ratio=float(b[k] / a[k]) if valid[k] else 0.,
                         profile=prof.max(axis=0))
    return out


def decon(wc, wl, dt, f1, f2, eps=0.05):
    n = len(wl)
    N = 8 * n
    Wc, Wl = np.fft.rfft(wc, N), np.fft.rfft(wl, N)
    f = np.fft.rfftfreq(N, dt)
    band = np.ones_like(f)
    band = bandpass(np.fft.irfft(band, N), dt, f1, f2)   # taper shape only
    band = np.abs(np.fft.rfft(band))
    band /= band.max() + 1e-30
    H = Wl * np.conj(Wc) / (np.abs(Wc) ** 2 + eps * np.max(np.abs(Wc) ** 2)) * band
    h = np.roll(np.fft.irfft(H, N), N // 2)
    t = (np.arange(N) - N // 2) * dt
    return t, h / (np.abs(h).max() + 1e-30)


def report(tag, r):
    s, p, o = r['single'], r['same'], r['opposite']
    print(f"{tag}")
    print(f"  single term          NCC {s['ncc']:.3f}  (s {s['s']:+.1f} ms)")
    print(f"  + same polarity      NCC {p['ncc']:.3f}  tau {p['tau']:5.1f} ms  "
          f"b/a {p['ratio']:+.2f}  (s {p['s']:+.1f} ms)")
    print(f"  + opposite polarity  NCC {o['ncc']:.3f}  tau {o['tau']:5.1f} ms  "
          f"b/a {o['ratio']:+.2f}  (s {o['s']:+.1f} ms)")
    gp, go = p['ncc'] - s['ncc'], o['ncc'] - s['ncc']
    win = 'same' if gp > go else 'opposite'
    if max(gp, go) < 0.005:
        win = 'undetermined (no second term needed)'
    print(f"  gain over single term: same {gp:+.3f}, opposite {go:+.3f} -> {win}")
    return win


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs = [('primary', args.clear, args.layer, 1.0)]
    if args.clear_mult and args.layer_mult:
        pairs.append(('multiple', args.clear_mult, args.layer_mult, 2.0))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tag, fc, fl, expect in pairs:
        wc = np.load(fc).astype(float)
        wl = np.load(fl).astype(float)
        if len(wc) != len(wl):
            raise SystemExit(f"{tag}: lengths differ ({len(wc)} vs {len(wl)})")
        dt = args.dt if args.dt else args.wl_ms * 1e-3 / len(wc)
        wcb, wlb = bandpass(wc, dt, args.f1, args.f2), bandpass(wl, dt, args.f1, args.f2)
        r = fit_two_term(wcb, wlb, dt, args.s_max, args.tau_min, args.tau_max, args.step)
        print(f"\n[{tag}] n {len(wc)}, dt {dt*1e3:.2f} ms, band {args.f1:.0f}-{args.f2:.0f} Hz"
              f"  (expected |b/a| relative to the primary: x{expect:.0f})")
        report(f"  {tag}", r)
        ls = '-' if tag == 'primary' else '--'
        ax[0].plot(r['tau'], r['same']['profile'], 'r' + ls, label=f'{tag}, same')
        ax[0].plot(r['tau'], r['opposite']['profile'], 'b' + ls, label=f'{tag}, opposite')
        ax[0].axhline(r['single']['ncc'], color='0.5', ls=ls, lw=0.8)
        t, h = decon(wc, wl, dt, args.f1, args.f2, args.eps)
        m = (t > -0.05) & (t < 0.12)
        ax[1].plot(t[m] * 1e3, h[m], 'k' + ls, label=tag)
    ax[0].set_xlabel('tau (ms)'); ax[0].set_ylabel('best NCC over s')
    ax[0].set_title('two-term fit, sign of b constrained (grey: single term)', fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].axhline(0, color='0.7', lw=0.6)
    ax[1].set_xlabel('lag (ms)'); ax[1].set_ylabel('normalised')
    ax[1].set_title('layer deconvolved by clear (water level %.2f)' % args.eps, fontsize=9)
    ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(args.out + '.png', dpi=135)
    print(f"\nfigure: {args.out}.png")


def selftest():
    rng = np.random.default_rng(1)
    dt, n = 0.004, 50
    t = np.arange(400) * dt

    def pulse(t0, fp=35.):
        u = np.pi * fp * (t - t0)
        return -2 * u * np.exp(-u ** 2)            # peak then trough

    def source(t0):
        s = pulse(t0) - 0.9 * pulse(t0 + 0.010) + 0.5 * pulse(t0 + 0.134)
        return bandpass(s, dt, 6, 90)

    t0 = 0.030
    wc = source(t0)[:n]
    cases = [('same, tau 34 ms', [0.034], +0.5),
             ('opposite, tau 22 ms', [0.022], -0.5),
             ('same, tau smeared 32-37 ms (one bin)', list(np.linspace(0.032, 0.037, 11)), +0.5),
             ('same, tau smeared 25-45 ms', list(np.linspace(0.025, 0.045, 21)), +0.5),
             ('opposite, tau smeared 25-45 ms', list(np.linspace(0.025, 0.045, 21)), -0.5),
             ('no layer', [0.030], 0.0)]
    for name, taus, b in cases:
        wl = np.mean([source(t0 + 0.005) + b * source(t0 + 0.005 + tt) for tt in taus], axis=0)[:n]
        wl = wl + 0.10 * np.std(wl) * rng.standard_normal(n)
        r = fit_two_term(bandpass(wc, dt, 8, 70), bandpass(wl, dt, 8, 70), dt)
        win = report(f"truth: {name}, b/a {b:+.1f}", r)
        print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clear'); ap.add_argument('--layer')
    ap.add_argument('--clear_mult'); ap.add_argument('--layer_mult')
    ap.add_argument('--dt', type=float, default=None, help='s; default wl_ms / n')
    ap.add_argument('--wl_ms', type=float, default=200.)
    ap.add_argument('--f1', type=float, default=8.); ap.add_argument('--f2', type=float, default=70.)
    ap.add_argument('--s_max', type=float, default=30.)
    ap.add_argument('--tau_min', type=float, default=15.); ap.add_argument('--tau_max', type=float, default=90.)
    ap.add_argument('--step', type=float, default=0.5)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--out', default='polarity')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.clear and a.layer):
            ap.error('--clear and --layer are required')
        run(a)
