#!/usr/bin/env python3
"""pick_consensus -- seafloor picking by agreement between methods that fail in
opposite directions.

Why.  With the seafloor known, the two simpler pickers fail in opposite ways
on the synthetic matrix.  The autocorrelation, given one wide bracket of lags,
is early in 79 per cent of its failures, at about a quarter of the true
two-way time: it locks onto a short periodicity of the sub-bottom.  Semblance
initialised within +-200 ms of the nominal depth is late: in shallow water the
lower edge of that window sits on the seafloor and the first coherent event
below it wins, 88 ms late at 100 m.  Errors of opposite sign from unrelated
mechanisms are the case where a consensus is worth building.  The picker need
not be accurate, only pick the right event: the extraction is unaffected by
tracking errors from -100 to +50 ms.

The discriminant.  The seafloor is the only reflector that moves out at the
velocity of water, which the direct arrival measures independently.  A
sub-bottom reflector at the same two-way time has a flatter hyperbola: at
400 m of water, 660 m of offset and a sediment at 1800 m/s the two arrivals
differ by 44 ms, several wavelet periods.  Semblance along the water
hyperbola, with the velocity fixed, separates them.

Far offsets stretch the wavelet, by (t(x) - t0) / t0, and beyond about 0.3
the event is smeared; near offsets carry no moveout.  Both bounds follow from
t0, so they are recomputed for every candidate:

    x_max = vw * t0 * sqrt((1 + s_max)^2 - 1)

At 400 m of water that is 660 m, at 100 m it is 166 m, two or three channels:
in shallow water the test is weak, and the score records it.

How it works.

    1. candidates: the strongest lobes of the stacked autocorrelation, plus
       the peaks of a water-velocity semblance scan over the same times, so
       that a seafloor the autocorrelation misses can still be proposed
    2. score, per candidate:
         semblance along the water hyperbola at t0        (the discriminant)
         semblance at 2 t0, where the first multiple must be (a sub-bottom
             reflector has no companion at exactly twice its time)
         the polarity of that multiple relative to the primary, which the free
             surface fixes
         the separation from the direct arrival, which must leave the window
             usable
    3. the winner is refined per shot by cross-correlation with the stack

Nothing here needs the nominal depth.

    python -m seaweed.pick_consensus --selftest
"""
import numpy as np

# CONVENTION OFFSET.  The picker returns the time at which the stacked event
# peaks along the water hyperbola, which is later than the onset convention
# the tracking is built on: against the true seafloor of 216 synthetic
# records (main grid and secondary axes), the median lateness is 9 ms.  It is
# subtracted here, once, as part of the picker's definition.  It is measured
# where the answer is known and carries nothing from any real line, so it is
# not a shift tuned per survey.
CONVENTION_MS = 9.0

AGC_MS = 300.0            # amplitude equalisation before any of the tests
S_MAX = 0.30              # fractional NMO stretch allowed
WIN_MS = 24.0             # semblance window, about one period
N_CAND = 6                # candidates taken from each proposer
MIN_CH = 3                # fewest channels a candidate may be scored on
SEP_MIN_MS = 50.0         # separation from the direct arrival at full weight


# ------------------------------------------------------------------ helpers
def _norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def agc(g, dt, win_ms=AGC_MS):
    """Running-RMS equalisation along time, per trace.

    Without it every test here is dominated by whatever is loudest.  On Viking
    the direct arrival and the shallow events are an order of magnitude above
    the water bottom, so the semblance scan peaked on short lags and the
    seafloor candidate never competed.  AGC does not change polarity or
    moveout, which are what the scoring actually reads.
    """
    g = np.asarray(g, float)
    w = max(3, int(round(win_ms * 1e-3 / dt)) | 1)
    k = np.ones(w) / w
    e = np.apply_along_axis(lambda v: np.convolve(v ** 2, k, mode="same"),
                            1, g)
    return g / (np.sqrt(e) + 1e-12 * np.sqrt(np.mean(g ** 2) + 1e-30))


# Apparent dips scanned.  A free dip is too permissive: its linear term can
# imitate a change of t0, and with +-0.35 the scan moved a pick 270 ms early
# as a dip of 14 degrees.  The range is kept to seafloor dips (+-0.20 is 11.5
# degrees), the multiple must fit at the same dip, which removes the
# trade-off, and a small dip is preferred where the evidence is equal.
SIN_DIP = np.arange(-0.20, 0.201, 0.025)


def hyperbola(t0, x, vw, order=1, sin_dip=0.0):
    """Two-way time of the order-th water-layer event at offset x.

    A dipping seafloor does not give a symmetric hyperbola in a common-shot
    gather.  For a plane reflector of apparent dip theta,

        t(x)^2 = t0^2 + (x/vw)^2 + 2 t0 (x/vw) sin(theta)

    and the linear term is not small: on a 2 degree synthetic seafloor, fitting
    a symmetric hyperbola put the apparent t0 100 ms below the true one.  The
    dip is unknown a priori, so it is scanned, and the apparent dip found is
    reported.
    """
    tau = x / vw
    return np.sqrt(np.maximum((order * t0) ** 2 + tau ** 2
                              + 2.0 * order * t0 * tau * sin_dip, 1e-12))


def usable_channels(t0, offs, vw, s_max=S_MAX):
    """Channels used to score a candidate at t0, and the stretch that cost.

    Semblance on one channel is 1 by construction, so a candidate is never
    scored on fewer than MIN_CH: in shallow water, where the stretch limit
    leaves one or two, the nearest MIN_CH are taken anyway and the stretch they
    reach is returned.  Paying in stretch and declaring it is better than
    accepting a degenerate semblance, which lets a wrong candidate win.
    """
    x_max = vw * t0 * np.sqrt((1.0 + s_max) ** 2 - 1.0)
    k = np.flatnonzero(np.abs(offs) <= x_max)
    if k.size < MIN_CH:
        k = np.argsort(np.abs(offs))[:min(MIN_CH, offs.size)]
        k = np.sort(k)
    x = np.abs(offs[k]).max()
    stretch = float(hyperbola(t0, x, vw) / t0 - 1.0)
    return k, stretch


def semblance_at(g, dt, offs, t0, vw, order=1, s_max=S_MAX, win_ms=WIN_MS,
                 sin_dip=0.0):
    """Semblance along the water hyperbola of the given order.

    g is (shots, samples, channels).  The velocity is FIXED, not scanned: this
    asks whether an event at t0 moves out like water, not what velocity fits
    it best.
    """
    S, T, C = g.shape
    ch, _st = usable_channels(t0, offs[:C], vw, s_max)
    if ch.size < 1:
        return 0.0, 0
    w = max(1, int(round(win_ms * 1e-3 / dt)) // 2)
    tt = hyperbola(t0, offs[ch], vw, order, sin_dip)
    k = np.round(tt / dt).astype(int)
    if k.min() - w < 0 or k.max() + w >= T:
        return 0.0, int(ch.size)
    idx = k[:, None] + np.arange(-w, w + 1)[None, :]        # (nch, 2w+1)
    A = g[:, :, ch]                                          # (S, T, nch)
    A = np.take_along_axis(
        A.transpose(0, 2, 1).reshape(S * ch.size, T),
        np.tile(idx, (S, 1)), axis=1).reshape(S, ch.size, -1)
    num = (A.sum(axis=1) ** 2).sum()
    den = ch.size * (A ** 2).sum()
    return (float(num / den) if den > 0 else 0.0), int(ch.size)


def stack_window(g, dt, offs, t0, vw, order=1, s_max=S_MAX, win_ms=WIN_MS,
                 sin_dip=0.0):
    """The moveout-corrected stack of the window at t0, for polarity tests."""
    S, T, C = g.shape
    ch, _st = usable_channels(t0, offs[:C], vw, s_max)
    w = max(1, int(round(win_ms * 1e-3 / dt)))
    tt = hyperbola(t0, offs[ch], vw, order, sin_dip)
    k = np.round(tt / dt).astype(int)
    if k.min() - w < 0 or k.max() + w >= T:
        return None
    out = np.zeros(2 * w + 1)
    for c, kc in zip(ch, k):
        out += g[:, kc - w:kc + w + 1, c].sum(axis=0)
    return out


def autocorr_candidates(g, dt, lo_s, hi_s, n=N_CAND):
    """Lags of the strongest lobes of the autocorrelation of the near channel,
    stacked over shots.  Several, not one: the point of the consensus is that
    the proposer does not decide."""
    x = g[:, :, 0].astype(float)
    x = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * x.shape[1])))
    F = np.fft.rfft(x, nfft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, nfft, axis=1)[:, :x.shape[1]].mean(axis=0)
    ac = _norm(ac)
    klo, khi = int(lo_s / dt), min(int(hi_s / dt), ac.size - 2)
    if khi <= klo + 2:
        return []
    seg = ac[klo:khi]
    pk = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])) + 1
    if pk.size == 0:
        return []
    pk = pk[np.argsort(seg[pk])[::-1]][:n]
    return sorted((klo + int(p)) * dt for p in pk)


def semblance_candidates(g, dt, offs, vw, lo_s, hi_s, n=N_CAND, step_ms=4.0):
    """Peaks of a water-velocity semblance scan.  This proposer can find a
    seafloor whose autocorrelation lobe is buried, which is the failure the
    autocorrelation picker cannot recover from on its own."""
    ts = np.arange(lo_s, hi_s, step_ms * 1e-3)
    if ts.size < 3:
        return []
    sv = np.array([max(semblance_at(g, dt, offs, t, vw, sin_dip=d)[0]
                       for d in SIN_DIP[::2]) for t in ts])
    pk = np.flatnonzero((sv[1:-1] > sv[:-2]) & (sv[1:-1] > sv[2:])) + 1
    # peaks too close to the direct arrival are dropped before the n are kept:
    # score_candidate weighs them down anyway, and on the real lines they
    # could fill most of the places the scan has to offer
    x0 = float(np.abs(offs[0]))
    pk = pk[[hyperbola(ts[p], x0, vw) - x0 / vw >= SEP_MIN_MS * 1e-3
             for p in pk]] if pk.size else pk
    if pk.size == 0:
        return []
    pk = pk[np.argsort(sv[pk])[::-1]][:n]
    return sorted(float(ts[p]) for p in pk)


def score_candidate(g, dt, offs, t0, vw, s_max=S_MAX):
    """Evidence for t0 being the seafloor, from quantities the proposers did
    not produce.  Returns the score and its parts, so a pick can be explained
    rather than just reported."""
    # the apparent dip is scanned, not assumed: the best one is part of the
    # answer and the semblance at it is the evidence
    sv = [(semblance_at(g, dt, offs, t0, vw, 1, s_max, sin_dip=d)[0], d)
          for d in SIN_DIP]
    sem, dip = max(sv)
    _s2, nch = semblance_at(g, dt, offs, t0, vw, 1, s_max, sin_dip=dip)
    # the multiple is scored at the dip the primary chose: a candidate that
    # needs one dip for the primary and another for its multiple is not a
    # seafloor, and this is the constraint that removes the degeneracy
    semm = semblance_at(g, dt, offs, t0, vw, 2, s_max, sin_dip=dip)[0]
    wp = stack_window(g, dt, offs, t0, vw, order=1, s_max=s_max, sin_dip=dip)
    wm = stack_window(g, dt, offs, t0, vw, order=2, s_max=s_max, sin_dip=dip)
    pol = 0.0
    if wp is not None and wm is not None:
        a, b = wp - wp.mean(), wm - wm.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        pol = float(a @ b / d) if d > 1e-30 else 0.0
    # the direct arrival must be far enough not to be inside the window
    x0 = float(np.abs(offs[0]))
    sep = hyperbola(t0, x0, vw) - x0 / vw
    sep_ok = float(np.clip(sep / (SEP_MIN_MS * 1e-3), 0.0, 1.0))
    # how much stretch the scoring channels actually reached: beyond s_max the
    # semblance is measured on a smeared wavelet and is worth less
    _ch, st = usable_channels(t0, np.asarray(offs, float)[:g.shape[2]], vw, s_max)
    conf = float(np.clip(s_max / max(st, 1e-6), 0.0, 1.0))
    # where the evidence is equal, the flatter explanation wins
    flat = 1.0 - 0.5 * abs(dip) / float(SIN_DIP.max())
    score = (sem * (1.0 + 0.5 * semm) * (1.0 + 0.5 * max(0.0, -pol))
             * sep_ok * conf * flat)
    return score, dict(t0=t0, semblance=sem, semblance_multiple=semm,
                       multiple_polarity=pol, sep_ms=sep * 1e3,
                       n_channels=nch, stretch=st, sin_dip=float(dip))


def pick_t0(g, dt, offs, vw, lo_s=0.10, hi_s=1.50, s_max=S_MAX, refine=True):
    """The seafloor time per shot, by consensus.

    Returns (t0 per shot, report).  The report holds every candidate with its
    score and parts, so a failure can be read afterwards instead of guessed.
    """
    offs = np.asarray(offs, float)
    g = agc(g, dt)
    hi_s = min(hi_s, (g.shape[1] * dt) / 2.2)
    cand = sorted(set(np.round(
        autocorr_candidates(g, dt, lo_s, hi_s)
        + semblance_candidates(g, dt, offs, vw, lo_s, hi_s), 4)))
    if not cand:
        raise RuntimeError("no candidate seafloor time")
    scored = [score_candidate(g, dt, offs, t, vw, s_max) for t in cand]
    best = int(np.argmax([s for s, _ in scored]))
    t0 = float(cand[best])
    rep = dict(t0=t0, candidates=[d for _, d in scored],
               scores=[s for s, _ in scored],
               agree_ms=float('nan'))
    # how far the runner-up sits: candidates that agree are the reassuring case
    if len(cand) > 1:
        order = np.argsort([s for s, _ in scored])[::-1]
        rep["agree_ms"] = float(abs(cand[order[0]] - cand[order[1]]) * 1e3)
    t0s = np.full(g.shape[0], t0 - CONVENTION_MS * 1e-3)
    if refine:
        t0s = refine_per_shot(g, dt, offs, t0s, vw)
    return t0s, rep


def _running_median(x, w):
    w = max(1, int(w) | 1)
    h = w // 2
    y = np.empty_like(x)
    for i in range(x.size):
        y[i] = np.median(x[max(0, i - h):min(x.size, i + h + 1)])
    return y


def pick_line(g, dt, offs, vw, group=60, tol_ms=60.0, lo_s=0.10, hi_s=1.50,
              s_max=S_MAX):
    """The seafloor along a whole line, group by group, with continuity.

    Each group is scored on its own, and nothing would stop it from taking a
    candidate its neighbours reject: on Viking that gave one group at 175 m and
    another at 480 m on a seafloor within 330-385 m.  So the first pass keeps
    every candidate and its score, and the repair below lets each group choose
    only among candidates near the smoothed curve of its neighbours; a group
    with none takes the smoothed value.

    Returns (t0 per shot, report per group).
    """
    S = g.shape[0]
    offs = np.asarray(offs, float)
    bounds = [(a, min(S, a + group)) for a in range(0, S, group)]
    centres = np.array([0.5 * (a + b) for a, b in bounds])
    cand, scores = [], []
    for a, b in bounds:
        gg = agc(np.asarray(g[a:b], float), dt)
        try:
            hi = min(hi_s, (gg.shape[1] * dt) / 2.2)
            cs = sorted(set(np.round(
                autocorr_candidates(gg, dt, lo_s, hi)
                + semblance_candidates(gg, dt, offs, vw, lo_s, hi), 4)))
            sc = [score_candidate(gg, dt, offs, t, vw, s_max)[0] for t in cs]
        except Exception:
            cs, sc = [], []
        cand.append(cs); scores.append(sc)
    first = np.array([cs[int(np.argmax(sc))] if cs else np.nan
                      for cs, sc in zip(cand, scores)])
    if not np.any(np.isfinite(first)):
        raise RuntimeError("no group produced a candidate")

    # Iterative repair.  The semblance peaks are broad, several candidates
    # score within a few per cent of each other and noise decides between them,
    # so neighbouring groups can pick 50 ms apart on a seafloor that moves by 5.
    # Each pass smooths the group curve with a running median and lets every
    # group re-choose only among the candidates near its own smoothed value; the
    # constraint is local, so a real slope is not flattened.
    ref = np.where(np.isfinite(first), first, np.nanmedian(first))
    for _ in range(3):
        sm = _running_median(ref, 5)
        new_ref = ref.copy()
        for j, (cs, sc) in enumerate(zip(cand, scores)):
            near = [(s_, t_) for s_, t_ in zip(sc, cs)
                    if abs(t_ - sm[j]) * 1e3 <= tol_ms]
            new_ref[j] = max(near)[1] if near else sm[j]
        ref = new_ref
    src = ["own" if abs(r - f) < 1e-9 else "repaired"
           for r, f in zip(ref, first)]

    # the group values are interpolated between group centres, not held
    # constant across each group, which would draw a staircase
    t0 = np.interp(np.arange(S), centres, ref) - CONVENTION_MS * 1e-3
    rep = [dict(first=a, last=b, t0=float(r), unconstrained=float(f),
                source=q, n_candidates=len(cs))
           for (a, b), r, f, q, cs in zip(bounds, ref, first, src, cand)]
    t0 = refine_per_shot(g, dt, offs, t0, vw)
    return t0, dict(reference=float(np.median(ref)), groups=rep)


def refine_per_shot(g, dt, offs, t0s, vw, win_ms=60.0):
    """Cross-correlation of each shot's near-channel window against the stack.

    It moves the pick by at most half the window, so it cannot undo a wrong
    candidate: the candidate has to be right first.
    """
    S, T, _ = g.shape
    w = int(round(win_ms * 1e-3 / dt))
    x0 = float(np.abs(offs[0]))
    k = np.round(hyperbola(t0s, x0, vw) / dt).astype(int)
    if k.min() - w < 0 or k.max() + w >= T:
        return t0s
    seg = np.stack([g[s, k[s] - w:k[s] + w + 1, 0] for s in range(S)])
    pilot = seg.mean(axis=0)
    out = t0s.copy()
    lim = w // 2
    for s in range(S):
        best, bl = -2.0, 0
        for L in range(-lim, lim + 1):
            a = seg[s]
            b = np.roll(pilot, L)
            a = a - a.mean(); b = b - b.mean()
            d = np.linalg.norm(a) * np.linalg.norm(b)
            v = float(a @ b / d) if d > 1e-30 else 0.0
            if v > best:
                best, bl = v, L
        out[s] = t0s[s] - bl * dt
    return out


# ------------------------------------------------------------------ self test
def _synthetic(t0=0.53, vw=1500.0, v_sed=1900.0, S=20, C=48, dx=25.0,
               x0=125.0, dt=0.002, T=1600, seed=0):
    """A shot gather with a water bottom, its multiple, and a sub-bottom
    reflector placed to be the trap: at a quarter of the seafloor time, which
    is where the autocorrelation picker went on the matrix."""
    rng = np.random.default_rng(seed)
    offs = x0 + dx * np.arange(C)
    t = np.arange(T) * dt
    ric = (1 - 2 * (np.pi * 35 * (t[:120] - 0.06)) ** 2) * np.exp(
        -(np.pi * 35 * (t[:120] - 0.06)) ** 2)
    g = np.zeros((S, T, C), np.float32)

    kpk = int(round(0.06 / dt))          # where the peak sits inside ric

    def put(time_s, amp, chan):
        k = int(round(time_s / dt)) - kpk
        if 0 <= k and k + ric.size < T:
            g[:, k:k + ric.size, chan] += amp * ric

    for c, x in enumerate(offs):
        put(hyperbola(t0, x, vw), 1.0, c)                  # seafloor
        put(hyperbola(t0, x, vw, 2), -0.45, c)             # first multiple
        for j in (1, 2, 3):                                # sub-bottom, flat
            put(np.sqrt((0.25 * t0 * j) ** 2 + (x / v_sed) ** 2), 0.55, c)
        put(x / vw, 0.8, c)                                # direct arrival
    g += 0.02 * rng.normal(size=g.shape).astype(np.float32)
    return g, dt, offs, vw, t0


def selftest():
    ok = True
    for t0 in (0.13, 0.29, 0.53, 1.33):
        T = int((2.6 * t0 + 0.8) / 0.002)
        g, dt, offs, vw, truth = _synthetic(t0=t0, T=T)
        ac = autocorr_candidates(g, dt, 0.10, min(1.50, T * dt / 2.2))
        t0s, rep = pick_t0(g, dt, offs, vw)
        err = (np.median(t0s) - truth) * 1e3
        near = min((abs(c - truth) * 1e3 for c in ac), default=float('inf'))
        d = [c for c in rep["candidates"] if abs(c["t0"] - rep["t0"]) < 1e-9][0]
        print("t0 %5.0f ms | autocorr best-of %2d, closest %6.1f ms | "
              "consensus %+7.1f ms | semblance %.3f, multiple %.3f, "
              "polarity %+.2f, %d ch, stretch %.2f, sin(dip) %+.2f"
              % (truth * 1e3, len(ac), near, err, d["semblance"],
                 d["semblance_multiple"], d["multiple_polarity"],
                 d["n_channels"], d["stretch"], d["sin_dip"]))
        ok = ok and abs(err) < 25.0
    print("self test", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
