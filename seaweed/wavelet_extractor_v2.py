# =============================================================================
#  Marine source-wavelet extraction from the water-bottom reflection
#
#  Events that travel only through the water column -- direct arrival,
#  water-bottom primary and its multiples -- carry the source signature
#  essentially intact, because water has very high Q; they are modified only
#  by geometrical spreading, the reflection coefficient and the ghosts.
#
#  The module returns the effective wavelet  w = S * G_s * G_r,  the signature
#  with its ghosts, which is what deconvolution of the recorded data needs.
#
#  Design decisions that are not obvious, each forced by a measured failure:
#
#   * No per-shot onset detector.  On clean data the pre-arrival background
#     is tiny, and a threshold crossing fires on the acausal precursor of the
#     zero-phase band-pass at the start of the search window.  Windows are
#     anchored to the physical arrival time instead.
#
#   * Window wider than the wavelet: lead + L_w + tail, so that it contains
#     the wavelet rather than locating it.
#
#   * Non-circular shifts: np.roll would wrap the tail onto the head.
#
#   * Origin decided on the stack, walking backwards from the envelope peak:
#     one decision where the SNR is highest, immune to the acausal precursor.
#
#   * Tracking by cross-correlation, not by onset: an STA/LTA with windows in
#     samples behaves differently at each sampling rate (+34 ms of bias at
#     f_max = 45 Hz).
#
#   * Multiples with imposed polarity: a water-layer multiple has sign
#     (-1)^order, set by the free-surface bounce.
# =============================================================================
import warnings

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly

VW_DEFAULT = 1500.


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def shift_pad(x, s):
    """Shift x by s samples, zero-filling (NOT circular). s > 0 delays."""
    out = np.zeros_like(x)
    if s == 0:
        return x.copy()
    if s > 0:
        if s < len(x):
            out[s:] = x[:-s]
    else:
        if -s < len(x):
            out[:s] = x[-s:]
    return out


def envelope(x):
    """Analytic envelope with padding: hilbert is FFT-based, so a strong onset
    transient wraps around circularly and corrupts the tail."""
    n = len(x)
    xp = np.concatenate([np.zeros(n), x, np.zeros(n)])
    return np.abs(hilbert(xp))[n:2 * n]


def bandpass(x, dt, band, axis=-1):
    nyq = 0.5 / dt
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.95)
    b, a = butter(4, [lo, hi], 'band')
    return filtfilt(b, a, x, axis=axis)


def arrival_time(t0, offset, vw=VW_DEFAULT, order=0):
    """order=-1 direct arrival, 0 primary, 1 first water-layer multiple, ...

    The direct arrival does not cross the seafloor, so its duration is that of
    the source alone, free of geological contributions -- useful as an
    INDEPENDENT length estimator.  Note that its ghost differs: the ray is
    nearly horizontal, so the ghost delay 2 z cos(theta)/V tends to zero.  Use
    it for DURATION, not for shape or phase.
    """
    if order < 0:
        return np.asarray(offset, float) / vw + 0.0 * np.asarray(t0, float)
    return np.sqrt(((order + 1) * t0) ** 2 + (offset / vw) ** 2)


# --------------------------------------------------------------------------- #
#  1. Water velocity from the direct arrival  (CROSS-CHECK, not primary)
# --------------------------------------------------------------------------- #
def estimate_vw_direct(gathers, dt, offsets, vw_guess=VW_DEFAULT, search_ms=60.,
                       thr_frac=0.3, n_shots=None, D_guess=None, snr_gate=4.0,
                       n_ch_max=None):
    """Water velocity from the direct arrival, by threshold picking.  A cross-check.

    The primary estimate of Vw comes from semblance on the water-bottom
    reflection, which sums energy along a hyperbola over many traces.  Picking
    the direct arrival depends on a per-trace threshold and overestimates on
    real data (1574 m/s on IT90-65 against 1495 from semblance), because the
    direct wave decays as 1/sqrt(r) and at far offsets the threshold latches
    onto pre-arrival noise.  Windows proportional to offset, iterating the
    window centre, and fitting with an intercept all made it worse.

    What makes it usable is an SNR gate: traces where the direct arrival does
    not emerge from the pre-arrival noise are dropped.  Without it the estimate
    is about 1575 m/s whatever the true velocity between 1430 and 1560; with it
    the error is a constant +3.5 m/s up to 40% noise.

    Returns (vw, x0, n_picks).  x0 is a diagnostic computed separately: it
    lumps the error of the nominal near offset with the picking bias
    (x0 = dx - V dt), and if it exceeds the channel spacing the near offset of
    the field report needs checking.
    """
    S, T, C = gathers.shape
    S = min(S, n_shots) if n_shots else S
    # Near channels only.  The direct wave decays as 1/sqrt(r), so at far
    # offsets it does not stand above the noise.  Note that the two velocity
    # estimates of this module measure different things: the direct arrival
    # travels through the surface layer, the water-bottom reflection averages
    # the whole column.  In a stratified column they legitimately differ (on
    # IT90-65 about 1445 against 1495-1510 m/s), and the reflection value is the
    # one for the moveout and for t0 -> depth.
    if n_ch_max is not None:
        C = max(2, min(C, int(n_ch_max)))
    off = np.asarray(offsets, float)[:C]

    def pick(v_ref, sep_min_s):
        ts, xs = [], []
        for s in range(S):
            for j in range(C):
                x = off[j]
                if D_guess is not None:
                    sep = (np.sqrt(x ** 2 + 4.0 * D_guess ** 2) - x) / v_ref
                    if sep < sep_min_s:
                        continue          # direct and primary merged
                    win = max(2, int(round(min(0.5 * sep, search_ms * 1e-3) / dt)))
                else:
                    win = max(2, int(round(search_ms * 1e-3 / dt)))
                c = int(round((x / v_ref) / dt))
                lo, hi = max(0, c - win), min(T, c + win)
                if hi - lo < 5:
                    continue
                seg = np.abs(gathers[s, lo:hi, j]); pk = seg.max()
                if pk < 1e-12:
                    continue
                # SNR gate: the peak is compared with the noise measured before the
                # window, and the trace is dropped if the direct arrival does not stand out
                n_pre = min(lo, max(20, win))
                if n_pre >= 20:
                    rms_pre = float(np.sqrt(np.mean(
                        gathers[s, lo - n_pre:lo, j] ** 2)))
                    if pk < snr_gate * rms_pre:
                        continue
                # first sustained crossing (about 6 ms), not an isolated one, which
                # pre-arrival noise triggers and biases early
                above = seg > thr_frac * pk
                run = max(2, int(round(0.006 / dt)))
                k = None
                for i2 in range(len(above) - run):
                    if above[i2] and above[i2:i2 + run].all():
                        k = i2; break
                if k is None:
                    continue
                ts.append((lo + k) * dt); xs.append(x)
        if len(ts) < 8:
            return None, 0.0, 0
        t = np.asarray(ts); x = np.asarray(xs)
        good = t > 3 * dt
        t, x = t[good], x[good]
        if len(t) < 8:
            return None, 0.0, 0
        # fit through the origin, with one outlier-rejection pass
        v = float(x @ t / (t @ t))
        res = np.abs(x - v * t)
        keep = res < 2.5 * np.median(res) + 1e-12
        if keep.sum() >= 8:
            t, x = t[keep], x[keep]
            v = float(x @ t / (t @ t))
        # intercept computed separately, as a diagnostic only
        x0 = 0.0
        if np.ptp(t) > 0.15:
            A = np.vstack([t, np.ones_like(t)]).T
            _, x0 = np.linalg.lstsq(A, x, rcond=None)[0]
        return v, float(x0), int(len(t))

    if D_guess is None:
        v, x0, n = pick(vw_guess, 0.0)
    else:
        # two passes: the first uses only offsets with a wide direct/primary
        # separation, the second all offsets, anchored to the first
        v1, _, n1 = pick(vw_guess, 0.030)
        v, x0, n = pick(v1 if v1 else vw_guess, 0.008)
        if v is None:
            v, x0, n = v1, 0.0, n1
    if v is None or not (1350. < v < 1650.):
        import warnings
        warnings.warn(f"estimate_vw_direct: implausible estimate "
                      f"({'None' if v is None else f'{v:.0f} m/s'}); falling "
                      f"back to vw_guess={vw_guess:.0f}. Prefer semblance.",
                      RuntimeWarning)
        return float(vw_guess), 0.0, 0
    return float(v), float(x0), n


def estimate_vw_semblance_direct(gathers, dt, offsets, v_range=(1400., 1600.),
                                 nv=81, n_shots=None, band=(6., 90.),
                                 win_ms=40., t_shift_ms=(-40., 40.), n_t=17,
                                 max_offset=None):
    """Water velocity from the direct arrival by linear semblance.

    The direct wave has linear moveout t = t_i + x/V, so the scan is over
    (V, t_i), a slant stack: the velocity that best flattens the direct arrival
    maximises coherence over all traces.  It avoids the per-trace threshold of
    picking, and degrades gracefully instead of returning a confident wrong
    number.

    max_offset: the direct arrival weakens as 1/sqrt(r) and eventually merges
    with refracted energy; restricting the scan to the offsets where it is
    clearly visible sharpens the peak.

    Returns (vw, qc).
    """
    S, T, C = gathers.shape
    S = min(S, n_shots) if n_shots else S
    off = np.asarray(offsets, float)
    sel = np.arange(C) if max_offset is None else np.where(off <= max_offset)[0]
    if len(sel) < 5:
        return None, dict(error="too few offsets in range")
    g = bandpass(gathers[:S, :, :], dt, band, axis=1)

    hw = max(1, int(round(win_ms * 1e-3 / dt / 2)))
    Vs = np.linspace(v_range[0], v_range[1], nv)
    Ts = np.linspace(t_shift_ms[0], t_shift_ms[1], n_t) * 1e-3
    Sm = np.zeros((nv, n_t))
    for iv, V in enumerate(Vs):
        for it, ti in enumerate(Ts):
            num = den = 0.0; n = 0
            for s in range(S):
                k = np.round((off[sel] / V + ti) / dt).astype(int)
                ok = (k - hw >= 0) & (k + hw < T)
                if ok.sum() < 5:
                    continue
                kk = k[ok]; jj = sel[ok]
                for j in range(-hw, hw + 1):
                    a = g[s, kk + j, jj]
                    num += a.sum() ** 2; den += (a ** 2).sum() * ok.sum()
                n += 1
            Sm[iv, it] = num / (den + 1e-30) if n else 0.0
    iv, it = np.unravel_index(np.argmax(Sm), Sm.shape)
    # parabolic refinement on the velocity axis
    v_best = float(Vs[iv])
    if 0 < iv < nv - 1:
        y0, y1, y2 = Sm[iv - 1, it], Sm[iv, it], Sm[iv + 1, it]
        den2 = (y0 - 2 * y1 + y2)
        if abs(den2) > 1e-30:
            v_best += 0.5 * (y0 - y2) / den2 * (Vs[1] - Vs[0])
    qc = dict(peak=float(Sm[iv, it]), t_shift_ms=float(Ts[it] * 1e3),
              v_grid=Vs, semblance=Sm[:, it],
              n_offsets=int(len(sel)),
              # width of the peak at 90% of its maximum: narrow means well constrained
              v_width=float(np.ptp(Vs[Sm[:, it] >= 0.9 * Sm[iv, it]])
                            if np.any(Sm[:, it] >= 0.9 * Sm[iv, it]) else np.nan))
    return v_best, qc


def estimate_vw_direct_semblance(gathers, dt, offsets, v_range=(1400., 1600.),
                                 nv=81, n_shots=None, D_guess=None,
                                 band=(6., 90.), win_ms=24., x_min=None,
                                 x_max=None, n_ch_max=None):
    """Water velocity from the direct arrival by linear semblance.

    Scans trial velocities and measures the coherence of the amplitudes along
    the linear moveout t = x / v; the velocity that best flattens the direct
    arrival maximises semblance.  It uses the coherence of many traces rather
    than a per-trace threshold, which at far offsets latches onto pre-arrival
    noise (1574 m/s against 1495 from reflection semblance on IT90-65).

    Offsets where the direct arrival and the water-bottom primary have merged
    are excluded, using D_guess to compute their separation.
    """
    S, T, C = gathers.shape
    S = min(S, n_shots) if n_shots else S
    if n_ch_max is not None:
        C = max(2, min(C, int(n_ch_max)))
    off = np.asarray(offsets, float)[:C]
    g = bandpass(gathers[:S, :, :C], dt, band, axis=1)

    ok = np.ones(C, bool)
    if x_min is not None:
        ok &= off >= x_min
    if x_max is not None:
        ok &= off <= x_max
    if D_guess is not None:
        # keep only offsets where the primary is at least one window away
        sep = (np.sqrt(off ** 2 + 4.0 * D_guess ** 2) - off) / v_range[0]
        ok &= sep > win_ms * 1e-3
    idx = np.where(ok)[0]
    if len(idx) < 5:
        return float(np.mean(v_range)), 0.0, 0

    hw = max(1, int(round(win_ms * 1e-3 / dt / 2)))
    vs = np.linspace(v_range[0], v_range[1], nv)
    sem = np.zeros(nv)
    for iv, v in enumerate(vs):
        num = den = 0.0; cnt = 0
        for s in range(S):
            k = np.round(off[idx] / v / dt).astype(int)
            good = (k - hw >= 0) & (k + hw < T)
            if good.sum() < 5:
                continue
            kk = k[good]; jj = idx[good]
            # Neidell-Taner semblance over a short time window along the line
            for j2 in range(-hw, hw + 1):
                a = g[s, kk + j2, jj]
                num += a.sum() ** 2; den += (a ** 2).sum()
            cnt += good.sum()
        sem[iv] = num / (cnt * den + 1e-30) if cnt and den > 0 else 0.0

    ip = int(np.argmax(sem))
    # parabolic interpolation on the semblance peak for sub-sample precision
    if 0 < ip < nv - 1:
        y0, y1, y2 = sem[ip - 1], sem[ip], sem[ip + 1]
        d = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / d if abs(d) > 1e-30 else 0.0
        v_best = float(vs[ip] + shift * (vs[1] - vs[0]))
    else:
        v_best = float(vs[ip])
    return v_best, float(sem[ip]), int(len(idx))


# --------------------------------------------------------------------------- #
#  2. Seafloor two-way time from the autocorrelation  (no prior needed)
# --------------------------------------------------------------------------- #
def estimate_t0_autocorr(gathers, dt, offsets=None, n_ch=1, n_shots=None,
                         t0_range=(0.100, 2.000), band=(6., 90.), group=20,
                         wl_guess_ms=200., require_negative=True):
    """Seafloor t0 from the autocorrelation of the near trace, with no prior.

    Primary and multiples form a sequence periodic in t0, the sub-bottom
    geology does not, so the autocorrelation shows a lobe at t0 and one at
    2 t0.  Compared with semblance around a supplied depth:
      - no initial estimate is needed (with a depth wrong by 200 m on MS-29 the
        semblance latched onto the edge of its window);
      - it measures a time, independently of Vw;
      - polarity adds a constraint: the lobe at t0 is negative, because the
        free surface reverses the multiple, and the one at 2 t0 positive.

    group: shots per group.  Over a dipping seafloor a sum over the whole line
    blurs the peak, so the estimate is made per group and interpolated.

    Returns (t0 per shot [s], diagnostics).
    """
    S, T, C = gathers.shape
    S = min(S, n_shots) if n_shots else S
    nch = max(1, min(n_ch, C))
    g = bandpass(gathers[:S, :, :nch], dt, band, axis=1)

    lo = max(int(round(t0_range[0] / dt)), int(round(1.2 * wl_guess_ms * 1e-3 / dt)))
    hi = min(int(round(t0_range[1] / dt)), T // 2 - 1)
    if hi <= lo + 5:
        return None, dict(error="empty search range")

    cent, t0c, scores = [], [], []
    for s0 in range(0, S, group):
        s1 = min(S, s0 + group)
        acc = np.zeros(T)
        for s in range(s0, s1):
            for c in range(nch):
                tr = g[s, :, c].astype(float)
                n = np.linalg.norm(tr)
                if n < 1e-12:
                    continue
                a = np.correlate(tr, tr, 'full')[T - 1:]
                acc += a / (a[0] + 1e-30)
        if not np.any(acc):
            continue
        acc /= (s1 - s0) * nch
        seg = acc[lo:hi]
        # the lobe at t0 is negative: take the minimum, then check that the
        # positive lobe at 2 t0 is where predicted
        k = int(np.argmin(seg)) if require_negative else int(np.argmax(np.abs(seg)))
        t0k = (lo + k) * dt
        k2 = int(round(2 * t0k / dt))
        conf2 = float(acc[k2]) if k2 < len(acc) else float('nan')
        cent.append(0.5 * (s0 + s1 - 1)); t0c.append(t0k)
        scores.append(dict(lobe1=float(seg[k]), lobe2=conf2))
    if not t0c:
        return None, dict(error="no usable group")

    t0c = np.asarray(t0c); cent = np.asarray(cent, float)
    med = np.median(t0c); mad = np.median(np.abs(t0c - med)) + 1e-6
    keep = np.abs(t0c - med) < max(4 * mad, 0.050)
    if keep.sum() >= 1:
        cent, t0c = cent[keep], t0c[keep]
    t0 = (np.interp(np.arange(S), cent, t0c) if len(cent) > 1
          else np.full(S, float(t0c[0])))
    qc = dict(n_groups=int(len(t0c)), n_rejected=int((~keep).sum()),
              t0_median_s=float(med), t0_spread_ms=float(np.ptp(t0c) * 1e3),
              lobe1=float(np.median([s["lobe1"] for s in scores])),
              lobe2=float(np.nanmedian([s["lobe2"] for s in scores])),
              search_ms=(lo * dt * 1e3, hi * dt * 1e3))
    return t0, qc


def statistical_spectrum(gathers, dt, t0, offsets=None, n_ch=8, win_ms=700.,
                         lag_ms=400., below_ms=250., band=(6., 90.), taper=0.1,
                         nfft=4096):
    """Wavelet amplitude spectrum estimated statistically, as an independent check.

    Assuming white reflectivity, the autocorrelation of a trace window below
    the seafloor equals, up to a constant, that of the wavelet, so the square
    root of its power spectrum gives |W(f)|.  It shares no assumption with the
    extraction: it uses neither the water bottom, nor the tracking, nor the
    alignment.

    It determines only the modulus of the spectrum; the phase would have to be
    assumed, and an airgun signature is mixed-phase.  The window starts
    below_ms beneath the seafloor and stops before the water-layer multiple,
    which would violate the whiteness assumption.  Nothing is stacked, so the
    estimate keeps the incoherent noise that the extracted wavelet attenuates
    by sqrt(N): the comparison is meaningful only where the noise does not
    dominate.

    On synthetics with ideal white reflectivity it recovers the true amplitude
    spectrum with r = 0.997.
    """
    S, T, C = gathers.shape
    nch = max(1, min(n_ch, C))
    g = bandpass(gathers[:, :, :nch], dt, band, axis=1)
    W = int(round(win_ms * 1e-3 / dt)); Lg = int(round(lag_ms * 1e-3 / dt))
    below = int(round(below_ms * 1e-3 / dt))
    acc = np.zeros(Lg + 1); n = 0
    for s in range(S):
        k0 = int(round(float(np.atleast_1d(t0)[min(s, len(np.atleast_1d(t0)) - 1)])
                       / dt)) + below
        for c in range(nch):
            if k0 < 0 or k0 + W > T:
                continue
            tr = g[s, k0:k0 + W, c].astype(float)
            if np.std(tr) < 1e-12:
                continue
            tr = tr - tr.mean()
            m = int(taper * len(tr))
            if m > 2:
                win = np.hanning(2 * m)
                tr[:m] *= win[:m]; tr[-m:] *= win[m:]
            a = np.correlate(tr, tr, 'full')[len(tr) - 1:][:Lg + 1]
            if a[0] > 1e-30:
                acc += a / a[0]; n += 1
    if n == 0:
        return None, None, 0
    acc /= n
    # The autocorrelation is laid out with lag zero at index 0 and the
    # negative half wrapped to the tail.  A layout centred on the middle adds a
    # linear phase, the transform is no longer real, and its real part is
    # |W|^2 cos(phi) instead of |W|^2 (on ideal white reflectivity r = 0.42
    # against 0.997).
    full = np.zeros(nfft)
    full[:len(acc)] = acc
    full[-(len(acc) - 1):] = acc[1:][::-1]
    A = np.sqrt(np.maximum(np.fft.rfft(full).real, 0.0))
    return np.fft.rfftfreq(nfft, dt), A, n


# --------------------------------------------------------------------------- #
#  3. Seafloor tracking
# --------------------------------------------------------------------------- #
def track_seafloor_xcorr(gathers, dt, offsets, t0_init_s, vw=VW_DEFAULT,
                         n_ch=3, win_ms=200., search_ms=80., band=(6., 90.),
                         pilot_update=0.25, n_group=20, reject_frac=0.35,
                         t0_shift_ms=0.0, n_pass=3, return_qc=False):
    """Seafloor t0 (s) per shot, locked to the common body of the waveform
    rather than to an onset.

    2D window: each estimate uses n_ch near channels of n_group adjacent shots
    together, so that noisy channels and shots are filled in by their
    neighbours.  Windows correlating with the pilot below reject_frac are
    discarded.

    Multiple passes.  The first pass is causal: it starts at shot 0 with no
    pilot (initial criterion max|seg|) and propagates forward, so the first
    shots stay mis-locked until the pilot forms.  Later passes start from a
    formed pilot and restart each shot from its own previous value, so the
    early shots are corrected and drift does not accumulate.  Iteration stops
    on convergence (median change below half a sample) or after n_pass passes.

    A pass recovers only an initialisation error smaller than search_ms;
    beyond that the right event is outside the range explored.  Dense anchors
    are the remedy (run_real, --n_anchor), not interpolation between passes,
    which smooths away real structure.

    t0_shift_ms: shift applied to the result, negative to move earlier.  Do not
    combine it with an anchor-based drift correction applied afterwards: the
    drift is (anchor - tracked), so the shift would cancel; run_real applies
    it after the drift correction.  The cross-correlation locks onto the body
    of the waveform, so t0 falls some tens of ms after the first break (a
    constant -80 ms on clean synthetics).  On the primary that is harmless,
    since the origin is re-estimated on the stack; on the multiple it is
    amplified by (k+1) and can truncate the tail, and moving 10-20 ms earlier
    protects both.
    """
    S, T, C = gathers.shape
    nch = max(1, min(n_ch, C))
    W = int(round(win_ms * 1e-3 / dt)); R = int(round(search_ms * 1e-3 / dt))
    g = bandpass(gathers[:, :, :nch], dt, band, axis=1)

    def cut(s, c, centre):
        a = centre - W // 2; b = a + W
        if a < 0 or b > T:
            return None
        seg = g[s, a:b, c]
        n = np.linalg.norm(seg)
        return seg / n if n > 1e-12 else None

    t0 = np.full(S, float(t0_init_s))
    pilot = None; score = np.zeros(S); n_used = np.zeros(S, int)
    n_done = 0
    for ip in range(max(1, n_pass)):
        t0_prev = t0.copy()
        prev = None
        for s in range(S):
            # pass 1 propagates from the neighbour (causal, it forms the pilot); later
            # passes restart each shot from its own value
            t0_guess = (t0_prev[s] if ip > 0 else
                        (float(t0_init_s) if prev is None else prev))
            prof = np.full(2 * R + 1, -np.inf)
            for il, lag in enumerate(range(-R, R + 1)):
                tot, nseg = 0.0, 0
                for ss in range(max(0, s - n_group // 2),
                                min(S, s + n_group // 2 + 1)):
                    for c in range(nch):
                        centre = int(round(arrival_time(t0_guess, offsets[c], vw) / dt))
                        seg = cut(ss, c, centre + lag)
                        if seg is None:
                            continue
                        v = (float(seg @ pilot) if pilot is not None
                             else float(np.abs(seg).max()))
                        if pilot is not None and v < reject_frac:
                            continue
                        tot += v; nseg += 1
                if nseg:
                    prof[il] = tot / nseg
            ip_best = int(np.argmax(prof))
            best_tot = float(prof[ip_best]); best_lag = ip_best - R
            # sub-sample lag by parabolic interpolation on the correlation peak:
            # integer lags quantise the seafloor at dt (1.5 m for dt = 2 ms)
            sub = 0.0
            if 0 < ip_best < 2 * R and np.all(np.isfinite(prof[ip_best-1:ip_best+2])):
                y0, y1, y2 = prof[ip_best - 1], prof[ip_best], prof[ip_best + 1]
                den = y0 - 2 * y1 + y2
                if abs(den) > 1e-30:
                    sub = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            centre0 = int(round(arrival_time(t0_guess, offsets[0], vw) / dt))
            centre = centre0 + best_lag
            acc, nacc = np.zeros(W), 0
            for c in range(nch):
                cc = int(round(arrival_time(t0_guess, offsets[c], vw) / dt)) + best_lag
                seg = cut(s, c, cc)
                if seg is None:
                    continue
                if pilot is not None and float(seg @ pilot) < reject_frac:
                    continue
                acc += seg; nacc += 1
            n_used[s] = nacc; score[s] = best_tot if np.isfinite(best_tot) else 0.
            if nacc:
                acc /= (np.linalg.norm(acc) + 1e-12)
                pilot = acc.copy() if pilot is None else (
                    (1 - pilot_update) * pilot + pilot_update * acc)
                pilot /= (np.linalg.norm(pilot) + 1e-12)
            t_meas = (centre + sub) * dt
            t0[s] = np.sqrt(max(t_meas ** 2 - (offsets[0] / vw) ** 2, 1e-12))
            prev = t0[s]
        n_done = ip + 1
        if ip > 0 and np.median(np.abs(t0 - t0_prev)) < 0.5 * dt:
            break

    if t0_shift_ms:
        t0 = np.maximum(t0 + t0_shift_ms * 1e-3, 1e-3)
    if return_qc:
        return t0, dict(n_pass_done=n_done, score=score, n_used=n_used,
                        n_low=int((score < reject_frac).sum()))
    return t0


# --------------------------------------------------------------------------- #
#  4. Wavelet origin, decided on the stack
# --------------------------------------------------------------------------- #
def find_origin(stack, dt, noise_seg=None, frac_peak=0.08, floor_k=4.0,
                guard_ms=8.0):
    """Start index of the wavelet.  Walk BACKWARDS from the envelope peak until
    it drops below threshold: this is immune to the acausal precursor of
    zero-phase filters, which defeats any first-crossing criterion.

    threshold = max(frac_peak * peak, floor_k * RMS of the pre-arrival noise)
    guard_ms: back off slightly so the rising edge is not clipped.
    """
    env = envelope(np.asarray(stack, float))
    ipk = int(np.argmax(env))
    thr = frac_peak * env[ipk]
    if noise_seg is not None and len(noise_seg) > 5:
        thr = max(thr, floor_k * float(np.sqrt(np.mean(np.asarray(noise_seg) ** 2))))
    i = ipk
    while i > 0 and env[i] > thr:
        i -= 1
    return max(0, i - int(round(guard_ms * 1e-3 / dt))), ipk, float(thr)


# --------------------------------------------------------------------------- #
#  5. Extraction
# --------------------------------------------------------------------------- #
def extract_wavelet(gathers, dt, t0_seafloor, offsets, w_len_ms=200.,
                    weights='coherence',
                    refuse_padding=False, pad_warn_frac=0.02,
                    vw=VW_DEFAULT, n_ch=1, order=0, band=(6., 90.),
                    lead_ms=80., tail_ms=120., up=4, n_iter=4,
                    refine_ms=40., shot_start=0, n_shots=None,
                    origin_offset_ms=None, template=None, template_search_ms=80.,
                    template_expected_sign=None, ch_start=0,
                    spreading_alpha=None, align="single", align_group=20):
    """Extract the wavelet from the primary (order=0) or the multiple (order=1).

    t0_seafloor: (S,) vertical two-way time of the seafloor per shot, in SECONDS.
    n_ch: near channels per shot (1 = nearest only, most vertical rays; >1
          raises the fold but introduces residual moveout).
    lead/tail: margins around the predicted arrival.  The window spans
          lead + w_len + tail: it must CONTAIN the wavelet, not locate it.

    Returns (w, qc).  w has length int(w_len_ms/dt) and unit norm.
    """
    S_tot, T, C = gathers.shape
    s1 = min(S_tot, shot_start + n_shots) if n_shots else S_tot
    sl = slice(shot_start, s1)
    g = gathers[sl]; t0s = np.asarray(t0_seafloor)[sl]
    S = g.shape[0]
    nch = max(1, min(n_ch, C))

    Lw = int(round(w_len_ms * 1e-3 / dt))
    lead = int(round(lead_ms * 1e-3 / dt))
    tail = int(round(tail_ms * 1e-3 / dt))
    # The window is cut at the estimated origin, i0 = lead + delta, so the
    # returned wavelet fits only while delta <= tail.  On the synthetic matrix
    # delta has median +40 ms but p99 +614 ms, so no practical tail covers it: a
    # window that runs off the stack means the origin estimator has locked onto
    # a later event, and frac_padded reports it as that flag.
    WIN = lead + Lw + tail

    gb = bandpass(g, dt, band, axis=1)

    segs, noise, used = [], [], []
    for s in range(S):
        for c in range(ch_start, min(ch_start + nch, C)):
            t_arr = arrival_time(float(t0s[s]), offsets[c], vw, order)
            k0 = int(round(t_arr / dt)) - lead
            if k0 < 0 or k0 + WIN > T:
                continue
            seg = gb[s, k0:k0 + WIN, c]
            if np.std(seg) < 1e-12 or np.any(~np.isfinite(seg)):
                continue
            if spreading_alpha is not None:
                # Geometrical spreading.  Windows are not amplitude-normalised before
                # stacking, so near channels dominate the sum (near/far ratio 2.0 at
                # D = 400 m over 120 channels in 3D); correcting equalises them.  Whether
                # that helps is not obvious, since near channels also have the most vertical
                # rays, and on 200 synthetic lines it made no resolved difference (paired
                # +0.0005, p = 0.068).  Off by default; alpha = 1.0 for 3D field data, 0.5
                # for the 2D synthetics.
                r = np.sqrt(offsets[c] ** 2 + ((order + 1) * 2 *
                            float(t0s[s]) * vw / 2.) ** 2)
                seg = seg * r ** spreading_alpha
            segs.append(seg); noise.append(seg[:max(4, lead // 2)])
            used.append((s, c))
    if len(segs) < 3:
        return None, dict(error="too few valid windows", n_used=len(segs))
    segs = np.asarray(segs)

    # --- fine alignment: pilot cross-correlation, non-circular shifts ---
    # Residual-statics scheme: a pilot is built by stacking the windows, each
    # window is cross-correlated against it and shifted by the lag of the peak,
    # and the pilot is rebuilt from the aligned windows at each iteration.  The
    # window itself is subtracted from the pilot first: otherwise it correlates
    # partly with its own contribution and the shift is biased towards zero,
    # most where the SNR is poor.
    su = resample_poly(segs, up, 1, axis=1)
    Rf = int(round(refine_ms * 1e-3 / dt * up))
    align_tdom_ms = float("nan")
    if align == "single":
        ref = su.mean(0); lags = np.zeros(len(su), int)
        for it in range(n_iter):
            al = np.empty_like(su)
            for i, tr in enumerate(su):
                pil = ref - tr / len(su)
                pil = pil / (np.linalg.norm(pil) + 1e-12)
                trn = tr / (np.linalg.norm(tr) + 1e-12)
                best, blag = -2.0, 0
                for L in range(-Rf, Rf + 1):
                    v = float(shift_pad(trn, L) @ pil)
                    if v > best:
                        best, blag = v, L
                lags[i] = blag; al[i] = shift_pad(tr, blag)
            ref = al.mean(0)
    elif align == "two_stage":
        # Two stages, against the bistable state of the east half of Viking,
        # where single windows jump a cycle and drag the stack with them.
        # 1. bulk shift: the windows of align_group adjacent shots move together,
        #    their stack correlated with the line stack within +-refine_ms;
        # 2. refinement: each window within +-T_dom/2 of its group's shift, so no
        #    window can leave its cycle alone.  T_dom = 4 x the first zero of the
        #    autocorrelation of the bulk-aligned stack.
        grp = np.array([s for s, _ in used]) // max(1, int(align_group))
        bulk = np.zeros(len(su), int)
        cur = su
        for it in range(2):
            pil = cur.mean(0); pil = pil / (np.linalg.norm(pil) + 1e-12)
            for gid in np.unique(grp):
                m = np.flatnonzero(grp == gid)
                gs = su[m].mean(0); gs = gs / (np.linalg.norm(gs) + 1e-12)
                best, blag = -2.0, 0
                for L in range(-Rf, Rf + 1):
                    v = float(shift_pad(gs, L) @ pil)
                    if v > best:
                        best, blag = v, L
                bulk[m] = blag
            cur = np.array([shift_pad(tr, l) for tr, l in zip(su, bulk)])
        st0 = cur.mean(0); st0 = st0 - st0.mean()
        ac = np.correlate(st0, st0, "full")[len(st0) - 1:]
        iz = int(np.argmax(ac <= 0))
        align_tdom_ms = 4e3 * iz * dt / up if iz > 0 else 2 * refine_ms
        Rb = max(1, int(round(0.5 * align_tdom_ms * 1e-3 / dt * up)))
        ref = cur.mean(0); lags = bulk.copy()
        for it in range(n_iter):
            al = np.empty_like(su)
            for i, tr in enumerate(su):
                pil = ref - shift_pad(tr, lags[i]) / len(su)
                pil = pil / (np.linalg.norm(pil) + 1e-12)
                trn = tr / (np.linalg.norm(tr) + 1e-12)
                best, blag = -2.0, bulk[i]
                for L in range(bulk[i] - Rb, bulk[i] + Rb + 1):
                    v = float(shift_pad(trn, L) @ pil)
                    if v > best:
                        best, blag = v, L
                lags[i] = blag; al[i] = shift_pad(tr, blag)
            ref = al.mean(0)
    else:
        raise ValueError("align must be 'single' or 'two_stage', got %r" % (align,))

    coh = np.array([float(np.corrcoef(t, ref)[0, 1]) if np.std(t) > 1e-12 else 0.
                    for t in al])
    coh = np.nan_to_num(coh)
    if weights == 'coherence':
        # each window weighted by its correlation with the stack, the default.
        # It downweights the windows that differ from the mean, including those
        # whose contamination differs, whose diversity would otherwise average out
        wts = np.clip(coh, 0, None)
    elif weights == 'uniform':
        wts = np.ones_like(coh)
    elif weights == 'snr':
        # each window judged on its own amplitude against the noise before the
        # arrival, not on how much it resembles the others
        n0 = max(4, int(round(0.6 * lead)))
        amp = np.array([float(np.sqrt(np.mean(t ** 2))) for t in al])
        nse = np.array([float(np.sqrt(np.mean(t[:n0] ** 2))) for t in al])
        wts = np.clip(amp / (nse + 1e-12), 0, None)
        wts = wts / (wts.max() + 1e-12)
    else:
        raise ValueError("weights must be 'coherence', 'uniform' or 'snr', "
                         "got %r" % (weights,))
    st_up = (wts[:, None] * al).sum(0) / (wts.sum() + 1e-12)

    st = resample_poly(st_up, 1, up)
    # no second band-pass on the stack: the gather was band-passed before
    # windowing, and filtering again squares the response (-21 per cent at
    # 80 Hz, -50 per cent at 90 Hz); resample_poly already anti-aliases
    st = st - st.mean()

    # --- origin -------------------------------------------------------------
    if template is not None:
        # Matched filter: the primary wavelet locates the multiple.  A bias on t0
        # is amplified by (order+1) on the multiple, and its window is already
        # displaced, so the origin cannot be inherited.  The lag also gives a
        # measured Dt = t_mult - t_pri, an independent water depth.
        #
        # Polarity imposed, not measured: a water-layer multiple has sign
        # (-1)^order.  Maximising |c| instead lets the filter select deep sub-bottom
        # reflections, which share the primary's polarity and are not attenuated by
        # R_wb^2; with the constraint, a low score is itself the diagnosis.  The
        # score to maximise and the correlation value are kept in separate
        # variables.
        tpl = np.asarray(template, float)
        tpl = (tpl - tpl.mean()) / (np.linalg.norm(tpl - tpl.mean()) + 1e-12)
        Rs = int(round(template_search_ms * 1e-3 / dt))
        esign = template_expected_sign
        best_score = -np.inf
        best, i0 = 0.0, lead
        for k in range(max(0, lead - Rs), min(len(st) - len(tpl), lead + Rs) + 1):
            seg = st[k:k + len(tpl)]
            seg = seg - seg.mean()
            n_ = np.linalg.norm(seg)
            if n_ < 1e-12:
                continue
            c = float(seg @ tpl / n_)
            score = abs(c) if esign is None else c * esign
            if score > best_score:
                best_score, best, i0 = score, c, k
        # the sign is kept, not absorbed by abs(): a positive sign means the
        # filter locked onto something that is not a water-layer multiple
        ipk, thr = int(np.argmax(envelope(st))), float(best)
    elif origin_offset_ms is None:
        i0, ipk, thr = find_origin(st, dt,
                                   noise_seg=np.concatenate(noise) if noise else None)
    else:
        # origin inherited (offset relative to the nominal lead), not re-estimated:
        # the wavelet is the same and the separation is known from the moveout, so
        # re-estimating on a weaker event only adds noise
        i0 = lead + int(round(origin_offset_ms * 1e-3 / dt))
        i0 = int(np.clip(i0, 0, max(0, len(st) - 1)))
        ipk, thr = int(np.argmax(envelope(st))), float('nan')

    w = st[i0:i0 + Lw]
    n_padded = max(0, Lw - len(w))
    if n_padded:
        # The window ran past the end of the stack.  Those samples were never
        # measured, and zero-filling is not neutral: a correlation against a zero
        # tail is easier than against a wrong one.  The fraction is reported.
        frac = n_padded / float(Lw)
        msg = ("extract_wavelet: %d of %d samples (%.0f%%) of the wavelet were "
               "not measured and have been zero-filled; the origin fell %.0f ms "
               "past the nominal lead and the window ran off the end of the "
               "stack" % (n_padded, Lw, 100 * frac,
                          (i0 - lead) * dt * 1e3))
        if refuse_padding:
            raise ValueError(msg + ". Called with refuse_padding=True.")
        if frac > pad_warn_frac:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        w = np.pad(w, (0, n_padded))
    # absolute amplitude kept before normalising: the multiple-to-primary
    # ratio, from which R_wb is estimated, needs it
    amp_rms = float(np.sqrt(np.mean(w ** 2)))
    amp_peak = float(np.max(np.abs(w)))
    w_raw = w.copy()
    w = w / (np.linalg.norm(w) + 1e-12)

    qc = dict(n_windows=len(segs), n_shots=S, n_ch=nch, order=order,
              weights=weights,
              # how much of the returned wavelet was never measured
              n_padded=int(n_padded), frac_padded=float(n_padded) / max(Lw, 1),
              amp_rms=amp_rms, amp_peak=amp_peak, w_raw=w_raw,
              stack_coherence=float(coh.mean()),
              coh_per_window=coh,
              align_residual_ms_mean=float(np.abs(lags).mean() * dt / up * 1e3),
              align_residual_ms_max=float(np.abs(lags).max() * dt / up * 1e3),
              # scatter of the applied shifts, np.std(lags).  It is not the residual
              # misalignment after the shifts and is not the stacking low-pass
              # exp(-2 pi^2 f^2 sigma^2): a large scatter is what a successful alignment
              # of a line with variable statics looks like.  Measured on Viking, the
              # implied residual sigma is 0.2-1.4 ms, at most -1.5 dB at 65 Hz.
              align_scatter_ms=float(np.std(lags) * dt / up * 1e3),
              # the stack weighted mean of the applied shifts.  The stack moves with
              # them, so an event at stack sample i sits at window sample
              # i - lag_mean in the data: origin_minus_lead_ms does not include it,
              # and an absolute time read off the stack needs it subtracted
              lag_mean_ms=float((wts * lags).sum() / (wts.sum() + 1e-12) * dt / up * 1e3),
              origin_samp=int(i0), origin_ms=float(i0 * dt * 1e3),
              peak_samp=int(ipk), origin_thr=thr,
              lead_samp=lead, win_samp=WIN,
              spreading_alpha=spreading_alpha,
              # how far the origin found sits from the nominal lead: about 0 when the
              # physical anchor was right, large when t0 is wrong
              origin_minus_lead_ms=float((i0 - lead) * dt * 1e3),
              template_score=(float(abs(thr)) if template is not None else None),
              template_sign=(int(np.sign(thr)) if template is not None else None),
              stack_upsampled=ref, aligned_segments=al, windows_used=used,
              # the pre-arrival segment of every window, the noise of find_origin
              noise_segments=np.asarray(noise), align=align,
              lags_ms=lags * dt / up * 1e3,
              align_tdom_ms=align_tdom_ms)
    return w.astype(np.float32), qc
