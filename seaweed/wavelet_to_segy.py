#!/usr/bin/env python3
"""wavelet_to_segy -- write an extracted wavelet (.npy) as a SEG-Y rev1 file.

The file is assembled directly with numpy, so segyio is not needed.

The wavelet is a short mixed-phase series whose sample 0 is the origin of the
wavelet, not a recording time.  SEG-Y has no negative times, so the origin is
carried in two places: the trace is optionally padded before the wavelet
(--pad_ms), and the delay recording time in the trace header (bytes 109-110)
is set to minus that padding, so a package that honours delrt puts the origin
back at t = 0.  Packages that ignore delrt see the wavelet starting at
--pad_ms into the trace, which is stated in the textual header.

Examples
--------
# minimal: one trace, IBM float, 4 ms
python -m seaweed.wavelet_to_segy line_wavelet.npy line_wavelet.sgy --dt 0.004

# for a deterministic deconvolution flow: 20 identical traces, 500 ms long,
# wavelet origin 100 ms in, IEEE float
python -m seaweed.wavelet_to_segy line_wavelet.npy line_wav.sgy --dt 0.004 \\
        --ntraces 20 --length_ms 500 --pad_ms 100 --format ieee
"""

import argparse
import os
import struct
import sys

import numpy as np


# --------------------------------------------------------------------------
# IBM 32-bit floating point (SEG-Y format code 1)
# --------------------------------------------------------------------------
def ieee_to_ibm(x):
    """Convert IEEE float64/32 array to IBM 360 single-precision, as uint32."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.uint32)

    nonzero = (x != 0.0) & np.isfinite(x)
    if not np.any(nonzero):
        return out

    v = x[nonzero]
    sign = (v < 0).astype(np.uint32) << 31
    a = np.abs(v)

    # exponent base 16, biased by 64
    exp16 = np.floor(np.log2(a) / 4.0).astype(np.int64) + 1
    mant = a / np.power(16.0, exp16.astype(np.float64))

    # guard against rounding pushing the mantissa to 1.0 or below 1/16
    too_big = mant >= 1.0
    exp16[too_big] += 1
    mant[too_big] /= 16.0
    too_small = mant < (1.0 / 16.0)
    exp16[too_small] -= 1
    mant[too_small] *= 16.0

    mant24 = np.round(mant * 16777216.0).astype(np.int64)
    carry = mant24 >= 16777216
    mant24[carry] = mant24[carry] // 16
    exp16[carry] += 1

    biased = exp16 + 64
    if np.any(biased < 0) or np.any(biased > 127):
        raise ValueError("value out of IBM float range; rescale the wavelet "
                         "or write IEEE (--format ieee)")

    out[nonzero] = (sign
                    | (biased.astype(np.uint32) << 24)
                    | (mant24.astype(np.uint32) & 0x00FFFFFF))
    return out


def ibm_to_ieee(u):
    """Inverse of ieee_to_ibm, used by the self-test."""
    u = np.asarray(u, dtype=np.uint32)
    sign = np.where((u >> 31) & 1, -1.0, 1.0)
    exp16 = ((u >> 24) & 0x7F).astype(np.int64) - 64
    mant = (u & 0x00FFFFFF).astype(np.float64) / 16777216.0
    val = sign * mant * np.power(16.0, exp16.astype(np.float64))
    return np.where(u == 0, 0.0, val)


# --------------------------------------------------------------------------
# headers
# --------------------------------------------------------------------------
def textual_header(lines, ebcdic=True):
    """3200-byte textual header from a list of strings."""
    card = []
    for i in range(40):
        text = lines[i] if i < len(lines) else ""
        card.append(("C%02d %s" % (i + 1, text))[:80].ljust(80))
    s = "".join(card)
    if ebcdic:
        return s.encode("cp500")
    return s.encode("ascii")


def binary_header(ns, dt_us, fmt_code, ntraces):
    h = bytearray(400)

    def i32(off, val):          # off is the 1-based byte position in the file
        struct.pack_into(">i", h, off - 3201, int(val))

    def i16(off, val):
        struct.pack_into(">h", h, off - 3201, int(val))

    i32(3201, 1)                # job id
    i32(3205, 1)                # line number
    i32(3209, 1)                # reel number
    i16(3213, ntraces)          # traces per ensemble
    i16(3215, 0)                # auxiliary traces
    i16(3217, dt_us)            # sample interval, microseconds
    i16(3219, dt_us)            # sample interval of original field recording
    i16(3221, ns)               # samples per trace
    i16(3223, ns)               # samples per trace, original
    i16(3225, fmt_code)         # 1 = IBM float, 5 = IEEE float
    i16(3227, 1)                # ensemble fold
    i16(3229, 1)                # trace sorting: as recorded
    i16(3255, 1)                # measurement system: metres
    i16(3501, 0x0100)           # SEG-Y revision 1.0
    i16(3503, 1)                # fixed length trace flag
    i16(3505, 0)                # no extended textual headers
    return bytes(h)


def trace_header(ns, dt_us, delrt_ms, tracl, tracr):
    h = bytearray(240)

    def i32(off, val):          # off is the 1-based byte position in the trace
        struct.pack_into(">i", h, off - 1, int(val))

    def i16(off, val):
        struct.pack_into(">h", h, off - 1, int(val))

    i32(1, tracl)               # trace sequence number within line
    i32(5, tracr)               # trace sequence number within file
    i32(9, 1)                   # original field record number
    i32(13, tracl)              # trace number within field record
    i32(17, 1)                  # energy source point number
    i32(21, tracl)              # CDP ensemble number
    i32(25, 1)                  # trace number within ensemble
    i16(29, 1)                  # trace identification: seismic data
    i16(35, 1)                  # number of vertically summed traces
    i16(39, 0)                  # distance from source to receiver
    i16(71, 1)                  # coordinate scalar
    i16(89, 1)                  # coordinate units: length
    i16(109, delrt_ms)          # delay recording time, ms (may be negative)
    i16(115, ns)                # number of samples in this trace
    i16(117, dt_us)             # sample interval, microseconds
    i16(119, 1)                 # gain type: fixed
    return bytes(h)


# --------------------------------------------------------------------------
def load_wavelet(path, key=None):
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if key is None:
            keys = list(obj.keys())
            if len(keys) != 1:
                raise SystemExit("the .npz holds %d arrays %s; choose one with "
                                 "--key" % (len(keys), keys))
            key = keys[0]
        obj = obj[key]
    w = np.asarray(obj, dtype=np.float64).squeeze()
    if w.ndim != 1:
        raise SystemExit("expected a 1-D wavelet, got shape %s" % (w.shape,))
    return w


def main():
    ap = argparse.ArgumentParser(
        description="write an extracted wavelet (.npy/.npz) as SEG-Y rev1")
    ap.add_argument("npy", help="input wavelet, .npy or .npz")
    ap.add_argument("sgy", help="output SEG-Y file")
    ap.add_argument("--dt", type=float, required=True,
                    help="sample interval in SECONDS (e.g. 0.004)")
    ap.add_argument("--key", default=None,
                    help="array name, if the input is an .npz with several")
    ap.add_argument("--ntraces", type=int, default=1,
                    help="number of identical traces to write (default 1)")
    ap.add_argument("--length_ms", type=float, default=None,
                    help="total trace length in ms; the wavelet is zero-padded "
                         "at the end (default: exactly the wavelet)")
    ap.add_argument("--pad_ms", type=float, default=0.0,
                    help="zeros inserted BEFORE the wavelet, in ms; delrt is "
                         "set to minus this value")
    ap.add_argument("--format", choices=["ibm", "ieee"], default="ibm",
                    help="sample format: ibm (code 1, safest for legacy "
                         "packages) or ieee (code 5). Default ibm")
    ap.add_argument("--scale", type=float, default=None,
                    help="multiply the wavelet by this factor before writing")
    ap.add_argument("--normalise", action="store_true",
                    help="scale so that max|w| = 1 before writing")
    ap.add_argument("--ascii_header", action="store_true",
                    help="write the textual header in ASCII instead of EBCDIC")
    ap.add_argument("--note", default="",
                    help="one extra line for the textual header")
    args = ap.parse_args()

    w = load_wavelet(args.npy, args.key)
    if args.normalise:
        peak = np.max(np.abs(w))
        if peak > 0:
            w = w / peak
    if args.scale is not None:
        w = w * args.scale

    dt_us = int(round(args.dt * 1e6))
    if not (1 <= dt_us <= 65535):
        raise SystemExit("sample interval %d us does not fit the SEG-Y "
                         "16-bit field" % dt_us)

    npad = int(round(args.pad_ms / 1000.0 / args.dt))
    if npad < 0:
        raise SystemExit("--pad_ms must be positive")
    trace = np.concatenate([np.zeros(npad), w])

    if args.length_ms is not None:
        ns = int(round(args.length_ms / 1000.0 / args.dt))
        if ns < trace.size:
            raise SystemExit("--length_ms gives %d samples, shorter than the "
                             "padded wavelet (%d)" % (ns, trace.size))
        trace = np.concatenate([trace, np.zeros(ns - trace.size)])
    ns = trace.size
    if ns > 32767:
        raise SystemExit("%d samples exceeds the SEG-Y 16-bit sample count"
                         % ns)

    delrt = int(round(-args.pad_ms))
    fmt_code = 1 if args.format == "ibm" else 5

    if fmt_code == 1:
        payload = ieee_to_ibm(trace).astype(">u4").tobytes()
    else:
        payload = trace.astype(">f4").tobytes()

    lines = [
        "SOURCE WAVELET EXTRACTED FROM THE WATER-BOTTOM REFLECTION (SEAWEED)",
        "SOURCE FILE  : %s" % os.path.basename(args.npy),
        "SAMPLES      : %d      SAMPLE INTERVAL : %.4f S" % (ns, args.dt),
        "TRACES       : %d      FORMAT : %s" % (args.ntraces, args.format.upper()),
        "WAVELET ORIGIN AT SAMPLE %d, I.E. %.1f MS INTO THE TRACE." % (npad, args.pad_ms),
        "DELAY RECORDING TIME (BYTES 109-110) SET TO %d MS SO THAT THE" % delrt,
        "ORIGIN FALLS AT T=0 IN PACKAGES THAT HONOUR IT.",
        "AMPLITUDES ARE %s." % ("NORMALISED TO UNIT PEAK" if args.normalise
                                else "AS EXTRACTED"),
        "THE WAVELET IS MIXED PHASE AND INCLUDES THE SOURCE AND RECEIVER",
        "GHOSTS; IT IS NOT A ZERO-PHASE OR MINIMUM-PHASE EQUIVALENT.",
    ]
    if args.note:
        lines.append(args.note.upper()[:75])

    with open(args.sgy, "wb") as fh:
        fh.write(textual_header(lines, ebcdic=not args.ascii_header))
        fh.write(binary_header(ns, dt_us, fmt_code, args.ntraces))
        for i in range(args.ntraces):
            fh.write(trace_header(ns, dt_us, delrt, i + 1, i + 1))
            fh.write(payload)

    size = os.path.getsize(args.sgy)
    expect = 3600 + args.ntraces * (240 + ns * 4)
    print("wrote %s" % args.sgy)
    print("  %d trace(s), %d samples, dt = %d us, format code %d"
          % (args.ntraces, ns, dt_us, fmt_code))
    print("  wavelet origin at sample %d (%.1f ms), delrt = %d ms"
          % (npad, args.pad_ms, delrt))
    print("  peak amplitude %.6g, size %d bytes (expected %d)"
          % (np.max(np.abs(trace)), size, expect))
    if size != expect:
        print("  WARNING: size mismatch", file=sys.stderr)


if __name__ == "__main__":
    main()
