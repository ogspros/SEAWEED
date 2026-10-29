#!/usr/bin/env python3
"""The real-line numbers of the paper that do not need the Viking signature:
the comparison of the tracking with the manual picks, Table 4 (whole line,
every shot), the stage-1 anchors, the agreement between the cells of the span
experiment (Sect. 4.2), and the Viking families (coherence, agreement within a
family, the two halves of Sect. 3.4).  Prints markdown.

    python validation/real/tables.py LINES RUNS [MANUAL]

LINES, RUNS: the outputs of lines.sh and runs.sh; MANUAL: the output of
manual_picks.py, without which the manual comparison is skipped.
"""
import glob
import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO]
from seaweed.h3_track import CONVENTION_MS                              # noqa: E402
from seaweed.wavelet_extractor_v2 import arrival_time                  # noqa: E402

SPREAD = os.path.join(REPO, "validation", "archive", "wavelet_spread.py")
NAMES = {"viking": "Viking", "it065": "IT90-65", "mepa": "SISOC-MEPA", "ms29": "MS-29"}
LINES = {"viking": dict(dt=0.004, dxsp=25, band=["--band", "8", "65"], cells=()),
         "it065": dict(dt=0.002, dxsp=25, band=["--band", "8", "65"], cells=("S0100", "S0200", "S0400", "S0600")),
         "mepa": dict(dt=0.002, dxsp=25, band=["--band", "15", "100", "--bands", "15", "30", "45", "60", "75", "100"],
                      cells=("S0100", "S0200", "S0300")),
         "ms29": dict(dt=0.004, dxsp=200, band=["--band", "8", "45", "--bands", "8", "15", "22", "30", "38", "45"],
                      cells=("S0100", "S0200"))}
# T_dom of the manual comparison (ms): anchor_qc on the stage-1 tracking
TDOM = {"viking": 28.1, "it065": 24.4, "mepa": 18.5, "ms29": 51.7}
VBAND = ["--dt", "0.004", "--dxsp", "25", "--band", "8", "65"]


def grab(s, pat, cast=float, n=1):
    m = re.search(pat, s)
    return tuple(cast(m.group(i)) for i in range(1, n + 1)) if m else (None,) * n


def ws(files, args):
    return subprocess.run([sys.executable, SPREAD, *files, *args], capture_output=True, text=True).stdout


def spread(files, args):
    if len(files) < 2:
        return None
    m = re.search(r"mean pairwise NCC = ([\d.]+)", ws(files, args))
    return float(m.group(1)) if m else None


def wavelet_peak(w, dt):
    w = np.asarray(w, float); n = 8 * w.size
    f = np.fft.rfftfreq(n, dt)
    return float(f[np.argmax(np.abs(np.fft.rfft(w, n)))])


def vs_manual(line, track, manual):
    """d = tracked arrival at the near trace - manual pick (ms): median, IQR,
    share within T_dom/2 of the median, RMS of the 100-shot block medians (median
    removed) and blocks beyond T_dom/2."""
    mm = np.load(os.path.join(manual, "%s_manual.npz" % line), allow_pickle=True)
    man = np.asarray(mm["man_s"], float)
    if line == "mepa":
        man = np.where(mm["constant"], np.nan, man)
    z = np.load(track, allow_pickle=True)
    S = man.size; t0 = np.asarray(z["t0"], float)[:S]
    d = (arrival_time(t0, float(np.asarray(z["offs"])[0]), float(z["vw"]), 0) - man) * 1e3
    k = np.isfinite(d); dd = d[k]; idx = np.arange(S)[k]
    med = float(np.median(dd)); td = TDOM[line]
    bm = []
    for b0 in range(0, S, 100):
        m = (idx >= b0) & (idx < b0 + 100)
        if m.sum() >= 10:
            bm.append(float(np.median(dd[m])) - med)
    bm = np.array(bm)
    return dict(med=med, iqr=float(np.subtract(*np.percentile(dd, [75, 25]))),
                inside=float(np.mean(np.abs(dd - med) <= td / 2)), n_out=int(np.sum(np.abs(bm) > td / 2)),
                n_blk=int(bm.size), rms100=float(np.sqrt(np.mean(bm ** 2))))


def table4(d, L):
    P = LINES[L]; s = open(os.path.join(d, L, "full.log"), errors="replace").read(); g = grab; r = {}
    r["coh"], = g(s, r"\[4\] PRIMARY: \d+ windows \| coherence ([\d.]+)")
    r["ol"], r["res"] = g(s, r"origin-lead ([+-]?\d+) ms \| align\. residual ([\d.]+)", float, 2)
    r["D0"], r["D1"] = g(s, r"tracked seafloor: t0 \d+-\d+ ms -> D (\d+)-(\d+) m", float, 2)
    r["dep"], = g(s, r"departure ([+-][\d.]+)%")
    r["R1"], = g(s, r">>> R_wb ([\d.]+) \(spreading")
    r["R2"], = g(s, r"R_wb from 2nd/1st multiple: ([\d.]+)")
    r["rstat"], = g(s, r"white-reflectivity assumption\): r = ([+-][\d.]+)")
    r["rstat_hi"], = g(s, r"dominates: r = ([+-][\d.]+)")
    r["gain"], = g(s, r"\[2b\] amplitude trend between primary and first multiple: RMS ~ t\^([+-][\d.]+)")
    r["stacks"] = g(s, r"partial stacks of 50 shots: (\d+)/(\d+) beyond", int, 2)
    r["gap"], = g(s, r"largest gap between origins ([\d.]+) ms")
    r["bist"] = "BISTABLE" in s
    w = os.path.join(d, L, "full_wavelet.npy")
    r["fpk"] = wavelet_peak(np.load(w), P["dt"])
    r["dur"], = g(ws([w], ["--dt", str(P["dt"]), "--dxsp", str(P["dxsp"]), *P["band"]]),
                  r"effective wavelet duration \(5-95% of the energy\): (\d+) ms")
    r["disc"] = 200 * abs(r["R2"] - r["R1"]) / (r["R2"] + r["R1"]) if r["R1"] and r["R2"] else None
    sm = json.load(open(os.path.join(d, L, "summary.json")))
    dm = sm["domain"]
    r["dom"] = "%s (%s, %.0f ms; separation min %.0f ms, median %.0f ms; %d isolated shots below)" % (
        dm["flag"], dm.get("active"), dm["threshold_ms"], dm["separation_min_ms"], dm["separation_median_ms"],
        dm.get("isolated_shots_below", 0))
    r["h3"] = "applicable %s, fallback %s (%.1f %%)" % (sm["h3_applicable"], sm["h3_fallback"],
                                                       100 * sm["h3_unreliable_fraction"])
    za = np.load(os.path.join(d, L, "stage1_anchors.npz"), allow_pickle=True)
    r["anch"] = "%d of %d groups kept (%d shots per group)" % (int(np.sum(za["group_kept"])), za["group_kept"].size,
                                                               sm["group"])
    return r


def fmt(x, f):
    return "-" if x is None else f % x


def main():
    lines, runs = sys.argv[1], sys.argv[2]
    manual = sys.argv[3] if len(sys.argv) > 3 else None
    LL = tuple(LINES)
    hdr = "| | " + " | ".join(NAMES[L] for L in LL) + " |\n|---|" + "---|" * len(LL)
    print("# Real lines, final recipe\n")
    print("Lines `%s`, runs `%s`.\n" % (lines, runs))
    if manual:
        print("## Tracking against the manual picks\n")
        print("d = tracked arrival at the near trace minus the manual pick (ms).  Within T_dom/2: |d - median(d)| <= "
              "T_dom/2.  RMS100: RMS of the 100-shot block medians of d, median removed.  T_dom (ms): %s.\n"
              % ", ".join("%s %.1f" % (NAMES[L], TDOM[L]) for L in LL))
        M = {L: {k: vs_manual(L, os.path.join(lines, L, k + ".npz"), manual) for k in ("stage1_track", "track")}
             for L in LL if os.path.exists(os.path.join(manual, "%s_manual.npz" % L))}
        print("| | " + " | ".join(NAMES[L] for L in M) + " |\n|---|" + "---|" * len(M))
        for k, lab in (("stage1_track", "stage-1 tracking"), ("track", "onset tracking")):
            for q, f in (("median", lambda m: "%+.1f" % m["med"]), ("IQR", lambda m: "%.2f" % m["iqr"]),
                         ("within T_dom/2", lambda m: "%.3f" % m["inside"]),
                         ("RMS100", lambda m: "%.2f" % m["rms100"]),
                         ("100-shot blocks beyond T_dom/2", lambda m: "%d/%d" % (m["n_out"], m["n_blk"]))):
                print("| %s: %s | %s |" % (lab, q, " | ".join(f(M[L][k]) for L in M)))
    print("\n## The onset convention in arrival time at the near trace\n")
    print("The constant is added to the vertical t0 (%+.0f ms); at the near trace it is the median over the shots of "
          "arrival(t0) - arrival(t0 - constant), with the final track.\n" % -CONVENTION_MS["h2"])
    print("| " + " | ".join(NAMES[L] for L in LL) + " |\n|" + "---|" * len(LL))
    cv = []
    for L in LL:
        z = np.load(os.path.join(lines, L, "track.npz"), allow_pickle=True)
        t0 = np.asarray(z["t0"], float); x0 = float(np.asarray(z["offs"])[0]); vw = float(z["vw"]); c = -CONVENTION_MS["h2"] * 1e-3
        cv.append(float(np.median(arrival_time(t0, x0, vw) - arrival_time(t0 - c, x0, vw))) * 1e3)
    print("| " + " | ".join("%.2f ms" % x for x in cv) + " |")
    T = {L: table4(lines, L) for L in LL}
    print("\n## Table 4 (whole line, every shot)\n")
    print(hdr)
    for name, f in (("tracked water depth (m)", lambda r: "%.0f-%.0f" % (r["D0"], r["D1"])),
                    ("peak frequency (Hz)", lambda r: "%.0f" % r["fpk"]),
                    ("wavelet duration, 5-95 % of the energy (ms)", lambda r: fmt(r["dur"], "%.0f")),
                    ("depth discrepancy from the multiple", lambda r: fmt(r["dep"], "%+.1f %%")),
                    ("R_wb, first multiple / primary", lambda r: fmt(r["R1"], "%.3f")),
                    ("R_wb, second / first multiple", lambda r: fmt(r["R2"], "%.3f")),
                    ("relative discrepancy", lambda r: fmt(r["disc"], "%.0f %%")),
                    ("statistical spectrum, r (6-90 Hz)", lambda r: fmt(r["rstat"], "%.3f")),
                    ("statistical spectrum, r (20-90 Hz)", lambda r: fmt(r["rstat_hi"], "%.3f")),
                    ("stack coherence", lambda r: fmt(r["coh"], "%.3f")),
                    ("origin lead (ms)", lambda r: fmt(r["ol"], "%+.0f")),
                    ("alignment residual (ms)", lambda r: fmt(r["res"], "%.1f")),
                    ("partial stacks of 50 within 2 samples", lambda r: "%d/%d%s" % (
                        r["stacks"][1] - r["stacks"][0], r["stacks"][1], " BISTABLE" if r["bist"] else "")),
                    ("largest gap between partial-stack origins (ms)", lambda r: fmt(r["gap"], "%.0f")),
                    ("amplitude trend primary-multiple", lambda r: fmt(r["gain"], "t^%+.2f")),
                    ("stage-1 anchors", lambda r: r["anch"]),
                    ("onset tracking", lambda r: r["h3"]),
                    ("operating domain", lambda r: r["dom"])):
        print("| %s | %s |" % (name, " | ".join(f(T[L]) for L in LL)))
    print("\n## Sect. 4.2: agreement between cells (wavelet_spread.py)\n")
    print("| line | cell | cells | mean pairwise NCC, broadband | by band |\n|---|---|---|---|---|")
    for L in LL:
        P = LINES[L]
        for c in P["cells"]:
            fs = sorted(glob.glob(os.path.join(runs, "archive", "%s_n100_*_%s_k*_wavelet.npy" % (L, c))))
            s = ws(fs, ["--dt", str(P["dt"]), "--dxsp", str(P["dxsp"]), *P["band"]]) if len(fs) > 1 else ""
            m = re.search(r"mean pairwise NCC = ([\d.]+) \+- ([\d.]+)", s)
            b = re.findall(r"^\s+(\d+-\d+)\s+([\d.]+)\s+[\d.]+\s*$", s, re.M)
            print("| %s | %s | %d | %s | %s |" % (NAMES[L], c, len(fs), "%s ± %s" % m.groups() if m else "-",
                                                ", ".join("%s %s" % x for x in b) or "-"))
    n = {L: len(glob.glob(os.path.join(runs, "archive", "%s_n100_*_wavelet.npy" % L))) for L in LL}
    print("\nExtractions of the span experiment: %s; %d in all." % (
        ", ".join("%s %d" % (NAMES[L], n[L]) for L in LL if n[L]), sum(n.values())))
    print("\n## Viking families (no signature)\n")
    print("| family | runs | coherence (mean) | mean pairwise agreement (8-65 Hz) |\n|---|---|---|---|")
    fams = [("A contiguous 100, 2.5 km", "viking/A_n100_L2.5km_blk*"), ("D every 5th, 12.4 km", "viking/D_n100_L12.4km_comb*"),
            ("C every 10th, 24.8 km", "viking/C_n100_L25km_comb*"), ("B contiguous 200, 5 km", "viking/B_n200_L5km_blk*"),
            ("E contiguous 500, 12.5 km", "viking/E_n500_L12.5km_blk*"), ("F whole line, 1001", "viking/F_n1001_L25km_full"),
            ("G coherence-selected", "viking/G_n*_auto"), ("C combs, 4 channels", "viking_nch4/C_n100_L25km_comb*"),
            ("C combs, 8 channels", "viking_nch8/C_n100_L25km_comb*"), ("C combs, 16 channels", "viking_nch16/C_n100_L25km_comb*")]
    for name, g in fams:
        wv = sorted(glob.glob(os.path.join(runs, g + "_wavelet.npy")))
        coh = [float(np.load(w.replace("_wavelet.npy", "_sel.npz"))["coherence"]) for w in wv]
        a = spread(wv, VBAND)
        print("| %s | %d | %.3f | %s |" % (name, len(wv), np.mean(coh) if coh else np.nan, "%.3f" % a if a else "-"))
    D = sorted(glob.glob(os.path.join(runs, "viking/D_n100_L12.4km_comb*_wavelet.npy")))
    west, east = D[:5], D[5:]
    cross = [spread([a, b], VBAND) for a, b in itertools.product(west, east)]
    nv = len(glob.glob(os.path.join(runs, "viking", "*_wavelet.npy")))
    print("\nViking selections of Table 4: %d." % nv)
    print("\nSect. 3.4, halves: D combs within the west half %.3f, within the east half %.3f, across halves %.3f "
          "(mean of %d pairs)" % (spread(west, VBAND), spread(east, VBAND), np.mean(cross), len(cross)))


if __name__ == "__main__":
    main()
