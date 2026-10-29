#!/usr/bin/env python3
"""The Viking numbers of the paper against the recorded signature: Table 3,
Sects. 2.2, 3.3 and 3.4.  Reads the outputs of runs.sh (RUNS) and ncc.sh (OUT).

    python validation/real/ncc_summary.py RUNS OUT
"""
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import mannwhitneyu

R, O = sys.argv[1], sys.argv[2]


def rows(path):
    """(name, extracted, minimum, zero) of a viking_phase_test output."""
    out = []
    for l in open(path):
        m = re.match(r"^(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+\d+/\d+", l)
        if m:
            out.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return out


def ms(v):
    v = np.asarray(v, float)
    return "%.3f ± %.3f" % (v.mean(), v.std(ddof=1)) if v.size > 1 else "%.3f" % v.mean()


def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean([np.sign(y - x) for x in a for y in b]))


def coh(fam):
    return np.mean([float(np.load(p)["coherence"]) for p in glob.glob(os.path.join(R, "viking", fam + "_*_sel.npz"))])


F = {f: rows(os.path.join(O, "families", f + ".txt")) for f in "ABCDEFG"}
v = {f: [x[1] for x in F[f]] for f in F}
print("# NCC against the recorded Viking signature, final recipe\n")
print("Runs `%s`; accuracy at +-30 ms, the phase comparison at +-60 ms.\n" % R)
print("## Table 3 (fold and span controls, 8-65 Hz)\n")
print("| selection | span (km) | n | NCC | coherence |\n|---|---|---|---|---|")
for f, lab, L in (("A", "contiguous", 2.5), ("D", "every 5th shot", 12.4), ("C", "every 10th shot", 24.8),
                  ("B", "fold 200", 5.0), ("E", "fold 500", 12.5), ("F", "fold 1001", 25.0)):
    print("| %s | %.1f | %d | %s | %.3f |" % (lab, L, len(v[f]), ms(v[f]), coh(f)))
print("\n## Abstract, Sect. 3.3, Fig. 3\n")
print("- whole-line combs (C), mean NCC 8-65 Hz: %s" % ms(v["C"]))
p60 = rows(os.path.join(O, "phase_C_60ms.txt"))
d = np.array([x[2] - x[1] for x in p60])
print("- minimum-phase equivalent minus extracted, C combs, +-60 ms: %+.3f ± %.3f" % (d.mean(), d.std(ddof=1)))
b = {f: [x[1] for x in rows(os.path.join(O, "families", f + "_8-18Hz.txt"))] for f in "ABCDEFG"}
allb = [x for f in b for x in b[f]]
print("- 8-18 Hz agreement over every tested selection: %.2f-%.2f (family means: %s)" % (
    min(allb), max(allb), ", ".join("%s %.2f" % (f, np.mean(b[f])) for f in b if b[f])))
tl = open(os.path.join(O, "thin_layer_C.txt")).read()
print("- single delayed copy (C combs), `thin_layer_C.txt`, last lines:")
print("\n```\n" + "\n".join(tl.strip().splitlines()[-8:]) + "\n```\n")
print("## Sect. 3.4\n")
a = dict((x[0], x[1]) for x in F["A"]); bb = dict((x[0], x[1]) for x in F["B"])
print("- 100-shot contiguous blocks (A), range: %.3f-%.3f" % (min(v["A"]), max(v["A"])))
k = [n for n in bb if n.endswith("blk2")]
print("- 200-shot block, shots 400-599 (B blk2): %s; best B block: %s %.3f" % (
    "%.3f" % bb[k[0]] if k else "-", max(bb, key=bb.get), max(bb.values())))
u1 = mannwhitneyu(v["C"], v["A"], alternative="greater"); u2 = mannwhitneyu(v["C"], v["D"], alternative="greater")
print("- A (2.5 km) against C (24.8 km), one-sided Mann-Whitney and Cliff's delta: p = %.2g, delta = %.2f" % (u1.pvalue, cliff(v["A"], v["C"])))
print("- D (12.4 km) against C: p = %.2g, delta = %.2f" % (u2.pvalue, cliff(v["D"], v["C"])))
D = sorted(F["D"], key=lambda x: int(re.search(r"comb(\d+)", x[0]).group(1)))
print("- D combs, NCC: west half (combs 0-4) %s, east half (combs 5-9) %s" % (ms([x[1] for x in D[:5]]), ms([x[1] for x in D[5:]])))
print("- E blocks (halves, 500 shots): %s" % ", ".join("%s %.3f" % (x[0], x[1]) for x in F["E"]))
print("- G coherence-selected: %s" % ", ".join("%s %.3f" % (x[0], x[1]) for x in F["G"]))
for f in ("A", "B"):
    t = open(os.path.join(O, "thin_layer_%s.txt" % f)).read().strip().splitlines()
    print("- single-reflector fit, family %s: `thin_layer_%s.txt`" % (f, f))
print("\n## Sect. 2.2, channel count (C combs)\n")
print("| channels | NCC |\n|---|---|")
print("| 1 | %s |" % ms(v["C"]))
for nc in (4, 8, 16):
    print("| %d | %s |" % (nc, ms([x[1] for x in rows(os.path.join(O, "nch", "C_nch%d.txt" % nc))])))
print("\n## Sect. 3.4, tracking perturbations\n")
print("| perturbation | A blocks: NCC (change) | C combs: NCC (change) |\n|---|---|---|")
for p in sorted(glob.glob(os.path.join(O, "pert", "*_A.txt"))):
    n = os.path.basename(p)[:-6]
    va = [x[1] for x in rows(p)]; vc = [x[1] for x in rows(p.replace("_A.txt", "_C.txt"))]
    print("| %s | %.3f (%+.3f) | %.3f (%+.3f) |" % (n, np.mean(va), np.mean(va) - np.mean(v["A"]), np.mean(vc), np.mean(vc) - np.mean(v["C"])))
print("\n## Figures\n")
print("Figs. 3 and 4: `%s/figure/figures/`; their printed numbers in `figure/figures.log`." % O)
