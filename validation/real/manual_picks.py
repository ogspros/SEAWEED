#!/usr/bin/env python3
"""Manual seafloor picks per shot of the cube, for the comparison of the
tracking with the manual picks (tables.py).

The cube was written by run_real.load_segy: unique FieldRecord values in order,
the first --nshot of them, each shot's traces sorted by |offset|, incomplete
shots skipped.  The same selection is rebuilt from the SEG-Y headers, and the
near trace (smallest |offset|) of every kept shot gives its CDP.  The manual
pick at a shot is the pick (two columns, CDP and TWT in ms) interpolated at that
CDP: a two-way time at the near offset, not converted.  Zero times are missing
picks and are dropped; runs of one value longer than 100 CDP are marked
`constant` (on SISOC-MEPA they are excluded from the comparison).

    DATA=DIR python validation/real/manual_picks.py OUT

writes OUT/<line>_manual.npz for every line whose pick file is present.
"""
import os
import sys

import numpy as np
import segyio

D = os.environ.get("DATA", ".")
LINES = {
    "viking": ("seismic.sgy", "viking_pick/viking_cube.npy", "manual_picks/Viking_WB_Pick.txt"),
    "it065": ("SGY/IT065_Raw_geom_100_3150.sgy", "span_experiment/it065_cube.npy", "manual_picks/IT065_WBT_PIck.txt"),
    "mepa": ("SGY/MEPA_RAW_geom_100_2031_filt.sgy", "span_experiment/mepa_cube.npy", "manual_picks/MEPA_WBT.txt"),
    "ms29": ("SGY/MS29_SP300_1157.sgy", "span_experiment/ms29_cube.npy", "manual_picks/MS29_Wb_Pick.txt"),
}
CONST_RUN = 100


def near_cdp(path, nshot, C):
    """CDP and |offset| of the near trace of every shot the cube kept."""
    B = segyio.TraceField
    with segyio.open(path, "r", ignore_geometry=True) as f:
        ffid = f.attributes(B.FieldRecord)[:]
        offs = f.attributes(B.offset)[:]
        cdp = f.attributes(B.CDP)[:]
    kept = []
    for sid in np.unique(ffid)[:nshot]:
        idx = np.where(ffid == sid)[0]
        if idx.size != C:
            continue
        k = idx[np.argsort(np.abs(offs[idx]), kind="stable")[0]]
        kept.append((int(sid), int(cdp[k]), float(abs(offs[k]))))
    return np.array(kept)


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        sys.exit(out + " exists")
    os.makedirs(out)
    for line, (sgy, cube, pk) in LINES.items():
        sgy, cube, pk = (os.path.join(D, p) for p in (sgy, cube, pk))
        if not os.path.exists(pk):
            print("%s: no pick file %s, skipped" % (line, pk))
            continue
        meta = np.load(cube + ".meta.npz", allow_pickle=True)
        S, _, C = np.load(cube, mmap_mode="r").shape
        kept = near_cdp(sgy, int(meta["nshot"]), C)
        if len(kept) != S:
            sys.exit("%s: %d shots from the headers, %d in the cube" % (line, len(kept), S))
        ffid, cdp, off = kept[:, 0].astype(int), kept[:, 1].astype(int), kept[:, 2]
        p = np.loadtxt(pk)
        pc, pt = p[:, 0], p[:, 1]
        z = pt == 0
        pc, pt = pc[~z], pt[~z]
        order = np.argsort(pc); pc, pt = pc[order], pt[order]
        const = np.zeros(S, bool); a = 0
        for i in range(1, pc.size + 1):
            if i == pc.size or pt[i] != pt[a]:
                if i - a > CONST_RUN:
                    const |= (cdp >= pc[a]) & (cdp <= pc[i - 1])
                a = i
        cover = (cdp >= pc.min()) & (cdp <= pc.max())
        man = np.where(cover, np.interp(cdp, pc, pt), np.nan) * 1e-3
        np.savez(os.path.join(out, "%s_manual.npz" % line), ffid=ffid, cdp=cdp, near_offset=off, man_s=man,
                 constant=const, covered=cover, picks=os.path.basename(pk),
                 note="manual TWT at the near offset, per cube shot")
        print("%s: %d shots, %d inside the picked range, %d zero picks dropped, %d shots in constant runs"
              % (line, S, cover.sum(), z.sum(), const.sum()))


if __name__ == "__main__":
    main()
