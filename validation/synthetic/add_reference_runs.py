"""
add_reference_runs.py — the control the secondary axes never had.

THE GAP
-------
Every secondary axis is described as "one parameter varied on the reference
cell", and the text says the structural elements "sit slightly above the
reference cell".  There is no reference cell in the matrix: its depth,
D = 400 m, is not one of the grid depths, so the secondary families are compared
against nothing, or implicitly against grid cells at a different depth, which
mixes the axis with the depth.

ONE CONTROL PER SEED
--------------------
The secondary families do not share their seeds -- sec_n_ch_16_r000 and
sec_slope_deg_2p0_r000 are different geologies -- so a single set of twenty
controls would be paired with one family and unpaired with the rest.  Unpaired
is not good enough here: the scatter between realisations of one cell is the
largest effect in the matrix (0.406 against 0.944 on one grid cell) and would
swamp the few hundredths the secondary axes are meant to resolve.

So one control is written per distinct seed.  Every secondary run then has a
control with the same procedural geology and one parameter different.  At four
minutes a model and thirty shards that is about twenty minutes.

WHAT IS RESET, AND WHAT IS NOT
------------------------------
Only the parameters the configuration lists under `reference:`, and only inside
`params`.  Derived fields are left alone except `n_ch_used` and
`n_ch_truncated`, recomputed from `n_ch_max`.  Resetting every derived field
whose name begins with the varied parameter would turn `n_ch_max`, which is
what the geometry allows and not what was asked for, from 32 into 1, and
`bool("1")` would make `n_ch_truncated` True.  This is why the script prints a
diff and writes nothing until told to.

    python3 validation/synthetic/add_reference_runs.py wavelet_matrix_v2 --config validation/synthetic/matrix_v2.yaml
    python3 validation/synthetic/add_reference_runs.py wavelet_matrix_v2 --config validation/synthetic/matrix_v2.yaml --write

then run the new ids the usual way, since the runner skips what is already done:

    for s in $(seq 0 29); do
      python3 validation/synthetic/run_matrix_v2.py --config validation/synthetic/matrix_v2.yaml --shard $s --n_shards 30 &
    done; wait
"""
import argparse
import json
import os

import yaml

# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--prefix", default="sec_",
                    help="the families that need a control")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    path = os.path.join(a.root, a.manifest)
    runs = [json.loads(l) for l in open(path)]
    print("%d entries in %s" % (len(runs), path))
    if any(r["run_id"].startswith("ref_") for r in runs):
        raise SystemExit("reference entries already present")

    ref = yaml.safe_load(open(a.config))["reference"]
    print("reference cell from %s:" % a.config)
    for k in sorted(ref):
        print("    %-16s %s" % (k, ref[k]))

    sec = [r for r in runs if r["run_id"].startswith(a.prefix)]
    if not sec:
        raise SystemExit("no entries with prefix %r" % a.prefix)
    by_seed = {}
    for r in sec:
        by_seed.setdefault(r["seed"], []).append(r)
    print("\n%d secondary entries, %d distinct seeds -> %d controls"
          % (len(sec), len(by_seed), len(by_seed)))

    out, touched = [], set()
    for i, (seed, group) in enumerate(sorted(by_seed.items())):
        t = json.loads(json.dumps(group[0]))
        t["run_id"] = "ref_s%04d" % i
        for k, v in ref.items():
            if k in t.get("params", {}):
                if t["params"][k] != v:
                    touched.add("params." + k)
                t["params"][k] = v
        d = t.get("derived", {})
        if "n_ch_used" in d and "n_ch" in ref:
            n_max = int(d.get("n_ch_max", d["n_ch_used"]))
            want = int(ref["n_ch"])
            d["n_ch_used"] = min(want, n_max)
            if "n_ch_truncated" in d:
                d["n_ch_truncated"] = bool(want > n_max)
            touched.add("derived.n_ch_used")
        out.append(t)
    print("fields reset: %s" % ", ".join(sorted(touched)))

    print("\nfirst control against the entry it was derived from:")
    A = flat(sorted(by_seed.items())[0][1][0])
    B = flat(out[0])
    for k in sorted(set(A) | set(B)):
        if A.get(k) != B.get(k):
            print("  %-28s %-20s -> %s" % (k, A.get(k), B.get(k)))
    print("  (%d further fields identical)"
          % sum(1 for k in set(A) | set(B) if A.get(k) == B.get(k)))

    print("\n%d entries would be appended, ref_s0000 .. ref_s%04d"
          % (len(out), len(out) - 1))
    if not a.write:
        print("\nNothing written.  Check the diff above, and in particular that")
        print("n_ch_max is NOT in it: it is what the geometry allows, not what")
        print("was asked for, and resetting it would make it 1.")
        return
    with open(path, "a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("appended to %s" % path)


if __name__ == "__main__":
    main()
