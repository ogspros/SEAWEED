#!/usr/bin/env python3
"""The synthetic records behind the paper: the full rerun with the final
recipe, with the chain records of the models whose chain had stopped replaced
by a rerun of those models.

What it is.  The full rerun (rerun_final.sh) was made with the method frozen
before two robustness corrections.  In it the chain stopped on 79 models
(stopped_models.json: 55 of v2, 24 of the shallow matrix): run_real failed
when the first multiple could not be extracted, and anchors_from_consensus
when fewer than four anchors were left.  run_matrix_v6 then scored those
records on the true seafloor.  After the corrections, which leave every other
record identical, those 79 models were rerun (rerun_models.sh) and their
chain records (passes v6 and shallow_chain) replaced the stopped ones, in
place; every other record, the known-seafloor passes of those models
included, is the one of the full rerun.  The operating-domain fields of each
record are those of the code that ran it: the 79 rerun records carry the
later flag.  With the present code the chain does not stop, and a new full
rerun needs no merge.

    python validation/synthetic/merge_rerun.py FULL RERUN OUT [--ids stopped_models.json]

FULL: the output of rerun_final.sh; RERUN: the output of rerun_models.sh;
OUT: a new directory, written as results_merged_shard000.jsonl and, with the
stored wavelets merged the same way, results_merged_shard000_wavelets.npz.
"""
import argparse
import glob
import json
import os

import numpy as np

CHAIN = ("v6", "shallow_chain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full"); ap.add_argument("rerun"); ap.add_argument("out")
    ap.add_argument("--ids", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopped_models.json"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(a.out + " exists")
    ids = {i for v in json.load(open(a.ids)).values() for i in v}
    rep = {}
    for f in sorted(glob.glob(os.path.join(a.rerun, "*", "results_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r["pass_name"] in CHAIN and r["run_id"] in ids:
                rep[(r["pass_name"], r["run_id"])] = l
    os.makedirs(a.out)
    n = 0
    with open(os.path.join(a.out, "results_merged_shard000.jsonl"), "w") as fh:
        for f in sorted(glob.glob(os.path.join(a.full, "results_*_shard*.jsonl"))):
            for l in open(f):
                r = json.loads(l); k = (r["pass_name"], r["run_id"])
                if k in rep:
                    l = rep[k]; n += 1
                fh.write(l)
    W = {}
    for f in sorted(glob.glob(os.path.join(a.full, "results_*_shard*_wavelets.npz"))):
        with np.load(f) as z:
            W.update({k: z[k] for k in z.files})
    for f in sorted(glob.glob(os.path.join(a.rerun, "*", "results_*_wavelets.npz"))):
        with np.load(f) as z:
            W.update({k: z[k] for k in z.files if k.split("|")[0] in CHAIN and k.split("|")[1] in ids})
    np.savez_compressed(os.path.join(a.out, "results_merged_shard000_wavelets.npz"), **W)
    missing = sorted(ids - {i for _, i in rep})
    print("%d chain records replaced from %d rerun models; models without a rerun record: %s"
          % (n, len({i for _, i in rep}), missing or "none"))


if __name__ == "__main__":
    main()
