#!/usr/bin/env python3
"""Picker tests of the paper (56 main-grid + 160 secondary-axis records) with
the final recipe, next to stage 1.  The records are the cubes of
picker_records.py; nothing is regenerated here.

Per record:
  stage 1, paper    the whole record as one group (pick_consensus, today's code),
                    winner - 9 ms, as eval_picker scored it
  stage 1, final    anchors_from_consensus as in seafloor_pipeline (group from
                    the rule, end-of-direct rejection): the median of its anchors
  stage-1 tracking  run_real on those anchors (n_group 20)
  complete method   the tracking of seafloor_pipeline (onset tracking with its fallback)
Error = median over the record of (pick - true t0), in ms; scored in the
window the extraction tolerates (-100 to +50 ms), in +-80 ms, and on the cycle
(|error| < T_dom true / 2).

    python validation/synthetic/picker_test.py run RECORDS OUT [--jobs 10]
    python validation/synthetic/picker_test.py redo RECORDS OLD NEW [--jobs 8]
    python validation/synthetic/picker_test.py report OUT

redo reruns the records of OLD whose pipeline stopped and writes NEW with
those records replaced and every other one copied from OLD.  The picker
numbers of the paper are such a merge: the run with the method frozen before
the robustness corrections of 25 October 2026, in which the pipeline stopped
on 15 records, and the rerun of those 15 after the corrections.  With the
present code the pipeline does not stop, and run alone gives the same pick
errors; the operating-domain flags are those of the code that ran.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT]


def stage1_record(g, dt, offs, vw):
    """The stage-1 anchor of a record taken as one group: the group code of
    anchors_from_consensus, the winner minus the convention offset."""
    from seaweed import pick_consensus as PC
    gg = PC.agc(np.asarray(g).astype(float), dt)
    hi = min(1.50, (gg.shape[1] * dt) / 2.2)
    cands = sorted(set(np.round(
        PC.autocorr_candidates(gg, dt, 0.10, hi)
        + PC.semblance_candidates(gg, dt, offs, vw, 0.10, hi), 4)))
    scored = [PC.score_candidate(gg, dt, offs, t, vw) for t in cands]
    j = int(np.argmax([s for s, _ in scored]))
    return float(cands[j]) - PC.CONVENTION_MS * 1e-3, float(scored[j][1]["semblance"])


def one(args):
    cube, out = args
    from seaweed.wavelet_extractor_v2 import estimate_vw_direct
    uid = os.path.basename(cube)[:-len("_cube.npy")]
    o = os.path.join(out, uid); os.makedirs(o)
    g = np.load(cube, mmap_mode="r"); meta = np.load(cube + ".meta.npz", allow_pickle=True)
    tr = np.load(cube.replace("_cube.npy", "_truth.npz"), allow_pickle=True)
    dt = float(meta["dt"]); offs = np.asarray(meta["off_hdr"], float)
    vw, _x, _n = estimate_vw_direct(np.asarray(g), dt, offs, D_guess=float(np.median(tr["D"])))
    res = dict(uid=uid, kind=str(tr["kind"]), T_ms=float(tr["tdom_true_ms"]), t0_true_ms=float(np.median(tr["t0_true"])) * 1e3)
    try:
        t1, _ = stage1_record(g, dt, offs, vw)
        res["s1_paper_err_ms"] = t1 * 1e3 - res["t0_true_ms"]
    except Exception as e:
        res["s1_paper_error"] = str(e)[:80]
    fp = os.path.join(o, "first_pass.npz")
    np.savez(fp, t0=np.zeros(g.shape[0]), vw=vw, dt=dt, offs=offs)
    rc = subprocess.run([sys.executable, "-m", "seaweed.seafloor_pipeline", os.path.join(o, "p"), "--segy", uid,
                         "--nshot", str(g.shape[0]), "--dxsp", "25", "--load_cube", cube, "--first_pass", fp,
                         "--n_ch_track", "3"], capture_output=True, text=True, cwd=ROOT,
                        env=dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1"))
    open(os.path.join(o, "pipeline.log"), "w").write(rc.stdout + rc.stderr)
    p = os.path.join(o, "p")
    if rc.returncode:
        res["pipeline_error"] = rc.returncode; return res
    t0t = np.asarray(tr["t0_true"], float)
    a = np.load(os.path.join(p, "stage1_anchors.npz"), allow_pickle=True)
    res["s1_final_err_ms"] = float(np.median(np.asarray(a["t"]) - t0t[np.asarray(a["i"], int)])) * 1e3
    for k, f in (("s1_track", "stage1_track.npz"), ("complete", "track.npz")):
        z = np.load(os.path.join(p, f), allow_pickle=True)
        res[k + "_err_ms"] = float(np.median(np.asarray(z["t0"], float) - t0t)) * 1e3
    s = json.load(open(os.path.join(p, "summary.json")))
    res.update(h3_applicable=s["h3_applicable"], h3_fallback_reason=s["h3_fallback_reason"], group=s["group"],
               domain=s["domain"]["flag"])
    return res


def cubes(records):
    """The cubes of the picker tests: grid and sec records, not calib."""
    return [c for c in sorted(glob.glob(os.path.join(records, "*_cube.npy")))
            if str(np.load(c.replace("_cube.npy", "_truth.npz"), allow_pickle=True)["kind"]) in ("grid", "sec")]


def run(records, out, jobs):
    from multiprocessing import Pool
    os.makedirs(out)
    J = [(c, out) for c in cubes(records)]
    with Pool(jobs) as pool, open(os.path.join(out, "records.jsonl"), "w") as fh:
        for r in pool.imap_unordered(one, J):
            fh.write(json.dumps(r) + "\n"); fh.flush()


def report(out):
    R = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
    print("# Picker tests with the final recipe (`%s`, %d records)\n" % (out, len(R)))
    print("errors: %s\n" % (", ".join(r["uid"] for r in R if "pipeline_error" in r or "s1_paper_error" in r) or "none"))
    print("| records | chain | n | within -100/+50 ms | within +-80 ms | on the cycle (\\|e\\| < T/2) | median error (ms) |")
    print("|---|---|---|---|---|---|---|")
    for lab, sel in (("main grid (56)", lambda r: r["kind"] == "grid"), ("secondary axes (160)", lambda r: r["kind"] == "sec")):
        for name, k in (("stage 1, paper (record as one group, -9 ms; today's code)", "s1_paper_err_ms"),
                        ("stage 1, final (anchors, rule G, end of the direct)", "s1_final_err_ms"),
                        ("stage-1 tracking", "s1_track_err_ms"), ("complete method (onset tracking, h2)", "complete_err_ms")):
            E = [(r[k], r["T_ms"]) for r in R if sel(r) and k in r]
            e = np.array([x for x, _ in E]); T = np.array([t for _, t in E])
            print("| %s | %s | %d | %.0f %% | %.0f %% | %.0f %% | %+.1f |" % (
                lab, name, e.size, 100 * np.mean((e >= -100) & (e <= 50)), 100 * np.mean(np.abs(e) <= 80),
                100 * np.mean(np.abs(e) < T / 2), np.median(e)))
    for lab, sel in (("main grid", lambda r: r["kind"] == "grid"), ("secondary axes", lambda r: r["kind"] == "sec")):
        L = [r for r in R if sel(r) and "h3_applicable" in r]
        na = [r for r in L if not r["h3_applicable"]]
        print("\n%s: onset tracking not applicable to the whole record on %d of %d (%.0f %%): %s; groups of %s shots; domain: %s"
              % (lab, len(na), len(L), 100 * len(na) / max(len(L), 1),
                 ", ".join("%s %d" % (k, sum(r["h3_fallback_reason"] == k for r in na)) for k in sorted({r["h3_fallback_reason"] for r in na})),
                 sorted({r["group"] for r in L}),
                 ", ".join("%s %d" % (k, sum(r["domain"] == k for r in L)) for k in sorted({r["domain"] for r in L}))))


def redo(records, old, new, jobs):
    from multiprocessing import Pool
    R = [json.loads(l) for l in open(os.path.join(old, "records.jsonl"))]
    stop = [r["uid"] for r in R if "pipeline_error" in r]
    os.makedirs(new)
    with Pool(jobs) as pool:
        again = pool.map(one, [(os.path.join(records, u + "_cube.npy"), new) for u in stop])
    by = {r["uid"]: r for r in again}
    with open(os.path.join(new, "records.jsonl"), "w") as fh:
        for r in R:
            fh.write(json.dumps(by.get(r["uid"], r)) + "\n")
    print("%d stopped records rerun; still failing: %s" % (len(stop), [r["uid"] for r in again if "pipeline_error" in r]))


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else None
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], jobs or 10)
    elif sys.argv[1] == "redo":
        redo(sys.argv[2], sys.argv[3], sys.argv[4], jobs or 8)
    else:
        report(sys.argv[2])
