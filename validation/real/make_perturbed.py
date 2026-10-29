"""Controlled perturbations of the final Viking tracking, one new folder each,
PREFIX<name>/track.npz, for the tracking-sensitivity test of Sect. 3.4.

  const_{p,m}{10,20,40}   t0 + s, s = +-10, +-20, +-40 ms
  ramp{20,40}             linear from -A/2 to +A/2 over the line (total change A)
  sin300_{20,40}          A sin(2 pi i / 300), period 300 shots
  noise{4,10}             independent Gaussian per shot, sigma 4 / 10 ms, seed 12345

    python validation/real/make_perturbed.py SRC_TRACK PREFIX
"""
import json
import os
import sys

import numpy as np

SRC = sys.argv[1]; PREFIX = sys.argv[2]
z = np.load(SRC, allow_pickle=True)
t0 = np.asarray(z["t0"], float); S = t0.size; i = np.arange(S)
P = {}
for s in (10, 20, 40):
    P["const_p%d" % s] = np.full(S, s * 1e-3); P["const_m%d" % s] = np.full(S, -s * 1e-3)
for A in (20, 40):
    P["ramp%d" % A] = (i / (S - 1) - 0.5) * A * 1e-3
    P["sin300_%d" % A] = A * 1e-3 * np.sin(2 * np.pi * i / 300.)
rng = np.random.default_rng(12345)
for sg in (4, 10):
    P["noise%d" % sg] = rng.normal(0., sg * 1e-3, S)
for name, d in P.items():
    o = PREFIX + name
    if os.path.exists(o):
        sys.exit(o + " exists, not overwritten")
    os.mkdir(o)
    p = json.loads(str(z["params"]))
    p["perturbation"] = dict(source=os.path.basename(SRC), name=name, rms_ms=float(np.sqrt(np.mean(d ** 2)) * 1e3),
                             range_ms=[float(d.min() * 1e3), float(d.max() * 1e3)])
    out = {k: z[k] for k in z.files}
    out.update(t0=np.maximum(t0 + d, 1e-3), params=json.dumps(p))
    np.savez(os.path.join(o, "track.npz"), **out)
    print(o, "rms %.1f ms, range %+.0f..%+.0f ms" % (p["perturbation"]["rms_ms"], *p["perturbation"]["range_ms"]))
