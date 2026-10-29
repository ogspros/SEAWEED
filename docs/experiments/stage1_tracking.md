# Stage-1 tracking

The stage-1 tracking gives the onset tracking its prior and its fallback. It
is built in two steps: anchors from the consensus picker, one per group of
shots (`anchors_from_consensus`, `pick_consensus`), then the cross-correlation
tracking of `run_real` between them, with the drift relative to the anchors
removed.

## The final rules

- **Groups.** G = min(max(10, round(2500 m / dxsp)), floor(N / 4)) shots, with
  dxsp the shot spacing and N the shots of the line; at most 48 shots of a
  group, spread over it, are scored.
- **Candidates.** The strongest lobes of the autocorrelation of the near
  channel and the peaks of a water-velocity semblance scan, six of each,
  after automatic gain control. Semblance peaks less than 50 ms after the
  direct arrival at the near trace are dropped before the six are chosen.
- **Direct arrival.** Every candidate whose arrival at the near trace falls
  before the end of the direct, t_d,end, is discarded before scoring.
  t_d,end is measured once per line: the envelope of the stacked near traces
  back under max(0.08 × peak, 4 × noise RMS), plus 8 ms.
- **Score.** For each remaining candidate t0:

  score = S_p · (1 + 0.5 S_m) · (1 + 0.5 max(0, −ρ)) · sep · conf · flat

  - S_p, the semblance of the primary along the water moveout, at the best
    of 17 apparent dips (|sin θ| ≤ 0.20);
  - S_m, the semblance of the first multiple at 2 t0, at the same dip;
  - ρ, the correlation of the moveout-corrected primary and multiple windows;
    only reversed polarity raises the score;
  - sep = clip(separation from the direct / 50 ms, 0, 1);
  - conf = clip(0.30 / stretch reached, 0, 1), at least three channels;
  - flat = 1 − 0.5 |sin θ| / 0.20, the flatter explanation where the
    evidence is equal.
- **Anchors.** The best candidate of a group, minus the 9-ms convention
  offset, is kept if its semblance is at least 0.35 and it lies within 120 ms
  of the running median of five groups. With two or three anchors the line
  goes on with them; with fewer, the whole line is picked as one group and
  flagged.
- **Tracking.** Cross-correlation against a pilot over max(5, round(500 m /
  dxsp)) adjacent shots and the three nearest channels; the drift relative to
  the anchors is smoothed and removed.

## How the rules were reached

| date | step | outcome |
|---|---|---|
| 15 Oct | T1: the anchors of the first version regenerated on Viking | the tracking of the paper reproduced array for array |
| 16–20 Oct | T2: a tolerance between groups of one dominant period, and a rule on the residual from neighbouring anchors | suspended on 20 October: its premise, 28 ms of relief between two Viking groups, was a tracking artefact (the raw onset gives the same time in both) |
| 20–21 Oct | T4: provenance of the anchors of the first version | not reconstructible with certainty from the files. IT90-65 reproduced with 150-shot groups and a 120-ms tolerance except one group (+16 ms); SISOC-MEPA with 100-shot groups, 18 of 19 anchors identical (one at +4 ms), the same at 60 and 120 ms. 120 ms adopted; 48 scored shots per group, the only value that reproduces Viking |
| 21 Oct | double `t0_shift` in `run_real` corrected (declared as changing results) | Viking unchanged (shift 0); on the archive lines the shift had been applied twice |
| 21 Oct | semblance peaks within 50 ms of the direct dropped (declared as changing results) | one MS-29 group moved (1131 → 1147 ms); SISOC-MEPA unchanged |
| 22 Oct | the stage-1 tracking regenerated from these anchors on the three development lines | the reference for everything that followed (`baseline-stage1`) |
| 23 Oct | the group rule of 2500 m, fixed before any run, with its criterion: on each line, against the group sizes in use, \|ΔIQR\| ≤ 1 ms against the manual picks, \|Δ within T_dom/2\| ≤ 0.02, partial stacks in agreement not fewer | adopted: IT90-65 ΔIQR +0.01 ms, −0.003; MS-29 −0.04 ms, +0.002; Viking and MEPA unchanged by construction. Half the rule on Viking (G = 50) put four anchors on the direct arrival (207–211 ms instead of about 480 ms) |
| 23–24 Oct | candidates on the direct arrival discarded: first below 2 T_dom, replaced the next day by the measured end of the direct | Viking gains one anchor (shot 850, 519 ms, whose group had won on the direct); IT90-65 gains four and its group 1050 leaves the direct (135 → 1059 ms); MEPA and MS-29 unchanged. Viking with G = 50 recovers (onset tracking IQR 213 → 3.7 ms) |
| 25 Oct | robustness: fewer than four anchors after rejection no longer stops the program | results unchanged wherever it did not stop |

## What the stage-1 tracking achieved

Against the manual picks (`manual_picks.md`), the stage-1 tracking had an
IQR of 18.4, 8.5 and 15.5 ms on Viking, IT90-65 and SISOC-MEPA, with slow
errors of tens of milliseconds over long stretches (IT90-65 shots 800–1199,
SISOC-MEPA 1450–1699). On the 216 synthetic records of known seafloor its
anchor was on the right cycle (within half a period of the true time) on 64 %
of the main grid and 59 % of the secondary axes; on bench C, 14 of 36 R0
lines and 11 of 36 R1 lines met the tracking thresholds. These are the
reasons for the onset tracking (`onset_tracking.md`).

The 96 % and 86 % of the paper (records whose extraction is unaffected by a
picking error between −100 and +50 ms) and the cycle precision above measure
different things: an anchor one cycle off leaves the extraction of one
record nearly unchanged, but anchors on different cycles along a line make a
step in the tracking.
