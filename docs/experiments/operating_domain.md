# Operating domain

The extraction needs the seafloor primary to arrive after the direct arrival
has died away. The operating domain states this as a flag on each line; it is
reported in the log and in `summary.json`, and never changes the tracking or
the wavelet.

## Definition (final, `h3_track.validity_domain`)

For each shot, Δt is the separation between the seafloor arrival at the near
trace and the direct arrival x0 / Vw. A shot is inside when both conditions
hold:

1. Δt ≥ 5 T_dom, with T_dom the dominant period of the line;
2. t_p ≥ t_d,end: the seafloor arrives after the end of the direct, measured
   on the stacked near traces of the whole line (`stage1_tracking.md`).

Both are one threshold, Δt ≥ max(5 T_dom, t_d,end − x0 / Vw); the flag
reports which of the two is active. T_dom and t_d,end are constant along the
line.

Δt is the smallest of three estimates, shot by shot, so that a tracking
locked on a later event cannot look deeper, and inside:

- the onset tracking;
- the stage-1 anchors, interpolated along the line;
- the first energy after the end of the direct on the near trace (first
  crossing of eight times the noise RMS).

All three are measured after the end of the direct, so a seafloor inside a
long direct is invisible to them. A fourth, independent test covers that
case: the water-layer reverberation period from the autocorrelation of the
near trace, without tracking and without rejection of the direct. A shot
whose autocorrelation puts the seafloor before the end of the direct is
outside whatever its Δt. Its search starts at 240 ms of two-way time, so it
cannot report a shallower seafloor.

Aggregation along the line:

- **outside** when more than 50 % of the shots are outside (by either test);
- otherwise **partially outside** when a 100-shot block has its median Δt
  below the threshold, or more than half of its shots fail the autocorrelation
  test;
- otherwise **inside**. Shots outside that belong to no such block are
  reported as isolated shots; they count towards the 50 % but do not make a
  line partially outside.

## How it was reached

| date | step |
|---|---|
| 23 Oct | Condition 1 with s* fixed by a rule written before the numbers: the smallest separation, in classes of T_dom, from which the complete method has at least 90 % of its anchors on the right cycle on the 216 synthetic records. Classes after the merge rule: [0, 5), [5, 8), [8, ∞) T_dom, with 13, 10 and 193 records and 0.750, 0.941 and 0.923 of the anchors on the right cycle; s* = 5 |
| 24 Oct | Condition 2 added. On Viking and IT90-65 the real direct arrival, with a slow component near 11 Hz, lasts 243 and 278 ms after x0 / Vw (8.7 and 11.4 T_dom), beyond 5 T_dom; on the synthetics it lasts 39 ms in the median (1.4 T_dom) and never beyond |
| 25 Oct | Δt taken as the smallest of the three estimates, after a synthetic record at 180 m whose tracking was 273 ms late had been flagged inside |
| 26 Oct | The autocorrelation test, with criteria fixed before the run: (1) the four off-cycle records at the edge of the domain (errors of +429, +273, +59 and +96 ms) flagged outside; (2) at most 5 % flagged outside by this test among records with a true separation of at least twice the threshold; (3) no real line outside. Criterion 1 failed (2 of 4), criteria 2 (1 of 202 records, 0.5 %) and 3 passed. Adopted by decision (`rule_overrides.md`) |
| 26 Oct | Rule (c): "partially outside" from the 100-shot blocks only. Decided after the run, in which IT90-65, SISOC-MEPA and MS-29 had become "partially outside" through isolated shots (1.3 %, 0.6 % and 0.1 %) under the rule of 23 October, by which one shot was enough |

## On the four lines

All four lines are inside, with the end of the direct the active condition
on Viking and IT90-65 and 5 T_dom on SISOC-MEPA and MS-29:

| line | direct x0 / Vw | end of the direct | 5 T_dom | Δt, minimum / median | isolated shots outside |
|---|---|---|---|---|---|
| Viking | 181 ms | 424 ms | 140 ms | 313 / 332 ms | 0 |
| IT90-65 | 86 ms | 364 ms | 122 ms | 560 / 833 ms | 39 (1.3 %, autocorrelation) |
| SISOC-MEPA | 33 ms | 76 ms | 92 ms | 951 / 1030 ms | 12 (0.6 %, first energy) |
| MS-29 | 198 ms | 280 ms | 258 ms | 740 / 946 ms | 1 |

The separations are those of the tracking of 24 October; the final values
are printed by `validation/real/tables.py`.

## Limits declared with it

- s* rests on a class of ten records.
- The domain flags shallow water, not slope or acquisition geometry: 22 of
  the 31 synthetic records where the onset tracking is not applicable lie
  above 8 T_dom.
- Below s* the complete method does worse than the onset tracking alone,
  because the stage-1 fallback is wrong there as well.
- A seafloor within a few tens of milliseconds of the end of a long direct is
  not recognised by these measures (two synthetic records at 220 m).
