# Manual picks

Manual seafloor picks of the four lines, made by an interpreter on the
near-offset sections, are the reference against which every tracking was
evaluated. They were never used to
choose a parameter. They are not part of the repository.

## Format and correspondence

Each pick file has two columns, CDP and two-way time in ms, picked on the
onset of the seafloor on the near-offset section. The pick at a shot is the
pick interpolated at the CDP of the shot's near trace
(`validation/real/manual_picks.py`): the cube selection of `run_real` is
rebuilt from the SEG-Y headers (unique field records in order, incomplete
shots skipped, traces sorted by |offset|), and the near trace of every kept
shot gives its CDP. The correspondence was verified on all four lines on 22
October:

1. the number of kept shots equals the cube: 1001, 3051, 1932 and 800;
2. the near trace read from the SEG-Y equals channel 0 of the cube, sample by
   sample, on 25 shots spread over each line;
3. the near offset equals that of the cube on every shot: 262, 125, 50 and
   300 m;
4. the CDP of the near trace is strictly increasing at a constant step.

Zero times (the last five picks of MS-29) are missing picks and are dropped.
On SISOC-MEPA the picked time is quantised to 1 ms, and runs of one value
longer than 100 CDP (seven runs, 337 shots) are excluded as interpolated
stretches; shorter runs are quantisation on a gently dipping seafloor.

## Comparison

d = tracked arrival time at the near trace − manual pick, in ms; the tracked
vertical time is converted with the water hyperbola at the near offset and
the water velocity of the tracking. Reported per line:

- the median of d, which absorbs the convention of the picking (manual picks
  sit on the onset);
- the IQR of d;
- the share of shots within T_dom/2 of the median;
- the RMS of the medians of 100-shot blocks, median removed: the slow
  component that matters for the multiple windows and R_wb;
- the number of 100-shot blocks beyond T_dom/2.

T_dom is that of `anchor_qc` on the stage-1 tracking: 28.1, 24.4, 18.5 and
51.7 ms on Viking, IT90-65, SISOC-MEPA and MS-29.

## What the comparison showed (22 October)

| | Viking | IT90-65 | SISOC-MEPA |
|---|---|---|---|
| shots with a manual pick | 935 / 1001 | 3050 / 3051 | 1595 / 1932 |
| raw onset of the near trace: median / IQR / within T_dom/2 | +2.1 / 4.0 / 1.00 | −1.0 / 2.4 / 0.95 | +5.0 / 3.0 / 0.91 |
| stage-1 tracking: median / IQR / within T_dom/2 | +8.2 / 18.4 / 0.91 | −3.2 / 8.5 / 0.83 | +7.5 / 15.5 / 0.60 |
| stage-1 tracking: RMS of 100-shot blocks, blocks beyond | 9.4 ms, 1/10 | 11.7 ms, 4/31 | 11.0 ms, 8/20 |

The raw onset follows the manual picks closely where it can be measured; the
stage-1 tracking carries slow errors of tens of milliseconds. This is what
the onset tracking was built on (`onset_tracking.md`). MS-29 was compared for
the first time in the adoption test of 23 October. The final values for the
four lines are printed by `validation/real/tables.py`.
