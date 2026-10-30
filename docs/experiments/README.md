# Experimental record

This folder documents how the final recipe of SEAWEED was developed and
evaluated between 15 and 29 October 2026: which protocols were written, which
criteria were fixed before the runs they judged, what the runs gave, and where
a rule was set aside by a declared decision. It is a summary of the internal
working record, which is in Italian, holds machine paths and is not published;
that record is available from the authors on request.

| file | content |
|---|---|
| [`stage1_tracking.md`](stage1_tracking.md) | the stage-1 tracking: anchors from the consensus picker, their rejection rules, the group size, the end of the direct arrival |
| [`onset_tracking.md`](onset_tracking.md) | the onset tracking: the hybrid pickers H1/H2, the protocol of H3 and H3.1, the formal evaluations, the convention constant |
| [`operating_domain.md`](operating_domain.md) | the operating domain: its two conditions, the three estimates of the separation, the autocorrelation test, the classes it was calibrated on |
| [`rule_overrides.md`](rule_overrides.md) | the two rules set aside by decision and the two rules changed after their outcome, each with the rule as fixed, the outcome, the decision and the reason |
| [`manual_picks.md`](manual_picks.md) | the manual seafloor picks and how the tracking was compared with them |
| [`known_limits.md`](known_limits.md) | the limits found and declared, not corrected |

The numbers of the paper, each with the script that produces it, are in
[`../RUNS.md`](../RUNS.md).

## How the work was done

- **Criteria before runs.** Every protocol, threshold and decision rule was
  written down, dated, before the code or the run it was meant to judge.
  Where a criterion was changed afterwards, the change is recorded as such,
  with its date and reason (`rule_overrides.md`).
- **The recorded Viking signature was used once.** The NCC against the
  far-field signature of Viking Graben was computed only after the method had
  been frozen (26 October 2026), for every Viking run at once. No parameter
  was chosen by looking at it; the development used internal diagnostics,
  the synthetic truth and the manual picks. The values of the first version
  of the paper were known, and are noted wherever a run of that version
  entered a comparison.
- **A held-out line.** MS-29 was not used during the development of the
  onset tracking. It was opened for its adoption test (23 October 2026) and
  treated as the other lines afterwards.
- **Every run in a new directory.** Inputs were linked, never copied or
  modified; no output directory was reused.
- **Code changes under test.** Before and after every change to the code:
  the static checks and the smoke test of `tests/`, which compares the ten
  whole-line Viking combs byte for byte. A change that alters results was
  committed separately and declared as such.
- **Frozen method.** After 26 October 2026 (internal tag `method-final`) the
  package `seaweed/` was not changed. The later decisions concern line flags,
  figures and text only.

## Tags

| tag | date | state |
|---|---|---|
| `v1.0-as-run` | 18 Sep 2026 | the code that produced the first version of the paper (internal) |
| `baseline-stage1` | 22 Oct 2026 | the stage-1 tracking regenerated from the anchors, the reference the onset tracking was compared with (internal) |
| `h3.1-adopted` | 23 Oct 2026 | the onset tracking adopted (internal) |
| `method-final` | 26 Oct 2026 | the method frozen before the NCC against the signature (internal) |
| `v1.0.0` | 29 Oct 2026 | the first public release; `seaweed/` functionally identical to `method-final` (see `CHANGELOG.md`) |

## Glossary of internal names

| internal name | in the paper | meaning |
|---|---|---|
| Stage 1, stage-1 anchors | stage-1 tracking | anchors from the consensus picker, one per group of shots, and the cross-correlation tracking between them |
| base, `baseline-stage1` | stage-1 tracking | the method before the onset tracking, kept as the reference and as the fallback |
| H1, H2 | — | hybrid pickers of 21 October, not adopted (`onset_tracking.md`) |
| H3, H3.1 | onset tracking | dense anchors from the onset of the near trace, the stage-1 tracking as a prior only; H3.1 is H3 with the convention constant |
| h2, h2m | — | the two onset variants: near channel only (h2, adopted), median of channels 1–3 (h2m) |
| validity domain | operating domain | the separation between direct arrival and seafloor needed for the extraction (`operating_domain.md`) |
| fourth separation | autocorrelation test | the water-layer reverberation period, a fourth, independent check of the operating domain |
| rule (c) | — | "partially outside" decided from 100-shot blocks only; isolated shots are counted |
| `recipe paper`, `recipe final` | first version, final recipe | the defaults of `run_real`: single alignment and 20-shot tracking window, or two-stage alignment, bistability check and the 500-m window |
| T1–T4 | — | the first tests of the defaults: clean anchors, the drift window, the tolerance between groups, the provenance of the anchors of the first version |
| C10 | — | the synthetic referee: 216 records with known seafloor (56 of the main grid, 160 of the secondary axes) and 24 calibration records |
| bench C | — | 72 synthetic lines of 60 shots with explicit bathymetry, classes R0 (no sub-bottom reflector) and R1 (a reflector about one period below the seafloor) |
| criteria 1–11, a–e | — | the acceptance criteria of the onset tracking (`onset_tracking.md`) |
| T_dom | dominant period | four times the first zero of the autocorrelation of the onset stack (`anchor_qc`) |
| convention constant | convention offset | the median lateness of a picking rule on calibration records, removed once at the output: the picker is 9 ms late, the onset rule 11 ms early |
| families A–G, C combs | Table 3 | the Viking selections: contiguous blocks (A, B, E), combs (C, D), whole line (F), coherence-selected (G) |
| cells | span experiment | stretches of 100–600 shots on the archive lines, each sampled by a comb of 100 shots |
| end of the direct | t_d,end | where the envelope of the stacked near traces returns under the threshold of the origin rule, plus 8 ms |
