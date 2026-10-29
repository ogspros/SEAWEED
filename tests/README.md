# The test that keeps the numbers fixed

No cleaning of this repository may change a number of the paper. That is the
whole rule, and this directory is what makes it checkable instead of hoped for.

The check is an md5 comparison on the ten whole-line combs of family C, the
runs behind `0.823 ± 0.006` in Sect. 4.4 and in the abstract. They go through
the entire extraction, on real data, with the defaults the paper describes, so
a change anywhere in the chain moves at least one of the ten files. Ten
wavelets of two hundred bytes each are enough to notice.

## Order of operations

1. Tag the state that produced the results, before touching anything:
   `git tag -a v1.0-as-run -m "the code that produced the paper"`.
2. On the machine that holds the data, freeze the products:
   `bash tests/freeze_reference.sh`. Commit what it writes.
3. Clean, refactor, remove dead branches.
4. After each step: `bash tests/check_static.sh`, then
   `bash tests/smoke_test.sh`.

Step 2 comes before step 3. A reference frozen after a modification proves
that the modification agrees with itself.

## The three commands

| command | needs the data | what it answers |
|---|---|---|
| `bash tests/check_static.sh` | no | does everything still compile, and do the family letters and the shared constants still agree |
| `bash tests/smoke_test.sh` | yes | do the ten extractions still give the same bytes |
| `bash tests/freeze_reference.sh` | yes | records what the comparison is made against |
| `python3 tests/test_h3.py` | no | do the onset-tracking pieces, the recipes of `run_real` and the unit fallback of `h3_track` behave |
| `RECIPE=final bash tests/smoke_test.sh` | yes | does the final recipe (`seaweed.seafloor_pipeline`, then the ten combs with `--recipe final`) still give the frozen onset tracking and the same bytes, against `tests/reference_final*` |
| `GEN=DIR UNITS=DIR bash tests/test_fallback_extraction.sh OUT` | yes (synthetic cubes and their stage-1 anchors) | where the onset tracking is not applicable to a whole unit, does the pipeline give the stage-1 tracking, flagged, and still extract with `--align two_stage --bistab_check` |

The plain `bash tests/smoke_test.sh` stays the non-regression test of the
paper: `--recipe paper`, the default of `run_real`, keeps the old defaults.

`tests/smoke_test.sh` loads the frozen tracking, so the ten runs take minutes.
Its exit codes are meant to be read:

- **0** — every wavelet is bit-for-bit the one behind the paper. The change is
  cosmetic and can be kept.
- **2** — no wavelet moved beyond `RTOL` (`1e-6` of its peak by default, about
  eight units in the last place of a `float32`), but the bytes differ. This is
  what a different numpy, BLAS or thread count produces. Compare
  `tests/reference_env.txt` with the machine in front of you before reading it
  as a result of the change.
- **1** — a wavelet, or a quality-control scalar of its `_sel.npz`, has moved.
  The change is not cosmetic. Either revert it, or re-read every number of the
  paper that depends on it.

## What is compared

- `tests/reference_md5.txt` — the hashes of record, bare file names so that
  `md5sum -c` works from inside any run directory. This is the check the rule
  is written on.
- `tests/reference/` — a copy of the ten wavelets and their `_sel.npz`, about
  4 kB, so that a comparison is possible without the seismic data and so that
  a difference can be described rather than only detected.
- `tests/reference_stats.tsv` — peak, r.m.s., sample of the minimum, stack
  coherence, alignment residual and shot selection of each run.
- `tests/reference_inputs.md5` — the cube, the anchors and the tracking. When
  a wavelet moves, this says first whether an input moved. A wavelet that
  changes because the anchors changed says nothing about the code.
- `tests/reference_env.txt` — commit, interpreter, library versions, BLAS and
  thread variables.

The `_sel.npz` scalars are compared as well. A wavelet can come out identical
while the stack coherence quoted in Table 3 has moved, because the two are
computed in the same pass.

## What is not covered

- **The tracking and the picker.** The ten runs load the frozen tracking, so
  they exercise stages 3 to 5. `FULL=1 bash tests/smoke_test.sh` rebuilds the
  tracking from the stage-1 anchors first and compares it array by array,
  which is the only way stages 0 to 2 are tested here. A `.npz` records the
  time it was written, so its md5 is not reproducible and is not used.
- **The synthetic matrix.** Tables 1 and 2 come from `run_matrix_v5.py` and
  `run_matrix_v6.py`, which are hours of computation. To extend the test to
  them, freeze one cell of the reference axis and compare its `results.jsonl`
  record by record.
- **The figures.** A PDF differs at every write; comparing the numbers a
  figure is drawn from is the meaningful check, and `tests/reference_stats.tsv`
  already holds them for family C.
- **The archive lines.** MS-29, IT90-65 and SISOC-MEPA are not re-run. Their
  runs are documented in `docs/RUNS.md` and their logs are in `logs/`.

## When it fails

Read the verdict of the numerical comparison before reading the diff. `CHANGED` with
`lag` different from zero means the origin of the wavelet moved, which is the
estimator of stage 5. `CHANGED` with `ncc` still at 1.000 means a scale factor,
usually a normalisation that was introduced by the change. `MISSING` means the
run did not produce a wavelet at all; its log is in `tests/_smoke_out/`.

To compare two directories of runs directly, at any time:

```
python3 tests/compare_wavelets.py viking_conv41 some_other_run
```

## Extending it

The family and the number of runs are variables. To freeze another family,
`FAMILY=A_n100_L2.5km_blk bash tests/freeze_reference.sh`, and the smoke test
follows the same pattern through `COMBS` and the selection it builds. Keep the
frozen family small: the check has to be cheap enough to run after every
modification, or it will not be run.
