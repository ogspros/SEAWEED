# The synthetic numbers with the final recipe

The scripts of this folder generated the synthetic data of the paper and
scored the first version of the workflow (`run_matrix_v5.py`,
`run_matrix_v6.py`, the `tables_*.py`). The final recipe is scored by the
drivers below, which reuse those scripts unchanged. `DATA` names the directory
of the published runs: the configurations `matrix_*.yaml` and the manifests
`wavelet_matrix_*/manifest.jsonl`, read only.

## Matrices: Table 2 and Sect. 3.2

```
DATA=DIR bash validation/synthetic/rerun_final.sh out/v7                   # 1340 models, 48 processes, about 7 hours
python validation/synthetic/tables_final.py out/v7 out/tables
```

`rerun_final.sh` runs `run_matrix_v7.py --recipe final` on every model of the
v2 matrix (860), the shallow-water matrix (400) and the towing-depth matrix
(80). `tables_final.py` runs the table scripts of the first version on those
records and writes `summary_final.md` for what they cannot see: chain failures
counted as failures, the records where the onset tracking is not applicable,
the operating-domain flags, the bistability check.

**How the records of the paper were assembled.** The full rerun behind the
paper was made with the method frozen before two robustness corrections (25
October 2026). In it the chain stopped on 79 models (`stopped_models.json`:
55 of v2, 24 of the shallow matrix): `run_real` failed when the first
multiple could not be extracted, and `anchors_from_consensus` when fewer than
four anchors were left; `run_matrix_v6` then scored those records on the true
seafloor. After the corrections, which leave every other record identical,
those 79 models were rerun, and their chain records replaced the stopped ones
in place:

```
DATA=DIR bash validation/synthetic/rerun_models.sh validation/synthetic/stopped_models.json out/stopped
python validation/synthetic/merge_rerun.py out/v7 out/stopped out/merged
python validation/synthetic/tables_final.py out/merged out/tables
DATA=DIR python validation/synthetic/paper_numbers.py out/merged out/numbers
```

`paper_numbers.py` prints the numbers of Sect. 3.2 that the table scripts do
not: the first-reflector rows, the known-seafloor failures by first-reflector
depth, the median NCC per deliberate picking error, and makes Fig. 2 with
`make_fig_synthetic.py` on the stored wavelets.

The operating-domain fields of each record are those of the code that ran it,
so the 79 rerun records carry a later flag than the others; Table 2 does not
depend on them. With the present code the chain does not stop, and a new full
rerun needs no merge.

## Picker tests: Sect. 3.2

```
python validation/synthetic/picker_records.py --data DIR --out out/records  # 216 records, about 15 min each on one core
python validation/synthetic/picker_test.py run out/records out/picker
python validation/synthetic/picker_test.py report out/picker
```

`picker_records.py` regenerates the 56 main-grid and 160 secondary-axis
records of the picker tests from their seeds (`picker_run_ids.json`, which
also lists the 24 calibration records of the onset convention, `--kinds
calib`). `picker_test.py` scores, on each record, the stage-1 picker as in the
first version, the final stage-1 anchors, the stage-1 tracking and the
complete method. The numbers of the paper are, likewise, the run with the
method frozen before the corrections, in which the pipeline stopped on 15
records, with those 15 rerun after the corrections (`picker_test.py redo`).

## Long direct arrival: the operating domain

```
DATA=DIR python validation/synthetic/long_direct.py run out/records out/long_direct
python validation/synthetic/long_direct.py report out/long_direct
```

Adds the slow component of the Viking direct arrival to the records near the
edge of the operating domain and runs the complete method with and without it.
A robustness check, not a validation.

```
python validation/synthetic/domain_criteria.py out/records out/picker out/long_direct out/lines
```

The three criteria of the autocorrelation test of the operating domain (Sect.
2.4): the four off-cycle edge records flagged outside (2 of 4), at most 5 %
flagged outside by the test among records well inside the domain (1 of 202),
no real line outside (`out/lines` from `validation/real/lines.sh`).
