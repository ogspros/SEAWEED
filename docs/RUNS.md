# Which script produces which number

Every number of the paper, with the public script that produces it and the
output where that script prints it. The drivers read the data from `DATA`
(`validation/real/README.md`, `validation/synthetic/README.md`) and write into
a new directory. Section numbers are those of the paper.

The logs the numbers of the paper's first version were read from (code of the
internal tag `v1.0-as-run`) are in `logs/`; those numbers are not listed here.

## Sect. 2: method

| item | value | script → output |
|---|---|---|
| group time minus 9 ms, the median delay on 216 synthetic records | 9 ms | first-version picker test (`eval_picker.py`); constant of `seaweed/pick_consensus.py` |
| group time on the correct cycle on synthetic records | about 60 % (68 % / 58 %) | `validation/synthetic/picker_test.py report` |
| IT90-65: direct-arrival and semblance water velocity | 1452.8, 1453 m/s | `validation/real/lines.sh` → `it065/` logs of the first pass and the stage-1 tracking |
| onset 11 ms before the true seafloor, 24 calibration records | 11 ms | `picker_records.py --kinds calib`; constant of `seaweed/h3_track.py` |
| Viking: t_p exceeds t_0 by 33 ms at 262 m and 345 m | 33 ms | geometry |
| channel count on Viking, C combs: 1 / 4 / 8 / 16 channels | 0.826 / 0.811 / 0.815 / 0.821 (−0.015, −0.011, −0.005) | `validation/real/ncc.sh` → `summary.md` |
| direct arrival: 9–11 T_dom on Viking and IT90-65 | 8.7, 11.4 T_dom | `validation/real/tables.py` (operating domain: end of the direct) with T_dom of `anchor_qc` |
| operating-domain flag: wrong-cycle edge cases recognised; false alarms among records well inside | 2 of 4; 1 of 202 | `validation/synthetic/domain_criteria.py` |

## Sect. 3.1–3.2: synthetics (Table 2, Fig. 2)

Records: `validation/synthetic/rerun_final.sh`, with the 79 models whose chain
stopped rerun and merged (`rerun_models.sh`, `merge_rerun.py`). Tables:
`tables_final.py` (which runs `tables_v5.py`, `compare_v5_v6.py`,
`tables_v5_extra.py` unchanged) and `paper_numbers.py`.

| item | value | output |
|---|---|---|
| 800 models, four SNR levels; 24 + 16 cells of 20 realisations | design | `matrix_v2.yaml` and its manifest |
| evaluation domain at 30 dB | 620 records | `tables_v5.txt` §1 |
| Table 2, known seafloor: evaluable, median NCC, failures; mean | 502/620, 0.875, 118 (19.0 %); 0.845 ± 0.113 | `tables_v5.txt` §1, `compare_v5_v6.txt` |
| Table 2, automatic picking: evaluable, median NCC, failures | 483/620, 0.879, 137 (22.1 %) | `compare_v5_v6.txt` |
| records lost / recovered; pairs evaluable in both; median paired difference | 20 / 1; 482; 0.000 | `compare_v5_v6.txt` |
| failures with the first reflector at 150 m | 96 of 118 | `paper_numbers.md` |
| zero padding in failures / in evaluable records | 71 % / 0 % | `tables_v5.txt` §1 |
| first reflector 80 → 300 m: mean NCC; evaluable of 160 | 0.809 → 0.867; 129, 132, 160 | `paper_numbers.md` |
| dip: mean, median, failures; roughness: failures | 0.639 ± 0.282, 0.790, 35 %; 32 % | `tables_v5.txt` §2 |
| fault, paired: mean, median, evaluable pairs, Wilcoxon p | +0.030, +0.011, 9 of 20, 0.0078 | `tables_v5.txt` §3 |
| folds, anomalous body | p = 0.13, 0.19 | `tables_v5.txt` §3 |
| Fig. 2: evaluable main-grid records with known seafloor | 257 | `paper_numbers.py` → `fig2/fig_synthetic.log` |
| deliberate picking errors −100…+50 ms: median NCC, failure rates | 0.872–0.886, 19–21 % | `paper_numbers.md` |
| at +100 ms | no record scored | `tables_v5.txt` §6 |
| complete chain: median picking error; inside −100…+50 ms | −2 ms; 93 % | `compare_v5_v6.txt` |
| onset anchors not placed | 63 records (10 %) | `summary_final.md` |
| picker tests, 56 / 160 records: stage 1 inside the interval; on the correct cycle | 96 % / 89 %; 68 % / 58 % | `picker_test.py report` (stage 1, final) |
| complete tracking: correct cycle; inside the interval | 88 % / 88 %; 98 % / 90 % | `picker_test.py report` |
| shallow-water grid: depths, m; mean NCC; failures | 78–258 m, 0.25–1.37; 0.77–0.86; 5–22 % | `tables_v5_extra.txt` |
| shallow-water chain: evaluable, failures, onset anchors not placed | 296/400, 26.0 %, 202 | `tables_shallow_chain.txt`, `summary_final.md` |
| noise: effective SNR, mean NCC, coherence | 11 → 41 dB, 0.834 → 0.845, 0.812 → 0.995 | `tables_v5.txt` §5 |
| towing depth: NCC change between the extreme configurations | −0.27 | `tables_v5_extra.txt` |

## Sect. 3.3: tracking against the manual picks (Table 3)

`validation/real/lines.sh`, `manual_picks.py`, `tables.py` ("Tracking against
the manual picks"); partial stacks in the Table 4 block.

| line | IQR stage 1 / final (ms) | median final (ms) | within T_dom/2 stage 1 / final | partial stacks |
|---|---|---|---|---|
| Viking | 7.33 / 3.87 | +11.8 | 0.800 / 0.958 | 20/20 |
| IT90-65 | 11.91 / 2.56 | +7.4 | 0.844 / 0.931 | 60/61 |
| SISOC-MEPA | 15.49 / 2.46 | +11.3 | 0.599 / 0.947 | 39/39 |
| MS-29 | 27.12 / 3.93 | +21.3 | 0.726 / 0.983 | 16/16 |

MS-29, RMS of the 100-shot block medians, stage 1 → final: 17.59 → 1.73 ms;
T_dom/2 = 51.7 / 2 ms. R_wb discrepancies with the stage-1 track (117 % and
59 %) come from the stage-1 extractions of the same runs (`stage1_full_line`
logs in `lines.sh`).

## Sects. 3.4–3.5: Viking Graben (Table 4, Figs. 3 and 4)

`validation/real/runs.sh`, `ncc.sh` → `summary.md` and the files it reads;
signature-free numbers from `tables.py`.

| item | value | output |
|---|---|---|
| recovered water depth | 332–362 m | `tables.py`, Table 4 block |
| whole-line combs, mean NCC 8–65 Hz | 0.826 ± 0.005 | `summary.md`, `families/C.txt` |
| 8–18 Hz band, every tested selection | 0.22–0.32 | `where_disagreement.log` |
| single delayed copy: delay, amplitude, NCC gain | 32 ms, 0.30, +0.037 | `thin_layer_C.txt` |
| minimum-phase equivalent minus extracted, ±60 ms | +0.031 ± 0.003 | `summary.md`, `phase_C_60ms.txt` |
| Table 4, NCC: 2.5 / 12.4 / 24.8 km | 0.693 ± 0.095 / 0.740 ± 0.030 / 0.826 ± 0.005 | `summary.md` |
| Table 4, NCC: fold 200 / 500 / 1001 | 0.715 ± 0.101 / 0.738 ± 0.037 / 0.824 | `summary.md` |
| Table 4, coherence | 0.888, 0.817, 0.725, 0.853, 0.818, 0.724 | `summary.md`, `tables.py` |
| A against C; D against C: one-sided Mann–Whitney p, Cliff's δ | 0.0046, 0.70; 9.1 × 10⁻⁵, 1.00 | `summary.md` |
| 100-shot contiguous blocks; best 200-shot block (400–599) | 0.545–0.855; 0.877 | `summary.md` |
| combs within either half / across halves | 0.994–0.997 / 0.527 | `tables.py` |
| tracking perturbations: A blocks; C combs | −0.004…+0.002, −0.060; ≤ 0.004, −0.145, −0.091, −0.150 | `summary.md`, `pert/` |
| Figs. 3 and 4 | | `figure/figures/`, numbers in `figure/figures.log` |

## Sect. 4: archive lines (Table 5, Figs. 5 and 6)

`validation/real/lines.sh`, `runs.sh`, `tables.py`, `figures.sh`. IT90-65 and
MS-29 are tracked on the three nearest channels, the default.

| item | MS-29 / IT90-65 / SISOC-MEPA | output |
|---|---|---|
| tracked water depth (m) | 694–1126 / 454–887 / 737–847 | `tables.py` |
| peak frequency (Hz); duration 5–95 % (ms) | 18 / 38 / 66; 100 / 26 / 18 | `tables.py` |
| depth discrepancy from the multiple | +0.8 / −0.8 / −0.1 % | `tables.py` |
| R_wb first / primary; second / first; discrepancy | 1.161 / 0.378 / 0.301; 0.517 / 0.340 / 0.320; 77 / 11 / 6 % | `tables.py` |
| statistical spectrum r, 6–90 / 20–90 Hz | 0.882 / 0.821 / 0.887; 0.915 / 0.960 / 0.891 | `tables.py` |
| Sect. 4.2, cell agreement: SISOC-MEPA 2.5 → 7.5 km | 0.870 ± 0.057 → 0.954 ± 0.014 | `tables.py` |
| IT90-65 2.5 → 15 km | 0.913 ± 0.036 → 0.940 ± 0.030 | `tables.py` |
| MS-29 19.8 → 39.8 km: broadband; 38–45 Hz | 0.883 → 0.885; 0.939 → 0.925 | `tables.py` |
| Sect. 4.3: fitted z_r, ε, r over 303 samples, notch | 25.00 ± 0.11 m, 1.19, 0.93, 30.20 Hz | `figures.sh` → `fig_deghost.log` |
| measured ratio variation; maximum gain of the model | 7.1 dB; 1.37 | `fig_deghost.log`; `ms29_deghost_check.log` |
| depth sweep maximum; flat interval | 24.70 m; 24.05–25.45 m | `fig_deghost.log` |
| wavelet vs section before deghosting; after | 0.967; 0.854 | `fig_deghost.log`, `ms29_deghost_check.log` |
| notch depth, wavelet / section | 7.2 / 6.5 dB | `ms29_deghost_check.log` |
| Figs. 5 and 6 | | `figures.sh` |

## Numbers checked against a public script after the first list

Values as in the paper after the corrections of 29 October 2026.

| item | paper | script → output | state |
|---|---|---|---|
| synthetics, channel count (Sect. 2): 1 (reference configuration, different realisations) / 4 / 16 / 32 channels | 0.867 / 0.872 / 0.895 / 0.884 | `paper_numbers.py` → "Channel count" (mean NCC, known seafloor, 30 dB) | reproduced |
| bench of 72 lines: complete tracking / stage 1 (Sect. 3.2) | 46 / 25 | `bench_lines.py run`, `report` | reproduced (24 + 22, 14 + 11) |
| multiple checks unavailable (Sect. 3.2) | 174 of 2480, all on the dip axis | `paper_numbers.py` → "Multiple checks unavailable" | reproduced |
| span experiment (Sect. 4.2) | 62 extractions | `tables.py` → "Extractions of the span experiment" (IT90-65 28, SISOC-MEPA 22, MS-29 12) | reproduced |
| onset convention in arrival time (Sect. 3.3) | 10.3–11.0 ms | `tables.py` → "The onset convention in arrival time" | reproduced (10.28, 10.95, 10.99, 10.83 ms) |
| modelled response variation (Sect. 4.3) | 5.8 dB | `deghost_gain.py` (in `figures.sh`) | reproduced (5.83 dB; measured 7.10 dB) |
| sub-bottom delays, Viking halves (Sect. 3.5) | 32 / 20 ms | `validation/viking/viking_reflector_delay.py` | reproduced (west 32 ms, 24–56; east 20 ms, 16–24; raw-onset alignment) |
| 5 T_dom class (Sect. 2.4) | ten records | `domain_criteria.py` → "Separation classes" | reproduced (13, 10, 193 records) |
| 8–18 Hz band (Sect. 3.4) | 0.22–0.32 | `ncc.sh` → `where_disagreement.log` | reproduced |
| domain flag (Sect. 2.4) | 2 of 4; 1 of 202 | `domain_criteria.py` | reproduced |
