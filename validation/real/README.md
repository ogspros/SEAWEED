# The real lines with the final recipe

The drivers behind every real-line number and figure of the paper. They run
from the repository and read the data from `DATA` (default: the current
directory); each writes into a new directory and refuses an existing one.

```
export DATA=/path/to/data
bash validation/real/lines.sh   out/lines                      # about 45 minutes
bash validation/real/runs.sh    out/lines out/runs             # a few minutes with JOBS=12
bash validation/real/ncc.sh     out/runs out/ncc               # a few minutes
python validation/real/manual_picks.py out/manual              # if the manual picks are available
python validation/real/tables.py out/lines out/runs out/manual > out/tables.md
bash validation/real/figures.sh out/lines out/runs out/figures
```

| script | produces |
|---|---|
| `lines.sh` | `seaweed.seafloor_pipeline` on the four whole lines, with the line flags of the paper: stage-1 anchors, stage-1 tracking, onset tracking, the whole-line wavelet and its checks, the operating-domain flag |
| `runs.sh` | on those trackings, with `--recipe final`: the 41 Viking selections of Table 3, the C combs with 4, 8 and 16 channels, the A blocks and C combs on ten perturbed Viking trackings (`make_perturbed.py`), and the cells of the span experiment on the archive lines |
| `ncc.sh` | every NCC against the recorded Viking signature (`validation/viking/viking_phase_test.py --maxlag_ms 30`; the phase comparison at 60), the single-reflector fits, the agreement by sub-band and time window (`where_disagreement.log`), Figs. 3 and 4, and `summary.md` (`ncc_summary.py`) |
| `manual_picks.py` | the manual picks per shot of each cube, from the SEG-Y headers |
| `tables.py` | the tracking against the manual picks, Table 4, the stage-1 anchors, the cell agreement of Sect. 4.2 and the Viking families |
| `figures.sh` | Figs. 5 and 6 at 300 dpi, and the receiver-deghost check of MS-29 |

## Inputs under `DATA`

| file | content |
|---|---|
| `seismic.sgy`, `FarField.dat` | Viking Graben line 12 and its recorded far-field signature |
| `SGY/IT065_Raw_geom_100_3150.sgy`, `SGY/MEPA_RAW_geom_100_2031_filt.sgy`, `SGY/MS29_SP300_1157.sgy` | the archive lines |
| `viking_pick/viking_cube.npy`, `span_experiment/<line>_cube.npy` (each with `.meta.npz`) | the cubes of the first version of the workflow (`run_real --save_cube`) |
| `viking_conv41/viking_track_conv.npz`, `span_experiment/<line>_track.npz` | the first-pass trackings of the first version; only their sampling interval, offsets and water velocity are read |
| `manual_picks/*.txt` | the manual picks, two columns: CDP and two-way time in ms (optional) |
| `Spettro_NO_deghost.txt`, `Spettro_deghost.txt` | the MS-29 stack spectra before and after the receiver deghost of the earlier reprocessing |

The cubes and first passes are those of `validation/viking/viking_rerun_conv.sh`
and `validation/archive/run_span_experiment.sh`. Without them,
`seafloor_pipeline` builds both from the SEG-Y (step 0); the water velocity
then comes from a new first pass and the numbers can move in the last digits.

## Line flags

| line | flags | shot spacing |
|---|---|---|
| Viking | `--near 262 --dx 25 --nshot 1012 --stretch_semb 0.15` | 25 m |
| IT90-65 | `--near 125 --dx 25 --nshot 3051 --n_ch_direct 40` | 25 m |
| SISOC-MEPA | `--near 50 --dx 12.5 --nshot 1932` | 25 m |
| MS-29 | `--near 300 --dx 100 --nshot 800 --n_anchor 20` | 200 m |

Everything else is a default of the final recipe; the tracking uses the
three nearest channels on every line. With a loaded tracking, the flags that
act on the first pass (`--stretch_semb`, `--n_anchor`, `--n_ch_direct`) have no
effect on the extraction.

These drivers were checked against the runs the paper's numbers were read
from: the same inputs give the same trackings and wavelets, byte for byte.
