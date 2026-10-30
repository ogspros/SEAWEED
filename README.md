# SEAWEED

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23780745.svg)](https://doi.org/10.5281/zenodo.23780745)

**SEAfloor WavelEt Extraction and Diagnostics** — an open-source workflow for
automated extraction and diagnostics of effective marine seismic wavelets.

SEAWEED implements an automated workflow with default parameters for
extracting the effective wavelet from marine seismic data when a recorded
source signature is unavailable. It identifies and tracks the seafloor, aligns
and stacks the water-bottom reflection on the near channel, locates the origin
of the stacked wavelet, and tests the result against the water-layer multiples
and a statistical estimate of the amplitude spectrum.

The method and its validation are described in the accompanying paper, which
is under submission (see [Citation](#citation)). This repository holds the
workflow, the validation scripts and the record of the runs behind the paper.

## What it returns, and what it does not

SEAWEED returns the **effective wavelet** `w = S·Gs·Gr`: the source signature
with its source and receiver ghosts, as seen at near offset by the streamer. It
does not deghost; recovering the bare signature is a separate, ill-conditioned
step that SEAWEED does not attempt.

Alongside the wavelet it reports physical and statistical checks:

- the first and second water-layer multiples, found by matched filters with
  templates `-w` and `+w`; their lag gives a second water-depth estimate, and
  the amplitude ratios give two estimates of the seafloor reflection
  coefficient `R_wb`;
- the correlation of the wavelet's amplitude spectrum with a statistical
  estimate from below-seafloor windows;
- stack coherence, alignment residual, the stability of the origin across
  partial stacks, and the separation between the direct arrival and the
  primary.

These checks are conditional and partial. The multiple checks assume the right
event was identified; the statistical spectrum constrains amplitude, not
phase; stack coherence measures consistency, not accuracy. None of them is a
certificate of waveform accuracy on a line without a recorded signature. The
method returns a waveform, but the phase tests of the paper establish no
advantage over a minimum-phase wavelet with the same amplitude spectrum.

## The workflow

The final recipe of the paper is run by one command,
`python -m seaweed.seafloor_pipeline`, which chains the steps below. Every
threshold is fixed, or scaled to the shot spacing of the line; nothing is
tuned per survey.

1. **First pass.** `run_real` reads the SEG-Y once, keeps it as a cube for the
   next steps, and measures the sampling interval, the offsets and the water
   velocity (semblance on the primary, with an independent estimate from the
   direct arrival).
2. **Seafloor anchors** (`anchors_from_consensus`, `pick_consensus`). The line
   is divided into groups of about 2500 m (at least 10 shots, at least four
   groups). In each group the autocorrelation and a water-velocity semblance
   scan propose candidates for the two-way water time `t0`; candidates
   arriving at the near trace before the end of the direct arrival are
   dropped. Each remaining candidate is scored by semblance along the water
   moveout with a scanned apparent dip, by the first multiple at `2 t0` at the
   same dip and by its polarity. A group's best candidate becomes an anchor if
   its semblance is at least 0.35 and it lies within 120 ms of the running
   median of its neighbours.
3. **Stage-1 tracking.** Pilot cross-correlation over 500 m of adjacent shots
   follows the seafloor between the anchors; the drift relative to the
   anchors is interpolated and removed.
4. **Onset tracking** (`anchor_qc`, `h3_track`). The dominant period `T_dom`
   is measured on the line. Dense anchors, one every 250 m, are taken from the
   onset of the near trace within ±2.5 `T_dom` of the stage-1 curve; the
   cross-correlation only carries the seafloor between them, and the onset
   refines each shot where it is reliable. Where the onsets cannot be
   measured, the stage-1 tracking of that stretch is kept and flagged. The
   operating domain — a separation from the direct arrival of at least
   max(5 `T_dom`, end of the direct) — is reported, not enforced.
5. **Extraction** (`run_real --recipe final`). A window on the near channel at
   the reflected travel time of each shot is aligned in two stages (groups of
   adjacent shots shifted together within ±40 ms, then each window within
   ±`T_dom`/2 of its group) and stacked with coherence weights. The origin is
   found by walking back from the envelope peak of the stack until it drops
   below four times the pre-arrival noise or 0.08 of the peak. Partial stacks
   of 50 shots test whether that origin is stable; an unstable one is
   flagged, not corrected.

## Installation

Python 3.10 or later.

```
git clone https://github.com/ogspros/SEAWEED.git
cd SEAWEED
pip install .              # or  pip install -e .  to edit the code in place
```

This installs the workflow and its four dependencies: numpy, scipy,
matplotlib and segyio. The results of the paper were computed with Python
3.10.20, numpy 2.2.5, scipy 1.15.3 and matplotlib 3.10.9; the full record is in
[`tests/reference_final_env.txt`](tests/reference_final_env.txt).

If the numerical libraries of an existing environment must not change, install
without touching them:

```
pip install --no-deps -e .
```

Nothing has to be installed to run from the repository root: `python -m
seaweed.seafloor_pipeline` finds the package there, and the shell drivers in
`validation/` set `PYTHONPATH` themselves.

The scripts in `validation/synthetic/` additionally need PyYAML and
[Devito](https://www.devitoproject.org), including the `examples` package that
ships with it (`pip install ".[validation]"` installs both from PyPI).

## Running it on your data

### What the input must be

- **A SEG-Y file of marine shot gathers.** Each trace carries its field
  record number (trace header bytes 9–12, `FieldRecord`) and its
  source-receiver offset (bytes 37–40, `offset`); the sampling interval is
  read from the binary header. A shot is the set of traces sharing a field
  record; within it the traces are sorted by |offset|, and a shot with a
  different number of traces from the first one is skipped. Shots are taken
  in the order of their field-record numbers.
- **The near offset** (`--near`, m): read from the headers if omitted.
- **The channel spacing** (`--dx`, m).
- **The shot spacing** (`--dxsp`, m): it sets the group size of the stage-1
  anchors (about 2.5 km of line, at least 10 shots and at least four groups)
  and the tracking window (about 500 m of shots, at least 5).
- **The number of shots to load** (`--nshot`), from the first field record or
  from `--shot0`.

The data need no preprocessing: no gain, no filter, no NMO. The workflow
uses the nearest channels only (the near channel for the wavelet, the three
nearest for tracking, more for the velocity scans).

### The command, with the defaults

```
python -m seaweed.seafloor_pipeline line_out --segy line.sgy --near 125 --dx 25 \
    --nshot 1000 --dxsp 25
```

`line_out` must not exist. The run takes from minutes to about an hour,
depending on the number of shots; every step writes its log there.

### What it writes

| file | content |
|---|---|
| `full_wavelet.npy` | the effective wavelet of the whole line: 200 ms at the sampling interval of the data, peak-normalised |
| `full_wavelet_multiple.npy` | the wavelet extracted from the first multiple, with the sign it was measured with (absent when the multiple cannot be extracted) |
| `full.png` | the quality-control figure of the extraction, on the final (onset) tracking: see below |
| `stage1_full_line.png` | the same figure for the extraction on the provisional stage-1 tracking, for comparison |
| `full.log` | every number of the extraction and of its checks |
| `full_sel.npz` | the shots used, per-window weights, stack coherence, alignment residual, water depth and velocity, the bistability check |
| `track.npz` | the seafloor time of every shot, with the parameters that made it; `--load_track` reuses it |
| `stage1_anchors.npz`, `stage1_track.npz` | the stage-1 anchors (rejected groups included) and tracking |
| `qc.npz` | the onset QC of the stage-1 tracking, with the dominant period |
| `summary.json` | the flags of the run |

### How to read it

Read `summary.json` and the last lines of the console output first: every
flag raised is there.

| flag | where | meaning | what to do |
|---|---|---|---|
| separation margin | `full.log`, step [3] | the water depth against D_min, the depth at which direct arrival and primary are one wavelet length apart at the near offset | below D_min the extraction still runs; check the windows on the near-offset section |
| operating domain | `summary.json` → `domain.flag` | "in the domain", "partially outside" (100-shot blocks listed in `blocks_below`) or "outside"; the separation between direct arrival and seafloor against max(5 T_dom, end of the direct) | outside or partially outside: the seafloor may sit inside the direct arrival; do not use those shots without inspection |
| onset tracking applicable | `h3_applicable`, `h3_fallback`, `h3_fallback_reason`, `h3_unreliable_fraction` | where the onsets cannot be measured, stretches (or the whole line) use the stage-1 tracking | a fallback stretch is less precise; compare the tracked section with the data there |
| stage-1 fallback | `stage1_fallback`, `stage1_n_kept` | fewer than four valid anchor groups: "reduced" (2–3 groups) or "single group" (constant drift, seafloor slope ignored) | check the tracking on a dipping line |
| bistability | `bistable`; `full.log`, "partial stacks of 50 shots" | the origins of partial stacks of 50 shots disagree with the full stack: the stack holds two states one cycle apart | inspect the aligned windows; extract on stretches separately |
| multiple available | `multiple_available`, `multiple_reason` | the record is too short for the multiple windows, or the matched filter failed | the multiple checks (second depth, R_wb) are missing; the wavelet is still saved |

The checks of `full.log` then say how far the result is consistent:

- **depth from the multiple**: should agree with the tracked depth within a
  few per cent; at the search boundary it is a bound;
- **R_wb** from the primary and first multiple, and from the first and second
  multiples: their discrepancy tests the amplitude assumptions; a value above
  one breaks them;
- **statistical spectrum**: the correlation of the wavelet's amplitude
  spectrum with that of the below-seafloor windows, over 6–90 and 20–90 Hz;
- **stack coherence and alignment residual**.

None of them certifies the waveform. Look at the quality-control figures.
`full.png` is drawn on the final tracking, `stage1_full_line.png` on the
provisional stage-1 tracking; comparing the two shows where the onset tracking
moved the seafloor. Their panels:

- (a) the near trace with the tracked seafloor and the first multiple at 2 t0;
  in `full.png` also the onset anchors (circles), the stage-1 anchors used
  where the onset was unreliable (crosses), the stage-1 fallback stretches
  (shaded), and in the title the operating domain and the result of the
  partial-stack check;
- (b) the tracked water depth, shot by shot and as a running mean;
- (c) the first 40 aligned windows;
- (d) the coherence of each window with the pilot, its weight in the stack;
- (e) the wavelets from the primary and from the first multiple;
- (f) their amplitude spectra and the statistical estimate.

Then compare extractions from different stretches of the line, band by band:

```
python -m seaweed.run_real --segy line.sgy --near 125 --dx 25 --nshot 1000 \
    --load_cube line_out/cube.npy --load_track line_out/track.npz \
    --recipe final --dxsp 25 --sp_first 0 --sp_last 1000 --sp_step 10 \
    --out line_comb0
```

Shots spread along the line (`--sp_step`) give a more stable wavelet than a
contiguous block of the same size.

### Options

`python -m seaweed.seafloor_pipeline --help` and `python -m seaweed.run_real
--help` list all of them.

| option | default | when to change it |
|---|---|---|
| `--near`, `--dx` | from the headers / required | always give them if the headers are not reliable |
| `--dxsp` | required | the shot spacing of the line |
| `--group` | about 2.5 km of shots | shorter groups on lines where the seafloor changes fast; at least four groups |
| `--nshot`, `--shot0` | required / first record | to process part of a line |
| `--n_ch_track` | 3 | the channels used by the tracking |
| `--load_cube` | — | reuse the cube written by a previous run (`cube.npy`), instead of reading the SEG-Y |
| `--first_pass`, `--load_anchors` | — | reuse the first pass or the stage-1 anchors of a previous run |
| `--sp_first`, `--sp_last`, `--sp_step` (`run_real`) | whole line | the shots of an extraction; `--sp_last` is exclusive |
| `--ref`, `--ref_dt` (`run_real`) | — | a known wavelet to compare against |
| `--wl_ms` (`run_real`) | 200 | the wavelet length |

Flags that `seafloor_pipeline` does not know are handed to every `run_real`
call.

To hand the wavelet to other software:

```
python -m seaweed.wavelet_to_segy line_out/full_wavelet.npy line_wavelet.sgy --dt 0.002
```

### A complete example: Viking Graben line 12

The Mobil AVO Viking Graben line 12 is distributed freely by the SEG, with
the far-field signature of its source (Keys and Foster, 1998):
<https://wiki.seg.org/wiki/Mobil_AVO_viking_graben_line_12>. Download the
shot gathers (SEG-Y) into a directory, here `viking/seismic.sgy`. The line has
1001 shots 25 m apart, 120 channels 25 m apart, a 262-m near offset and 4-ms
sampling.

```
python -m seaweed.seafloor_pipeline viking_out --segy viking/seismic.sgy --near 262 \
    --dx 25 --nshot 1012 --dxsp 25
```

`--nshot 1012` is the value of the paper, which loads the 1001 shots of the
line.

The run ends with lines like

```
onset tracking applicable; fallback on 0.0% of the line
operating domain: in the domain (separation from the direct 319 ms median, ...)
```

and `viking_out/full_wavelet.npy` is the wavelet. With the cube and first pass
of the paper the ten whole-line combs agree with the recorded signature at a
mean NCC of 0.826 ± 0.005 over 8–65 Hz; `validation/real/` runs that
comparison (`ncc.sh`), and `validation/viking/viking_phase_test.py
FarField.dat DIR --fam C --maxlag_ms 30` compares any set of extractions with
the signature.

## Repository layout

```
seaweed/                  the workflow
validation/
    synthetic/            the Devito generator, the synthetic matrices, their tables and Fig. 2
    viking/               the Viking Graben runs against the recorded signature, Figs. 3 and 4
    archive/              MS-29, IT90-65 and SISOC-MEPA, Figs. 5 and 6, the span experiment
    real/                 the four real lines with the final recipe: every real-line number and figure
docs/
    RUNS.md               which script produces which number of the paper
    experiments/          how the final recipe was developed: protocols, criteria, outcomes, decisions
    LAYOUT.md             where each script of the original working directory went
logs/                     the logs of the runs behind the first version of the paper, before the final recipe
tests/                    the checks that a change to the code leaves the results unchanged
```

## Reproducing the paper

The data are not in the repository.

- **Viking Graben line 12** and its recorded far-field signature are publicly
  distributed (Keys and Foster, 1998).
- **MS-29** and **SISOC-MEPA** raw data are available through
  <https://snap.ogs.it>, with processed data at
  <https://data.mendeley.com/datasets/xgv9nd49xg>.
- **IT90-65** is available through the Seismic Data Library System,
  <https://sdls.ogs.it>.

[`docs/RUNS.md`](docs/RUNS.md) names, for every number of the paper, the
script that produces it. The real lines are run by
[`validation/real/`](validation/real/README.md), the synthetic matrices by
`validation/synthetic/run_matrix_v7.py --recipe final`.

[`docs/experiments/`](docs/experiments/README.md) records how the final recipe
was reached: the protocols, the criteria fixed before each run, the outcomes,
and the four times a rule fixed before a run was set aside or changed after a
result had been seen (`rule_overrides.md`): two overrides of a decision rule
and two changes to a criterion or a flag rule. The internal working record it
summarises, in Italian and with the paths of the machine it was run on, is
available from the authors on request.

`tests/` holds two frozen references, each checked byte for byte on the
Viking Graben data (see [`tests/README.md`](tests/README.md)):

- `RECIPE=final bash tests/smoke_test.sh` runs the final recipe — the
  pipeline above, then ten whole-line combs with `--recipe final` — and
  compares the onset tracking and the ten wavelets with
  `tests/reference_final/`;
- `bash tests/smoke_test.sh` re-runs the ten combs of the first version
  (`--recipe paper`, the defaults of the paper's first version) and compares
  them with `tests/reference/`.

`bash tests/check_static.sh` and `python3 tests/test_h3.py` need no data.

## Citation

If you use SEAWEED, please cite the paper:

> Brancatelli G., Busetti M., Forlin E., Geletti R., Lipari V., Pignelli R.
> SEAWEED: an open-source workflow for automated extraction and diagnostics of
> effective marine seismic wavelets. *Bulletin of Geophysics and
> Oceanography*, submitted.

and the archived version of the software you used
([`CITATION.cff`](CITATION.cff)), archived on Zenodo with DOI
[10.5281/zenodo.23780745](https://doi.org/10.5281/zenodo.23780745).

## Authors

The SEAWEED authors, who hold the copyright of this software:

- Giuseppe Brancatelli, National Institute of Oceanography and Applied
  Geophysics – OGS, Trieste, Italy (corresponding author,
  gbrancatelli@ogs.it)
- Martina Busetti, OGS
- Edy Forlin, OGS
- Riccardo Geletti, OGS
- Vincenzo Lipari, OGS
- Rogerio Pignelli, OGS and University of Trieste

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
