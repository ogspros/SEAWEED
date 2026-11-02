# Known limits

Limits found during the work and declared rather than corrected. Each is
stated where the paper discusses the corresponding step.

- **Seafloor close to the end of a long direct arrival.** A seafloor within a
  few tens of milliseconds of the end of the direct is not recognised by the
  operating-domain tests: the three separations are measured after the end
  of the direct, and the autocorrelation test cannot separate a seafloor that
  only touches it (synthetic records at 220 m, 3 ms inside the direct, and
  one whose autocorrelation locked on a later lobe).
- **The operating domain is calibrated on short direct arrivals.** s* = 5 T_dom
  comes from synthetic records whose direct lasts about 1.4 T_dom, and rests on
  a class of ten records; the real direct of Viking and IT90-65 lasts 9–11
  T_dom, which is why the end of the direct is a second condition.
- **The domain flags shallow water, not slope or geometry.** Most synthetic
  records where the onset tracking is not applicable lie well inside the
  domain, on the slope and geometry axes.
- **Synthetic records on a steep slope: the seafloor outside the search
  interval.** On the 10-degree dip axis the record length is set from the
  nominal depth of the model (400 m, 2.19 s), but under the shots the seafloor
  lies at 1041-1098 m (t0 1.39-1.46 s).  The picker searches up to the record
  length divided by 2.2 (0.99 s), so that the multiple stays inside the record:
  the true seafloor is never a candidate, and the stage-1 anchors fall about
  1.3 s early, in the coda of the direct arrival.  The run flags these records
  (onset tracking not applicable, operating domain "outside"), but it does not
  say that the search interval cannot contain the seafloor.
- **The end of the direct can be measured early.** On those records the
  direct arrival peaks at 92 ms at the near trace and its coda lasts to about
  280 ms, but its end is measured at 122 ms: the envelope of the stacked near
  traces falls below the threshold between two bursts of the coda.  Candidates
  in the later coda are then not discarded.
- **Below the domain the fallback is also wrong.** Where the onset tracking is
  not applicable in shallow water, the stage-1 tracking it falls back on is on
  the right cycle on about a quarter of the records.
- **Shallow-water chain.** On the 400 records of the shallow-water matrix the
  onset tracking is not applicable on 202 (50.5 %); those records are extracted
  on the stage-1 tracking and flagged.
- **Stage-1 anchors on the direct arrival.** Groups whose best candidate is
  the direct arrival, and agree with each other, are not removed by the
  neighbour check; the end-of-direct rejection is what removes them, and it
  is inactive where the end of the direct cannot be measured.
- **Multiple cycle at coarse sampling.** At 4-ms sampling the matched filter of
  the first multiple can lock one cycle late, in the stage-1 and in the onset
  tracking alike; R_wb on Viking then measures that lock, not the tracking.
- **The bistability check is verified only on real lines.** A synthetic record
  of 20 shots has a single block of 50, so the check never triggers there.
- **The multiple checks are not calibrated as failure detectors.** No
  multiple-absent, misplaced-window or noise-only controls were run.

## Known issues of 1.0.0

Found after the release, on an external line and on synthetic records built
to reproduce it. Not corrected in 1.0.0.

- **Stage 1 on a source coda, with a varying seafloor.** Where the source
  signature has a long periodic coda, flat in time along the line, and the
  seafloor varies by hundreds of milliseconds within a stage-1 group, the
  consensus picker can lock on the coda. The coda arrives after the measured
  end of the direct, so it is not discarded; its periodicity mimics the
  water-layer multiple; and its semblance at a single t0 is higher than that
  of a seafloor that changes within the group. The neighbour check then
  rejects the few groups that are on the seafloor. *Symptom:* stage-1 anchors
  at the same time along the whole line (`group_t0` in `stage1_anchors.npz`),
  and `domain.flag` "outside the domain, T_dom not measurable". *Remedy*
  (assisted processing): pick the anchors with
  `python -m seaweed.pick_seafloor --lag T0MIN T0MAX ... --save_anchors anchors.npz`,
  with the lag range read off the near-offset section (and `--group_m`
  shorter than the default 800 m where the seafloor varies fast), then run the
  pipeline with `--load_anchors anchors.npz`.
- **Alignment on a very rough seafloor.** Where t0 varies by more than
  T_dom/2 across the adjacent shots of a tracking group, the two-stage
  alignment does not follow it: the stack coherence is low and the
  bistability check is often raised, while on a regular stretch of the same
  line the extraction is clean. *Remedy:* extract on stretches where the
  seafloor is regular (`run_real --sp_first --sp_last`, as in the README)
  and compare them.
- **The onset constant depends on the noise.** The onset tracking applies
  a fixed convention constant (`CONVENTION_MS` in `h3_track`), calibrated on
  synthetic records; the threshold of the onset rule rises with the noise, so
  on noisier records the onset is measured later. On a flat synthetic control
  the final tracking moves from −1.8 ms against the truth without noise to
  +9.8 ms with noise at 0.044 of the direct-arrival amplitude. *Expected
  effect:* a constant offset of the tracking, not a change of its shape; not
  yet verified on real lines.
- **False bistability on a short last block.** The bistability check
  compares partial stacks of 50 shots; the last block of a line can be much
  shorter, and its origin is less reliable. On a flat, noise-free synthetic
  record with a correct tracking, the last block (18 shots) was 74 samples
  off the others: one block of six is already above the 10 % limit, and
  `bistable` is true. If only the last block is off (`full.log`, "partial
  stacks of 50 shots"), the flag does not mean that the stack holds two
  states.
