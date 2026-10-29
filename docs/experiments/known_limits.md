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
