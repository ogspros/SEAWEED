# Changelog

## Unreleased

- Documentation only: the known issues of 1.0.0, in
  `docs/experiments/known_limits.md` and below; the README points to them.

- Docstrings only: the package docstring says "default parameters", as the
  README, and the usage line of `seafloor_pipeline` puts `OUTDIR` first, as
  the README examples (with it last, the value of `--near` was taken as the
  output directory).

## 1.0.0

First public release: the final recipe of the paper, run by
`python -m seaweed.seafloor_pipeline`. The earlier defaults remain available
as `run_real --recipe paper`. `seaweed/` is functionally identical to the
internal tag `method-final`: only comments, docstrings, printed messages and
the quality-control figure differ, and the trackings and wavelets are the same,
byte for byte.

### Known issues

Found after the release; described in `docs/experiments/known_limits.md`
("Known issues of 1.0.0"):

- Stage 1 can lock on a long periodic source coda where the seafloor varies
  within a group: anchors at the same time along the whole line, "T_dom not
  measurable". Remedy: anchors from `pick_seafloor --lag` with
  `--load_anchors` (assisted processing).
- On a very rough seafloor the alignment does not follow it (low coherence,
  bistability). Remedy: extract on regular stretches.
- The convention constant of the onset tracking does not depend on the noise,
  while the measured onset does: expected effect, a constant offset of the
  tracking (to be verified on real lines).
- A short last block can raise a false bistability flag.
