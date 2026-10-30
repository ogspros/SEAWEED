# Changelog

## Unreleased

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
