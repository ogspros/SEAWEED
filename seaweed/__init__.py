"""SEAWEED: SEAfloor WavelEt Extraction and Diagnostics.

An automated workflow, with fixed default parameters, that extracts the
effective source wavelet -- source signature with source and receiver ghosts
-- from the water-bottom reflection of marine seismic data, and tests it
against the water-layer multiples and a statistical amplitude spectrum.

Command-line entry points, each run as ``python -m seaweed.<name>``:

    run_real                 SEG-Y to wavelet: tracking, extraction and checks
    anchors_from_consensus   stage-1 seafloor anchors from the consensus picker
    pick_consensus           the consensus picker itself
    pick_seafloor            line-wide autocorrelation picker
    wavelet_to_segy          write an extracted wavelet as a SEG-Y file

Library modules:

    wavelet_extractor_v2     tracking, alignment, stacking, origin, validators
    seafloor_robust          semblance along the water moveout
    wavelet_extractor_qc     onset detection used by the semblance

Importing the package does not import any of them, so ``import seaweed`` is
cheap and has no side effects.
"""

__version__ = "1.0.0"


def code_version():
    """The package version, with the git commit when run from a clone
    ("1.0.0+5b48df2", "-dirty" if the tree has uncommitted changes).  Written
    into the anchors and tracking files, so that each says which code made it."""
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "describe", "--always", "--dirty"],
                           cwd=here, capture_output=True, text=True,
                           timeout=5).stdout.strip()
    except Exception:
        h = ""
    return __version__ + ("+" + h if h else "")
