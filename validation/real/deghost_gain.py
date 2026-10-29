#!/usr/bin/env python3
"""Sect. 4.3: how much the earlier MS-29 deghost operator corrects over the fit
band, measured and modelled, with the functions of fig_deghost.py unchanged.

  measured  max / min of the ratio of the section spectra after and before
            deghosting, 8-45 Hz (the 7.1 dB printed by fig_deghost.py)
  modelled  max / min of the fitted damped inverse ghost c / sqrt(G^2 + eps^2)
            over the same band, with its maximum gain c / eps at the notch

    python validation/real/deghost_gain.py NO_DEGHOST DEGHOST
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive"))
import fig_deghost as F                                                 # noqa: E402

f, A = F.load_spectrum(sys.argv[1]); _, B = F.load_spectrum(sys.argv[2])
ratio = B / np.maximum(A, 1e-30)
fit = F.fit_operator(f, ratio, F.VW_ECHOS, *F.BAND); m = fit["mask"]
op = fit["c"] / np.sqrt(F.ghost(f[m], fit["z"], F.VW_ECHOS) ** 2 + fit["eps"] ** 2)
print("fit: c %.4f, z_r %.3f m, eps %.4f, band %g-%g Hz, %d points" % (fit["c"], fit["z"], fit["eps"], *F.BAND, m.sum()))
print("measured ratio: max %.4f at %.2f Hz, min %.4f at %.2f Hz, variation %.2f dB"
      % (ratio[m].max(), f[m][ratio[m].argmax()], ratio[m].min(), f[m][ratio[m].argmin()],
         20 * np.log10(ratio[m].max() / ratio[m].min())))
print("modelled response: max %.4f at %.2f Hz, min %.4f at %.2f Hz, variation %.2f dB"
      % (op.max(), f[m][op.argmax()], op.min(), f[m][op.argmin()], 20 * np.log10(op.max() / op.min())))
