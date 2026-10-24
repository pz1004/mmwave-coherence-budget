#!/usr/bin/env python3
"""csirs.py's one-sided lower bound on an efficiency, with the level and the scan window as
arguments.  Declared in protocol.json (analysis 'csirs_cfg_refbounds').

csirs.lower_bound fixes its level at csirs.ALPHA_LB = 0.05/7 and scans [eta - 0.25, eta] for the
single pass-to-fail transition of the crossed bootstrap's test.  csirs_cfg.py and refbounds.py need
the same bound at other settings: the configured CSI-RS sweep, whose efficiency can fall further
below its bound than 0.25 allows, and the fast references, whose bounds hold together over seven
scenarios and four references (0.05/28).  The two functions below are csirs.py's passes_boot and
lower_bound (csirs.py:201-234) with alpha and window as arguments; they call csirs.test and the t
quantile exactly as csirs.py does, so at alpha = 0.05/7 and window = 0.25 they return csirs.json's
bounds bit for bit (gate H2 of refbounds.py, G2 of csirs_cfg.py).  Once the scan brackets the
single transition, bisection finds it to 1e-6 whatever the window.
"""
import numpy as np
from scipy import stats

import csirs

SCAN = csirs.SCAN


def passes_boot(r, alpha):
    """At most alpha of the bootstrap replicates of the contrast at or below zero."""
    return r['diff'] > 0 and r['ci95'][0] > 0 and r['p'] / 2 <= alpha


def lower_bound(selk, gse, blocks, nb, eta, se_lin, tag, alpha, window):
    """min(e_boot, e_t), both one-sided at alpha: e_boot the largest e at which the crossed
    bootstrap of mean(se_selected - e se_genie) puts at most alpha of its replicates at or below
    zero, by a SCAN-point scan of [eta - window, eta] that must show one transition and bisection
    to 1e-6, on csirs.test's stream for `tag`; e_t = eta - t_(S-1)(1 - alpha) se_lin."""
    lo = max(0.0, eta - window)
    grid = np.linspace(lo, eta, SCAN)
    ok = [passes_boot(csirs.test(selk, gse, blocks, nb, e, tag), alpha) for e in grid]
    assert ok[0], (tag, "the scan's lowest point does not pass; widen the window")
    flips = sum(a != b for a, b in zip(ok, ok[1:]))
    assert flips <= 1, (tag, "more than one pass-to-fail transition", ok)
    if all(ok):
        e_b = float(grid[-1])
    else:
        i = ok.index(False)
        a, b = float(grid[i - 1]), float(grid[i])
        while b - a > 1e-6:
            m = 0.5 * (a + b)
            a, b = (m, b) if passes_boot(csirs.test(selk, gse, blocks, nb, m, tag), alpha) \
                else (a, m)
        e_b = a
    e_t = eta - float(stats.t.ppf(1 - alpha, selk.shape[0] - 1)) * se_lin
    return min(e_b, e_t), e_b, e_t, flips
