#!/usr/bin/env python3
"""The calibrated p-value of rule R14, shared by calib_holm.py and the paper's generator.
Nothing here simulates; it reads the null distributions calib_holm.py stored.

For a recorded p-value p of a test in scenario g, the Monte Carlo p-value under configuration c is
(1 + #{P* <= p}) / (N + 1), with P* the test's p-value in N null replications of c on g's own
blocks: exactly valid when the data follow c.  R14's calibrated p-value is the largest of these over
a set of configurations, plus beta (zero for Section V's rule; 0.001 for rule R12, whose set is
chosen by a bound on its seed share), and never below p itself.  The null distributions are stored
as counts on a grid, and a p-value is counted at the first grid point above it by more than 1e-12,
so ties and rounding count against the verdict.
"""
import hashlib, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NULL, SUMMARY = 'calib_holm_null.npz', 'calib_holm.json'
TOL = 1e-12
STATS = ('max', 'joint', 'boot')


def grid():
    """The thresholds: steps of 1e-5 up to 0.01, of 1e-4 up to 0.1 and of 1e-3 up to 1."""
    return np.array([k / 100000 for k in range(1, 1001)] + [k / 10000 for k in range(101, 1001)]
                    + [k / 1000 for k in range(101, 1001)])


def counts(values, g):
    """#{v <= g_i} at every grid point, from an array of null p-values (inf allowed)."""
    v = np.sort(np.asarray(values, dtype=float))
    return np.searchsorted(v, g, side='right').astype(np.int64)


def index(g, p):
    """The grid points at which p-values are counted: the first above each by more than TOL."""
    return np.searchsorted(g, np.asarray(p, dtype=float) + TOL, side='left')


def mc_p(cnt, g, n, p):
    """Monte Carlo p-values at p (scalar or array) from one configuration's counts; 1 for any p
    at or above the last grid point, for inf and for nan."""
    p = np.asarray(p, dtype=float)
    i = index(g, p)
    ok = (i < g.size) & (p < 1)
    out = np.ones(p.shape)
    out[ok] = np.minimum(1.0, (1 + cnt[i[ok]]) / (n + 1))
    return out


def calibrated_from(cnts, g, n, p, beta=0.0):
    """R14's calibrated p-value from the counts of a set of configurations: the largest Monte
    Carlo p-value, plus beta, never below p and never above 1."""
    p = np.asarray(p, dtype=float)
    q = np.max([mc_p(c, g, n, p) for c in cnts], axis=0)
    return np.minimum(1.0, np.maximum(np.where(np.isnan(p), 1.0, p), q + beta))


class Null:
    """The stored null distributions, checked against the summary that records their hash."""

    def __init__(self, here=HERE):
        path = os.path.join(here, NULL)
        S = json.load(open(os.path.join(here, SUMMARY)))
        h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        assert h == S['null_npz_sha256'], f"{NULL} is not the file {SUMMARY} records"
        z = np.load(path)
        self.geoms = [str(x) for x in z['geoms']]
        self.configs = [str(x) for x in z['configs']]
        self.grid = z['grid']
        assert np.array_equal(self.grid, grid()), "the stored grid is not calib_cal.grid()"
        self.n = int(z['n_cal'])
        self.cnt = {s: z[f'counts_{s}'] for s in STATS}
        self.summary = S

    def calibrated(self, geom, p, stat='max', configs=None, beta=0.0):
        """R14's calibrated p-value of p in scenario `geom`."""
        gi = self.geoms.index(geom)
        cs = self.configs if configs is None else configs
        cnts = [self.cnt[stat][gi, self.configs.index(c)] for c in cs]
        return float(calibrated_from(cnts, self.grid, self.n, p, beta))
