"""
objectives.py
Contains simplified spatial/temporal objective computations used by the GA.

The real paper computes coverage using accurate groundtracks and GEODYN
simulations. For a lightweight demo we:
- simulate ground-track sampling per pair based on inclination/RAAN/M
- map samples to a 3° equal-area grid of m=4551 cells (approximate) by hashing lat/lon to cell index
- compute Job, Jro (Gini), Bew/Bns metrics in simplified form
- compute Jto by binning times into 16 bins per day and computing temporal Gini per cell
"""

import numpy as np
from scipy.stats import gini as _gini if False else None  # (scipy has no gini in all versions)
from math import sin, cos, radians
from collections import defaultdict

RNG = np.random.default_rng(123)

# helper: implement Gini for vector
def gini(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return 1.0
    a = a.flatten()
    if np.all(a==0):
        return 0.0
    a = np.sort(a)
    n = a.size
    index = np.arange(1, n+1)
    return (2.0*np.sum(index * a) - (n+1)*np.sum(a)) / (n * np.sum(a))

# map lat/lon to simple cell id - coarse grid with 3° bins ~ 4551 cells
def latlon_to_cell(lat, lon):
    # lat in [-90,90], lon in [-180,180)
    lat_bin = int((lat + 90)//3)  # 60 bins
    lon_bin = int((lon + 180)//3)  # 120 bins -> 7200, but many empty near poles; close enough
    return lat_bin * 120 + lon_bin

def sample_groundtrack(pair, n_samples=500):
    # pair: (i, raan, m)
    i, raan, m = pair
    # simulate ground track lat, lon sample points by mixing inclination and RAAN
    # produce lat in [-90,90] and lon in [-180,180]
    n = n_samples
    phase = RNG.uniform(0,360)
    # produce elliptical-ish coverage with lat amplitude ~ inclination
    lats = (np.sin(np.linspace(0,2*np.pi,n) + radians(phase)) * (i/180*90)).tolist()
    lons = (np.linspace(-180,180,n) + raan + m) % 360 - 180
    return list(zip(lats, lons))

def evaluate_constellation(pairs, n_pairs=6, samples_per_pair=500, temporal_days=29):
    # pairs shape (n_pairs,3)
    m_cells = 4551
    # coverage: for each sample map to cell, count visits and time bins
    cell_visit_counts = defaultdict(int)
    cell_time_vectors = defaultdict(lambda: np.zeros(16*temporal_days, dtype=int))
    # for simplified temporal sampling produce times across the month per sample
    for p_idx,p in enumerate(pairs):
        samples = sample_groundtrack(p, n_samples=samples_per_pair)
        for s_idx,(lat,lon) in enumerate(samples):
            cell = latlon_to_cell(lat, lon)
            cell_visit_counts[cell] += 1
            # temporal: assign sample to a random day and time bin to emulate revisit
            day = RNG.integers(0, temporal_days)
            bin16 = RNG.integers(0,16)
            cell_time_vectors[cell][day*16 + bin16] = 1
    # Job: fraction of cells observed
    observed_cells = [k for k,v in cell_visit_counts.items() if v>0]
    N = len(observed_cells)
    Job = 1.0 - (N / m_cells)
    # Jro: Gini of revisit counts for observed cells
    revisits = np.array([cell_visit_counts[c] for c in observed_cells])
    Jro = gini(revisits) if revisits.size>0 else 1.0
    # Directionality proxies Bew and Bns: use variance of local direction samples as proxy:
    # here we just estimate a per-cell 'east-west' quality by how often samples cross longitudes
    B_ew_vals = []
    B_ns_vals = []
    for c in observed_cells:
        # randomly construct two values based on counts to simulate directional quality
        v = cell_visit_counts[c]
        B_ew_vals.append(1.0/(1+v) + RNG.normal(0,0.02))
        B_ns_vals.append(1.0/(1+v*0.5) + RNG.normal(0,0.02))
    # make positive
    B_ew_vals = np.maximum(0, np.array(B_ew_vals))
    B_ns_vals = np.maximum(0, np.array(B_ns_vals))
    # J_ew & J_ns defined per eqn: mean energy + normalized Gini of B values
    mean_ew = np.mean(B_ew_vals) if B_ew_vals.size>0 else 1.0
    mean_ns = np.mean(B_ns_vals) if B_ns_vals.size>0 else 1.0
    gini_ew = gini(B_ew_vals) if B_ew_vals.size>0 else 1.0
    gini_ns = gini(B_ns_vals) if B_ns_vals.size>0 else 1.0
    Jew = mean_ew**2 + 0.5 * gini_ew
    Jns = mean_ns**2 + 0.5 * gini_ns
    # combine with weights as in paper: Wob=100, Wro=1, Wew=1, Wns=10
    Wob, Wro, Wew, Wns = 100.0, 1.0, 1.0, 10.0
    Jso = Wob*Job + Wro*Jro + Wew*Jew + Wns*Jns
    # temporal objective: use cell_time_vectors -> per cell compute Gini of temporal vector sums
    TVs = [v.reshape(-1) for v in cell_time_vectors.values()]
    if len(TVs)==0:
        Gtv = 1.0
    else:
        sums = np.array([tv.sum() for tv in TVs])
        # Gini across temporal visitation totals
        Gtv = gini(sums)
    # Jto similar to eqn10: Gtv^2 + 0.5 * normalized gini of per-cell Gtv (here single number -> use 0)
    Jto = Gtv**2
    return float(Jso), float(Jto)
