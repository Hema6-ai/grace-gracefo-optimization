"""
utils.py
Utility functions: nondominated_sort and crowding distance used by NSGA-like algorithm.
"""

import numpy as np

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def nondominated_sort(objs):
    # objs: (N, M)
    N = objs.shape[0]
    S = [[] for _ in range(N)]
    n = [0]*N
    rank = [0]*N
    fronts = []
    for p in range(N):
        S[p] = []
        n[p] = 0
        for q in range(N):
            if p==q: continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if len(fronts)==0:
                fronts.append([])
            fronts[0].append(p)
    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i+1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts

def crowding_distance(objs):
    # objs shape (k, M)
    k, m = objs.shape
    if k==0:
        return np.array([])
    dist = np.zeros(k)
    for j in range(m):
        order = np.argsort(objs[:,j])
        sorted_vals = objs[order,j]
        dist[order[0]] = dist[order[-1]] = np.inf
        denom = sorted_vals[-1] - sorted_vals[0]
        if denom == 0:
            continue
        for i in range(1,k-1):
            dist[order[i]] += (sorted_vals[i+1] - sorted_vals[i-1]) / denom
    # replace inf with big number for sorting
    dist[np.isinf(dist)] = 1e9
    return dist
