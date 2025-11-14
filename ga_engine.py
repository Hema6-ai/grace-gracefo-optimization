
---

### `ga_engine.py`
```python
"""
ga_engine.py
Simple GA for multi-objective constellation optimization.
Implements a lightweight NSGA-II-like routine:
- population encoded directly as sets of (i, RAAN, M) for each pair (6 pairs)
- computes J_so and J_to via objectives.objectives
- selection via nondominated sorting and crowding distance
- SBX-like crossover and gaussian mutation
"""

import numpy as np
from objectives import evaluate_constellation
from utils import nondominated_sort, crowding_distance

RNG = np.random.default_rng(42)

def random_individual(n_pairs=6):
    # Each pair: inclination [0,180], RAAN [0,360], M [0,360]
    # Represent as flat array: [i1, raan1, m1, i2, raan2, m2, ...]
    return np.concatenate([
        RNG.uniform(0,180,size=n_pairs),
        RNG.uniform(0,360,size=n_pairs),
        RNG.uniform(0,360,size=n_pairs)
    ])

def decode(ind):
    # returns shape (n_pairs,3)
    n = ind.size // 3
    return np.vstack([ind[:n], ind[n:2*n], ind[2*n:]]).T

def init_population(pop_size, n_pairs=6):
    return np.array([random_individual(n_pairs) for _ in range(pop_size)])

def crossover(a, b, eta=10):
    # Simple simulated binary crossover (SBX) on array a,b
    u = RNG.random(a.shape)
    beta = np.empty_like(a)
    mask = u <= 0.5
    beta[mask] = (2*u[mask])**(1/(eta+1))
    beta[~mask] = (1/(2*(1-u[~mask])))**(1/(eta+1))
    child1 = 0.5*((1+beta)*a + (1-beta)*b)
    child2 = 0.5*((1-beta)*a + (1+beta)*b)
    # enforce ranges
    child1 = enforce_bounds(child1)
    child2 = enforce_bounds(child2)
    return child1, child2

def mutate(ind, mut_rate=0.1, sigma_scale=0.05):
    # gaussian mutation scaled to variable ranges
    n = ind.size // 3
    out = ind.copy()
    # incl (0..180)
    for i in range(n):
        if RNG.random() < mut_rate:
            out[i] += RNG.normal(0, 180*sigma_scale)
    # raan (0..360)
    for i in range(n,2*n):
        if RNG.random() < mut_rate:
            out[i] += RNG.normal(0, 360*sigma_scale)
    # mean anomaly
    for i in range(2*n,3*n):
        if RNG.random() < mut_rate:
            out[i] += RNG.normal(0, 360*sigma_scale)
    return enforce_bounds(out)

def enforce_bounds(ind):
    n = ind.size // 3
    ind[:n] = np.clip(ind[:n], 0, 180)
    ind[n:2*n] = np.mod(ind[n:2*n], 360)
    ind[2*n:3*n] = np.mod(ind[2*n:3*n], 360)
    return ind

def run_nsga(pop_size=100, gens=20, n_pairs=6, crossover_rate=0.9, mut_rate=0.15, verbose=False):
    pop = init_population(pop_size, n_pairs)
    # store objectives
    objs = np.zeros((pop_size,2))
    for i,ind in enumerate(pop):
        jso,jto = evaluate_constellation(decode(ind), n_pairs=n_pairs)
        objs[i] = [jso,jto]
    history = []
    for g in range(1, gens+1):
        # create offspring
        offspring = []
        while len(offspring) < pop_size:
            # tournament selection (2)
            a = pop[RNG.integers(0,pop_size)]
            b = pop[RNG.integers(0,pop_size)]
            # crossover
            if RNG.random() < crossover_rate:
                c1,c2 = crossover(a,b)
            else:
                c1,c2 = a.copy(), b.copy()
            c1 = mutate(c1, mut_rate)
            c2 = mutate(c2, mut_rate)
            offspring.append(c1); offspring.append(c2)
        offspring = np.array(offspring[:pop_size])
        # evaluate offspring
        off_objs = np.zeros((pop_size,2))
        for i,ind in enumerate(offspring):
            jso,jto = evaluate_constellation(decode(ind), n_pairs=n_pairs)
            off_objs[i] = [jso,jto]
        # combine
        combined = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, off_objs])
        # nondominated sort
        fronts = nondominated_sort(combined_objs)
        new_pop = []
        new_objs = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(combined[idx])
                    new_objs.append(combined_objs[idx])
            else:
                # fill rest by crowding distance
                remaining = pop_size - len(new_pop)
                cd = crowding_distance(combined_objs[front])
                order = np.argsort(-cd)  # descending
                for idx_local in order[:remaining]:
                    idx = front[idx_local]
                    new_pop.append(combined[idx])
                    new_objs.append(combined_objs[idx])
                break
        pop = np.array(new_pop)
        objs = np.array(new_objs)
        history.append((g, pop.copy(), objs.copy()))
        if verbose:
            print(f"gen {g} best approx Jso {objs[:,0].min():.4f} Jto {objs[:,1].min():.4f}")
    return history
