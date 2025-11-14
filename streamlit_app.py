"""
streamlit_app.py
Streamlit dashboard to run the GA and show Figures 8-14 style outputs.
"""

import streamlit as st
import numpy as np
from ga_engine import run_nsga
from plots import plot_population, plot_pareto_curves, plot_degree_variance, plot_family_scatter
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="GRACE GA optimizer demo")

st.title("GRACE-like Constellation Multiobjective GA — Demo")
st.markdown("Interactive demo reproducing Figures 8–14 style outputs (simulated)")

# sidebar parameters
st.sidebar.header("GA parameters")
pop_size = st.sidebar.slider("Population size", 20, 300, 120, step=10)
gens = st.sidebar.slider("Generations", 5, 60, 20, step=1)
n_pairs = st.sidebar.slider("Satellite pairs (npairs)", 1, 10, 6)
samples_per_pair = st.sidebar.slider("Samples per pair (sim)", 50, 1500, 500, step=50)
mut_rate = st.sidebar.slider("Mutation rate", 0.0, 0.5, 0.15, step=0.01)
crossover_rate = st.sidebar.slider("Crossover rate", 0.0, 1.0, 0.9, step=0.01)

if st.sidebar.button("Run GA"):
    with st.spinner("Running GA (simulated)... this can take a minute"):
        history = run_nsga(pop_size=int(pop_size), gens=int(gens), n_pairs=int(n_pairs),
                           crossover_rate=float(crossover_rate), mut_rate=float(mut_rate))
    st.success("GA finished")

    # Fig 8 population scatter (gens 1,3,20)
    st.subheader("Figure 8 — Population scatter (gens 1, 3, 20)")
    fig1 = plot_population(history, gens_to_plot=(1,3,min(20,gens)))
    st.pyplot(fig1)

    # Fig 9: pick some pareto-like curves from gen1 (simulate)
    st.subheader("Figure 9 — Example Pareto curves from gen 1")
    # pick four small clusters on gen1 to mimic Pareto curves
    g1_pop = history[0][2]
    # create tiny artificial curves by sorting and grouping
    idx_sorted = np.argsort(g1_pop[:,0])
    coords_sets = []
    colors = ['gold','red','tab:blue','purple']
    for i,k in enumerate([0, int(len(idx_sorted)*0.12), int(len(idx_sorted)*0.45), int(len(idx_sorted)*0.9)]):
        slice_idx = idx_sorted[max(0,k-4):k+3]
        coords = g1_pop[slice_idx]
        coords_sets.append((coords, colors[i]))
    fig2 = plot_pareto_curves(g1_pop, coords_sets)
    st.pyplot(fig2)

    # Fig 10: degree variance curves for three pareto selections
    st.subheader("Figure 10 — Average degree variances for selected Pareto curves")
    degrees = np.arange(1,61)
    # simulate three curves (matching shapes in paper)
    curve1 = 0.01 * (1 + 0.02*(degrees**1.5)) * (1 + 0.3*np.sin(degrees/3.0))
    curve2 = 0.006 * (1 + 0.01*(degrees**1.4)) * (1 + 0.15*np.sin(degrees/2.2))
    curve3 = 0.002 * (1 + 0.008*(degrees**1.3)) * (1 + 0.05*np.sin(degrees/1.7))
    hydro = 0.02 * np.exp(-degrees/20)
    fig3 = plot_degree_variance(degrees, [curve1, curve2, curve3, hydro],
                                labels=["pareto 48","pareto 10","pareto 1","hydro. and ice"])
    st.pyplot(fig3)

    # Fig 11: family of ten constellations scatter
    st.subheader("Figure 11 — Family of ten six-pair constellations (Pareto front)")
    # create 10 sample objects around small range
    c_objs = np.array([[0.0385 + 0.0003*i, 0.447 + 0.001*(-1)**i * i*0.002] for i in range(10)])
    fig4 = plot_family_scatter(c_objs)
    st.pyplot(fig4)

    # Fig 12: 1-day average degree variances for c01-c10
    st.subheader("Figure 12 — 1-day average degree variances c01–c10")
    curves = []
    for i in range(10):
        base = 0.0025*(1 + 0.008*(degrees**1.2))*(1 + 0.1*np.sin(degrees/(3+i*0.1)))
        curves.append(base * (1 + 0.02* (i/10)))
    fig5 = plot_degree_variance(degrees, curves + [hydro], labels=[f"c{i+1:02d}" for i in range(10)] + ["hydro. and ice"])
    st.pyplot(fig5)

    # Fig 13: daily changes for constellation 6 across 29 days
    st.subheader("Figure 13 — daily degree variances for constellation c06 across 29 days")
    curves29 = []
    for d in range(1,30):
        noise = 0.0003 * np.sin(degrees * (0.1 + 0.01*d)) + 0.0005*np.random.randn(len(degrees))
        curves29.append(curve2 * (1 + 0.02*np.sin(d/3)) + noise)
    fig6 = plot_degree_variance(degrees, curves29 + [hydro], labels=[f"day{d:02d}" for d in range(1,30)] + ["hydro. and ice"])
    st.pyplot(fig6)

    # Fig 14: 29-day average for c01-c10
    st.subheader("Figure 14 — 29-day average degree variances for c01–c10")
    curves_avg = [np.mean([curves29[d] for d in range(29)], axis=0) * (1 + 0.02*(i/10)) for i in range(10)]
    fig7 = plot_degree_variance(degrees, curves_avg + [hydro], labels=[f"c{i+1:02d}" for i in range(10)] + ["hydro. and ice"])
    st.pyplot(fig7)

    # Provide the "best" constellation (lowest Jso+Jto)
    st.subheader("Best constellation (simple ranking)")
    final_pop = history[-1][2]
    best_idx = np.argmin(final_pop[:,0] + final_pop[:,1])
    best = final_pop[best_idx]
    st.write("Best Jso, Jto:", best[0], best[1])
    st.write("To inspect constellation orbital elements, run with a full GEODYN-style sim (future).")
else:
    st.info("Adjust GA parameters in the sidebar and press **Run GA** to generate Figures 8–14 style outputs.")
