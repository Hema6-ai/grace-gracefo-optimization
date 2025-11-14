"""
plots.py
Plotting helpers to create figures 8-14 style outputs from GA history.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_population(history, gens_to_plot=(1,3,20)):
    """
    Scatter of Jso vs Jto for selected generations. history is list of tuples (g, pop, objs)
    """
    fig, ax = plt.subplots(figsize=(7,4.5))
    colors = {gens_to_plot[i]: c for i,c in zip(range(len(gens_to_plot)), ['tab:blue','tab:orange','tab:green'])}
    for (g,p,o) in history:
        if g in gens_to_plot:
            ax.scatter(o[:,0], o[:,1], label=f'igen={g}', s=40, alpha=0.9)
    ax.set_xlabel(r'$J_{so}$ [−]')
    ax.set_ylabel(r'$J_{to}$ [−]')
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_pareto_curves(hist_first_gen_objs, pareto_sets):
    """
    hist_first_gen_objs: (N,2) objects from first generation
    pareto_sets: list of indices or small arrays representing selected pareto curves (we'll pass coordinates)
    """
    fig, ax = plt.subplots(figsize=(7,4.5))
    for coords, c in pareto_sets:
        ax.plot(coords[:,0], coords[:,1], '-o', color=c)
    ax.set_xlabel(r'$J_{so}$')
    ax.set_ylabel(r'$J_{to}$')
    ax.grid(True, linestyle=':', alpha=0.6)
    return fig

def plot_degree_variance(degrees, curves, labels, logy=True):
    fig, ax = plt.subplots(figsize=(8,4))
    for y,label in zip(curves, labels):
        ax.plot(degrees, y, label=label)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Geoid Height [cm]')
    if logy:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True, linestyle=':')
    return fig

def plot_family_scatter(objs_c10):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(objs_c10[:,0], objs_c10[:,1], s=50)
    ax.set_xlabel(r'$J_{so}$')
    ax.set_ylabel(r'$J_{to}$')
    ax.grid(True, linestyle=':')
    return fig
