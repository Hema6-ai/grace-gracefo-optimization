# GRACE GA Constellation Optimizer (demo)

This repository is a self-contained Streamlit demo that implements a simplified multi-objective GA for designing GRACE-like constellations and reproduces the Figures 8–14 style outputs from:

> Deccia et al., *Using a Multiobjective Genetic Algorithm to Design Satellite Constellations for Recovering Earth System Mass Change*, Remote Sens. 2022.

Features:
- Interactive Streamlit UI to set GA / mission parameters.
- Simulated ground-track sampling, spatial/temporal objective computations (J_so, J_to).
- Lightweight NSGA-II-like GA (nondominated sorting + crossover + mutation).
- Plots: population scatter (gens 1,3,20); Pareto curves; degree variance plots for Pareto selections; family of constellations; daily/29-day degree variance plots.

How to run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
