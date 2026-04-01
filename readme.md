# SOCF — Self-Optimizing Control Framework


[GitHub forks](https://img.shields.io/github/forks/Saketdhake/Self_Optimizing_Contro)
![GitHub License](https://img.shields.io/github/license/Saketdhake/Self_Optimizing_Control)
![GitHub Repo stars](https://img.shields.io/github/stars/Saketdhake/Self_Optimizing_Control)

A Python package for **Self-Optimizing Control (SOC)** analysis built on [Pyomo](https://www.pyomo.org/) and solved with [IPOPT](https://coin-or.github.io/Ipopt/).

## 📖 Documentation

**Full documentation, theory, API reference, and examples are available at:**

👉 **[https://saketdhake.github.io/Self_Optimizing_Control/](https://saketdhake.github.io/Self_Optimizing_Control/)**

## Overview

SOCF automates the SOC screening workflow:

1. **Scenario generation** — nominal + single-factor disturbance sampling
2. **Base optimization** — solve the true economic objective at each scenario
3. **Metric evaluation** — compute user-defined KPIs at each optimum
4. **Loss evaluation** — for each candidate CV design, compute economic loss
5. **Ranking** — rank designs by average loss, flagging infeasible ones

## Quick start

```bash
git clone https://github.com/Saketdhake/Self_Optimizing_Control.git
cd Self_Optimizing_Control
pip install -e .
```

```python
from socf.core import run_self_optimizing_model

results_df, loss_matrix = run_self_optimizing_model(
    build_model=build_base_model,
    disturbances=disturbances,
    user_designs=controlled_variables,
    metrics=cvs,
)
```

## Requirements

- Python ≥ 3.10
- IPOPT solver (install via `conda install -c conda-forge ipopt`)
- numpy, pandas, pyomo (installed automatically)

See the [Installation guide](https://saketdhake.github.io/Self_Optimizing_Control/installation.html) for detailed setup instructions.

## Examples

- [CSTR Quickstart](https://saketdhake.github.io/Self_Optimizing_Control/examples/cstr_quickstart.html) — A→B→C reactor with 8 candidate CV designs
- [Evaporator](Examples/Test_evaporator.ipynb) — Multi-variable evaporator system

## Citation

```bibtex
@misc{Dhake2026,
  author = {Saket P. Dhake},
  title  = {SOCF: Self-Optimizing Control Framework in Python},
  year   = {2026},
  url    = {https://github.com/Saketdhake/Self_Optimizing_Control}
}
```

## License

