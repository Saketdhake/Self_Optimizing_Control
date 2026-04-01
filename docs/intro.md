# SOCF Documentation

![GitHub License](https://img.shields.io/github/license/Saketdhake/Self_Optimizing_Control)
![GitHub Repo stars](https://img.shields.io/github/stars/Saketdhake/Self_Optimizing_Control)

**SOCF** (Self-Optimizing Control Framework) is a Python package for Self-Optimizing Control analysis built on top of [Pyomo](https://www.pyomo.org/). It automates the full SOC workflow:

1. **Scenario generation** — nominal + single-factor disturbance sampling
2. **Base optimization** — solve the true economic objective at each scenario using IPOPT
3. **Metric evaluation** — compute user-defined KPIs at each optimum
4. **Loss evaluation** — for each candidate CV design, compute the economic loss relative to the true optimum
5. **Ranking** — rank designs by average loss, flagging infeasible ones

The package is designed for process systems engineers who want to screen controlled variable (CV) candidates quickly and systematically.

## Quick start

```python
from socf.core import run_self_optimizing_model

results_df, loss_matrix = run_self_optimizing_model(
    build_model=build_base_model,
    disturbances=disturbances,
    user_designs=controlled_variables,
    metrics=cvs,
)
```

## Contents

```{tableofcontents}
```

## Citing SOCF

```bibtex
@misc{Dhake2026,
  author = {Saket P. Dhake},
  title  = {SOCF: Self-Optimizing Control Framework in Python},
  year   = {2026},
  url    = {https://github.com/Saketdhake/Self_Optimizing_Control}
}
```