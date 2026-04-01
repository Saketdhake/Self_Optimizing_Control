# API reference

## Main entry point

The primary function you call from user code:

```{eval-rst}
.. autofunction:: socf.core.run_self_optimizing_model
```

## Internal functions

These are called internally by `run_self_optimizing_model` but are documented here for advanced users and contributors.

### Scenario generation

```{eval-rst}
.. autofunction:: socf.core._generate_scenarios
```

### Solver setup

```{eval-rst}
.. autofunction:: socf.core._make_solver
```

```{eval-rst}
.. autofunction:: socf.core._is_good_termination
```

### Base optimization

```{eval-rst}
.. autofunction:: socf.core._solve_base_objectives
```

### Metric evaluation

```{eval-rst}
.. autofunction:: socf.core._evaluate_metrics
```

### Loss computation

```{eval-rst}
.. autofunction:: socf.core._solve_one_loss_job
```

```{eval-rst}
.. autofunction:: socf.core._build_raw_loss_matrix
```

### Post-processing

```{eval-rst}
.. autofunction:: socf.core._mark_infeasible_designs
```

```{eval-rst}
.. autofunction:: socf.core._append_average_and_ranking
```

### Data classes

```{eval-rst}
.. autoclass:: socf.core._LossJob
   :members:
```