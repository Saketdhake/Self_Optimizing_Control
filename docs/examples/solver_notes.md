# Solver notes

## IPOPT

SOCF uses [IPOPT](https://coin-or.github.io/Ipopt/) (Interior Point OPTimizer) as its nonlinear solver. IPOPT is called via Pyomo's `SolverFactory("ipopt")` interface.

### Installation

The easiest way to install IPOPT is via Conda:

```bash
conda install -c conda-forge ipopt
```

Alternatively, you can build from source or use system package managers (see the [IPOPT documentation](https://coin-or.github.io/Ipopt/INSTALL.html)).

### Verifying IPOPT is available

```python
import pyomo.environ as pyo
solver = pyo.SolverFactory("ipopt")
print(solver.available())  # Should print True
```

If this prints `False`, SOCF will raise a `RuntimeError` at the start of `run_self_optimizing_model`.

## Termination conditions

SOCF treats the following IPOPT termination conditions as acceptable:

- `optimal` — the standard success
- `locallyOptimal` — common for nonlinear problems; a local minimum/maximum was found
- `feasible` — a feasible point was found (not necessarily optimal, but usable for screening)

Any other termination (e.g., `infeasible`, `maxIterations`) causes that (design, scenario) pair to be marked as `"infeasible"` in the loss matrix.

## Debugging tips

If you see many `"infeasible"` entries in the loss matrix:

1. **Run with `tee=True`** to see full IPOPT output:
   ```python
   results_df, loss_matrix = run_self_optimizing_model(
       ..., tee=True, parallel=False
   )
   ```

2. **Check variable bounds** — overly tight bounds can make the problem infeasible when disturbances push the optimal solution outside the feasible region.

3. **Check initial values** — IPOPT is sensitive to starting points for nonlinear problems. Try adjusting `initialize=` values in your model.

4. **Simplify first** — test with fewer disturbance scenarios and fewer CV designs to isolate which combination fails.

## Parallel execution

SOCF supports parallel loss evaluation via `ProcessPoolExecutor`:

```python
results_df, loss_matrix = run_self_optimizing_model(
    ..., parallel=True, n_workers=4
)
```

Each worker creates its own IPOPT instance. If parallel execution fails for any reason (e.g., pickling issues in Jupyter), SOCF automatically falls back to serial mode.

```{warning}
Parallel mode requires that `build_model` and all design/metric functions are importable from a `.py` module (not defined inline in a Jupyter notebook). This is a limitation of Python's `multiprocessing`.
```