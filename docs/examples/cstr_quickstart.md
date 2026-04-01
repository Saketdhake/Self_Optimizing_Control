# CSTR quickstart

This example demonstrates SOCF on a simple CSTR (Continuous Stirred-Tank Reactor) with consecutive reactions $A \to B \to C$. The objective is to maximize production of $B$.

## Problem setup

The model has:
- **Variables:** compositions $X_a, X_b, X_c \in [0,1]$ and hold-up $M \geq 0$
- **Parameters (disturbances):** feed flow rate $F$, feed compositions $Z_a$, reaction rate constants $K_a, K_b$
- **Objective:** maximize $J = 100 \cdot X_b$
- **Constraints:** steady-state mass balances

## Disturbances

```python
disturbances = {
    "F":  [1.0, 0.7],
    "Za": [0.8, 0.6, 1.0],
    "Ka": [1.0, 1.5],
    "Kb": [1.0, 1.5],
}
```

SOCF generates the nominal scenario (all first values) plus single-factor changes — 6 scenarios total.

## Candidate CV designs

Eight designs are tested, ranging from simple variable fixes (`M=1.0`, `Xa=0.4`) to ratio constraints (`Xb/Xa=0.5`) and linear combinations (`Xa+2Xb+3Xc=2.0`).

## Results

The loss matrix shows that **M/F = 1.0** (controlling the ratio of hold-up to feed flow) ranks #1 with the lowest average loss of 0.07. This means holding $M/F$ constant at its nominal value keeps the reactor near-optimal across all disturbance scenarios.

Controlling $X_b = 0.2$ (rank #8) is the worst choice — it becomes infeasible under some disturbance scenarios.

## Running the example

See the full notebook: [`Examples/CSTR.ipynb`](https://github.com/Saketdhake/Self_Optimizing_Control/blob/main/Examples/CSTR.ipynb)

```python
from socf.core import run_self_optimizing_model

results_df, loss_matrix = run_self_optimizing_model(
    build_model=build_base_model,
    disturbances=disturbances,
    user_designs=controlled_variables,
    metrics=cvs,
)
```