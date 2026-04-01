# Self-Optimizing Control (SOC) overview

## What is Self-Optimizing Control?

Self-Optimizing Control (SOC) is a methodology for selecting **controlled variables (CVs)** such that when they are held at constant setpoints, the resulting economic loss is acceptably small — even when disturbances shift the true optimal operating point.

The key idea: instead of running a real-time optimizer, find measurements (or combinations of measurements) to control so that the plant operates near-optimally *by construction*.

## The SOC problem

Consider a process with:
- **Degrees of freedom** $u$ (manipulated variables)
- **Disturbances** $d$ (uncontrolled inputs)
- **An economic objective** $J(u, d)$ to minimize (or maximize)

At the true optimum, the optimal inputs $u^*(d)$ change with every disturbance realization. SOC asks: can we find a controlled variable $c$ and a setpoint $c_s$ such that holding $c = c_s$ yields near-optimal $J$ for all expected $d$?

## Loss

The **loss** for a given CV design under a specific disturbance scenario is:

$$
L(d) = J\bigl(u_{\text{CV}}(d),\, d\bigr) - J^*(d)
$$

where $J^*$ is the truly optimal cost and $u_{\text{CV}}$ is the input that results from holding the CV at its nominal setpoint. A good CV design keeps $L$ small across all scenarios.

## How SOCF implements this

SOCF automates the SOC screening workflow:

1. **Generate scenarios** — nominal point plus single-factor disturbance changes (one parameter varies at a time while others stay nominal)
2. **Solve the base problem** — find $J^*(d)$ for each scenario using IPOPT
3. **Apply each CV design** — fix or constrain variables, re-solve, and compute the loss $L$
4. **Rank designs** — by average loss across scenarios; infeasible designs are pushed to the bottom

This approach follows the nonlinear programming-based SOC screening methodology. For the underlying theory, see {cite}`Skogestad2000` and {cite}`Halvorsen2003`.

## References

```{bibliography}
:filter: docname in docnames
```