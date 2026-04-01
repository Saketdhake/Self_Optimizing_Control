# Loss evaluation and CV selection

## How SOCF computes loss

For each candidate CV design and each disturbance scenario, SOCF:

1. Builds a fresh Pyomo model via the user's `build_model()` function
2. Applies the design rule (e.g., fixes a variable or adds a constraint)
3. Sets disturbance parameters to the scenario values
4. Solves with IPOPT
5. Computes loss relative to the base optimum:

For a **minimization** problem:

$$
L = J_{\text{design}} - J^*
$$

For a **maximization** problem:

$$
L = J^* - J_{\text{design}}
$$

In both cases, larger $L$ means worse performance.

## Infeasible scenarios

If IPOPT cannot find a feasible solution for a (design, scenario) pair, that cell in the loss matrix is marked `"infeasible"`. When computing the average loss for ranking:

- If a design has **any** infeasible entry, its average loss is labeled `"infeasible"`
- Infeasible designs are ranked last (pushed to the bottom)

This is intentionally conservative: a design that fails under even one scenario is penalized.

## Ranking

Designs are ranked by average loss using dense ranking (ties share the same rank). The design with the lowest average loss is ranked 1.

## Negative losses

In some edge cases, numerical noise or sign convention mismatches can produce negative loss values. The `mark_negative_loss_infeasible` option (default `False`) lets you treat these as infeasible if negative loss is meaningless for your problem.

## CV design patterns

SOCF supports any design that can be expressed as a mutation of the Pyomo model. Common patterns:

**Fix a single variable:**
```python
{"M=1.0": lambda m: m.M.fix(1.0)}
```

**Add a ratio constraint:**
```python
{"M/F=1.0": lambda m: m.add_component("mf_eq", pyo.Constraint(expr=m.M == m.F))}
```

**Fix a linear combination:**
```python
{"Xa+2Xb+3Xc=2.0": lambda m: m.add_component(
    "combo", pyo.Constraint(expr=m.Xa + 2*m.Xb + 3*m.Xc == 2.0)
)}
```

**Fix multiple variables simultaneously:**
```python
{"F3=24.72, F200=217.74": lambda m: [m.F3.fix(24.72), m.F200.fix(217.74)]}
```

Each design function receives the model and modifies it in-place. SOCF builds a fresh model for every (design, scenario) solve, so mutations don't leak between runs.