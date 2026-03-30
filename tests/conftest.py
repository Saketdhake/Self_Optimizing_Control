"""Shared fixtures for SOC engine tests."""

import pytest
import pandas as pd
import numpy as np

try:
    import pyomo.environ as pyo
    HAS_PYOMO = True
except ImportError:
    HAS_PYOMO = False

requires_pyomo = pytest.mark.skipif(not HAS_PYOMO, reason="Pyomo is not installed")


# -- Disturbance dicts --

@pytest.fixture
def simple_disturbances():
    return {"F": [1.0, 1.1], "T": [300, 320]}


@pytest.fixture
def three_param_disturbances():
    return {"F": [1.0, 1.1], "T": [300, 320], "Za": [0.4, 0.5]}


# -- Pre-built loss matrices for ranking / infeasibility tests --

@pytest.fixture
def clean_loss_matrix():
    return pd.DataFrame(
        {"Loss with Design A": [0.5, 1.0, 0.3],
         "Loss with Design B": [0.2, 0.4, 0.6]},
        index=["Nominal", "F=1.1", "T=320"],
    )


@pytest.fixture
def loss_matrix_with_infeasible():
    return pd.DataFrame(
        {"Loss with Design A": [0.5, "infeasible", 0.3],
         "Loss with Design B": [0.2, 0.4, 0.6]},
        index=["Nominal", "F=1.1", "T=320"],
    )


@pytest.fixture
def loss_matrix_with_nan():
    return pd.DataFrame(
        {"Loss with Design A": [0.5, np.nan, 0.3],
         "Loss with Design B": [0.2, 0.4, 0.6]},
        index=["Nominal", "F=1.1", "T=320"],
    )


@pytest.fixture
def loss_matrix_with_negatives():
    return pd.DataFrame(
        {"Loss with Design A": [0.5, -0.01, 0.3],
         "Loss with Design B": [0.2, 0.4, 0.6]},
        index=["Nominal", "F=1.1", "T=320"],
    )


# -- Toy Pyomo model + associated fixtures --

if HAS_PYOMO:

    @pytest.fixture
    def toy_build_model():
        """min J = (x - F)^2 + (y - T/100)^2, optimum is always J*=0."""
        def _build():
            m = pyo.ConcreteModel()
            m.F = pyo.Param(initialize=1.0, mutable=True)
            m.T = pyo.Param(initialize=300, mutable=True)
            m.x = pyo.Var(initialize=1.0)
            m.y = pyo.Var(initialize=3.0)
            m.J = pyo.Objective(
                expr=(m.x - m.F)**2 + (m.y - m.T / 100)**2,
                sense=pyo.minimize,
            )
            return m
        return _build

    @pytest.fixture
    def toy_disturbances():
        return {"F": [1.0, 2.0], "T": [300, 400]}

    @pytest.fixture
    def toy_metrics():
        return {
            "x_val": lambda m, sc: float(pyo.value(m.x)),
            "y_val": lambda m, sc: float(pyo.value(m.y)),
        }

    @pytest.fixture
    def toy_designs():
        def fix_x(m):
            m.x.fix(1.5)

        def fix_y(m):
            m.y.fix(3.5)

        return {"fix_x": fix_x, "fix_y": fix_y}