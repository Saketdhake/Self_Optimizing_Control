"""Integration tests — need Pyomo + IPOPT to run."""

import pytest
import pandas as pd

try:
    import pyomo.environ as pyo
    HAS_PYOMO = True
except ImportError:
    HAS_PYOMO = False

from socf.core import (
    _make_solver,
    _solve_base_objectives,
    _evaluate_metrics,
    _build_raw_loss_matrix,
    _generate_scenarios,
    run_self_optimizing_model,
)

pytestmark = pytest.mark.skipif(not HAS_PYOMO, reason="Pyomo not installed")


def _ipopt_ok():
    try:
        _make_solver()
        return True
    except RuntimeError:
        return False

requires_ipopt = pytest.mark.skipif(not _ipopt_ok(), reason="IPOPT solver not available")


# -- Solver creation --

@requires_ipopt
class TestMakeSolver:
    def test_creates_solver(self):
        assert _make_solver() is not None


# -- Base objectives --

@requires_ipopt
class TestBaseObjectives:
    def test_length_matches_scenarios(self, toy_build_model, toy_disturbances):
        combos, _, _ = _generate_scenarios(toy_disturbances)
        J_list = _solve_base_objectives(toy_build_model, combos, _make_solver())
        assert len(J_list) == len(combos)

    def test_toy_optimum_is_zero(self, toy_build_model, toy_disturbances):
        combos, _, _ = _generate_scenarios(toy_disturbances)
        J_list = _solve_base_objectives(toy_build_model, combos, _make_solver())
        for j in J_list:
            assert j == pytest.approx(0.0, abs=1e-4)

    def test_bad_param_raises(self, toy_build_model):
        with pytest.raises(AttributeError, match="nonexistent"):
            _solve_base_objectives(toy_build_model, [{"nonexistent": 42}], _make_solver())


# -- Metric evaluation --

@requires_ipopt
class TestMetrics:
    def _run(self, toy_build_model, toy_disturbances, toy_metrics):
        combos, names, _ = _generate_scenarios(toy_disturbances)
        return _evaluate_metrics(toy_build_model, toy_metrics, combos, names, _make_solver())

    def test_returns_dataframe(self, toy_build_model, toy_disturbances, toy_metrics):
        assert isinstance(self._run(toy_build_model, toy_disturbances, toy_metrics), pd.DataFrame)

    def test_rows_are_metric_names(self, toy_build_model, toy_disturbances, toy_metrics):
        df = self._run(toy_build_model, toy_disturbances, toy_metrics)
        assert list(df.index) == list(toy_metrics.keys())

    def test_columns_are_scenario_names(self, toy_build_model, toy_disturbances, toy_metrics):
        combos, names, _ = _generate_scenarios(toy_disturbances)
        df = self._run(toy_build_model, toy_disturbances, toy_metrics)
        assert list(df.columns) == names

    def test_nominal_values(self, toy_build_model, toy_disturbances, toy_metrics):
        # at nominal (F=1, T=300): x*=1.0, y*=3.0
        df = self._run(toy_build_model, toy_disturbances, toy_metrics)
        assert df.at["x_val", "Nominal"] == pytest.approx(1.0, abs=0.01)
        assert df.at["y_val", "Nominal"] == pytest.approx(3.0, abs=0.01)


# -- Loss matrix --

@requires_ipopt
class TestLossMatrix:
    def _run(self, toy_build_model, toy_disturbances, toy_designs):
        combos, names, _ = _generate_scenarios(toy_disturbances)
        solver = _make_solver()
        J_list = _solve_base_objectives(toy_build_model, combos, solver)
        return _build_raw_loss_matrix(toy_build_model, toy_designs, combos, names, J_list, solver)

    def test_shape(self, toy_build_model, toy_disturbances, toy_designs):
        combos, names, _ = _generate_scenarios(toy_disturbances)
        lm = self._run(toy_build_model, toy_disturbances, toy_designs)
        assert lm.shape == (len(names), len(toy_designs))

    def test_losses_non_negative(self, toy_build_model, toy_disturbances, toy_designs):
        lm = self._run(toy_build_model, toy_disturbances, toy_designs)
        for col in lm.columns:
            for val in lm[col]:
                if val != "infeasible":
                    assert float(val) >= -1e-6

    def test_known_loss_value(self, toy_build_model, toy_disturbances, toy_designs):
        # fix_x sets x=1.5; at nominal (F=1), loss = (1.5-1)^2 = 0.25
        lm = self._run(toy_build_model, toy_disturbances, toy_designs)
        assert float(lm.at["Nominal", "Loss with fix_x"]) == pytest.approx(0.25, abs=0.05)

    def test_mismatched_lengths_raises(self, toy_build_model, toy_designs):
        with pytest.raises(ValueError):
            _build_raw_loss_matrix(
                toy_build_model, toy_designs,
                combos=[{"F": 1.0, "T": 300}],
                scenario_names=["Nominal", "Extra"],
                J_list=[0.0],
                solver=_make_solver(),
            )


# -- Full pipeline --

@requires_ipopt
class TestFullPipeline:
    def _run(self, toy_build_model, toy_disturbances, toy_designs, toy_metrics, **kw):
        return run_self_optimizing_model(
            build_model=toy_build_model,
            disturbances=toy_disturbances,
            user_designs=toy_designs,
            metrics=toy_metrics,
            **kw,
        )

    def test_returns_two_dataframes(self, toy_build_model, toy_disturbances, toy_designs, toy_metrics):
        results_df, loss_matrix = self._run(toy_build_model, toy_disturbances, toy_designs, toy_metrics)
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(loss_matrix, pd.DataFrame)

    def test_loss_matrix_has_summary_rows(self, toy_build_model, toy_disturbances, toy_designs, toy_metrics):
        _, lm = self._run(toy_build_model, toy_disturbances, toy_designs, toy_metrics)
        assert "Average loss" in lm.index
        assert "Ranking" in lm.index

    def test_something_gets_rank_1(self, toy_build_model, toy_disturbances, toy_designs, toy_metrics):
        _, lm = self._run(toy_build_model, toy_disturbances, toy_designs, toy_metrics)
        assert 1 in lm.loc["Ranking"].values

    def test_metrics_shape(self, toy_build_model, toy_disturbances, toy_designs, toy_metrics):
        results_df, _ = self._run(toy_build_model, toy_disturbances, toy_designs, toy_metrics)
        combos, names, _ = _generate_scenarios(toy_disturbances)
        assert results_df.shape == (len(toy_metrics), len(names))

    def test_parallel_matches_serial(self, toy_build_model, toy_disturbances, toy_designs, toy_metrics):
        _, serial = self._run(toy_build_model, toy_disturbances, toy_designs, toy_metrics, parallel=False)
        _, parallel = self._run(toy_build_model, toy_disturbances, toy_designs, toy_metrics, parallel=True, n_workers=2)

        for col in serial.columns:
            s, p = serial.at["Average loss", col], parallel.at["Average loss", col]
            if s == "infeasible":
                assert p == "infeasible"
            else:
                assert float(s) == pytest.approx(float(p), abs=0.1)