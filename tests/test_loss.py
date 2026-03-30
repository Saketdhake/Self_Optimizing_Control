"""Tests for loss matrix post-processing: infeasibility marking, averages, ranking."""

import pytest
import pandas as pd

from socf.core import (
    _mark_infeasible_designs,
    _append_average_and_ranking,
    _is_good_termination,
)

try:
    from pyomo.opt import TerminationCondition
    HAS_PYOMO = True
except ImportError:
    HAS_PYOMO = False


# -- _is_good_termination --

@pytest.mark.skipif(not HAS_PYOMO, reason="Pyomo not installed")
class TestIsGoodTermination:
    def test_optimal(self):
        assert _is_good_termination(TerminationCondition.optimal)

    def test_locally_optimal(self):
        assert _is_good_termination(TerminationCondition.locallyOptimal)

    def test_feasible(self):
        assert _is_good_termination(TerminationCondition.feasible)

    def test_infeasible(self):
        assert not _is_good_termination(TerminationCondition.infeasible)

    def test_max_iterations(self):
        assert not _is_good_termination(TerminationCondition.maxIterations)


# -- _mark_infeasible_designs --

class TestMarkInfeasible:
    def test_nan_replaced_cellwise(self, loss_matrix_with_nan):
        result = _mark_infeasible_designs(loss_matrix_with_nan)
        assert result.at["F=1.1", "Loss with Design A"] == "infeasible"
        # non-NaN cells untouched
        assert result.at["Nominal", "Loss with Design A"] == 0.5
        assert result.at["F=1.1", "Loss with Design B"] == 0.4

    def test_nan_marks_whole_column(self, loss_matrix_with_nan):
        result = _mark_infeasible_designs(loss_matrix_with_nan, mark_all_scenarios=True)
        assert (result["Loss with Design A"] == "infeasible").all()
        assert result.at["Nominal", "Loss with Design B"] == 0.2

    def test_negatives_kept_by_default(self, loss_matrix_with_negatives):
        result = _mark_infeasible_designs(loss_matrix_with_negatives)
        assert result.at["F=1.1", "Loss with Design A"] == -0.01

    def test_negatives_flagged_when_asked(self, loss_matrix_with_negatives):
        result = _mark_infeasible_designs(loss_matrix_with_negatives, mark_negative=True)
        assert result.at["F=1.1", "Loss with Design A"] == "infeasible"

    def test_clean_matrix_passes_through(self, clean_loss_matrix):
        result = _mark_infeasible_designs(clean_loss_matrix)
        for col in clean_loss_matrix.columns:
            for idx in clean_loss_matrix.index:
                assert result.at[idx, col] == clean_loss_matrix.at[idx, col]

    def test_output_dtype_is_object(self, clean_loss_matrix):
        result = _mark_infeasible_designs(clean_loss_matrix)
        assert all(result[c].dtype == object for c in result.columns)


# -- _append_average_and_ranking --

class TestAverageAndRanking:
    def test_adds_summary_rows(self, clean_loss_matrix):
        result = _append_average_and_ranking(clean_loss_matrix)
        assert "Average loss" in result.index
        assert "Ranking" in result.index

    def test_average_values(self, clean_loss_matrix):
        # A: mean(0.5, 1.0, 0.3) = 0.6    B: mean(0.2, 0.4, 0.6) = 0.4
        result = _append_average_and_ranking(clean_loss_matrix)
        assert float(result.at["Average loss", "Loss with Design A"]) == pytest.approx(0.6, abs=0.01)
        assert float(result.at["Average loss", "Loss with Design B"]) == pytest.approx(0.4, abs=0.01)

    def test_lower_avg_gets_rank_1(self, clean_loss_matrix):
        result = _append_average_and_ranking(clean_loss_matrix)
        assert int(result.at["Ranking", "Loss with Design B"]) == 1
        assert int(result.at["Ranking", "Loss with Design A"]) == 2

    def test_infeasible_column_avg(self, loss_matrix_with_infeasible):
        result = _append_average_and_ranking(loss_matrix_with_infeasible)
        assert result.at["Average loss", "Loss with Design A"] == "infeasible"

    def test_infeasible_ranks_last(self, loss_matrix_with_infeasible):
        result = _append_average_and_ranking(loss_matrix_with_infeasible)
        assert int(result.at["Ranking", "Loss with Design A"]) > int(result.at["Ranking", "Loss with Design B"])

    def test_original_rows_preserved(self, clean_loss_matrix):
        result = _append_average_and_ranking(clean_loss_matrix)
        for idx in clean_loss_matrix.index:
            assert idx in result.index

    def test_all_infeasible(self):
        df = pd.DataFrame(
            {"Loss with A": ["infeasible", "infeasible"],
             "Loss with B": ["infeasible", "infeasible"]},
            index=["Nominal", "F=1.1"],
        )
        result = _append_average_and_ranking(df)
        assert result.at["Average loss", "Loss with A"] == "infeasible"
        assert result.at["Average loss", "Loss with B"] == "infeasible"

    def test_tied_averages_share_rank(self):
        df = pd.DataFrame(
            {"Loss with A": [0.5, 0.5],
             "Loss with B": [0.5, 0.5]},
            index=["Nominal", "F=1.1"],
        )
        result = _append_average_and_ranking(df)
        assert int(result.at["Ranking", "Loss with A"]) == int(result.at["Ranking", "Loss with B"]) == 1