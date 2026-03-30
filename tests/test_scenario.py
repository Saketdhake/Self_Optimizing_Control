"""Tests for _generate_scenarios — pure logic, no solver needed."""

from socf.core import _generate_scenarios


def test_two_param_count(simple_disturbances):
    # 2 params x 2 vals = 4 full combos, but only nominal + 2 single-factor = 3
    combos, names, _ = _generate_scenarios(simple_disturbances)
    assert len(combos) == 3
    assert len(names) == 3


def test_nominal_is_first(simple_disturbances):
    combos, names, nominal = _generate_scenarios(simple_disturbances)
    assert names[0] == "Nominal"
    assert combos[0] == nominal


def test_nominal_uses_first_values(simple_disturbances):
    _, _, nominal = _generate_scenarios(simple_disturbances)
    assert nominal == {"F": 1.0, "T": 300}


def test_excludes_multi_factor_combos(simple_disturbances):
    combos, _, nominal = _generate_scenarios(simple_disturbances)
    for c in combos:
        n_diffs = sum(c[p] != nominal[p] for p in nominal)
        assert n_diffs <= 1, f"multi-factor combo slipped through: {c}"


def test_scenario_naming(simple_disturbances):
    _, names, _ = _generate_scenarios(simple_disturbances)
    non_nominal = [n for n in names if n != "Nominal"]
    assert "F=1.1" in non_nominal
    assert "T=320" in non_nominal


def test_three_params(three_param_disturbances):
    # nominal + 3 single-factor = 4
    combos, names, _ = _generate_scenarios(three_param_disturbances)
    assert len(combos) == 4
    assert len(names) == 4


def test_single_param():
    dist = {"alpha": [0.1, 0.2, 0.3]}
    combos, names, nominal = _generate_scenarios(dist)
    assert len(combos) == 3
    assert nominal == {"alpha": 0.1}
    assert "alpha=0.2" in names
    assert "alpha=0.3" in names


def test_single_value_per_param_gives_only_nominal():
    dist = {"F": [1.0], "T": [300]}
    combos, names, _ = _generate_scenarios(dist)
    assert len(combos) == 1
    assert names == ["Nominal"]


def test_return_types(simple_disturbances):
    combos, names, nominal = _generate_scenarios(simple_disturbances)
    assert isinstance(combos, list) and all(isinstance(c, dict) for c in combos)
    assert isinstance(names, list) and all(isinstance(n, str) for n in names)
    assert isinstance(nominal, dict)