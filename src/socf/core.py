from __future__ import annotations

import itertools
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value, Objective
from pyomo.opt import TerminationCondition


def _generate_scenarios(disturbances: Dict[str, List[Any]]) -> Tuple[List[dict], List[str], dict]:
    """
    Build a small set of disturbance scenarios from user-supplied disturbance grids.

    The input `disturbances` is a mapping like:
        {"F": [1.0, 1.1], "T": [300, 320], "Za": [0.4, 0.5]}

    If you took the full Cartesian product, you'd get every combination of values.
    That explodes quickly. For SOC screening, a common “cheap” sampling is:
      1) keep the nominal point, and
      2) keep only *single-factor* deviations from nominal (one parameter changes at a time).

    This function does exactly that:
      - It forms the full product to define what “nominal” means.
      - Nominal is the *first* combination produced by itertools.product, which is driven
        by the order of `disturbances.keys()` and the order of each value list.
      - Then it filters the full set to keep combinations where at most one parameter differs
        from nominal.

    Parameters
    ----------
    disturbances:
        Dictionary mapping each disturbance/parameter name to a list of values to test.
        The first value in each list becomes the nominal value (because nominal is the first
        Cartesian product combination).

    Returns
    -------
    combos:
        List of scenarios, where each scenario is a dict {param_name: value}.
        Includes the nominal scenario plus the single-factor-change scenarios.
    scenario_names:
        Human-readable names aligned with `combos`.
        - "Nominal" for the nominal scenario
        - "<param>=<value>" for single-factor scenarios (e.g., "F=1.1")
    nominal:
        The nominal scenario dict.

    Notes
    -----
    - The ordering of scenarios depends on the order of `disturbances.keys()`. If you need a
      deterministic order across Python versions/environments, pass an OrderedDict or build
      the dict in a controlled way.
    - This sampling rule is intentionally limited; it does not capture interactions between
      multiple disturbances changing at the same time.
    """
    params = list(disturbances.keys())

    all_combos = [
        dict(zip(params, vals))
        for vals in itertools.product(*(disturbances[p] for p in params))
    ]

    nominal = all_combos[0]

    combos = [
        c for c in all_combos
        if sum(c[p] != nominal[p] for p in params) <= 1
    ]

    scenario_names = []
    for c in combos:
        diffs = [(p, c[p]) for p in params if c[p] != nominal[p]]
        if not diffs:
            scenario_names.append("Nominal")
        else:
            p, v = diffs[0]
            scenario_names.append(f"{p}={v}")

    return combos, scenario_names, nominal


def _make_solver():
    """
    Create an IPOPT solver object and apply user options (if any).

    This project assumes IPOPT is the default nonlinear solver. This helper does two things:
      - constructs the solver via `pyomo.SolverFactory("ipopt")`
      - checks that IPOPT is actually available, so failures happen early and clearly

    Parameters
    ----------
    solver_options:
        Optional dict of IPOPT options, written as you would in Pyomo:
            {"tol": 1e-8, "max_iter": 5000}
        These get assigned to `solver.options[k] = v`.

    Returns
    -------
    solver:
        A configured Pyomo solver instance for IPOPT.

    Raises
    ------
    RuntimeError
        If IPOPT is not available (not installed, not on PATH, or not configured).

    Examples
    --------
    >>> solver = _make_solver({"tol": 1e-8, "max_iter": 5000})
    >>> results = solver.solve(model)
    """
    solver = pyo.SolverFactory("ipopt")

    try:
        available = solver.available(exception_flag=False)
    except TypeError:
        # Some Pyomo versions don't accept exception_flag in available()
        available = solver.available()

    if not available:
        raise RuntimeError(
            "Default solver 'ipopt' is not available. "
            "Install IPOPT and ensure it's on PATH / correctly configured."
        )

    return solver


def _is_good_termination(tc) -> bool:
    """
    Decide whether a Pyomo solve termination condition should be treated as “good”.

    In practice, IPOPT can return different termination flags depending on the problem
    structure and tolerances. For this SOC workflow we treat the following as acceptable:
      - optimal
      - locallyOptimal (typical for nonlinear problems)
      - feasible (a feasible point found; not always ideal but usable in some pipelines)

    Parameters
    ----------
    tc:
        Termination condition value from `results.solver.termination_condition`.

    Returns
    -------
    bool
        True if the termination condition is acceptable, False otherwise.
    """
    return tc in (
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    )


def _solve_base_objectives(
    build_model: Callable[[], pyo.ConcreteModel],
    combos: List[dict],
    solver,
    tee: bool = False,
) -> List[float]:
    """
    Solve the *true* (economic) optimization problem for every scenario and record J*.

    For each scenario in `combos`, this function:
      1) builds a fresh model by calling `build_model()`
      2) sets each disturbance parameter to the scenario value
      3) solves the model (with the provided solver)
      4) reads the optimal objective value `m.J`

    The output `J_list[i]` is the base optimum for `combos[i]`. This becomes the
    reference point used later when computing SOC “loss” under different CV designs.

    Parameters
    ----------
    build_model:
        Zero-argument callable that returns a fully-formed Pyomo ConcreteModel.
        The model is expected to have attributes matching the disturbance names
        (e.g., if you pass disturbance key "F", the model must have `m.F`).
        The model is also expected to have the true objective available as `m.J`.
    combos:
        List of scenario dictionaries (e.g., produced by `_generate_scenarios`).
        Each dict maps parameter name -> value.
    solver:
        A configured Pyomo solver object (typically IPOPT).
    tee:
        If True, stream solver output to the console (useful for debugging).

    Returns
    -------
    J_list:
        List of base optimal objective values. Same ordering as `combos`.

    Raises
    ------
    AttributeError
        If the model does not contain a component named by one of the disturbance keys.
    RuntimeError
        If the solver does not return an acceptable termination condition for a scenario.
    """
    J_list: List[float] = []
    for c in combos:
        m = build_model()
        for p, val in c.items():
            if not hasattr(m, p):
                raise AttributeError(f"Model has no component named '{p}'.")
            getattr(m, p).set_value(val)

        results = solver.solve(m, tee=tee)
        if not _is_good_termination(results.solver.termination_condition):
            raise RuntimeError(
                f"Base optimum solve failed for scenario {c} with termination "
                f"{results.solver.termination_condition}."
            )

        J_list.append(value(m.J))
    return J_list


def _evaluate_metrics(
    build_model: Callable[[], pyo.ConcreteModel],
    metrics: Dict[str, Callable[[pyo.ConcreteModel, dict], float]],
    combos: List[dict],
    scenario_names: List[str],
    solver,
    tee: bool = False,
) -> pd.DataFrame:
    """
    Evaluate user-chosen “metrics” at the true optimum for each scenario.

    Think of `metrics` as post-processing functions. Each metric gets the solved model
    (at the true optimum) plus the scenario dict, and returns a number—examples:
      - purity, yield, conversion
      - constraint margin
      - any KPI you care about

    Internally, we solve the model for each scenario first (same as `_solve_base_objectives`),
    then call each metric function on that solved model.

    Parameters
    ----------
    build_model:
        Callable that returns a new Pyomo model.
    metrics:
        Dict mapping metric name -> callable(metric_fn).
        Each metric function must have signature:
            metric_fn(m: ConcreteModel, scenario: dict) -> float
    combos:
        List of scenario dicts (param -> value).
    scenario_names:
        Names aligned with `combos`. Used for column labels in the returned table.
    solver:
        Pyomo solver (typically IPOPT).
    tee:
        If True, prints solver output.

    Returns
    -------
    df:
        DataFrame with:
          - rows = metric names
          - columns = scenario names
          - values = metric values at the true optimum
        Values are rounded to 4 decimals.

    Raises
    ------
    AttributeError
        If the model is missing any disturbance component referenced by a scenario.
    RuntimeError
        If any scenario solve fails (non-acceptable termination condition).

    Notes
    -----
    The returned DataFrame is transposed relative to a typical “scenario-by-row” layout
    because it is often convenient to compare metrics across scenarios for each KPI.
    """
    results_rows = []
    for c in combos:
        m = build_model()
        for p, val in c.items():
            if not hasattr(m, p):
                raise AttributeError(f"Model has no component named '{p}'.")
            getattr(m, p).set_value(val)

        results = solver.solve(m, tee=tee)
        if not _is_good_termination(results.solver.termination_condition):
            raise RuntimeError(
                f"Metric evaluation solve failed for scenario {c} with termination "
                f"{results.solver.termination_condition}."
            )

        results_rows.append([fn(m, c) for fn in metrics.values()])

    df = pd.DataFrame(results_rows, index=scenario_names, columns=list(metrics.keys())).T
    return df.round(4)


@dataclass(frozen=True)
class _LossJob:
    """
    Small container describing a single loss-evaluation run.

    A “loss job” corresponds to:
      - one SOC design (e.g., a chosen set of controlled variables / constraints), and
      - one disturbance scenario, and
      - the base optimal objective value for that scenario (J_nom)

    This dataclass is intentionally simple and picklable, which makes it safe to ship
    into ProcessPoolExecutor workers.
    """
    design_label: str
    scenario_name: str
    scenario: dict
    J_nom: float


def _solve_one_loss_job(
    build_model: Callable[[], pyo.ConcreteModel],
    apply_design: Callable[[pyo.ConcreteModel], None],
    job: _LossJob,
    tee: bool = False,
) -> Tuple[str, str, Any]:
    """
    Solve one (design, scenario) pair and return the SOC loss.

    This is the “worker” function used in parallel evaluation. Steps:
      1) build a fresh model
      2) apply the SOC design (typically modifies constraints / fixes CVs / changes objective)
      3) set disturbances to scenario values
      4) solve with IPOPT
      5) compute loss relative to the base optimum J_nom

    Loss definition used here:
      - If the model objective is minimizing:
            loss = J_opt - J_nom
      - If the model objective is maximizing:
            loss = J_nom - J_opt
    so that “bigger loss = worse performance” in either case.

    If a solve fails (bad termination condition), we return the string "infeasible".
    That makes downstream display and ranking easier.

    Parameters
    ----------
    build_model:
        Callable returning a fresh Pyomo model.
    apply_design:
        Function that mutates the model in-place to represent a particular SOC design.
        Example: fix certain variables, add CV constraints, change objective, etc.
    job:
        _LossJob instance describing which design/scenario to run and the base objective J_nom.
    tee:
        If True, prints solver output (not usually recommended in heavy parallel runs).

    Returns
    -------
    (design_label, scenario_name, loss):
        design_label:
            The label passed in via the job.
        scenario_name:
            The scenario name passed in via the job.
        loss:
            Either a float (rounded to 2 decimals) or the string "infeasible".

    Raises
    ------
    ValueError
        If the model ends up with zero or multiple active objectives after applying the design.
    AttributeError
        If a disturbance component is missing on the model.
    """
    m = build_model()
    apply_design(m)

    active_objs = list(m.component_data_objects(Objective, active=True))
    if len(active_objs) != 1:
        raise ValueError(f"Model must have exactly 1 active Objective, found {len(active_objs)}.")
    obj = active_objs[0]

    for p, val in job.scenario.items():
        if not hasattr(m, p):
            raise AttributeError(f"Model has no component named '{p}'.")
        getattr(m, p).set_value(val)

    solver = _make_solver()
    results = solver.solve(m, tee=tee)
    tc = results.solver.termination_condition

    if not _is_good_termination(tc):
        return job.design_label, job.scenario_name, "infeasible"

    J_opt = value(obj.expr)
    loss = (J_opt - job.J_nom) if obj.is_minimizing() else (job.J_nom - J_opt)
    return job.design_label, job.scenario_name, round(float(loss), 2)


def _build_raw_loss_matrix(
    build_model: Callable[[], pyo.ConcreteModel],
    user_designs: Dict[str, Callable[[pyo.ConcreteModel], None]],
    combos: List[dict],
    scenario_names: List[str],
    J_list: List[float],
    solver,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    tee: bool = False,
) -> pd.DataFrame:
    """
    Compute the loss matrix across (designs × scenarios).

    Output is a DataFrame with:
      - index = scenario_names
      - columns = "Loss with <design label>"
      - each cell = numeric loss (float rounded to 2 decimals) or "infeasible"

    There are two execution modes:

    Serial mode (parallel=False)
    ----------------------------
    Reuses the `solver` object passed in, and runs jobs in nested loops.
    This is simpler and usually easier to debug.

    Parallel mode (parallel=True)
    -----------------------------
    Uses ProcessPoolExecutor and calls `_solve_one_loss_job` in each worker. Each worker
    constructs its own IPOPT solver instance (safer, because solver objects are not
    reliably shareable across processes).

    If parallel execution fails for any reason, the function falls back to serial mode.

    Parameters
    ----------
    build_model:
        Callable returning a new Pyomo model each time it is called.
    user_designs:
        Dict mapping design label -> apply_design function.
        Each apply_design(model) should mutate the model to enforce that design.
    combos:
        List of scenario dicts (param -> value).
    scenario_names:
        List of scenario names aligned with `combos`.
    J_list:
        Base optimal objective values aligned with `combos`. Usually returned by
        `_solve_base_objectives`.
    solver:
        A configured solver object (used only in serial mode).
    Parallel:
        If True, run the loss jobs in parallel using multiple processes.
    n_workers:
        Number of worker processes. If None, ProcessPoolExecutor uses a default based on CPU.
    tee:
        If True, prints solver logs.

    Returns
    -------
    loss_matrix:
        DataFrame of losses (dtype object because it can hold floats or "infeasible").

    Raises
    ------
    ValueError
        If scenario_names, combos, and J_list lengths do not match.
    """
    if not (len(scenario_names) == len(combos) == len(J_list)):
        raise ValueError("scenario_names, combos, and J_list must have the same length.")

    col_labels = [f"Loss with {lbl}" for lbl in user_designs]
    loss_matrix = pd.DataFrame(index=scenario_names, columns=col_labels, dtype=object)

    scenarios = list(zip(scenario_names, combos, J_list))

    jobs: List[_LossJob] = []
    for dlabel in user_designs.keys():
        for sname, sc, J_nom in scenarios:
            jobs.append(_LossJob(dlabel, sname, sc, float(J_nom)))

    def run_serial() -> None:
        """
        Serial executor: loops over jobs, builds model, applies design, solves, stores loss.
        """
        for job in jobs:
            apply_design = user_designs[job.design_label]

            m = build_model()
            apply_design(m)

            active_objs = list(m.component_data_objects(Objective, active=True))
            if len(active_objs) != 1:
                raise ValueError(f"Model must have exactly 1 active Objective, found {len(active_objs)}.")
            obj = active_objs[0]

            for p, val in job.scenario.items():
                if not hasattr(m, p):
                    raise AttributeError(f"Model has no component named '{p}'.")
                getattr(m, p).set_value(val)

            results = solver.solve(m, tee=tee)
            tc = results.solver.termination_condition

            if not _is_good_termination(tc):
                loss = "infeasible"
            else:
                J_opt = value(obj.expr)
                loss_val = (J_opt - job.J_nom) if obj.is_minimizing() else (job.J_nom - J_opt)
                loss = round(float(loss_val), 2)

            loss_matrix.at[job.scenario_name, f"Loss with {job.design_label}"] = loss

    def run_parallel() -> None:
        """
        Parallel executor: submits independent jobs to worker processes.
        """
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for job in jobs:
                apply_design = user_designs[job.design_label]
                futs.append(
                    ex.submit(
                        _solve_one_loss_job,
                        build_model,
                        apply_design,
                        job,
                        tee,
                    )
                )

            for fut in as_completed(futs):
                dlabel, sname, loss = fut.result()
                loss_matrix.at[sname, f"Loss with {dlabel}"] = loss

    if parallel:
        try:
            run_parallel()
        except Exception:
            # If anything goes wrong (pickling, solver crash, etc.), just fall back to serial.
            run_serial()
    else:
        run_serial()

    return loss_matrix

def _build_impl_error_loss_row(
    build_model: Callable[[], pyo.ConcreteModel],
    user_designs: Dict[str, Callable[[pyo.ConcreteModel], None]],
    impl_error_designs: Dict[str, Callable[[pyo.ConcreteModel], None]],
    nominal: dict,
    J_nom: float,
    solver,
    tee: bool = False,
) -> pd.Series:
    """
    Compute the implementation-error loss row for each design.

    For implementation error evaluation, the disturbances are kept at their
    nominal values but the controlled variable setpoint is perturbed by a
    known measurement/implementation error dc. This corresponds to disturbance
    "dc" in Skogestad's SOC procedure (Step 5/6).

    The user provides `impl_error_designs`, a dict with the same keys as
    `user_designs`, but each apply_design function fixes/constrains the CV
    at its *perturbed* value (cs + dc) rather than the nominal optimal value.

    For example, if the nominal design is "M=1.0" (fixing M at 1.0), the
    implementation error design would be "M=1.1" (fixing M at 1.0 + 0.1).

    Parameters
    ----------
    build_model:
        Callable returning a fresh Pyomo model.
    user_designs:
        Dict mapping design label -> apply_design function (nominal setpoints).
        Used only for the column labels.
    impl_error_designs:
        Dict mapping the SAME design labels -> apply_design function with the
        perturbed setpoint (cs + dc). Must have the same keys as `user_designs`.
    nominal:
        The nominal scenario dict (disturbance parameter values at nominal).
    J_nom:
        The base optimal objective value at the nominal scenario.
    solver:
        A configured Pyomo solver object.
    tee:
        If True, prints solver output.

    Returns
    -------
    row:
        pd.Series with index = "Loss with <design label>" for each design,
        and values = numeric loss or "infeasible".

    Raises
    ------
    KeyError
        If `impl_error_designs` does not contain a key present in `user_designs`.
    """
    col_labels = [f"Loss with {lbl}" for lbl in user_designs]
    row = pd.Series(index=col_labels, dtype=object)

    for dlabel in user_designs:
        if dlabel not in impl_error_designs:
            raise KeyError(
                f"impl_error_designs is missing key '{dlabel}'. "
                f"It must have the same keys as user_designs."
            )

        apply_design = impl_error_designs[dlabel]

        m = build_model()
        apply_design(m)

        active_objs = list(m.component_data_objects(Objective, active=True))
        if len(active_objs) != 1:
            raise ValueError(
                f"Model must have exactly 1 active Objective, found {len(active_objs)}."
            )
        obj = active_objs[0]

        # Set disturbances to nominal values
        for p, val in nominal.items():
            if not hasattr(m, p):
                raise AttributeError(f"Model has no component named '{p}'.")
            getattr(m, p).set_value(val)

        results = solver.solve(m, tee=tee)
        tc = results.solver.termination_condition

        if not _is_good_termination(tc):
            loss = "infeasible"
        else:
            J_opt = value(obj.expr)
            loss_val = (J_opt - J_nom) if obj.is_minimizing() else (J_nom - J_opt)
            loss = round(float(loss_val), 2)

        row[f"Loss with {dlabel}"] = loss

    return row


def _mark_infeasible_designs(
    loss_matrix: pd.DataFrame,
    mark_all_scenarios: bool = False,
    mark_negative: bool = False,
) -> pd.DataFrame:
    """
    Clean up / standardize infeasibility markers in a loss matrix.

    This function is about *presentation consistency* and easier downstream logic.
    Depending on what happened earlier, an “infeasible” outcome might appear as:
      - NaN (missing)
      - the string "infeasible"
      - (optionally) a negative numeric loss, if you decide negative losses are invalid

    What it does:
      - Always returns an object-typed DataFrame
      - Turns NaNs into "infeasible" (either cell-wise or whole-column)
      - Optionally flags negative numeric values as "infeasible"

    Parameters
    ----------
    loss_matrix:
        DataFrame produced by `_build_raw_loss_matrix`.
        Cells may contain floats, NaNs, or strings.
    mark_all_scenarios:
        If False (default):
            Replace only the NaN cells with "infeasible".
        If True:
            If a column has *any* NaN, mark the entire column as "infeasible".
            (This is stricter: “if design fails once, treat it as unusable everywhere.”)
    mark_negative:
        If True, convert any negative numeric loss to "infeasible".
        This is optional because negative loss can happen due to numerical noise,
        sign conventions, or if the “design objective” differs from the “true objective”.
        Only enable this if negative loss is definitely meaningless for your use case.

    Returns
    -------
    display:
        A cleaned DataFrame where infeasible entries are consistently labeled "infeasible".

    Notes
    -----
    This function does *not* compute averages or rankings; it just normalizes labels.
    """
    display = loss_matrix.copy().astype(object)

    if mark_all_scenarios:
        for col in display.columns:
            if pd.isna(display[col]).any():
                display[col] = "infeasible"
    else:
        display[pd.isna(display)] = "infeasible"

    if mark_negative:
        numeric = pd.to_numeric(display.stack(), errors="coerce").unstack()
        neg_mask = numeric < 0
        display = display.mask(neg_mask, "infeasible")

    return display


def _append_average_and_ranking(loss_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary rows: average loss and rank for each design.

    Rules used here are intentionally simple (and match what you asked earlier):
      - If a design has *any* infeasible entry in its column, then:
            Average loss = "infeasible"
        (so we don't pretend we can average partial information)
      - Otherwise:
            Average loss = mean of the scenario losses (numeric)
      - Ranking is dense rank (1, 2, 3, ...), ascending by average loss.
        Infeasible designs are pushed to the bottom.

    Parameters
    ----------
    loss_matrix:
        DataFrame containing numeric losses and/or the string "infeasible".
        Typically after `_mark_infeasible_designs`.

    Returns
    -------
    out:
        Copy of the input DataFrame with two extra rows appended:
          - "Average loss"
          - "Ranking"

    Notes
    -----
    - Ranking uses `dense` method, so ties get the same rank and the next rank
      increments by 1 (not by the number of tied items).
    - Internally we compute numeric averages only on feasible cells; infeasible columns
      are labeled "infeasible" in the average row.
    """
    is_infeasible_cell = loss_matrix.isna() | loss_matrix.eq("infeasible")
    numeric = loss_matrix.mask(is_infeasible_cell).apply(pd.to_numeric, errors="coerce")

    infeas_col = is_infeasible_cell.any(axis=0)
    avg_num = numeric.mean()

    avg_display = avg_num.round(2).astype(object)
    avg_display[infeas_col] = "infeasible"

    rank_base = avg_num.copy()
    if (~infeas_col).any():
        push = avg_num[~infeas_col].max() + 1
    else:
        push = 1.0
    rank_base[infeas_col] = push
    rankings = rank_base.rank(method="dense", ascending=True)

    out = loss_matrix.copy()
    out.loc["Average loss"] = avg_display
    out.loc["Ranking"] = rankings.astype(int)

    return out


def run_self_optimizing_model(
    build_model,
    disturbances,
    user_designs,
    metrics,
    solver="ipopt",
    parallel: bool = False,
    n_workers=None,
    mark_negative_loss_infeasible=False,
    impl_error_designs: Optional[Dict[str, Callable]] = None,
    tee: bool = False,
):
    """
    Run the full SOC workflow: scenarios → base optimum → metrics → loss matrix → ranking.

    This is the main entry point you call from the outside. Conceptually, it does:

      A) Scenario generation
         - Use `_generate_scenarios(disturbances)` to get:
             combos, scenario_names, nominal

      B) “True” optimization at each scenario
         - Use IPOPT to solve the base economic objective at each scenario
         - Store J* for each scenario (J_list)

      C) Metric evaluation
         - Solve again at each scenario and compute user metrics at the optimum
         - Returned as a metric-by-scenario table

      D) SOC design loss evaluation
         - For each candidate design in `user_designs`, solve under each scenario
         - Compute loss relative to the base optimum J*
         - Label failures as "infeasible"
         - Optionally run in parallel

      E) Post-processing
         - Normalize infeasible labels
         - Add average loss and rank per design

    Parameters
    ----------
    build_model:
        Callable that returns a new Pyomo ConcreteModel.
        Must define disturbances as mutable components that can be `set_value(...)`.
        The “true” base objective must be accessible as `m.J`.
    disturbances:
        Dict mapping disturbance name -> list of values.
        Used to generate nominal and single-factor-change scenarios.
    user_designs:
        Dict mapping design label -> apply_design function.
        Each apply_design(model) mutates the model to represent that SOC design.
    metrics:
        Dict mapping metric name -> metric function.
        Each metric function has signature (model, scenario_dict) -> float.
    parallel:
        If True, evaluate loss jobs with ProcessPoolExecutor.
    n_workers:
        Number of worker processes (only used when parallel=True).
    mark_negative_loss_infeasible:
        If True, treat negative numeric losses as infeasible during labeling.
    impl_error_designs:
        Optional dict mapping the SAME design labels as `user_designs` to
        apply_design functions that fix/constrain the CV at the *perturbed*
        setpoint (cs + dc). When provided, an additional "Impl. error (dc)"
        row is appended to the loss matrix.

        Example: if user_designs has "M=1.0" (fixing M at 1.0) and dc for M
        is 10% (i.e. 0.1), then impl_error_designs should have "M=1.0" mapped
        to a function that fixes M at 1.1.

        The implementation error loss is evaluated at the nominal disturbance
        scenario and is included in the average loss and ranking calculations.
    tee:
        If True, prints solver output (useful for debugging, noisy for large runs).

    Returns
    -------
    results_df:
        DataFrame of metrics at the true optimum:
            rows = metrics, columns = scenarios
    loss_matrix:
        DataFrame of scenario-wise losses for each design, plus:
            - "Average loss" row
            - "Ranking" row

    Raises
    ------
    RuntimeError
        If IPOPT is not available, or if base/metric solves fail for any scenario.
    AttributeError
        If disturbances refer to components that are missing on the built model.

    Tip
    -------------
    If you see lots of "infeasible" in the loss matrix, run once with `tee=True` and
    `parallel=False` to get cleaner solver logs and isolate which scenario/design fails.
    """
    combos, scenario_names, nominal = _generate_scenarios(disturbances)

    # Always IPOPT
    solver = _make_solver()

    # Base & metrics at true optimum
    J_list = _solve_base_objectives(build_model, combos, solver, tee=tee)
    results_df = _evaluate_metrics(build_model, metrics, combos, scenario_names, solver, tee=tee)

    # Loss matrix for each design (disturbance scenarios)
    loss_matrix = _build_raw_loss_matrix(
        build_model=build_model,
        user_designs=user_designs,
        combos=combos,
        scenario_names=scenario_names,
        J_list=J_list,
        solver=solver,
        parallel=parallel,
        n_workers=n_workers,
        tee=tee,
    )

    # Implementation error row (optional)
    if impl_error_designs is not None:
        # J_nom for the nominal scenario is J_list[0] (first combo is always nominal)
        J_nom_nominal = J_list[0]

        impl_row = _build_impl_error_loss_row(
            build_model=build_model,
            user_designs=user_designs,
            impl_error_designs=impl_error_designs,
            nominal=nominal,
            J_nom=J_nom_nominal,
            solver=solver,
            tee=tee,
        )

        loss_matrix.loc["Impl. error (dc)"] = impl_row

    # Normalize infeasible labels + ranking
    loss_matrix = _mark_infeasible_designs(loss_matrix, mark_negative=mark_negative_loss_infeasible)
    loss_matrix = _append_average_and_ranking(loss_matrix)

    return results_df, loss_matrix