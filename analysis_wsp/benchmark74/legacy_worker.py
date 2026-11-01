from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .common import ILS_PARAMETERS, atomic_write_json, utc_now


def lbbd_stage_limit(deadline: float, now: float, greedy_stages_remaining: int) -> float:
    remaining = max(0.0, deadline - now)
    return remaining / max(1, greedy_stages_remaining)


def lbbd_objective_audit(model_objective: float, validated_objective: int, lower_bound: float) -> dict[str, Any]:
    """Use the feasible deployment's independently recomputed score canonically.

    A time-limited Benders master may retain conservative DoesBurn values even
    though the returned deployment protects one or more additional vertices.
    The deterministic graph recomputation is the benchmark's ground truth; the
    master value remains in the result as audit telemetry.
    """
    model_value = int(round(model_objective))
    validated_value = int(validated_objective)
    bound = float(lower_bound)
    if bound > validated_value + 1e-6:
        raise ValueError(
            f"LBBD lower bound {bound} exceeds validated feasible objective {validated_value}"
        )
    denominator = max(1.0, abs(float(validated_value)))
    return {
        "objective": validated_value,
        "objective_source": "validated_recomputation",
        "model_objective": model_value,
        "objective_delta": validated_value - model_value,
        "objective_consistent": model_value == validated_value,
        "mip_gap": max(0.0, (validated_value - bound) / denominator),
    }


@contextmanager
def enforce_lbbd_global_budget(deadline: float, greedy_stages: int):
    import gurobipy as gp

    original = gp.Model.optimize
    telemetry: dict[str, Any] = {"optimize_calls": 0, "allocated_seconds": [], "baseline_mip_starts": 0}

    def bounded_optimize(model, callback=None):
        call = int(telemetry["optimize_calls"])
        if call == 0:
            variables = model.getVars()
            for variable in variables:
                variable.Start = 1.0 if variable.Obj > 0.0 else 0.0
            telemetry["baseline_mip_starts"] = 1
        stages_left = max(0, greedy_stages - call)
        limit = lbbd_stage_limit(deadline, time.monotonic(), stages_left)
        model.setParam("TimeLimit", limit)
        telemetry["optimize_calls"] = call + 1
        telemetry["allocated_seconds"].append(limit)
        return original(model, callback)

    gp.Model.optimize = bounded_optimize
    try:
        yield telemetry
    finally:
        gp.Model.optimize = original


def legacy_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "Logic-based-Benders-decomposition-for-wildfire-suppression"


def load_instance(path: Path) -> tuple[list[Any], dict[Any, float], list[Any], float, float, dict[float, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = [tuple(node) for node in data["Nodes"]]
    arcs = {ast.literal_eval(edge): float(weight) for edge, weight in data["Arcs"].items()}
    ignitions = [tuple(node) for node in data["Ignitions"]]
    resources = {float(key): int(value) for key, value in data["ResAtTime"].items()}
    return nodes, arcs, ignitions, float(data["Delay"]), float(data["ArrivalTimeTarget"]), resources


def validate_solution(
    nodes: list[Any],
    arcs: dict[Any, float],
    ignitions: list[Any],
    delay: float,
    horizon: float,
    resources: dict[float, int],
    assignments: list[tuple[Any, float]],
) -> dict[str, Any]:
    sys.path.insert(0, str(legacy_root()))
    from shortest_paths import ShortestPaths

    if len({node for node, _time in assignments}) != len(assignments):
        raise ValueError("A node received more than one resource")
    for release_time, capacity in resources.items():
        if sum(abs(time_value - release_time) < 1e-9 for _node, time_value in assignments) > capacity:
            raise ValueError(f"Capacity exceeded at release time {release_time}")
    in_arcs = {node: [] for node in nodes}
    out_arcs = {node: [] for node in nodes}
    for edge in arcs:
        out_arcs[edge[0]].append(edge)
        in_arcs[edge[1]].append(edge)
    protected = {node for node, _time in assignments}
    arrival, _path, _pred = ShortestPaths(nodes, arcs, in_arcs, out_arcs, protected, ignitions, delay)
    for node, release_time in assignments:
        if arrival[node] + 1e-9 < release_time:
            raise ValueError(f"Resource at {node} burns before release time {release_time}")
    objective = sum(arrival[node] < horizon for node in nodes)
    return {"objective_recomputed": int(objective), "assignments": len(assignments), "feasible": True}


def run_ils(instance: Path, seed: int, time_limit: int) -> dict[str, Any]:
    sys.path.insert(0, str(legacy_root()))
    from IteratedLocalSearch import FireILS

    if "Deadline" not in inspect.signature(FireILS).parameters:
        raise RuntimeError("Strict ILS deadline instrumentation is absent; run the benchmark74 bootstrap first")
    nodes, arcs, ignitions, delay, horizon, resources = load_instance(instance)
    np.random.seed(seed)
    started = time.monotonic()
    search_started = time.monotonic()
    deadline = search_started + time_limit
    solution, objective, telemetry = FireILS(
        nodes,
        arcs,
        resources,
        ignitions,
        delay,
        horizon,
        ILS_PARAMETERS["MultiStarts"],
        ILS_PARAMETERS["p1"],
        ILS_PARAMETERS["p2"],
        ILS_PARAMETERS["MaxNeighbours"],
        ILS_PARAMETERS["MaxModifications"],
        ILS_PARAMETERS["MaxFailures"],
        ILS_PARAMETERS["MaxNoImprovements"],
        ILS_PARAMETERS["MaxCandidates"],
        Deadline=deadline,
    )
    search_finished = time.monotonic()
    assignments = [(node, float(release)) for (node, release), value in solution.items() if value > 0.5]
    try:
        validation = validate_solution(nodes, arcs, ignitions, delay, horizon, resources, assignments)
    except ValueError as exc:
        baseline_validation = validate_solution(nodes, arcs, ignitions, delay, horizon, resources, [])
        return {
            "_status": "algorithm_failure",
            "algorithm": "ils",
            "seed": seed,
            "objective": None,
            "scored_objective": int(baseline_validation["objective_recomputed"]),
            "lower_bound": None,
            "solver_status": "invalid_final_candidate",
            "wall_seconds": time.monotonic() - started,
            "search_limit_seconds": time_limit,
            "search_wall_seconds": search_finished - search_started,
            "deadline_overshoot_seconds": max(0.0, search_finished - deadline),
            "termination_reason": "invalid_final_candidate",
            "complete_candidates": int(telemetry["complete_candidates"]),
            "failure_category": "infeasible_final_solution",
            "validation_error": str(exc),
            "invalid_candidate_assignments": len(assignments),
            "fallback_policy": "discard_invalid_and_score_no_deployment_baseline",
            "baseline_solution": [],
            "baseline_validation": baseline_validation,
        }
    if int(objective) != validation["objective_recomputed"]:
        raise ValueError(f"ILS objective {objective} differs from recomputation {validation['objective_recomputed']}")
    return {
        "algorithm": "ils",
        "seed": seed,
        "objective": int(objective),
        "lower_bound": None,
        "solver_status": telemetry["termination_reason"],
        "wall_seconds": time.monotonic() - started,
        "search_limit_seconds": time_limit,
        "search_wall_seconds": search_finished - search_started,
        "deadline_overshoot_seconds": max(0.0, search_finished - deadline),
        "termination_reason": telemetry["termination_reason"],
        "complete_candidates": int(telemetry["complete_candidates"]),
        "solution": [[list(node), release] for node, release in assignments],
        "validation": validation,
    }


def run_lbbd(instance: Path, seed: int, time_limit: int) -> dict[str, Any]:
    sys.path.insert(0, str(legacy_root()))
    from model_LBBD import FireLBBD

    nodes, arcs, ignitions, delay, horizon, resources = load_instance(instance)
    started = time.monotonic()
    deadline = started + time_limit
    with enforce_lbbd_global_budget(deadline, len(resources)) as budget:
        try:
            greedy, greedy_solution = FireLBBD(
                nodes, arcs, resources, ignitions, delay, horizon, time_limit, seed, None, True
            )
        except AttributeError as exc:
            if "Unable to retrieve attribute 'x'" not in str(exc):
                raise
            validation = validate_solution(nodes, arcs, ignitions, delay, horizon, resources, [])
            return {
                "algorithm": "lbbd", "seed": seed,
                "objective": int(validation["objective_recomputed"]),
                "objective_source": "validated_recomputation",
                "model_objective": None,
                "objective_delta": None,
                "objective_consistent": None,
                "lower_bound": None,
                "solver_status": "no_incumbent_baseline",
                "mip_gap": None,
                "model_mip_gap": None,
                "wall_seconds": time.monotonic() - started,
                "greedy_wall_seconds": time.monotonic() - started,
                "exact_started": False, "solution": [], "validation": validation,
                "optimality_cuts": 0, "feasibility_cuts": 0,
                "global_budget_seconds": time_limit,
                "optimize_calls": int(budget["optimize_calls"]),
                "optimize_time_allocations_seconds": budget["allocated_seconds"],
                "baseline_mip_starts": int(budget["baseline_mip_starts"]),
                "budget_policy": "global_remaining_divided_among_remaining_greedy_stages",
                "fallback_policy": "validated_no_deployment_baseline_when_no_incumbent",
            }
        elapsed = time.monotonic() - started
        remaining = max(0.0, deadline - time.monotonic())
        final_model = greedy
        final_solution = greedy_solution
        exact_started = False
        if remaining >= 1.0:
            exact_started = True
            try:
                exact, exact_solution = FireLBBD(
                    nodes, arcs, resources, ignitions, delay, horizon, remaining, seed, greedy_solution, False
                )
                if int(exact.SolCount) > 0:
                    final_model = exact
                    final_solution = exact_solution
            except AttributeError as exc:
                if "Unable to retrieve attribute 'x'" not in str(exc):
                    raise
                exact_started = False
    assignments = [(node, float(release)) for node, release in final_solution]
    validation = validate_solution(nodes, arcs, ignitions, delay, horizon, resources, assignments)
    scoring = lbbd_objective_audit(
        final_model.ObjVal,
        int(validation["objective_recomputed"]),
        final_model.ObjBound,
    )
    return {
        "algorithm": "lbbd",
        "seed": seed,
        **scoring,
        "lower_bound": float(final_model.ObjBound),
        "solver_status": int(final_model.Status),
        "model_mip_gap": float(final_model.MIPGap) if int(round(final_model.ObjVal)) else 0.0,
        "wall_seconds": time.monotonic() - started,
        "greedy_wall_seconds": elapsed,
        "exact_started": exact_started,
        "global_budget_seconds": time_limit,
        "optimize_calls": int(budget["optimize_calls"]),
        "optimize_time_allocations_seconds": budget["allocated_seconds"],
        "baseline_mip_starts": int(budget["baseline_mip_starts"]),
        "budget_policy": "global_remaining_divided_among_remaining_greedy_stages",
        "solution": [[list(node), release] for node, release in assignments],
        "validation": validation,
        "optimality_cuts": int(final_model._OptimalityCuts),
        "feasibility_cuts": int(final_model._FeasibilityCuts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ils", "lbbd"), required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timelimit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_ils(args.instance, args.seed, args.timelimit) if args.algorithm == "ils" else run_lbbd(args.instance, args.seed, args.timelimit)
    status = result.pop("_status", "completed")
    provenance = "measured" if status == "completed" else "measured_algorithm_failure"
    result.update({"status": status, "completed_utc": utc_now(), "result_provenance": provenance})
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
