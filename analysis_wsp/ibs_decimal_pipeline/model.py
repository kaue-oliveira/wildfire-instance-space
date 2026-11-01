"""Independent Dijkstra audit using Decimal input and exact centiseconds.

This code does not call any C++ objective or feasibility routine. It checks
every distinct logged allocation, not only the incumbent or final row.
"""
from __future__ import annotations

import ast
import heapq
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .common import read_csv


def centi(value):
    number = Decimal(str(value)) * 100
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"Time is not exactly representable in hundredths: {value}")
    return int(number)


def load_model(path, truncated=False):
    data = json.loads(Path(path).read_text(), parse_float=Decimal)
    ids = {tuple(node): i for i, node in enumerate(data["Nodes"])}
    if len(ids) != len(data["Nodes"]):
        raise ValueError("Duplicate terrain coordinates")
    conversion = (lambda v: int(Decimal(str(v)))) if truncated else centi
    adjacency = [[] for _ in ids]
    for edge, weight in data["Arcs"].items():
        source, target = ast.literal_eval(edge)
        adjacency[ids[source]].append((ids[target], conversion(weight)))
    releases = {}
    for instant, quantity in data["ResAtTime"].items():
        time = conversion(instant)
        if time in releases:
            raise ValueError("Release instants collide")
        releases[time] = int(quantity)
    return {"n": len(ids), "ids": ids, "adjacency": adjacency,
            "horizon": conversion(data["ArrivalTimeTarget"]), "delay": conversion(data["Delay"]),
            "ignition": ids[tuple(data["Ignitions"][0])], "releases": releases,
            "truncated": truncated}


def signature_allocations(signature, truncated=False):
    if signature == "EMPTY":
        return []
    result = []
    for token in signature.split("-"):
        vertex, instant = token.split("@")
        if truncated:
            time = int(instant)
        else:
            time = centi(instant)
        result.append((int(vertex), time))
    if result != sorted(set(result)):
        raise ValueError("Noncanonical or duplicated solution signature")
    return result


def evaluate(model, allocations):
    n = model["n"]
    if len({v for v, _ in allocations}) != len(allocations):
        raise ValueError("A vertex received multiple resources")
    if any(v < 0 or v >= n for v, _ in allocations):
        raise ValueError("Allocation vertex is outside terrain")
    for instant, count in Counter(t for _, t in allocations).items():
        if instant not in model["releases"] or count > model["releases"][instant]:
            raise ValueError("Allocation exceeds release capacity or has an invalid release time")
    protected = {v for v, _ in allocations}
    arrival = [10**40] * n
    arrival[model["ignition"]] = 0
    heap = [(0, model["ignition"])]
    while heap:
        instant, vertex = heapq.heappop(heap)
        if instant != arrival[vertex]:
            continue
        extra = model["delay"] if vertex in protected else 0
        for neighbor, weight in model["adjacency"][vertex]:
            candidate = instant + weight + extra
            if candidate < arrival[neighbor]:
                arrival[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    violations = sum(arrival[v] < t for v, t in allocations)
    if violations:
        raise ValueError(f"Resources were deployed after fire arrival: {violations}")
    return sum(t < model["horizon"] for t in arrival), arrival


def validate_trajectory(path, model, seed, cache=None):
    cache = {} if cache is None else cache
    rows = read_csv(path)
    if not rows:
        raise ValueError("Empty trajectory")
    for row in rows:
        if set(row) != {"Run", "Fitness", "Solution"} or int(row["Run"]) != seed:
            raise ValueError("Invalid trajectory columns or seed")
        fitness = Decimal(row["Fitness"])
        if fitness != fitness.to_integral_value() or not 0 <= fitness <= model["n"]:
            raise ValueError("Invalid objective")
        sig = row["Solution"]
        if sig not in cache:
            allocations = signature_allocations(sig, model["truncated"])
            cache[sig] = evaluate(model, allocations)[0]
        if cache[sig] != int(fitness):
            raise ValueError(f"Objective disagrees with independent Dijkstra: {sig}: {fitness} != {cache[sig]}")
    return {"rows": len(rows), "unique_states": len({r['Solution'] for r in rows}),
            "best_objective": min(int(r["Fitness"]) for r in rows),
            "last_objective": int(rows[-1]["Fitness"]), "all_states_feasible": True}, rows


def validate_saved_solution(path, model):
    data = json.loads(Path(path).read_text(), parse_float=Decimal)
    allocation = [(model["ids"][tuple(v)], centi(t)) for v, t in data["resourceAllocation"]]
    objective, arrival = evaluate(model, allocation)
    if objective != int(data["objv"]):
        raise ValueError("Saved best solution has a divergent objective")
    for vertex, instant in data["fireArrivalTime"]:
        observed = centi(instant)
        expected = arrival[model["ids"][tuple(vertex)]]
        if expected != observed:
            raise ValueError(f"Saved fire arrival differs from Dijkstra: {vertex}")
    signature = "-".join(f"{v}@{t // 100}.{t % 100:02d}" for v, t in sorted(allocation)) or "EMPTY"
    return objective, signature
