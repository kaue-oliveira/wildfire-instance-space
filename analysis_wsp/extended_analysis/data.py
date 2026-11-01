from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


META_COLUMNS = {"instance", "group", "experiment_group", "source_zip", "result_provenance"}
ALGORITHMS = ("CUTH", "IBS", "ILS", "LBBD", "MIP", "RS")


def load_wildfire_instances(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    instance_dir = root / "third_party" / "Wildfire-Suppression" / "instances"
    for zip_path in sorted(instance_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json"):
                    continue
                data = json.loads(archive.read(member).decode("utf-8"))
                name = Path(member).stem
                records.setdefault(
                    name,
                    {
                        "instance": name,
                        "group": "literature" if zip_path.stem == "literature" else "generated",
                        "experiment_group": zip_path.stem,
                        "source_zip": zip_path.name,
                        "result_provenance": "original_source",
                        "data": data,
                    },
                )
    if len(records) != 55:
        raise ValueError(f"Baseline esperado com 55 instâncias; encontrado: {len(records)}")
    return records


def instance_content_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dijkstra(n: int, arcs: list[tuple[int, int, float]], sources: list[int]) -> list[float]:
    import heapq

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, weight in arcs:
        adj[u].append((v, weight))
    dist = [math.inf] * n
    queue: list[tuple[float, int]] = []
    for source in sources:
        dist[source] = 0.0
        heapq.heappush(queue, (0.0, source))
    while queue:
        du, u = heapq.heappop(queue)
        if du != dist[u]:
            continue
        for v, weight in adj[u]:
            candidate = du + weight
            if candidate < dist[v]:
                dist[v] = candidate
                heapq.heappush(queue, (candidate, v))
    return dist


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {f"{prefix}_{key}": math.nan for key in ("mean", "std", "min", "max", "p25", "p50", "p75", "cv")}
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_p25": float(np.quantile(arr, 0.25)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p75": float(np.quantile(arr, 0.75)),
        f"{prefix}_cv": std / mean if mean else 0.0,
    }


def base_features(record: dict[str, Any]) -> dict[str, Any]:
    data = record["data"]
    n = int(data["|V|"])
    horizon = float(data["H"])
    arcs = [(int(u), int(v), float(w)) for u, v, w in data["arcs"]]
    times = [w for _, _, w in arcs]
    releases = [float(v) for v in data.get("t", [])]
    capacities = [float(v) for v in data.get("c", [])]
    delays = [float(v) for v in data.get("delta", [])]
    arrivals = dijkstra(n, arcs, [int(v) for v in data.get("I", [])])
    finite = np.asarray([v for v in arrivals if math.isfinite(v)], dtype=float)
    coords = data.get("distance", {}).get("coordinates", [])
    xs = [float(v[0]) for v in coords]
    ys = [float(v[1]) for v in coords]
    zs = [float(v[2]) for v in coords if len(v) > 2]
    weights = {(u, v): w for u, v, w in arcs}
    asymmetry: list[float] = []
    for (u, v), weight in weights.items():
        if u < v and (v, u) in weights:
            reverse = weights[(v, u)]
            denom = (weight + reverse) / 2.0
            asymmetry.append(abs(weight - reverse) / denom if denom else 0.0)
    burned_h = int(np.sum(finite < horizon))
    row: dict[str, Any] = {
        "instance": record["instance"],
        "group": record["group"],
        "experiment_group": record["experiment_group"],
        "source_zip": record["source_zip"],
        "result_provenance": "measured",
        "instance_sha256": instance_content_hash(data),
        "n_vertices": n,
        "n_arcs": len(arcs),
        "arc_density": len(arcs) / n if n else 0.0,
        "horizon": horizon,
        "total_resources": float(sum(capacities)),
        "decision_points": len(releases),
        "resources_per_decision": float(sum(capacities) / len(releases)) if releases else 0.0,
        "delta_mean": float(np.mean(delays)) if delays else 0.0,
        "delta_over_h": float(np.mean(delays) / horizon) if delays and horizon else 0.0,
        "first_release_over_h": min(releases) / horizon if releases and horizon else 0.0,
        "last_release_over_h": max(releases) / horizon if releases and horizon else 0.0,
        "release_window_over_h": (max(releases) - min(releases)) / horizon if len(releases) > 1 and horizon else 0.0,
        "free_burning_time": float(finite.max()) if finite.size else math.nan,
        "free_burning_over_h": float(finite.max() / horizon) if finite.size and horizon else math.nan,
        "free_burned_fraction_h": burned_h / n if n else 0.0,
        "arrival_q05_over_h": float(np.quantile(finite, 0.05) / horizon) if finite.size and horizon else math.nan,
        "arrival_q50_over_h": float(np.quantile(finite, 0.50) / horizon) if finite.size and horizon else math.nan,
        "arrival_q95_over_h": float(np.quantile(finite, 0.95) / horizon) if finite.size and horizon else math.nan,
        "directional_asymmetry_mean": float(np.mean(asymmetry)) if asymmetry else 0.0,
        "directional_asymmetry_max": float(np.max(asymmetry)) if asymmetry else 0.0,
        "x_span": max(xs) - min(xs) if xs else 0.0,
        "y_span": max(ys) - min(ys) if ys else 0.0,
        "z_span": max(zs) - min(zs) if zs else 0.0,
        "z_std": float(np.std(zs)) if zs else 0.0,
    }
    row.update(_stats(times, "arc_time"))
    row.update(_release_perimeter(n, arcs, arrivals, releases, horizon))
    return row


def _release_perimeter(
    n: int,
    arcs: list[tuple[int, int, float]],
    arrivals: list[float],
    releases: list[float],
    horizon: float,
) -> dict[str, float]:
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v, _ in arcs:
        adj[u].append(v)
    perimeters: list[int] = []
    windows: list[int] = []
    ordered = sorted(releases)
    for idx, time in enumerate(ordered):
        next_time = ordered[idx + 1] if idx + 1 < len(ordered) else horizon
        burned = [arrival < time for arrival in arrivals]
        perimeters.append(sum(burned[u] and any(not burned[v] for v in adj[u]) for u in range(n)))
        windows.append(sum(time <= arrival < next_time for arrival in arrivals))
    return {
        "release_perimeter_mean": float(np.mean(perimeters)) if perimeters else 0.0,
        "release_perimeter_max": float(np.max(perimeters)) if perimeters else 0.0,
        "release_front_window_mean": float(np.mean(windows)) if windows else 0.0,
        "release_front_window_max": float(np.max(windows)) if windows else 0.0,
    }


def load_original_features(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "analysis_wsp" / "data" / "features_instances.csv")
    if frame["instance"].nunique() != 55:
        raise ValueError("Tabela original não contém exatamente 55 instâncias únicas")
    frame["result_provenance"] = "original_source"
    return frame


def load_performance_matrix(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(root / "analysis_wsp" / "data" / "performance_by_algorithm.csv")
    raw["result_provenance"] = "original_source"
    pivot = raw.pivot(index="instance", columns="algorithm_norm", values="mean_gap").reset_index()
    missing = sorted(set(ALGORITHMS) - set(pivot.columns))
    if missing:
        raise ValueError(f"Algoritmos ausentes no baseline: {missing}")
    return raw, pivot[["instance", *ALGORITHMS]]


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in META_COLUMNS | {"instance_sha256"} and pd.api.types.is_numeric_dtype(frame[col])
    ]
