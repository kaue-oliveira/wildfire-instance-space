from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from .provenance import write_json

try:
    import igraph as ig
except ImportError:  # pragma: no cover - dependency is pinned for experiments
    ig = None


AGGREGATES = ("mean", "std", "min", "q25", "q50", "q75", "q90", "max")
TOPOLOGY_FAMILIES = {
    "degree": ("in_degree", "out_degree", "degree_ratio", "degree_imbalance"),
    "pagerank": ("pagerank_unweighted", "pagerank_strength"),
    "hits": ("hits_hub_unweighted", "hits_authority_unweighted", "hits_hub_strength", "hits_authority_strength"),
    "betweenness": ("betweenness_unweighted", "betweenness_distance"),
    "harmonic": ("harmonic_in_unweighted", "harmonic_out_unweighted", "harmonic_in_distance", "harmonic_out_distance"),
    "clustering": ("directed_clustering",),
}

# HITS is part of the measured feature definition. NetworkX 3.4.2 delegates
# ``hits`` to ARPACK through ``svds`` but does not pass a random state, so two
# calls with the same graph and the same ``nstart`` can return different
# vectors when the leading singular value is repeated. The fixed backend
# below keeps the same singular-vector definition and sum-to-one
# normalisation, while making the numerical solver state explicit.
HITS_BACKEND_ID = "scipy_svds_seeded_random_state_v1"
# A cache is valid only when every exact metric uses the same implementation as
# the one used to project GA candidates.  HITS alone is not enough to identify
# that contract: betweenness and harmonic centrality must also be canonical.
TOPOLOGY_BACKEND_ID = "hybrid_exact_igraph_betweenness_harmonic_scipy_svds_seeded_random_state_v2"


def _topology_cache_compatible(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("topology_backend_id") == TOPOLOGY_BACKEND_ID
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _topology_cache_has_features(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(payload.get("features"), dict) and bool(payload["features"])
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


@dataclass(frozen=True)
class TopologyResult:
    features: dict[str, float]
    diagnostics: dict[str, Any]


def graph_fingerprint(data: dict[str, Any]) -> str:
    canonical = {
        "n": int(data["|V|"]),
        "arcs": sorted((int(u), int(v), float(w)) for u, v, w in data["arcs"]),
    }
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_graph(data: dict[str, Any], weighted: bool = False) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(data["|V|"])))
    for u, v, travel_time in data["arcs"]:
        travel = float(travel_time)
        if not math.isfinite(travel) or travel <= 0:
            raise ValueError(f"Tempo de propagação inválido: {(u, v, travel)}")
        graph.add_edge(
            int(u),
            int(v),
            distance=travel,
            strength=1.0 / travel,
            weight=(1.0 / travel) if weighted else 1.0,
        )
    return graph


def _igraph_graph(data: dict[str, Any]) -> tuple[Any, list[float]]:
    """Build the compiled directed graph used by exact centralities."""
    if ig is None:
        raise RuntimeError("python-igraph is required for the exact compiled topology backend")
    n = int(data["|V|"])
    edges: list[tuple[int, int]] = []
    distances: list[float] = []
    for u, v, travel_time in data["arcs"]:
        travel = float(travel_time)
        if not math.isfinite(travel) or travel <= 0:
            raise ValueError(f"Tempo de propagação inválido: {(u, v, travel)}")
        edges.append((int(u), int(v)))
        distances.append(travel)
    return ig.Graph(n=n, edges=edges, directed=True), distances


def _igraph_exact_betweenness(data: dict[str, Any], weighted: bool) -> dict[str, float]:
    graph, distances = _igraph_graph(data)
    n = int(data["|V|"])
    values = np.asarray(
        graph.betweenness(directed=True, weights=distances if weighted else None),
        dtype=float,
    )
    # igraph returns unnormalised directed counts. This is NetworkX's exact
    # directed normalisation when endpoints=False.
    if n > 2:
        values /= (n - 1) * (n - 2)
    else:
        values[:] = 0.0
    return aggregate(values, "betweenness_distance" if weighted else "betweenness_unweighted")


def _igraph_exact_harmonic_direction(
    data: dict[str, Any],
    weighted: bool,
    mode: str,
) -> dict[int, float]:
    """Calculate one exact directed harmonic-centrality orientation.

    ``igraph.harmonic_centrality`` performs a separate all-sources shortest-path
    calculation for each orientation.  GA projection commonly needs only one
    orientation selected by SIFTED, so computing both would double the dominant
    exact-search cost without contributing to the candidate fitness.
    """
    if mode not in {"in", "out"}:
        raise ValueError(f"Modo de centralidade harmônica inválido: {mode}")
    graph, distances = _igraph_graph(data)
    weights = distances if weighted else None
    values = np.asarray(graph.harmonic_centrality(mode=mode, weights=weights, normalized=True), dtype=float)
    return {node: float(value) for node, value in enumerate(values)}


def _igraph_exact_harmonic(data: dict[str, Any], weighted: bool) -> tuple[dict[int, float], dict[int, float]]:
    incoming = _igraph_exact_harmonic_direction(data, weighted, "in")
    outgoing = _igraph_exact_harmonic_direction(data, weighted, "out")
    return (
        incoming,
        outgoing,
    )


def _igraph_exact_hits(data: dict[str, Any], weighted: bool) -> tuple[dict[int, float], dict[int, float]]:
    """Compiled HITS eigenvectors with NetworkX's sum-to-one normalisation."""
    graph, distances = _igraph_graph(data)
    strengths = [1.0 / value for value in distances] if weighted else None
    hubs = np.abs(np.asarray(graph.hub_score(weights=strengths, scale=False), dtype=float))
    authorities = np.abs(np.asarray(graph.authority_score(weights=strengths, scale=False), dtype=float))
    hub_sum, authority_sum = float(hubs.sum()), float(authorities.sum())
    if hub_sum <= 0 or authority_sum <= 0 or not np.isfinite(hubs).all() or not np.isfinite(authorities).all():
        raise RuntimeError("HITS compilado degenerou em vetor não normalizável")
    hubs /= hub_sum
    authorities /= authority_sum
    return (
        {node: float(value) for node, value in enumerate(hubs)},
        {node: float(value) for node, value in enumerate(authorities)},
    )


def aggregate(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"topo_{prefix}_{name}": math.nan for name in AGGREGATES}
    return {
        f"topo_{prefix}_mean": float(np.mean(arr)),
        f"topo_{prefix}_std": float(np.std(arr)),
        f"topo_{prefix}_min": float(np.min(arr)),
        f"topo_{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"topo_{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"topo_{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"topo_{prefix}_q90": float(np.quantile(arr, 0.90)),
        f"topo_{prefix}_max": float(np.max(arr)),
    }


def _hits_repeated(graph: nx.DiGraph, seeds: tuple[int, ...]) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    nodes = list(graph.nodes())
    adjacency = nx.adjacency_matrix(graph, nodelist=nodes, dtype=float)
    hubs_runs: list[np.ndarray] = []
    auth_runs: list[np.ndarray] = []
    failures: list[str] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        start = rng.uniform(0.5, 1.5, size=len(nodes))
        nstart = dict(zip(nodes, start / start.sum()))
        try:
            _, _, vt = svds(
                adjacency,
                k=1,
                v0=np.asarray([nstart[node] for node in nodes], dtype=float),
                maxiter=1000,
                tol=1e-12,
                random_state=int(seed),
            )
            authorities = np.asarray(vt.flatten().real, dtype=float)
            hubs = np.asarray(adjacency @ authorities, dtype=float).ravel()
            if not np.isfinite(hubs).all() or not np.isfinite(authorities).all():
                raise FloatingPointError("HITS retornou vetor não finito")
            hub_sum, authority_sum = float(hubs.sum()), float(authorities.sum())
            if hub_sum == 0.0 or authority_sum == 0.0:
                raise FloatingPointError("HITS retornou vetor com soma zero")
            hubs /= hub_sum
            authorities /= authority_sum
            hubs_runs.append(hubs)
            auth_runs.append(authorities)
        except Exception as exc:  # research code must retain the concrete failure
            failures.append(f"{type(exc).__name__}: {exc}")
    if not hubs_runs:
        nan_values = {node: math.nan for node in nodes}
        return nan_values, nan_values.copy(), {
            "backend": HITS_BACKEND_ID,
            "initialization_seeds": list(seeds),
            "failures": len(failures),
            "failure_messages": failures,
            "max_abs_range": math.inf,
            "stable_at_1e-6": False,
        }
    hubs_arr = np.vstack(hubs_runs)
    auth_arr = np.vstack(auth_runs)
    max_abs_range = max(
        float(np.nanmax(np.ptp(hubs_arr, axis=0))),
        float(np.nanmax(np.ptp(auth_arr, axis=0))),
    )
    hubs_mean = np.nanmean(hubs_arr, axis=0)
    auth_mean = np.nanmean(auth_arr, axis=0)
    return (
        dict(zip(nodes, hubs_mean)),
        dict(zip(nodes, auth_mean)),
        {
            "backend": HITS_BACKEND_ID,
            "initialization_seeds": list(seeds),
            "repetitions": len(seeds),
            "successful_repetitions": len(hubs_runs),
            "failures": len(failures),
            "failure_messages": failures,
            "max_abs_range": max_abs_range,
            "stable_at_1e-6": bool(max_abs_range <= 1e-6),
        },
    )


def _hits_topology(data: dict[str, Any]) -> TopologyResult:
    """Compute only HITS features for cache migration and audit isolation."""
    graph = build_graph(data, weighted=False)
    weighted_graph = build_graph(data, weighted=True)
    hubs, authorities, unweighted_diagnostics = _hits_repeated(graph, (11, 23, 37, 53, 71))
    hubs_weighted, authorities_weighted, weighted_diagnostics = _hits_repeated(
        weighted_graph,
        (11, 23, 37, 53, 71),
    )
    features: dict[str, float] = {}
    for values, name in (
        (hubs.values(), "hits_hub_unweighted"),
        (authorities.values(), "hits_authority_unweighted"),
        (hubs_weighted.values(), "hits_hub_strength"),
        (authorities_weighted.values(), "hits_authority_strength"),
    ):
        features.update(aggregate(values, name))
    return TopologyResult(
        features=features,
        diagnostics={
            "hits_unweighted": unweighted_diagnostics,
            "hits_strength": weighted_diagnostics,
            "topology_hits_backend_id": HITS_BACKEND_ID,
            "result_provenance": "measured",
        },
    )


def _harmonic(graph: nx.DiGraph, weighted: bool, batch_size: int = 256) -> tuple[dict[int, float], dict[int, float]]:
    """Exact directed harmonic centrality using bounded-memory sparse batches."""
    nodes = np.asarray(sorted(graph.nodes()), dtype=int)
    index = {int(node): position for position, node in enumerate(nodes)}
    rows, cols, weights = [], [], []
    for u, v, attrs in graph.edges(data=True):
        rows.append(index[int(u)])
        cols.append(index[int(v)])
        weights.append(float(attrs["distance"]) if weighted else 1.0)
    matrix = csr_matrix((weights, (rows, cols)), shape=(len(nodes), len(nodes)))
    incoming_sums = np.zeros(len(nodes), dtype=float)
    outgoing_sums = np.zeros(len(nodes), dtype=float)
    for start in range(0, len(nodes), batch_size):
        sources = np.arange(start, min(start + batch_size, len(nodes)), dtype=int)
        distances = sparse_dijkstra(matrix, directed=True, indices=sources, unweighted=not weighted)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = np.where(np.isfinite(distances) & (distances > 0), 1.0 / distances, 0.0)
        incoming_sums += inverse.sum(axis=0)
        outgoing_sums[sources] = inverse.sum(axis=1)
    divisor = max(1, len(nodes) - 1)
    return (
        {int(node): float(incoming_sums[position] / divisor) for position, node in enumerate(nodes)},
        {int(node): float(outgoing_sums[position] / divisor) for position, node in enumerate(nodes)},
    )


def _betweenness_features(
    data: dict[str, Any],
    *,
    k: int | None,
    seed: int,
) -> dict[str, float]:
    """Calculate only the two betweenness variants used by the audit.

    Keeping this operation separate is important: the landmark sensitivity study
    repeats betweenness hundreds of times and must not silently repeat PageRank,
    HITS, harmonic centrality and clustering at every `(k, seed)` pair.
    """
    if k is None:
        return {
            **_igraph_exact_betweenness(data, weighted=False),
            **_igraph_exact_betweenness(data, weighted=True),
        }
    graph = build_graph(data, weighted=False)
    weighted_graph = build_graph(data, weighted=True)
    between = nx.betweenness_centrality(
        graph,
        k=k,
        normalized=True,
        weight=None,
        endpoints=False,
        seed=seed,
    )
    between_weighted = nx.betweenness_centrality(
        weighted_graph,
        k=k,
        normalized=True,
        weight="distance",
        endpoints=False,
        seed=seed,
    )
    return {
        **aggregate(between.values(), "betweenness_unweighted"),
        **aggregate(between_weighted.values(), "betweenness_distance"),
    }


def _betweenness_variant_task(task: tuple[dict[str, Any], bool, int]) -> dict[str, float]:
    data, weighted, seed = task
    graph = build_graph(data, weighted=weighted)
    values = nx.betweenness_centrality(
        graph,
        k=None,
        normalized=True,
        weight="distance" if weighted else None,
        endpoints=False,
        seed=seed,
    )
    return aggregate(values.values(), "betweenness_distance" if weighted else "betweenness_unweighted")


def _harmonic_landmarks(
    graph: nx.DiGraph,
    *,
    k: int,
    seed: int,
    weighted: bool,
) -> tuple[dict[int, float], dict[int, float]]:
    """Landmark estimate of directed in/out harmonic centrality."""
    nodes = np.asarray(sorted(graph.nodes()), dtype=int)
    k = min(len(nodes), int(k))
    rng = np.random.default_rng(seed)
    sources = nodes if k == len(nodes) else np.sort(rng.choice(nodes, size=k, replace=False))
    index = {int(node): position for position, node in enumerate(nodes)}
    rows, cols, weights = [], [], []
    for u, v, attrs in graph.edges(data=True):
        rows.append(index[int(u)])
        cols.append(index[int(v)])
        weights.append(float(attrs["distance"]) if weighted else 1.0)
    matrix = csr_matrix((weights, (rows, cols)), shape=(len(nodes), len(nodes)))
    source_indices = np.asarray([index[int(source)] for source in sources], dtype=int)

    def estimate(adjacency: csr_matrix) -> dict[int, float]:
        distances = sparse_dijkstra(adjacency, directed=True, indices=source_indices)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = np.where(np.isfinite(distances) & (distances > 0), 1.0 / distances, 0.0)
        values = inverse.sum(axis=0) * len(nodes) / k / max(1, len(nodes) - 1)
        return {int(node): float(values[position]) for position, node in enumerate(nodes)}

    incoming = estimate(matrix)
    outgoing = estimate(matrix.transpose().tocsr())
    return incoming, outgoing


def compute_selected_topology_for_search(
    data: dict[str, Any],
    required_features: Iterable[str],
    landmarks: int,
    seed: int,
) -> TopologyResult:
    """Calculate only topology families that can influence a trained PILOT."""
    required = set(required_features)
    started = time.perf_counter()
    graph = build_graph(data, weighted=False)
    weighted_graph = build_graph(data, weighted=True)
    n = graph.number_of_nodes()
    exact_fallback = int(landmarks) <= 0
    k = n if exact_fallback else min(n, int(landmarks))
    features: dict[str, float] = {}

    def needed(token: str) -> bool:
        return any(column.startswith(f"topo_{token}_") for column in required)

    if any(needed(token) for token in ("in_degree", "out_degree", "degree_ratio", "degree_imbalance")):
        incoming = np.asarray([graph.in_degree(node) for node in graph], dtype=float)
        outgoing = np.asarray([graph.out_degree(node) for node in graph], dtype=float)
        total = incoming + outgoing
        values = {
            "in_degree": incoming,
            "out_degree": outgoing,
            "degree_ratio": np.divide(incoming, total, out=np.zeros_like(incoming), where=total != 0),
            "degree_imbalance": np.divide(incoming - outgoing, total, out=np.zeros_like(incoming), where=total != 0),
        }
        for name, array in values.items():
            if needed(name):
                features.update(aggregate(array, name))
    if needed("pagerank_unweighted"):
        features.update(
            aggregate(
                nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000, weight=None).values(),
                "pagerank_unweighted",
            ),
        )
    if needed("pagerank_strength"):
        features.update(
            aggregate(
                nx.pagerank(weighted_graph, alpha=0.85, tol=1e-12, max_iter=1000, weight="strength").values(),
                "pagerank_strength",
            ),
        )
    hit_seeds = (11, 23, 37, 53, 71)
    if needed("hits_hub_unweighted") or needed("hits_authority_unweighted"):
        hubs, auth, _ = _hits_repeated(graph, hit_seeds)
        if needed("hits_hub_unweighted"):
            features.update(aggregate(hubs.values(), "hits_hub_unweighted"))
        if needed("hits_authority_unweighted"):
            features.update(aggregate(auth.values(), "hits_authority_unweighted"))
    if needed("hits_hub_strength") or needed("hits_authority_strength"):
        hubs, auth, _ = _hits_repeated(weighted_graph, hit_seeds)
        if needed("hits_hub_strength"):
            features.update(aggregate(hubs.values(), "hits_hub_strength"))
        if needed("hits_authority_strength"):
            features.update(aggregate(auth.values(), "hits_authority_strength"))
    if needed("betweenness_unweighted") or needed("betweenness_distance"):
        if exact_fallback:
            between = {}
            if needed("betweenness_unweighted"):
                between.update(_igraph_exact_betweenness(data, weighted=False))
            if needed("betweenness_distance"):
                between.update(_igraph_exact_betweenness(data, weighted=True))
        else:
            between = _betweenness_features(data, k=k, seed=seed)
        features.update({key: value for key, value in between.items() if key in required})
    if needed("harmonic_in_unweighted") or needed("harmonic_out_unweighted"):
        if exact_fallback:
            hin = (
                _igraph_exact_harmonic_direction(data, weighted=False, mode="in")
                if needed("harmonic_in_unweighted")
                else {}
            )
            hout = (
                _igraph_exact_harmonic_direction(data, weighted=False, mode="out")
                if needed("harmonic_out_unweighted")
                else {}
            )
        else:
            hin, hout = _harmonic_landmarks(graph, k=k, seed=seed, weighted=False)
        if needed("harmonic_in_unweighted"):
            features.update(aggregate(hin.values(), "harmonic_in_unweighted"))
        if needed("harmonic_out_unweighted"):
            features.update(aggregate(hout.values(), "harmonic_out_unweighted"))
    if needed("harmonic_in_distance") or needed("harmonic_out_distance"):
        if exact_fallback:
            hin = (
                _igraph_exact_harmonic_direction(data, weighted=True, mode="in")
                if needed("harmonic_in_distance")
                else {}
            )
            hout = (
                _igraph_exact_harmonic_direction(data, weighted=True, mode="out")
                if needed("harmonic_out_distance")
                else {}
            )
        else:
            hin, hout = _harmonic_landmarks(weighted_graph, k=k, seed=seed, weighted=True)
        if needed("harmonic_in_distance"):
            features.update(aggregate(hin.values(), "harmonic_in_distance"))
        if needed("harmonic_out_distance"):
            features.update(aggregate(hout.values(), "harmonic_out_distance"))
    if needed("directed_clustering"):
        features.update(aggregate(nx.clustering(graph, weight=None).values(), "directed_clustering"))
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"Features topológicas selecionadas não calculadas: {missing}")
    return TopologyResult(
        features,
        {
            "mode": "selected_family_exact_compiled" if exact_fallback else "selected_family_search_approximation",
            "backend": "hybrid_exact" if exact_fallback else "networkx_scipy_landmarks",
            "backend_components": (
                {
                    "betweenness": "python-igraph",
                    "harmonic": "python-igraph",
                    "hits": HITS_BACKEND_ID,
                    "degree": "numpy_networkx",
                }
                if exact_fallback
                else None
            ),
            "landmarks": None if exact_fallback else k,
            "required_topology_features": sorted(required),
            "elapsed_seconds": time.perf_counter() - started,
            "result_provenance": "measured",
            "approximation_validation_status": "not_applicable_exact" if exact_fallback else "pending_for_nonexact_mode",
        },
    )


def compute_topology_for_search(data: dict[str, Any], landmarks: int, seed: int) -> TopologyResult:
    """Bounded-cost topology used only while searching candidate genotypes.

    The result is explicitly labelled approximate. Any candidate retained by the
    GA is subsequently recomputed through :func:`compute_topology` in exact mode.
    """
    started = time.perf_counter()
    graph = build_graph(data, weighted=False)
    weighted_graph = build_graph(data, weighted=True)
    n = graph.number_of_nodes()
    k = min(n, int(landmarks))
    in_degree = np.asarray([graph.in_degree(node) for node in graph], dtype=float)
    out_degree = np.asarray([graph.out_degree(node) for node in graph], dtype=float)
    total = in_degree + out_degree
    ratio = np.divide(in_degree, total, out=np.zeros_like(in_degree), where=total != 0)
    imbalance = np.divide(in_degree - out_degree, total, out=np.zeros_like(in_degree), where=total != 0)
    features: dict[str, float] = {}
    for values, name in (
        (in_degree, "in_degree"),
        (out_degree, "out_degree"),
        (ratio, "degree_ratio"),
        (imbalance, "degree_imbalance"),
    ):
        features.update(aggregate(values, name))
    features.update(aggregate(nx.pagerank(graph, weight=None).values(), "pagerank_unweighted"))
    features.update(aggregate(nx.pagerank(weighted_graph, weight="strength").values(), "pagerank_strength"))
    hit_seeds = (11, 23, 37, 53, 71)
    hubs, auth, hu_diag = _hits_repeated(graph, hit_seeds)
    hubs_w, auth_w, hw_diag = _hits_repeated(weighted_graph, hit_seeds)
    for values, name in (
        (hubs.values(), "hits_hub_unweighted"),
        (auth.values(), "hits_authority_unweighted"),
        (hubs_w.values(), "hits_hub_strength"),
        (auth_w.values(), "hits_authority_strength"),
    ):
        features.update(aggregate(values, name))
    features.update(_betweenness_features(data, k=k, seed=seed))
    hin, hout = _harmonic_landmarks(graph, k=k, seed=seed, weighted=False)
    hin_w, hout_w = _harmonic_landmarks(weighted_graph, k=k, seed=seed, weighted=True)
    for values, name in (
        (hin.values(), "harmonic_in_unweighted"),
        (hout.values(), "harmonic_out_unweighted"),
        (hin_w.values(), "harmonic_in_distance"),
        (hout_w.values(), "harmonic_out_distance"),
    ):
        features.update(aggregate(values, name))
    features.update(aggregate(nx.clustering(graph, weight=None).values(), "directed_clustering"))
    return TopologyResult(
        features,
        {
            "n_vertices": n,
            "n_arcs": graph.number_of_edges(),
            "betweenness_mode": "approximate_landmarks",
            "harmonic_mode": "approximate_landmarks",
            "landmarks": k,
            "seed": seed,
            "hits_unweighted": hu_diag,
            "hits_strength": hw_diag,
            "elapsed_seconds": time.perf_counter() - started,
            "result_provenance": "measured",
            "approximation_validation_status": "pending_for_nonexact_mode",
        },
    )


def compute_topology(
    data: dict[str, Any],
    exact_betweenness: bool = True,
    approximate_k: int | None = None,
    approximate_seed: int = 0,
    parallel_metrics: int = 1,
) -> TopologyResult:
    started = time.perf_counter()
    graph = build_graph(data, weighted=False)
    weighted_graph = build_graph(data, weighted=True)
    n = graph.number_of_nodes()
    in_degree = np.asarray([graph.in_degree(node) for node in graph], dtype=float)
    out_degree = np.asarray([graph.out_degree(node) for node in graph], dtype=float)
    total_degree = in_degree + out_degree
    ratio = np.divide(in_degree, total_degree, out=np.zeros_like(in_degree), where=total_degree != 0)
    imbalance = np.divide(in_degree - out_degree, total_degree, out=np.zeros_like(in_degree), where=total_degree != 0)

    features: dict[str, float] = {}
    features.update(aggregate(in_degree, "in_degree"))
    features.update(aggregate(out_degree, "out_degree"))
    features.update(aggregate(ratio, "degree_ratio"))
    features.update(aggregate(imbalance, "degree_imbalance"))

    pagerank = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000, weight=None)
    pagerank_weighted = nx.pagerank(weighted_graph, alpha=0.85, tol=1e-12, max_iter=1000, weight="strength")
    features.update(aggregate(pagerank.values(), "pagerank_unweighted"))
    features.update(aggregate(pagerank_weighted.values(), "pagerank_strength"))

    hits_seeds = (11, 23, 37, 53, 71)
    hubs, authorities, hits_unweighted_diag = _hits_repeated(graph, hits_seeds)
    hubs_weighted, authorities_weighted, hits_weighted_diag = _hits_repeated(weighted_graph, hits_seeds)
    features.update(aggregate(hubs.values(), "hits_hub_unweighted"))
    features.update(aggregate(authorities.values(), "hits_authority_unweighted"))
    features.update(aggregate(hubs_weighted.values(), "hits_hub_strength"))
    features.update(aggregate(authorities_weighted.values(), "hits_authority_strength"))

    k = None if exact_betweenness else min(n, int(approximate_k or max(32, math.ceil(math.sqrt(n)))))
    # Keep the baseline and GA candidate path on one canonical exact backend.
    # Running NetworkX in a parallel worker here looks equivalent, but can
    # produce different weighted shortest-path tie handling on real graphs.
    features.update(_betweenness_features(data, k=k, seed=approximate_seed))

    if exact_betweenness:
        harmonic_in, harmonic_out = _igraph_exact_harmonic(data, weighted=False)
        harmonic_in_weighted, harmonic_out_weighted = _igraph_exact_harmonic(data, weighted=True)
    else:
        harmonic_in, harmonic_out = _harmonic(graph, weighted=False)
        harmonic_in_weighted, harmonic_out_weighted = _harmonic(weighted_graph, weighted=True)
    features.update(aggregate(harmonic_in.values(), "harmonic_in_unweighted"))
    features.update(aggregate(harmonic_out.values(), "harmonic_out_unweighted"))
    features.update(aggregate(harmonic_in_weighted.values(), "harmonic_in_distance"))
    features.update(aggregate(harmonic_out_weighted.values(), "harmonic_out_distance"))

    clustering = nx.clustering(graph, weight=None)
    features.update(aggregate(clustering.values(), "directed_clustering"))
    elapsed = time.perf_counter() - started
    return TopologyResult(
        features=features,
        diagnostics={
            "n_vertices": n,
            "n_arcs": graph.number_of_edges(),
            "betweenness_mode": "exact" if k is None else "approximate",
            "betweenness_k": k,
            "betweenness_seed": approximate_seed if k is not None else None,
            "harmonic_mode": "exact",
            "harmonic_batch_size": 256,
            "hits_unweighted": hits_unweighted_diag,
            "hits_strength": hits_weighted_diag,
            "elapsed_seconds": elapsed,
            "result_provenance": "measured",
        },
    )


def _exact_cache_task(task: tuple[str, dict[str, Any], str, int, int]) -> dict[str, Any]:
    fingerprint, data, cache_path_text, max_exact_vertices, metric_workers = task
    started = time.perf_counter()
    result = compute_topology(data, exact_betweenness=True, parallel_metrics=metric_workers)
    result.diagnostics["exact_attempt_policy"] = f"exact_when_n_vertices<={max_exact_vertices}; larger graphs require convergence validation"
    result.diagnostics["approximation_validation_status"] = "not_applicable_exact"
    write_json(
        Path(cache_path_text),
        {
            "topology_backend_id": TOPOLOGY_BACKEND_ID,
            "features": result.features,
            "diagnostics": result.diagnostics,
        },
    )
    return {
        "graph_sha256": fingerprint,
        "n_vertices": int(data["|V|"]),
        "cache_path": cache_path_text,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "completed",
        "result_provenance": "measured",
    }


def precompute_exact_topology_cache(
    records: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    workers: int,
    minimum_vertices: int = 1601,
) -> list[dict[str, Any]]:
    """Compute independent large exact graphs in parallel before table assembly."""
    cache_dir = output_dir / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    maximum = max(int(record["data"]["|V|"]) for record in records.values())
    pending: list[tuple[str, dict[str, Any], str, int]] = []
    scheduled_fingerprints: set[str] = set()
    for _, record in sorted(records.items()):
        if int(record["data"]["|V|"]) < minimum_vertices:
            continue
        fingerprint = graph_fingerprint(record["data"])
        cache_path = cache_dir / f"{fingerprint}_exact.json"
        if (
            (not cache_path.exists() or not _topology_cache_compatible(cache_path))
            and not _topology_cache_has_features(cache_path)
            and fingerprint not in scheduled_fingerprints
        ):
            pending.append((fingerprint, record["data"], str(cache_path), maximum))
            scheduled_fingerprints.add(fingerprint)
    metric_workers = max(1, int(workers)) if len(pending) == 1 else 1
    tasks = [(*task, metric_workers) for task in pending]
    effective_workers = min(max(1, int(workers)), max(1, len(tasks)))
    if effective_workers == 1:
        results = [_exact_cache_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            results = list(executor.map(_exact_cache_task, tasks))
    write_json(
        output_dir / "large_exact_computation.json",
        {
            "minimum_vertices": minimum_vertices,
            "graphs_computed": len(results),
            "workers_requested": int(workers),
            "workers_effective": effective_workers,
            "results": results,
            "result_provenance": "measured",
        },
    )
    return results


def build_topology_table(
    records: dict[str, dict[str, Any]],
    output_dir: Path,
    exact_betweenness: bool = True,
    max_exact_vertices: int = 1600,
    fallback_landmarks: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = output_dir / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for name, record in sorted(records.items()):
        fingerprint = graph_fingerprint(record["data"])
        n_vertices = int(record["data"]["|V|"])
        use_exact = bool(exact_betweenness and n_vertices <= max_exact_vertices)
        mode_key = "exact" if use_exact else f"approx_k{min(n_vertices, fallback_landmarks)}"
        cache_path = cache_dir / f"{fingerprint}_{mode_key}.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and _topology_cache_compatible(cache_path):
            result = TopologyResult(features=cached["features"], diagnostics=cached["diagnostics"])
            cache_hit = True
        else:
            result = (
                compute_topology(record["data"], exact_betweenness=True)
                if use_exact
                else compute_topology_for_search(
                    record["data"],
                    landmarks=min(n_vertices, fallback_landmarks),
                    seed=11,
                )
            )
            result.diagnostics["exact_attempt_policy"] = (
                f"exact_when_n_vertices<={max_exact_vertices}; larger graphs require convergence validation"
            )
            result.diagnostics["approximation_validation_status"] = "not_applicable_exact" if use_exact else "pending"
            write_json(
                cache_path,
                {
                    "topology_backend_id": TOPOLOGY_BACKEND_ID,
                    "features": result.features,
                    "diagnostics": result.diagnostics,
                },
            )
            cache_hit = False
        rows.append(
            {
                "instance": name,
                "group": record["group"],
                "experiment_group": record["experiment_group"],
                "source_zip": record["source_zip"],
                "graph_sha256": fingerprint,
                "result_provenance": "measured",
                **result.features,
            },
        )
        diagnostics.append(
            {
                "instance": name,
                "graph_sha256": fingerprint,
                "cache_hit": cache_hit,
                **result.diagnostics,
            },
        )
    features_df = pd.DataFrame(rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    features_df.to_csv(output_dir / "topology_candidates.csv", index=False)
    diagnostics_df.to_csv(output_dir / "topology_computation_diagnostics.csv", index=False)
    return features_df, diagnostics_df


def validate_exact_topology_recomputation(
    records: dict[str, dict[str, Any]],
    baseline_topology: pd.DataFrame,
    output_path: Path,
    *,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Prove that the exact GA feature path reproduces the baseline table.

    The GA is allowed to search only in the same feature space used to train
    MATILDA.  This test runs before ISA variants and before target generation,
    so a stale cache or a backend mismatch cannot consume a long GA budget.
    """
    feature_columns = [column for column in baseline_topology if column.startswith("topo_")]
    if not feature_columns:
        raise ValueError("Tabela topológica sem features para validar")
    expected = baseline_topology.set_index("instance")
    rows: list[dict[str, Any]] = []
    for instance, record in sorted(records.items()):
        recomputed = compute_selected_topology_for_search(
            record["data"],
            feature_columns,
            landmarks=0,
            seed=0,
        ).features
        errors = {
            column: abs(float(recomputed[column]) - float(expected.loc[instance, column]))
            for column in feature_columns
        }
        worst_feature = max(errors, key=errors.get)
        rows.append(
            {
                "instance": instance,
                "max_absolute_error": float(errors[worst_feature]),
                "worst_feature": worst_feature,
                "result_provenance": "measured",
            },
        )
    frame = pd.DataFrame(rows)
    details_path = output_path.with_name(f"{output_path.stem}_by_instance.csv")
    frame.to_csv(details_path, index=False)
    maximum = float(frame["max_absolute_error"].max())
    result = {
        "status": "passed" if maximum <= tolerance else "failed",
        "instances_checked": len(frame),
        "features_checked": len(feature_columns),
        "tolerance": tolerance,
        "max_absolute_error": maximum,
        "details_csv": details_path.name,
        "topology_backend_id": TOPOLOGY_BACKEND_ID,
        "result_provenance": "measured",
    }
    write_json(output_path, result)
    if result["status"] != "passed":
        worst = frame.sort_values("max_absolute_error", ascending=False).iloc[0]
        raise RuntimeError(
            "Tabela topológica do baseline diverge do caminho exato do GA: "
            f"instância={worst['instance']}; feature={worst['worst_feature']}; "
            f"erro={worst['max_absolute_error']}; tolerância={tolerance}",
        )
    return result


def audit_and_prepare_topology(
    candidates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [column for column in candidates if column.startswith("topo_")]
    unstable_hits: dict[str, bool] = {}
    for field in ("hits_unweighted", "hits_strength"):
        if field in diagnostics:
            # CSV round-trip may stringify dicts; instability is also checked feature-wise below.
            unstable_hits[field] = diagnostics[field].astype(str).str.contains(
                "'stable_at_1e-6': False|\"stable_at_1e-6\": false",
                regex=True,
            ).any()
    audit_rows: list[dict[str, Any]] = []
    keep: list[str] = []
    exact_vectors: dict[bytes, str] = {}
    transformations: list[dict[str, Any]] = []
    prepared = candidates.copy()
    for column in feature_cols:
        series = pd.to_numeric(candidates[column], errors="coerce")
        values = series.to_numpy(dtype=float)
        finite = np.isfinite(values)
        unique = int(series.nunique(dropna=False))
        variance = float(np.nanvar(values))
        q25, q75 = np.nanquantile(values, (0.25, 0.75)) if finite.any() else (math.nan, math.nan)
        iqr = float(q75 - q25)
        reason = "kept"
        duplicate_of: str | None = None
        if not finite.all():
            reason = "invalid_nonfinite"
        elif unique <= 1 or variance == 0.0:
            reason = "constant_or_degenerate"
        elif np.unique(values - float(np.min(values)) + 1.0).size <= 1:
            # PRELIM at the pinned commit performs this exact shift before
            # Box-Cox. Sub-ulp variation around one collapses to a constant,
            # which would otherwise create a zero sigma and NaNs.
            reason = "prelim_numeric_degenerate"
        else:
            vector_key = np.ascontiguousarray(values).tobytes()
            if vector_key in exact_vectors:
                reason = "exact_duplicate"
                duplicate_of = exact_vectors[vector_key]
            else:
                exact_vectors[vector_key] = column
        if column.startswith("topo_hits_") and "_unweighted_" in column and unstable_hits.get("hits_unweighted", False):
            reason = "hits_numerically_unstable"
        if column.startswith("topo_hits_") and "_strength_" in column and unstable_hits.get("hits_strength", False):
            reason = "hits_numerically_unstable"
        skewness = float(series.skew()) if finite.all() else math.nan
        transform = "identity"
        if reason == "kept":
            keep.append(column)
        transformations.append(
            {
                "feature": column,
                "transformation": transform,
                "sample_skewness": skewness,
                "long_tail_flag_abs_skew_gt_2": bool(abs(skewness) > 2),
                "matilda_prelim": "positive_shift_boxcox_zscore",
                "result_provenance": "measured",
            },
        )
        audit_rows.append(
            {
                "feature": column,
                "family": topology_family(column),
                "unique_values": unique,
                "variance": variance,
                "iqr": iqr,
                "iqr_zero": bool(iqr == 0.0),
                "all_finite": bool(finite.all()),
                "sample_skewness": skewness,
                "decision": reason,
                "duplicate_of": duplicate_of,
                "result_provenance": "measured",
            },
        )
    audit = pd.DataFrame(audit_rows)
    transform_df = pd.DataFrame(transformations)
    meta = [column for column in prepared if not column.startswith("topo_")]
    selected = prepared[[*meta, *keep]].copy()
    audit.to_csv(output_dir / "topology_feature_audit.csv", index=False)
    transform_df.to_csv(output_dir / "topology_transformations.csv", index=False)
    selected.to_csv(output_dir / "topology_features_for_isa.csv", index=False)
    return selected, audit, transform_df


def topology_family(column: str) -> str:
    for family, prefixes in TOPOLOGY_FAMILIES.items():
        if any(column.startswith(f"topo_{prefix}_") for prefix in prefixes):
            return family
    return "unknown"


def rebuild_convergence_from_raw(
    output_dir: Path,
    metric: str,
    *,
    landmarks: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Re-aggregate a complete raw matrix without repeating centralities."""
    raw_path = output_dir / f"{metric}_convergence_raw.csv"
    frame = pd.read_csv(raw_path)
    required = {
        "instance",
        "feature",
        "k_requested",
        "k_effective",
        "seed",
        "exact",
        "approximate",
        "relative_error",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Matriz crua incompleta para {metric}: colunas/linhas ausentes")
    if set(frame["k_requested"].astype(int).unique()) != set(map(int, landmarks)):
        raise ValueError(f"Matriz crua incompleta para {metric}: níveis de landmarks divergentes")
    if set(frame["seed"].astype(int).unique()) != set(map(int, seeds)):
        raise ValueError(f"Matriz crua incompleta para {metric}: seeds divergentes")
    counts = frame.groupby("k_requested", sort=True).size()
    if counts.nunique() != 1 or frame.duplicated(["instance", "feature", "k_requested", "seed"]).any():
        raise ValueError(f"Matriz crua incompleta ou duplicada para {metric}")
    summary_rows: list[dict[str, Any]] = []
    for k_requested, group in frame.groupby("k_requested", sort=True):
        pairs = group.groupby(["instance", "feature"], as_index=False).agg(
            exact=("exact", "first"),
            approximate=("approximate", "mean"),
            seed_cv=("approximate", lambda x: float(np.std(x) / max(abs(np.mean(x)), 1e-12))),
        )
        rho = spearmanr(pairs["exact"], pairs["approximate"]).statistic if len(pairs) > 1 else math.nan
        median_error = float(group["relative_error"].median())
        p95_error = float(group["relative_error"].quantile(0.95))
        max_cv = float(pairs["seed_cv"].max())
        summary_rows.append(
            {
                "k_requested": int(k_requested),
                "k_effective_min": int(group["k_effective"].min()),
                "k_effective_max": int(group["k_effective"].max()),
                "spearman": float(rho),
                "median_relative_error": median_error,
                "p95_relative_error": p95_error,
                "max_seed_cv": max_cv,
                "passes": bool(rho >= 0.99 and median_error <= 0.01 and p95_error <= 0.05 and max_cv <= 0.01),
                "result_provenance": "measured",
            },
        )
    summary = pd.DataFrame(summary_rows).sort_values("k_requested")
    summary.to_csv(output_dir / f"{metric}_convergence_summary.csv", index=False)
    feature_rows: list[dict[str, Any]] = []
    for (feature, k_requested), group in frame.groupby(["feature", "k_requested"], sort=True):
        pairs = group.groupby("instance", as_index=False).agg(
            exact=("exact", "first"),
            approximate=("approximate", "mean"),
            seed_cv=("approximate", lambda x: float(np.std(x) / max(abs(np.mean(x)), 1e-12))),
        )
        rho = spearmanr(pairs["exact"], pairs["approximate"]).statistic if len(pairs) > 1 else math.nan
        median_error = float(group["relative_error"].median())
        p95_error = float(group["relative_error"].quantile(0.95))
        max_cv = float(pairs["seed_cv"].max())
        feature_rows.append(
            {
                "feature": feature,
                "k_requested": int(k_requested),
                "k_effective_min": int(group["k_effective"].min()),
                "k_effective_max": int(group["k_effective"].max()),
                "spearman": float(rho),
                "median_relative_error": median_error,
                "p95_relative_error": p95_error,
                "max_seed_cv": max_cv,
                "passes": bool(rho >= 0.99 and median_error <= 0.01 and p95_error <= 0.05 and max_cv <= 0.01),
                "result_provenance": "measured",
            },
        )
    feature_summary = pd.DataFrame(feature_rows).sort_values(["feature", "k_requested"])
    feature_summary.to_csv(output_dir / f"{metric}_convergence_by_feature.csv", index=False)
    feature_decisions: list[dict[str, Any]] = []
    for feature, group in feature_summary.groupby("feature", sort=True):
        group = group.sort_values("k_requested")
        feature_recommendation: int | None = None
        passes_feature = group["passes"].tolist()
        for index in range(1, len(group)):
            if passes_feature[index - 1] and passes_feature[index]:
                feature_recommendation = int(group.iloc[index]["k_requested"])
                break
        feature_decisions.append(
            {
                "feature": feature,
                "recommended_approximate_k": feature_recommendation,
                "status": "validated" if feature_recommendation is not None else "not_validated",
                "result_provenance": "measured",
            },
        )
    pd.DataFrame(feature_decisions).to_csv(output_dir / f"{metric}_convergence_feature_decisions.csv", index=False)
    recommended: int | None = None
    passes = summary["passes"].tolist()
    for index in range(1, len(summary)):
        if passes[index - 1] and passes[index]:
            recommended = int(summary.iloc[index]["k_requested"])
            break
    decision = {
        "final_mode": "exact",
        "recommended_approximate_k": recommended,
        "aggregation_axis": "k_requested",
        "acceptance_requires_two_consecutive_levels": True,
        "criteria": {"spearman": 0.99, "median_relative_error": 0.01, "p95_relative_error": 0.05, "max_seed_cv": 0.01},
        "raw_matrix_validation": {
            "rows": int(len(frame)),
            "rows_per_requested_level": int(counts.iloc[0]),
            "landmarks": sorted(map(int, landmarks)),
            "seeds": sorted(map(int, seeds)),
            "duplicates": 0,
        },
        "result_provenance": "measured",
    }
    write_json(output_dir / f"{metric}_convergence_decision.json", decision)
    return decision


def betweenness_convergence_study(
    records: dict[str, dict[str, Any]],
    exact_candidates: pd.DataFrame,
    output_dir: Path,
    landmarks: tuple[int, ...],
    seeds: tuple[int, ...],
    max_graphs_per_size: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_names: list[str] = []
    by_n: dict[int, list[str]] = {}
    for name, record in sorted(records.items()):
        n = int(record["data"]["|V|"])
        if n <= 1600:
            by_n.setdefault(n, []).append(name)
    for n in sorted(by_n):
        selected_names.extend(by_n[n][:max_graphs_per_size])
    exact = exact_candidates.set_index("instance")
    bet_cols = [column for column in exact.columns if column.startswith("topo_betweenness_")]
    rows: list[dict[str, Any]] = []
    for name in selected_names:
        record = records[name]
        n = int(record["data"]["|V|"])
        for k_requested in landmarks:
            k = min(n, k_requested)
            for seed in seeds:
                started = time.perf_counter()
                approximate_features = _betweenness_features(record["data"], k=k, seed=seed)
                elapsed = time.perf_counter() - started
                for column in bet_cols:
                    expected = float(exact.loc[name, column])
                    observed = float(approximate_features[column])
                    absolute_error = abs(observed - expected)
                    relative_error = absolute_error / max(abs(expected), 1e-12)
                    rows.append(
                        {
                            "instance": name,
                            "n_vertices": n,
                            "k_requested": k_requested,
                            "k_effective": k,
                            "seed": seed,
                            "feature": column,
                            "exact": expected,
                            "approximate": observed,
                            "absolute_error": absolute_error,
                            "relative_error": relative_error,
                            "elapsed_seconds": elapsed,
                            "result_provenance": "measured",
                        },
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "betweenness_convergence_raw.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for k_requested, group in frame.groupby("k_requested"):
        pairs = group.groupby(["instance", "feature"], as_index=False).agg(
            exact=("exact", "first"),
            approximate=("approximate", "mean"),
            seed_cv=("approximate", lambda x: float(np.std(x) / max(abs(np.mean(x)), 1e-12))),
        )
        rho = spearmanr(pairs["exact"], pairs["approximate"]).statistic if len(pairs) > 1 else math.nan
        summary_rows.append(
            {
                "k_requested": int(k_requested),
                "k_effective_min": int(group["k_effective"].min()),
                "k_effective_max": int(group["k_effective"].max()),
                "spearman": float(rho),
                "median_relative_error": float(group["relative_error"].median()),
                "p95_relative_error": float(group["relative_error"].quantile(0.95)),
                "max_seed_cv": float(pairs["seed_cv"].max()),
                "passes": bool(
                    rho >= 0.99
                    and group["relative_error"].median() <= 0.01
                    and group["relative_error"].quantile(0.95) <= 0.05
                    and pairs["seed_cv"].max() <= 0.01
                ),
                "result_provenance": "measured",
            },
        )
    summary = pd.DataFrame(summary_rows).sort_values("k_requested")
    summary.to_csv(output_dir / "betweenness_convergence_summary.csv", index=False)
    passing = summary["passes"].tolist()
    recommended: int | None = None
    for idx in range(1, len(summary)):
        if passing[idx - 1] and passing[idx]:
            recommended = int(summary.iloc[idx]["k_requested"])
            break
    decision = {
        "final_mode": "exact",
        "recommended_approximate_k": recommended,
        "aggregation_axis": "k_requested",
        "acceptance_requires_two_consecutive_levels": True,
        "criteria": {"spearman": 0.99, "median_relative_error": 0.01, "p95_relative_error": 0.05, "max_seed_cv": 0.01},
        "result_provenance": "measured",
    }
    write_json(output_dir / "betweenness_convergence_decision.json", decision)
    return frame, decision


def harmonic_convergence_study(
    records: dict[str, dict[str, Any]],
    exact_candidates: pd.DataFrame,
    output_dir: Path,
    landmarks: tuple[int, ...],
    seeds: tuple[int, ...],
    max_graphs_per_size: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected: list[str] = []
    by_n: dict[int, list[str]] = {}
    for name, record in sorted(records.items()):
        n = int(record["data"]["|V|"])
        if n <= 1600:
            by_n.setdefault(n, []).append(name)
    for n in sorted(by_n):
        selected.extend(by_n[n][:max_graphs_per_size])
    exact = exact_candidates.set_index("instance")
    harmonic_cols = [column for column in exact if column.startswith("topo_harmonic_")]
    rows: list[dict[str, Any]] = []
    for name in selected:
        data = records[name]["data"]
        graph = build_graph(data, weighted=False)
        weighted_graph = build_graph(data, weighted=True)
        n = graph.number_of_nodes()
        for requested in landmarks:
            k = min(n, requested)
            for seed in seeds:
                started = time.perf_counter()
                hin, hout = _harmonic_landmarks(graph, k=k, seed=seed, weighted=False)
                hin_w, hout_w = _harmonic_landmarks(weighted_graph, k=k, seed=seed, weighted=True)
                approximate: dict[str, float] = {}
                for values, metric in (
                    (hin.values(), "harmonic_in_unweighted"),
                    (hout.values(), "harmonic_out_unweighted"),
                    (hin_w.values(), "harmonic_in_distance"),
                    (hout_w.values(), "harmonic_out_distance"),
                ):
                    approximate.update(aggregate(values, metric))
                elapsed = time.perf_counter() - started
                for column in harmonic_cols:
                    expected = float(exact.loc[name, column])
                    observed = float(approximate[column])
                    absolute = abs(observed - expected)
                    rows.append(
                        {
                            "instance": name,
                            "n_vertices": n,
                            "k_requested": requested,
                            "k_effective": k,
                            "seed": seed,
                            "feature": column,
                            "exact": expected,
                            "approximate": observed,
                            "absolute_error": absolute,
                            "relative_error": absolute / max(abs(expected), 1e-12),
                            "elapsed_seconds": elapsed,
                            "result_provenance": "measured",
                        },
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "harmonic_convergence_raw.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for k_requested, group in frame.groupby("k_requested"):
        pairs = group.groupby(["instance", "feature"], as_index=False).agg(
            exact=("exact", "first"),
            approximate=("approximate", "mean"),
            seed_cv=("approximate", lambda x: float(np.std(x) / max(abs(np.mean(x)), 1e-12))),
        )
        rho = spearmanr(pairs["exact"], pairs["approximate"]).statistic if len(pairs) > 1 else math.nan
        med, p95, max_cv = group["relative_error"].median(), group["relative_error"].quantile(0.95), pairs["seed_cv"].max()
        summary_rows.append(
            {
                "k_requested": int(k_requested),
                "k_effective_min": int(group["k_effective"].min()),
                "k_effective_max": int(group["k_effective"].max()),
                "spearman": float(rho),
                "median_relative_error": float(med),
                "p95_relative_error": float(p95),
                "max_seed_cv": float(max_cv),
                "passes": bool(rho >= 0.99 and med <= 0.01 and p95 <= 0.05 and max_cv <= 0.01),
                "result_provenance": "measured",
            },
        )
    summary = pd.DataFrame(summary_rows).sort_values("k_requested")
    summary.to_csv(output_dir / "harmonic_convergence_summary.csv", index=False)
    recommended: int | None = None
    passes = summary["passes"].tolist()
    for index in range(1, len(summary)):
        if passes[index - 1] and passes[index]:
            recommended = int(summary.iloc[index]["k_requested"])
            break
    decision = {
        "final_mode": "exact",
        "recommended_approximate_k": recommended,
        "aggregation_axis": "k_requested",
        "acceptance_requires_two_consecutive_levels": True,
        "criteria": {"spearman": 0.99, "median_relative_error": 0.01, "p95_relative_error": 0.05, "max_seed_cv": 0.01},
        "result_provenance": "measured",
    }
    write_json(output_dir / "harmonic_convergence_decision.json", decision)
    return frame, decision
