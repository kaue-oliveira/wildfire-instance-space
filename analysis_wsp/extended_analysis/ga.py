from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy import stats
from scipy.stats import qmc
from shapely.geometry import MultiPoint, Point, Polygon

from .data import base_features, dijkstra, instance_content_hash, numeric_feature_columns
from .isa import _safe_name
from .provenance import stable_json_hash, write_json
from .topology import compute_selected_topology_for_search, compute_topology, graph_fingerprint


GENE_DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grid", ("Small", "Medium", "Large", "Huge")),
    ("slope", ("Flat", "Moderate", "Steep")),
    ("wind", ("Light", "Moderate", "Strong")),
    ("resources_delay", ("Low", "Medium", "High")),
    ("num_resources", ("Few", "Moderate", "Many")),
    ("resources_dist", ("Few", "Moderate", "Many")),
    ("first_release_time", ("Early", "Late", "VeryLate")),
    ("last_release_time", ("VeryEarly", "Early", "Late", "VeryLate")),
)


@dataclass(frozen=True)
class Genotype:
    grid: str
    slope: str
    wind: str
    resources_delay: str
    num_resources: str
    resources_dist: str
    first_release_time: str
    last_release_time: str
    generator_seed: int

    def identifier(self) -> str:
        digest = stable_json_hash(asdict(self))[:16]
        return f"ga_{self.grid.lower()}_{digest}"


@dataclass
class EvaluatedCandidate:
    genotype: Genotype
    identifier: str
    data: dict[str, Any]
    features: dict[str, float]
    z: np.ndarray
    fitness: float
    topology_diagnostics: dict[str, Any]


class TechnicalExhaustionError(RuntimeError):
    """The finite genotype domain could not provide another unique candidate."""


_GENERATOR_MODULE: Any | None = None
_GA_WORKER_CONTEXT: dict[str, Any] = {}


def _topology_cache_mode(*, exact: bool, search_landmarks: int) -> str:
    if exact:
        return "full_exact"
    return "selected_exact" if search_landmarks <= 0 else "search_approximate"


def _generator_module(root: Path) -> Any:
    global _GENERATOR_MODULE
    if _GENERATOR_MODULE is None:
        source = root / "third_party" / "Wildfire-Suppression" / "scripts" / "instance_generation.py"
        spec = importlib.util.spec_from_file_location("wsp_instance_generation_fixed", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Não foi possível carregar o gerador: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GENERATOR_MODULE = module
    return _GENERATOR_MODULE


def generate_wsp_data(root: Path, genotype: Genotype) -> dict[str, Any]:
    """Invoke the fixed WSP generator without its quadratic Python Dijkstra."""
    module = _generator_module(root)
    instance = module.Instance(
        grid=genotype.grid,
        wind=genotype.wind,
        slope=genotype.slope,
        first_release_time=genotype.first_release_time,
        last_release_time=genotype.last_release_time,
        num_resources=genotype.num_resources,
        resources_dist=genotype.resources_dist,
        resources_delay=genotype.resources_delay,
        seed=int(genotype.generator_seed),
    )
    vertices, arcs, coord_to_id, id_to_coord, heights, _wind, _speed, _r0, _rate, travel, ignition = instance.generate_graph()
    indexed_arcs = sorted(
        (int(coord_to_id[u]), int(coord_to_id[v]), float(travel[(u, v)]))
        for u, v in arcs
    )
    arrivals = dijkstra(len(vertices), indexed_arcs, [int(coord_to_id[ignition])])
    arrival_dict = {coordinate: arrivals[int(coord_to_id[coordinate])] for coordinate in vertices}
    horizon = instance.generate_horizon(vertices, arrival_dict)
    resources = instance.generate_resource_distribution(vertices, arrival_dict, horizon)
    coordinates = []
    for vertex_id in sorted(id_to_coord):
        x, y = id_to_coord[vertex_id]
        coordinates.append([float(instance.d * x), float(instance.d * y), float(heights[(x, y)])])
    return {
        "H": float(horizon),
        "|V|": len(vertices),
        "|R|": int(instance.Resources_decision_points),
        "I": [int(coord_to_id[ignition])],
        "Vb": "NA",
        "Vp": "NA",
        "w": "NA",
        "t": [float(item["t"]) for item in resources],
        "c": [int(item["c"]) for item in resources],
        "r": "NA",
        "z": "NA",
        "e": "NA",
        "delta": [float(item["delta"]) for item in resources],
        "arcs": [[u, v, value] for u, v, value in indexed_arcs],
        "distance": {"coordinates": coordinates},
    }


def random_genotype(rng: np.random.Generator) -> Genotype:
    values = [domain[int(rng.integers(0, len(domain)))] for _, domain in GENE_DOMAINS]
    seed = int(rng.integers(1, np.iinfo(np.int32).max))
    return Genotype(*values, seed)


def uniform_crossover(a: Genotype, b: Genotype, rng: np.random.Generator, probability: float) -> Genotype:
    av, bv = list(asdict(a).values()), list(asdict(b).values())
    if rng.random() >= probability:
        return a
    values = [bv[index] if rng.random() < 0.5 else av[index] for index in range(len(av))]
    return Genotype(*values)


def mutate(genotype: Genotype, rng: np.random.Generator, probability: float) -> Genotype:
    values = list(asdict(genotype).values())
    for index, (_, domain) in enumerate(GENE_DOMAINS):
        if rng.random() < probability:
            alternatives = [value for value in domain if value != values[index]]
            values[index] = alternatives[int(rng.integers(0, len(alternatives)))]
    if rng.random() < probability:
        values[-1] = int(rng.integers(1, np.iinfo(np.int32).max))
    return Genotype(*values)


def _candidate_metadata(features: dict[str, float], feature_columns: list[str], identifier: str) -> Any:
    from instancespace.data.metadata import Metadata

    missing = sorted(set(feature_columns) - set(features))
    if missing:
        raise ValueError(f"Candidato sem {len(missing)} features requeridas: {missing[:8]}")
    row: dict[str, Any] = {"instances": identifier, "source": "ga_generated"}
    for column in feature_columns:
        value = float(features[column])
        if not math.isfinite(value):
            raise ValueError(f"Feature não finita no candidato: {column}={value}")
        row[f"feature_{_safe_name(column)}"] = value
    return Metadata.from_data_frame(pd.DataFrame([row]))


class CandidateEvaluator:
    def __init__(
        self,
        root: Path,
        space: Any,
        feature_template: pd.DataFrame,
        output_dir: Path,
        *,
        search_landmarks: int = 0,
    ) -> None:
        self.root = root
        self.space = space
        self.model = space.model
        self.apply_prelim_bounds = bool(space.options.bound.flag)
        self.feature_columns = numeric_feature_columns(feature_template)
        self.selected_indices = np.asarray(self.model.sifted.selvars, dtype=int).ravel()
        self.selected_feature_columns = [self.feature_columns[index] for index in self.selected_indices]
        self.required_topology_features = [column for column in self.selected_feature_columns if column.startswith("topo_")]
        self.output_dir = output_dir
        self.search_landmarks = search_landmarks
        self.generated_dir = output_dir / "generated_candidates"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.existing_search_identifiers = {
            path.name.removesuffix("_search.json")
            for path in self.generated_dir.glob("*_search.json")
        }
        self.topology_cache_capacity = 512
        self.topology_cache: OrderedDict[tuple[str, str, int, int], Any] = OrderedDict()

    def _cached_topology(
        self,
        key: tuple[str, str, int, int],
        factory: Callable[[], Any],
    ) -> tuple[Any, bool]:
        if key in self.topology_cache:
            value = self.topology_cache.pop(key)
            self.topology_cache[key] = value
            return value, True
        value = factory()
        self.topology_cache[key] = value
        while len(self.topology_cache) > self.topology_cache_capacity:
            self.topology_cache.popitem(last=False)
        return value, False

    def _features(self, genotype: Genotype, data: dict[str, Any], exact: bool) -> tuple[dict[str, float], dict[str, Any]]:
        record = {
            "instance": genotype.identifier(),
            "group": "ga_generated",
            "experiment_group": "ga",
            "source_zip": "",
            "data": data,
        }
        base = base_features(record)
        fingerprint = graph_fingerprint(data)
        topology_key = (
            fingerprint,
            _topology_cache_mode(exact=exact, search_landmarks=self.search_landmarks),
            0 if exact else int(self.search_landmarks),
            0 if exact else int(genotype.generator_seed),
        )
        topo, topology_cache_hit = self._cached_topology(
            topology_key,
            lambda: (
                compute_topology(data, exact_betweenness=True)
                if exact
                else compute_selected_topology_for_search(
                    data,
                    self.required_topology_features,
                    self.search_landmarks,
                    int(genotype.generator_seed),
                )
            ),
        )
        merged = {**base, **topo.features}
        required = self.feature_columns if exact else self.selected_feature_columns
        diagnostics = {
            **topo.diagnostics,
            "graph_sha256": fingerprint,
            "topology_cache_hit": topology_cache_hit,
            "topology_cache_capacity": self.topology_cache_capacity,
        }
        return {column: float(merged[column]) for column in required}, diagnostics

    def _project(self, features: dict[str, float]) -> np.ndarray:
        raw = np.asarray([[features[column] for column in self.selected_feature_columns]], dtype=float)
        processed = raw.copy()
        prelim = self.model.prelim
        for local_index, original_index in enumerate(self.selected_indices):
            value = processed[:, local_index]
            if self.apply_prelim_bounds:
                value = np.clip(value, prelim.lo_bound[original_index], prelim.hi_bound[original_index])
            value = value - prelim.min_x[original_index] + 1
            value = stats.boxcox(value, prelim.lambda_x[original_index])
            processed[:, local_index] = (value - prelim.mu_x[original_index]) / prelim.sigma_x[original_index]
        return np.asarray((processed @ self.model.pilot.a.T)[0], dtype=float)

    def evaluate(self, genotype: Genotype, target: np.ndarray, exact: bool = False) -> EvaluatedCandidate:
        identifier = genotype.identifier()
        artifact_path = self.generated_dir / f"{identifier}_{'exact' if exact else 'search'}.json"
        expected_mode = (
            "full_exact"
            if exact
            else "search_exact_selected_topology"
            if self.search_landmarks <= 0
            else "search_approximation"
        )
        restored = False
        if not exact and artifact_path.is_file():
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                restored_features = {str(key): float(value) for key, value in payload["features"].items()}
                restored_z = np.asarray(payload["pilot_coordinates"], dtype=float)
                restored = (
                    payload.get("calculation_mode") == expected_mode
                    and payload.get("prelim_bound_applied") is self.apply_prelim_bounds
                    and set(self.selected_feature_columns).issubset(restored_features)
                    and restored_z.shape == (2,)
                    and np.isfinite(restored_z).all()
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                restored = False
            if restored:
                data = {}
                features = {column: restored_features[column] for column in self.selected_feature_columns}
                z = restored_z
                diagnostics = {
                    **dict(payload.get("topology_diagnostics", {})),
                    "candidate_artifact_cache_hit": True,
                }
        if not restored:
            data = generate_wsp_data(self.root, genotype)
            features, diagnostics = self._features(genotype, data, exact=exact)
            diagnostics = {**diagnostics, "candidate_artifact_cache_hit": False}
            z = self._project(features)
            payload = {
                "genotype": asdict(genotype),
                "instance_sha256": instance_content_hash(data),
                "calculation_mode": expected_mode,
                "prelim_bound_applied": self.apply_prelim_bounds,
                "topology_diagnostics": diagnostics,
                "features": features,
                "pilot_coordinates": z.tolist(),
                "feature_result_provenance": "measured",
                "coordinate_result_provenance": "measured",
            }
            write_json(artifact_path, payload)
            if exact:
                write_json(self.generated_dir / f"{identifier}.instance.json", data)
        if not exact:
            self.existing_search_identifiers.add(identifier)
        # Search graphs can contain thousands of vertices and are already fully
        # represented by the immutable on-disk artifact.  Keeping every graph in
        # an evaluator-wide cache causes memory to grow with the number of GA
        # evaluations.  Only exact finalists need their graph after projection.
        retained_data = data if exact else {}
        return EvaluatedCandidate(
            genotype=genotype,
            identifier=identifier,
            data=retained_data,
            features=features,
            z=z,
            fitness=float(np.linalg.norm(z - target)),
            topology_diagnostics=diagnostics,
        )


def validate_candidate_projection(
    evaluator: CandidateEvaluator,
    baseline_features: pd.DataFrame,
    baseline_coordinates: pd.DataFrame,
    output_path: Path,
    *,
    baseline_records: dict[str, dict[str, Any]] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Prove that GA projection reproduces the trained baseline PILOT space.

    The pinned pyInstanceSpace explorer clips against stored IQR bounds even
    when ``bound.flag`` is false, whereas PRELIM training correctly skips that
    operation.  Candidate projection follows the resolved training option and
    is accepted only if all 55 training coordinates are reproduced.
    """

    expected = baseline_coordinates.set_index("instance")
    baseline_by_instance = baseline_features.set_index("instance")
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for record in baseline_features.to_dict(orient="records"):
        instance = str(record["instance"])
        projected = evaluator._project(
            {column: float(record[column]) for column in evaluator.selected_feature_columns},
        )
        target = expected.loc[instance, ["z1", "z2"]].to_numpy(dtype=float)
        error = float(np.linalg.norm(projected - target))
        errors.append(error)
        rows.append(
            {
                "instance": instance,
                "projected_z1": float(projected[0]),
                "projected_z2": float(projected[1]),
                "expected_z1": float(target[0]),
                "expected_z2": float(target[1]),
                "euclidean_error": error,
                "result_provenance": "measured",
            },
        )
    array = np.asarray(errors, dtype=float)
    recomputed_coordinate_errors: list[float] = []
    selected_feature_errors: list[float] = []
    recomputed_rows: list[dict[str, Any]] = []
    if baseline_records is not None:
        placeholder = Genotype(
            "Small", "Flat", "Light", "Low", "Few", "Few", "Early", "VeryEarly", 0,
        )
        for instance, record in sorted(baseline_records.items()):
            recalculated, _ = evaluator._features(placeholder, record["data"], exact=False)
            expected_features = baseline_by_instance.loc[instance]
            absolute_errors = {
                column: abs(float(recalculated[column]) - float(expected_features[column]))
                for column in evaluator.selected_feature_columns
            }
            feature_error = max(absolute_errors.values(), default=0.0)
            selected_feature_errors.append(feature_error)
            projected = evaluator._project(recalculated)
            target = expected.loc[instance, ["z1", "z2"]].to_numpy(dtype=float)
            coordinate_error = float(np.linalg.norm(projected - target))
            recomputed_coordinate_errors.append(coordinate_error)
            recomputed_rows.append(
                {
                    "instance": instance,
                    "max_selected_feature_absolute_error": feature_error,
                    "recomputed_projection_euclidean_error": coordinate_error,
                    "result_provenance": "measured",
                },
            )
        pd.DataFrame(recomputed_rows).to_csv(
            output_path.with_name(f"{output_path.stem}_recomputed_features.csv"),
            index=False,
        )
    recomputed_array = np.asarray(recomputed_coordinate_errors, dtype=float)
    feature_array = np.asarray(selected_feature_errors, dtype=float)
    details_path = output_path.with_name(f"{output_path.stem}_by_instance.csv")
    pd.DataFrame(rows).to_csv(details_path, index=False)
    result = {
        "status": (
            "passed"
            if len(array)
            and float(array.max()) <= tolerance
            and (not len(recomputed_array) or float(recomputed_array.max()) <= tolerance)
            and (not len(feature_array) or float(feature_array.max()) <= tolerance)
            else "failed"
        ),
        "instances_checked": len(array),
        "tolerance": tolerance,
        "max_euclidean_error": float(array.max()) if len(array) else math.nan,
        "median_euclidean_error": float(np.median(array)) if len(array) else math.nan,
        "p95_euclidean_error": float(np.quantile(array, 0.95)) if len(array) else math.nan,
        "feature_recomputations_checked": len(recomputed_array),
        "max_recomputed_projection_euclidean_error": (
            float(recomputed_array.max()) if len(recomputed_array) else None
        ),
        "max_selected_feature_absolute_error": float(feature_array.max()) if len(feature_array) else None,
        "resolved_bound_flag": evaluator.apply_prelim_bounds,
        "projection_semantics": "match PRELIM training; apply clipping only when bound.flag=true",
        "pinned_explorer_behavior": "pyInstanceSpace 0.2.1 explore clips unconditionally; not used for GA projection",
        "details_csv": details_path.name,
        "recomputed_features_csv": (
            f"{output_path.stem}_recomputed_features.csv" if baseline_records is not None else None
        ),
        "result_provenance": "measured",
    }
    write_json(output_path, result)
    if result["status"] != "passed":
        raise RuntimeError(
            "Projeção GA diverge do PILOT do baseline: "
            f"erro direto={result['max_euclidean_error']}; "
            f"erro após recálculo={result['max_recomputed_projection_euclidean_error']}; "
            f"erro de feature={result['max_selected_feature_absolute_error']}; "
            f"tolerância={tolerance}",
        )
    return result


def _valid_polygon(points: np.ndarray) -> Polygon | None:
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
        return None
    polygon = Polygon(points[:, :2]).buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        polygon = MultiPoint(points[:, :2]).convex_hull
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None


def cloister_polygon(model: Any) -> Polygon | None:
    for candidate in (np.asarray(model.cloister.z_ecorr), np.asarray(model.cloister.z_edge)):
        polygon = _valid_polygon(candidate)
        if polygon is not None:
            return polygon
    return None


def validate_exact_domain_membership(
    coordinate: np.ndarray,
    *,
    empirical_hull: Any,
    cloister: Polygon | None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Confirm that an exact finalist remains in the internal population domain."""

    point = Point(float(coordinate[0]), float(coordinate[1]))
    inside_hull = bool(empirical_hull.buffer(tolerance).covers(point))
    inside_cloister = bool(cloister is None or cloister.buffer(tolerance).covers(point))
    outside_hull = not inside_hull
    confirmed = inside_hull and inside_cloister
    return {
        "inside_empirical_hull_exact": inside_hull,
        "outside_empirical_hull_exact": outside_hull,
        "inside_cloister_exact": inside_cloister,
        "domain_membership_confirmed_exact": confirmed,
        "domain_membership_tolerance": tolerance,
    }


def _sobol_in_geometry(geometry: Any, count: int, seed: int, pool_size: int) -> pd.DataFrame:
    if geometry is None or geometry.is_empty:
        return pd.DataFrame(columns=["point_id", "z1", "z2"])
    minx, miny, maxx, maxy = geometry.bounds
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    rows: list[dict[str, Any]] = []
    generated = 0
    batch_size = max(int(pool_size), 1024)
    maximum_draws = max(batch_size * 128, int(count) * 1024)
    # Preserve the original first Sobol batch (and therefore existing target
    # hashes), but continue deterministically if a thin geometry does not yield
    # the requested number of admissible points.  A coverage calculation must
    # never silently claim 10,000 points after observing fewer.
    while len(rows) < count and generated < maximum_draws:
        unit = sampler.random(batch_size)
        points = np.column_stack(
            [minx + unit[:, 0] * (maxx - minx), miny + unit[:, 1] * (maxy - miny)],
        )
        for local_index, (x, y) in enumerate(points):
            if geometry.covers(Point(float(x), float(y))):
                rows.append(
                    {
                        "point_id": f"sobol_{generated + local_index:08d}",
                        "z1": float(x),
                        "z2": float(y),
                    },
                )
                if len(rows) >= count:
                    break
        generated += batch_size
    if len(rows) < count:
        raise TechnicalExhaustionError(
            f"Amostragem Sobol obteve {len(rows)}/{count} pontos admissíveis após {generated} sorteios",
        )
    return pd.DataFrame(rows)


def leave_one_out_threshold(points: np.ndarray) -> float:
    distances = distance_matrix(points, points)
    np.fill_diagonal(distances, np.inf)
    return float(np.quantile(np.min(distances, axis=1), 0.90))


def select_maximin_targets(
    candidates: pd.DataFrame,
    occupied: np.ndarray,
    count: int,
    *,
    undercoverage_threshold: float | None,
) -> pd.DataFrame:
    if candidates.empty or count <= 0:
        return candidates.iloc[:0].copy()
    pool = candidates.copy().reset_index(drop=True)
    coords = pool[["z1", "z2"]].to_numpy(dtype=float)
    original_distance = np.min(distance_matrix(coords, occupied), axis=1)
    pool["empty_radius_original"] = original_distance
    if undercoverage_threshold is not None:
        pool = pool[pool["empty_radius_original"] > undercoverage_threshold].copy()
    selected: list[pd.Series] = []
    working_occupied = occupied.copy()
    while not pool.empty and len(selected) < count:
        coords = pool[["z1", "z2"]].to_numpy(dtype=float)
        score = np.min(distance_matrix(coords, working_occupied), axis=1)
        ranked = pool.assign(_score=score).sort_values(
            ["_score", "z1", "z2", "point_id"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        chosen = ranked.iloc[0].drop(labels="_score")
        selected.append(chosen)
        point = np.asarray([[float(chosen["z1"]), float(chosen["z2"])]])
        working_occupied = np.vstack([working_occupied, point])
        pool = pool[pool["point_id"] != chosen["point_id"]]
    result = pd.DataFrame(selected).reset_index(drop=True)
    if not result.empty:
        result["target_order"] = np.arange(len(result), dtype=int)
    return result


def make_initial_targets(
    baseline_coordinates: pd.DataFrame,
    model: Any,
    output_dir: Path,
    *,
    count: int,
    sobol_seed: int,
    sobol_points: int,
    config_hash: str,
) -> pd.DataFrame:
    points = baseline_coordinates[["z1", "z2"]].to_numpy(dtype=float)
    hull = MultiPoint(points).convex_hull
    cloister = cloister_polygon(model)
    geometry = hull if cloister is None else hull.intersection(cloister)
    threshold = leave_one_out_threshold(points)
    filename = "targets_internal_round0.csv"
    candidates = _sobol_in_geometry(geometry, sobol_points, sobol_seed, max(sobol_points * 8, 1024))
    targets = select_maximin_targets(candidates, points, count, undercoverage_threshold=threshold)
    baseline_hash = stable_json_hash(
        baseline_coordinates[["instance", "z1", "z2"]].sort_values("instance").to_dict(orient="records"),
    )
    if not targets.empty:
        targets["target_id"] = [f"internal_common_r0_t{index:02d}" for index in range(len(targets))]
        targets["round"] = 0
        targets["region"] = "internal"
        targets["baseline_sha256"] = baseline_hash
        targets["configuration_sha256"] = config_hash
        targets["result_provenance"] = "measured"
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_dir / filename, index=False)
    write_json(
        output_dir / f"{filename}.metadata.json",
        {
            "baseline_sha256": baseline_hash,
            "configuration_sha256": config_hash,
            "sobol_scramble": True,
            "sobol_seed": sobol_seed,
            "sobol_points_requested": sobol_points,
            "geometry": "empirical_hull_intersection_cloister",
            "undercoverage_threshold": threshold,
            "targets_created": len(targets),
            "csv_sha256": hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
            "result_provenance": "measured" if len(targets) else "missing",
        },
    )
    return targets


def make_adaptive_targets(
    baseline_coordinates: pd.DataFrame,
    accepted_coordinates: np.ndarray,
    model: Any,
    *,
    count: int,
    round_number: int,
    root_seed: int,
    sobol_points: int,
) -> pd.DataFrame:
    baseline = baseline_coordinates[["z1", "z2"]].to_numpy(dtype=float)
    occupied = np.vstack([baseline, accepted_coordinates]) if accepted_coordinates.size else baseline
    hull = MultiPoint(baseline).convex_hull
    cloister = cloister_polygon(model)
    geometry = hull if cloister is None else hull.intersection(cloister)
    threshold = leave_one_out_threshold(baseline)
    candidates = _sobol_in_geometry(
        geometry,
        sobol_points,
        seed=int(np.random.SeedSequence([root_seed, round_number, 991]).generate_state(1)[0]),
        pool_size=max(sobol_points * 8, 1024),
    )
    targets = select_maximin_targets(candidates, occupied, count, undercoverage_threshold=threshold)
    if not targets.empty:
        targets["target_id"] = [f"internal_seed{root_seed}_r{round_number}_t{index:02d}" for index in range(len(targets))]
        targets["round"] = round_number
        targets["region"] = "internal"
        targets["root_seed"] = root_seed
        targets["result_provenance"] = "measured"
    return targets


def _tournament(population: list[EvaluatedCandidate], rng: np.random.Generator, size: int) -> EvaluatedCandidate:
    indices = rng.choice(len(population), size=min(size, len(population)), replace=False)
    return min((population[int(index)] for index in indices), key=lambda candidate: (candidate.fitness, candidate.identifier))


def _select_survivors(
    current_population: list[EvaluatedCandidate],
    offspring: list[EvaluatedCandidate],
    population_size: int,
    elitism: int,
) -> list[EvaluatedCandidate]:
    """Keep previous elites and fill every remaining slot by fitness."""

    ordered_current = sorted(current_population, key=lambda candidate: (candidate.fitness, candidate.identifier))
    elites = ordered_current[: min(elitism, len(ordered_current))]
    elite_ids = {candidate.identifier for candidate in elites}
    ordered_offspring = sorted(offspring, key=lambda candidate: (candidate.fitness, candidate.identifier))
    remainder = [candidate for candidate in ordered_offspring if candidate.identifier not in elite_ids]
    return (elites + remainder)[:population_size]


def evaluation_auc(history: pd.DataFrame, reference_fitness: float, budget: int) -> float:
    if history.empty or reference_fitness <= 0:
        return math.nan
    ordered = history.sort_values("evaluation")
    x = ordered["evaluation"].to_numpy(dtype=float) / budget
    y = np.minimum(1.0, ordered["best_so_far_fitness"].to_numpy(dtype=float) / reference_fitness)
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[1.0], y])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        y = np.concatenate([y, [y[-1]]])
    elif x[-1] > 1.0:
        mask = x <= 1.0
        x, y = x[mask], y[mask]
        if x[-1] < 1.0:
            x = np.concatenate([x, [1.0]])
            y = np.concatenate([y, [y[-1]]])
    return float(np.trapz(y, x))


def wall_auc(history: pd.DataFrame, reference_fitness: float, reference_time: float) -> float:
    if history.empty or reference_fitness <= 0 or reference_time <= 0:
        return math.nan
    ordered = history.sort_values("wall_seconds")
    x = np.minimum(1.0, ordered["wall_seconds"].to_numpy(dtype=float) / reference_time)
    y = np.minimum(1.0, ordered["best_so_far_fitness"].to_numpy(dtype=float) / reference_fitness)
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[1.0], y])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        y = np.concatenate([y, [y[-1]]])
    return float(np.trapz(y, x))


def _quarantine_uncommitted_search_artifacts(
    evaluator: Any,
    target_dir: Path,
    history_path: Path,
) -> list[dict[str, Any]]:
    """Remove cache writes that occurred after the last committed history row.

    Candidate evaluation writes its reusable feature artifact immediately before
    appending the corresponding convergence row.  A process interruption in that
    narrow interval leaves a valid calculation in the cache but no committed GA
    evaluation.  Treating that artifact as a pre-existing duplicate on replay
    changes the RNG rejection path and makes the checkpoint non-reproducible.

    The artifact is preserved under the target checkpoint for auditability, but
    removed from the active cache so the interrupted evaluation is performed and
    committed normally.  Confirmed history rows and their artifacts are never
    touched.
    """
    generated_dir = getattr(evaluator, "generated_dir", None)
    existing = getattr(evaluator, "existing_search_identifiers", None)
    if not isinstance(generated_dir, Path) or not history_path.is_file():
        return []
    checkpoint_mtime_ns = history_path.stat().st_mtime_ns
    quarantine_dir = target_dir / "interrupted_uncommitted_cache"
    records: list[dict[str, Any]] = []
    for artifact in sorted(generated_dir.glob("*_search.json")):
        artifact_mtime_ns = artifact.stat().st_mtime_ns
        if artifact_mtime_ns <= checkpoint_mtime_ns:
            continue
        identifier = artifact.name.removesuffix("_search.json")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / f"{artifact.stem}_{artifact_mtime_ns}.json"
        artifact.replace(destination)
        if isinstance(existing, set):
            existing.discard(identifier)
        records.append(
            {
                "candidate_id": identifier,
                "original_path": str(artifact),
                "quarantined_path": str(destination),
                "artifact_mtime_ns": artifact_mtime_ns,
                "checkpoint_mtime_ns": checkpoint_mtime_ns,
                "sha256": digest,
                "reason": "artifact_written_after_last_committed_convergence_row",
                "result_provenance": "measured",
            },
        )
    if records:
        write_json(
            target_dir / "interrupted_uncommitted_cache_manifest.json",
            {
                "artifacts": records,
                "count": len(records),
                "history_path": str(history_path),
                "policy": "preserved_but_excluded_from_active_cache_and_evaluation_count",
                "result_provenance": "measured",
            },
        )
    return records


def run_ga_target(
    target_row: pd.Series,
    evaluator: CandidateEvaluator,
    output_dir: Path,
    *,
    root_seed: int,
    population_size: int,
    generations: int,
    local_evaluations: int,
    tournament_size: int,
    elitism: int,
    crossover_probability: float,
    mutation_probability: float,
    ga_priority: float = 0.8,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if not 0 < ga_priority <= 1:
        raise ValueError(f"ga_priority deve estar em (0, 1], recebido {ga_priority}")
    target = np.asarray([target_row["z1"], target_row["z2"]], dtype=float)
    reference = float(target_row["empty_radius_original"])
    budget = population_size * generations + local_evaluations
    target_order = int(target_row.get("target_order", 0))
    round_number = int(target_row.get("round", 0))
    seed_sequence = np.random.SeedSequence([root_seed, 1, round_number, target_order])
    rng = np.random.default_rng(seed_sequence)
    target_dir = output_dir / str(target_row["target_id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    history_path = target_dir / "convergence_raw.csv"
    try:
        saved_history = pd.read_csv(history_path)
    except (OSError, pd.errors.EmptyDataError):
        saved_history = pd.DataFrame()
    if not saved_history.empty:
        required_history_columns = {
            "evaluation",
            "candidate_id",
            "origin_queue",
            "generation",
            "wall_seconds",
            "fitness",
            "best_so_far_fitness",
        }
        if not required_history_columns.issubset(saved_history.columns):
            raise ValueError(f"Checkpoint de convergência incompleto: {history_path}")
        expected_evaluations = np.arange(1, len(saved_history) + 1)
        if not np.array_equal(saved_history["evaluation"].to_numpy(dtype=int), expected_evaluations):
            raise ValueError(f"Sequência de avaliações inválida no checkpoint: {history_path}")
    quarantined_artifacts = _quarantine_uncommitted_search_artifacts(evaluator, target_dir, history_path)
    seen: set[str] = set()
    histories: list[dict[str, Any]] = saved_history.to_dict(orient="records")
    replay_count = len(histories)
    replay_identifiers = set(saved_history.get("candidate_id", pd.Series(dtype=str)).astype(str))
    wall_offset = float(saved_history.iloc[-1]["wall_seconds"]) if replay_count else 0.0
    evaluation_cursor = 0
    current_population: list[EvaluatedCandidate] = []
    best: EvaluatedCandidate | None = None
    started = time.perf_counter()
    search_calculation_mode = (
        "search_exact_selected_topology"
        if getattr(evaluator, "search_landmarks", 0) <= 0
        else "search_approximation"
    )
    write_json(
        target_dir / "resume_state.json",
        {
            "recovered_evaluations": replay_count,
            "wall_seconds_recovered": wall_offset,
            "resume_replay_required": bool(replay_count),
            "candidate_artifacts_visible_at_start": len(getattr(evaluator, "existing_search_identifiers", set())),
            "uncommitted_artifacts_quarantined": len(quarantined_artifacts),
            "result_provenance": "measured",
        },
    )

    def unique_genotype(factory: Callable[[], Genotype]) -> Genotype:
        for _ in range(100000):
            genotype = factory()
            identifier = genotype.identifier()
            cached_before_target = identifier in getattr(evaluator, "existing_search_identifiers", set())
            if identifier not in seen and (not cached_before_target or identifier in replay_identifiers):
                seen.add(identifier)
                return genotype
        raise TechnicalExhaustionError("Esgotamento técnico ao procurar genótipo único")

    def evaluate(genotype: Genotype, origin: str, generation: int) -> EvaluatedCandidate:
        nonlocal best, evaluation_cursor
        evaluation_cursor += 1
        if evaluation_cursor <= replay_count:
            saved = saved_history.iloc[evaluation_cursor - 1]
            if (
                str(saved["candidate_id"]) != genotype.identifier()
                or str(saved["origin_queue"]) != origin
                or int(saved["generation"]) != generation
            ):
                raise RuntimeError(
                    f"Replay divergente na avaliação {evaluation_cursor}: "
                    f"esperado={saved['candidate_id']}, obtido={genotype.identifier()}",
                )
        candidate = evaluator.evaluate(genotype, target, exact=False)
        if best is None or (candidate.fitness, candidate.identifier) < (best.fitness, best.identifier):
            best = candidate
        if evaluation_cursor > replay_count:
            row = {
                "target_id": target_row["target_id"],
                "root_seed": root_seed,
                "round": round_number,
                "evaluation": evaluation_cursor,
                "wall_seconds": wall_offset + time.perf_counter() - started,
                "fitness": candidate.fitness,
                "best_so_far_fitness": best.fitness,
                "origin_queue": origin,
                "generation": generation,
                "candidate_id": candidate.identifier,
                "calculation_mode": search_calculation_mode,
                "result_provenance": "measured",
            }
            histories.append(row)
            pd.DataFrame([row]).to_csv(
                history_path,
                mode="a",
                header=not history_path.exists() or history_path.stat().st_size == 0,
                index=False,
            )
        return candidate

    technical_exhaustion: str | None = None
    ga_evaluations = 0
    local_completed = 0
    try:
        for generation in range(generations):
            ga_queue: list[Genotype] = []
            for _ in range(population_size):
                if generation == 0 or not current_population:
                    genotype = unique_genotype(lambda: random_genotype(rng))
                else:
                    parent_a = _tournament(current_population, rng, tournament_size)
                    parent_b = _tournament(current_population, rng, tournament_size)
                    genotype = unique_genotype(
                        lambda: mutate(
                            uniform_crossover(parent_a.genotype, parent_b.genotype, rng, crossover_probability),
                            rng,
                            mutation_probability,
                        ),
                    )
                ga_queue.append(genotype)

            offspring: list[EvaluatedCandidate] = []
            ga_cursor = 0
            while ga_cursor < len(ga_queue):
                dispatched = ga_evaluations + local_completed
                current_ratio = ga_evaluations / dispatched if dispatched else 0.0
                dispatch_ga = (
                    best is None
                    or local_completed >= local_evaluations
                    or current_ratio < ga_priority
                )
                if dispatch_ga:
                    genotype = ga_queue[ga_cursor]
                    ga_cursor += 1
                    offspring.append(evaluate(genotype, "ga", generation))
                    ga_evaluations += 1
                else:
                    assert best is not None
                    genotype = unique_genotype(lambda: mutate(best.genotype, rng, mutation_probability))
                    evaluate(genotype, "local_search", generation)
                    local_completed += 1
            current_population = _select_survivors(
                current_population,
                offspring,
                population_size,
                elitism,
            )

        # Complete the declared local-search budget after the final synchronous
        # GA generation if queue failover left any local slots undispatched.
        while local_completed < local_evaluations:
            assert best is not None
            genotype = unique_genotype(lambda: mutate(best.genotype, rng, mutation_probability))
            evaluate(genotype, "local_search", generations)
            local_completed += 1
    except TechnicalExhaustionError as exc:
        technical_exhaustion = f"{type(exc).__name__}: {exc}"

    history = pd.DataFrame(histories)
    if not history.empty:
        history[["evaluation", "fitness", "best_so_far_fitness", "candidate_id", "origin_queue", "generation"]].to_csv(
            target_dir / "convergence_by_evaluations_raw.csv",
            index=False,
        )
        history[["wall_seconds", "fitness", "best_so_far_fitness", "candidate_id", "origin_queue", "generation"]].to_csv(
            target_dir / "convergence_by_wall_time_raw.csv",
            index=False,
        )
        x_eval = history["evaluation"].to_numpy(dtype=float) / budget
        y_eval = np.minimum(1.0, history["best_so_far_fitness"].to_numpy(dtype=float) / reference)
        normalized_eval = pd.DataFrame({"x_e": np.concatenate([[0.0], x_eval]), "y": np.concatenate([[1.0], y_eval])})
        if normalized_eval.iloc[-1]["x_e"] < 1.0:
            normalized_eval.loc[len(normalized_eval)] = [1.0, normalized_eval.iloc[-1]["y"]]
        normalized_eval.to_csv(target_dir / "convergence_by_evaluations_normalized.csv", index=False)
    if best is None:
        result = {
            "target_id": target_row["target_id"],
            "root_seed": root_seed,
            "status": "missing",
            "reason": technical_exhaustion or "no_valid_evaluation",
            "evaluations_completed": 0,
            "evaluation_budget": budget,
            "result_provenance": "missing",
        }
        write_json(target_dir / "result.json", result)
        return result, None

    exact_error: str | None = None
    exact_best: EvaluatedCandidate | None = None
    try:
        exact_best = evaluator.evaluate(best.genotype, target, exact=True)
    except Exception as exc:  # exact validation failure must stay visible, never imputed
        exact_error = f"{type(exc).__name__}: {exc}"
    final_fitness = exact_best.fitness if exact_best is not None else math.nan
    if exact_best is None:
        classification = "missing"
    elif final_fitness <= 0.25 * reference:
        classification = "hit"
    elif final_fitness < reference:
        classification = "improved"
    else:
        classification = "missed"
    elapsed = wall_offset + time.perf_counter() - started
    selected_candidate_rows = history[history["candidate_id"] == best.identifier]
    selected_candidate_discovery = (
        float(selected_candidate_rows["wall_seconds"].min()) if not selected_candidate_rows.empty else None
    )
    search_time_to_improved = (
        float(history.loc[history["best_so_far_fitness"] < reference, "wall_seconds"].min())
        if (history["best_so_far_fitness"] < reference).any()
        else None
    )
    search_time_to_hit = (
        float(history.loc[history["best_so_far_fitness"] <= 0.25 * reference, "wall_seconds"].min())
        if (history["best_so_far_fitness"] <= 0.25 * reference).any()
        else None
    )
    result = {
        "target_id": target_row["target_id"],
        "region": target_row.get("region"),
        "round": round_number,
        "target_order": target_order,
        "root_seed": root_seed,
        "target_z1": float(target[0]),
        "target_z2": float(target[1]),
        "empty_radius_original": reference,
        "hit_threshold": 0.25 * reference,
        "best_search_candidate_id": best.identifier,
        "best_search_fitness": best.fitness,
        "best_exact_candidate_id": exact_best.identifier if exact_best else None,
        "best_exact_instance_sha256": instance_content_hash(exact_best.data) if exact_best else None,
        "best_exact_fitness": final_fitness,
        "exact_z1": float(exact_best.z[0]) if exact_best else None,
        "exact_z2": float(exact_best.z[1]) if exact_best else None,
        "classification": classification,
        "accepted": bool(exact_best is not None and classification in {"hit", "improved"}),
        "evaluations_completed": len(history),
        "evaluation_budget": budget,
        "ga_evaluations_completed": ga_evaluations,
        "local_search_evaluations_completed": local_completed,
        "ga_priority": ga_priority,
        "local_search_priority": 1.0 - ga_priority,
        "observed_ga_allocation_ratio": ga_evaluations / max(ga_evaluations + local_completed, 1),
        "dispatch_policy": "Bowly-inspired GA/local-search allocation adapted to synchronous generations and independent-seed workers",
        "technical_exhaustion": technical_exhaustion,
        "exact_validation_error": exact_error,
        "auc_eval": evaluation_auc(history, reference, budget),
        "wall_seconds": elapsed,
        "evaluations_per_second": len(history) / elapsed if elapsed else math.nan,
        "time_to_first_improved": selected_candidate_discovery if classification in {"hit", "improved"} else None,
        "time_to_hit": selected_candidate_discovery if classification == "hit" else None,
        "time_to_first_improved_search_curve": search_time_to_improved,
        "time_to_hit_search_curve": search_time_to_hit,
        "search_calculation_mode": search_calculation_mode,
        "time_semantics": (
            "time_to_first_improved/time_to_hit refer only to the final fully exact candidate; "
            "*_search_curve refers to fitness from the selected exact topology features"
            if search_calculation_mode == "search_exact_selected_topology"
            else "time_to_first_improved/time_to_hit refer only to the final fully exact candidate; "
            "*_search_curve refers to approximate search fitness"
        ),
        "seed_sequence_entropy": list(seed_sequence.entropy) if isinstance(seed_sequence.entropy, tuple | list) else seed_sequence.entropy,
        "seed_sequence_spawn_key": list(seed_sequence.spawn_key),
        "result_provenance": "measured" if exact_best is not None else "missing",
    }
    write_json(target_dir / "result.json", result)
    return result, exact_best.z if result["accepted"] and exact_best is not None else None


def coverage_metrics(
    baseline_coordinates: pd.DataFrame,
    accepted: np.ndarray,
    *,
    sobol_seed: int,
    sobol_points: int,
    cloister: Polygon | None = None,
) -> dict[str, Any]:
    baseline = baseline_coordinates[["z1", "z2"]].to_numpy(dtype=float)
    occupied = np.vstack([baseline, accepted]) if accepted.size else baseline
    hull = MultiPoint(baseline).convex_hull
    sampling_geometry = hull if cloister is None else hull.intersection(cloister)
    candidates = _sobol_in_geometry(
        sampling_geometry,
        sobol_points,
        sobol_seed,
        max(sobol_points * 8, 1024),
    )
    sample = candidates[["z1", "z2"]].to_numpy(dtype=float)
    nearest = np.min(distance_matrix(sample, occupied), axis=1) if len(sample) else np.asarray([])
    threshold = leave_one_out_threshold(baseline)
    metrics: dict[str, Any] = {
        "fill_distance": float(nearest.max()) if len(nearest) else math.nan,
        "nearest_mean": float(nearest.mean()) if len(nearest) else math.nan,
        "nearest_median": float(np.median(nearest)) if len(nearest) else math.nan,
        "nearest_q90": float(np.quantile(nearest, 0.9)) if len(nearest) else math.nan,
        "undercovered_fraction": float(np.mean(nearest > threshold)) if len(nearest) else math.nan,
        "undercoverage_threshold": threshold,
        "empirical_hull_area_context_only": float(hull.area),
        "coverage_domain": "empirical_hull" if cloister is None else "empirical_hull_intersection_cloister",
        "coverage_domain_area": float(sampling_geometry.area),
        "sobol_points_requested": int(sobol_points),
        "sobol_points_observed": int(len(sample)),
        "accepted_instances": int(len(accepted)),
        "result_provenance": "measured",
    }
    minx, miny, maxx, maxy = hull.bounds
    for grid in (8, 10, 12):
        ix = np.clip(((occupied[:, 0] - minx) / max(maxx - minx, 1e-12) * grid).astype(int), 0, grid - 1)
        iy = np.clip(((occupied[:, 1] - miny) / max(maxy - miny, 1e-12) * grid).astype(int), 0, grid - 1)
        metrics[f"occupancy_{grid}x{grid}"] = len(set(zip(ix, iy))) / (grid * grid)
    return metrics


def _run_ga_seed(
    experiment_dir: Path,
    initial_targets: pd.DataFrame,
    baseline_coordinates: pd.DataFrame,
    model: Any,
    evaluator_factory: Callable[[Path], CandidateEvaluator],
    config: Any,
    *,
    root_seed: int,
) -> list[dict[str, Any]]:
    target_budget = config.target_budget
    baseline_hull = MultiPoint(baseline_coordinates[["z1", "z2"]].to_numpy(dtype=float)).convex_hull
    cloister = cloister_polygon(model)
    seed_dir = experiment_dir / f"seed_{root_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_results_path = seed_dir / "ga_seed_results.csv"
    seed_results: list[dict[str, Any]] = pd.read_csv(seed_results_path).to_dict(orient="records") if seed_results_path.exists() else []
    known = {str(row["target_id"]): index for index, row in enumerate(seed_results)}
    evaluator = evaluator_factory(seed_dir)
    accepted: list[np.ndarray] = []
    targets_done = 0
    desired_budget = target_budget
    round_number = 0
    coverage_records: list[dict[str, Any]] = []
    current = initial_targets.copy()
    while targets_done < desired_budget and not current.empty:
        current = current.head(min(config.batch_size, desired_budget - targets_done)).copy()
        current.to_csv(seed_dir / f"targets_round_{round_number}.csv", index=False)
        for _, target_row in current.iterrows():
            result_path = seed_dir / str(target_row["target_id"]) / "result.json"
            expected_budget = config.ga_evaluation_budget
            saved: dict[str, Any] | None = None
            try:
                candidate_saved = json.loads(result_path.read_text(encoding="utf-8"))
                if int(candidate_saved.get("evaluations_completed", -1)) == expected_budget:
                    saved = candidate_saved
            except (OSError, json.JSONDecodeError, ValueError):
                pass
            if saved is None:
                result, coordinate = run_ga_target(
                    target_row,
                    evaluator,
                    seed_dir,
                    root_seed=root_seed,
                    population_size=config.population,
                    generations=config.generations,
                    local_evaluations=config.local_search_evaluations,
                    tournament_size=config.tournament_size,
                    elitism=config.elitism,
                    crossover_probability=config.crossover_probability,
                    mutation_probability=config.mutation_probability,
                    ga_priority=config.ga_priority,
                )
            else:
                result = saved
                coordinate = None
            exact_coordinate = (
                np.asarray([result["exact_z1"], result["exact_z2"]], dtype=float)
                if result.get("exact_z1") is not None and result.get("exact_z2") is not None
                else None
            )
            # Domain validity and search success answer different questions.
            # Evaluate membership for every exactly recomputed finalist so that
            # generator-validity rates are not conditioned on hit/improvement.
            if exact_coordinate is not None and np.isfinite(exact_coordinate).all():
                membership = validate_exact_domain_membership(
                    exact_coordinate,
                    empirical_hull=baseline_hull,
                    cloister=cloister,
                )
                result.update(membership)
                if result.get("accepted") and not membership["domain_membership_confirmed_exact"]:
                    result["classification_before_domain_validation"] = result.get("classification")
                    result["accepted"] = False
                    result["classification"] = "missed"
                    result["rejection_reason"] = "exact_projection_outside_internal_domain"
                coordinate = exact_coordinate if result.get("accepted") else None
                write_json(result_path, result)
            key = str(result["target_id"])
            if key in known:
                seed_results[known[key]] = result
            else:
                known[key] = len(seed_results)
                seed_results.append(result)
            targets_done += 1
            if coordinate is not None:
                accepted.append(coordinate)
        pd.DataFrame(seed_results).sort_values(["round", "target_order", "target_id"], kind="mergesort").to_csv(seed_results_path, index=False)
        accepted_array = np.vstack(accepted) if accepted else np.empty((0, 2))
        coverage = coverage_metrics(
            baseline_coordinates,
            accepted_array,
            sobol_seed=config.baseline_sobol_seed,
            sobol_points=config.sobol_points,
            cloister=cloister,
        )
        coverage.update({"root_seed": root_seed, "round": round_number})
        coverage_records.append(coverage)
        coverage_path = seed_dir / "coverage_history.jsonl"
        coverage_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in coverage_records),
            encoding="utf-8",
        )
        round_number += 1
        if targets_done >= desired_budget:
            break
        current = make_adaptive_targets(
            baseline_coordinates,
            accepted_array,
            model,
            count=min(config.batch_size, desired_budget - targets_done),
            round_number=round_number,
            root_seed=root_seed,
            sobol_points=config.sobol_points,
        )
    if targets_done < desired_budget:
        write_json(
            seed_dir / "target_shortfall.json",
            {
                "requested": desired_budget,
                "completed": targets_done,
                "reason": "no_remaining_target_in_defined_geometry",
                "result_provenance": "missing",
            },
        )
    return seed_results


def _ga_seed_worker(root_seed: int) -> list[dict[str, Any]]:
    if not _GA_WORKER_CONTEXT:
        raise RuntimeError("Contexto do worker GA não foi inicializado")
    return _run_ga_seed(root_seed=root_seed, **_GA_WORKER_CONTEXT)


def run_ga_experiment(
    root: Path,
    experiment_dir: Path,
    initial_targets: pd.DataFrame,
    baseline_coordinates: pd.DataFrame,
    model: Any,
    evaluator_factory: Callable[[Path], CandidateEvaluator],
    config: Any,
) -> pd.DataFrame:
    del root  # O caminho já está capturado pela fábrica do avaliador.
    global _GA_WORKER_CONTEXT
    _GA_WORKER_CONTEXT = {
        "experiment_dir": experiment_dir,
        "initial_targets": initial_targets,
        "baseline_coordinates": baseline_coordinates,
        "model": model,
        "evaluator_factory": evaluator_factory,
        "config": config,
    }
    seeds = [int(seed) for seed in config.root_seeds]
    workers = min(max(1, int(config.workers)), len(seeds))
    write_json(
        experiment_dir / "ga_parallel_execution.json",
        {
            "workers_requested": int(config.workers),
            "workers_effective": workers,
            "parallel_unit": "independent root seed",
            "root_seeds": seeds,
            "shared_evolutionary_state": False,
            "result_provenance": "measured",
        },
    )
    try:
        if workers == 1:
            grouped = [_ga_seed_worker(seed) for seed in seeds]
        else:
            # O contexto fork preserva o modelo MATILDA em memória e mantém em
            # processos separados o RNG global usado pelo gerador legado.
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as executor:
                grouped = list(executor.map(_ga_seed_worker, seeds))
    finally:
        _GA_WORKER_CONTEXT = {}
    all_results = [row for group in grouped for row in group]
    results = pd.DataFrame(all_results)
    if not results.empty:
        results = results.sort_values(["root_seed", "round", "target_order", "target_id"], kind="mergesort").reset_index(drop=True)
    results.to_csv(experiment_dir / "ga_target_results.csv", index=False)
    return results


def consolidate_wall_auc(experiment_dir: Path, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    updated = results.copy()
    updated["radius_class"] = pd.qcut(updated["empty_radius_original"], q=min(3, updated["empty_radius_original"].nunique()), duplicates="drop", labels=False)
    updated["radius_class_lower"] = updated.groupby("radius_class", dropna=False)["empty_radius_original"].transform("min")
    updated["radius_class_upper"] = updated.groupby("radius_class", dropna=False)["empty_radius_original"].transform("max")
    wall_values: dict[tuple[str, int], float] = {}
    common = updated[updated["round"] == 0]
    for target_id, group in common.groupby("target_id"):
        reference_time = float(group["wall_seconds"].max())
        for index, row in group.iterrows():
            path = experiment_dir / f"seed_{int(row.root_seed)}" / str(target_id) / "convergence_raw.csv"
            history = pd.read_csv(path)
            wall_values[(str(target_id), int(row.root_seed))] = wall_auc(history, float(row.empty_radius_original), reference_time)
            normalized = pd.DataFrame(
                {
                    "x_t": np.concatenate([[0.0], np.minimum(1.0, history["wall_seconds"].to_numpy(dtype=float) / reference_time)]),
                    "y": np.concatenate([[1.0], np.minimum(1.0, history["best_so_far_fitness"].to_numpy(dtype=float) / float(row.empty_radius_original))]),
                },
            )
            if normalized.iloc[-1]["x_t"] < 1.0:
                normalized.loc[len(normalized)] = [1.0, normalized.iloc[-1]["y"]]
            normalized["T_ref_seconds"] = reference_time
            normalized.to_csv(path.parent / "convergence_by_wall_time_normalized.csv", index=False)
    adaptive = updated[updated["round"] > 0]
    for (_round, radius_class), group in adaptive.groupby(["round", "radius_class"], dropna=False):
        reference_time = float(group["wall_seconds"].max())
        for _, row in group.iterrows():
            path = experiment_dir / f"seed_{int(row.root_seed)}" / str(row.target_id) / "convergence_raw.csv"
            history = pd.read_csv(path)
            wall_values[(str(row.target_id), int(row.root_seed))] = wall_auc(history, float(row.empty_radius_original), reference_time)
            normalized = pd.DataFrame(
                {
                    "x_t": np.concatenate([[0.0], np.minimum(1.0, history["wall_seconds"].to_numpy(dtype=float) / reference_time)]),
                    "y": np.concatenate([[1.0], np.minimum(1.0, history["best_so_far_fitness"].to_numpy(dtype=float) / float(row.empty_radius_original))]),
                },
            )
            if normalized.iloc[-1]["x_t"] < 1.0:
                normalized.loc[len(normalized)] = [1.0, normalized.iloc[-1]["y"]]
            normalized["T_ref_seconds"] = reference_time
            normalized.to_csv(path.parent / "convergence_by_wall_time_normalized.csv", index=False)
    updated["auc_wall"] = [wall_values.get((str(row.target_id), int(row.root_seed)), math.nan) for row in updated.itertuples()]
    updated.to_csv(experiment_dir / "ga_target_results_with_wall_auc.csv", index=False)
    numeric = ["best_exact_fitness", "auc_eval", "auc_wall", "wall_seconds", "evaluations_per_second"]
    summary_rows: list[dict[str, Any]] = []
    for (round_number, radius_class), group in updated.groupby(["round", "radius_class"], dropna=False):
        row: dict[str, Any] = {
            "round": int(round_number),
            "radius_class": None if pd.isna(radius_class) else int(radius_class),
            "radius_class_lower": float(group["radius_class_lower"].iloc[0]),
            "radius_class_upper": float(group["radius_class_upper"].iloc[0]),
            "n": len(group),
            "result_provenance": "measured",
        }
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            for name, value in (
                ("mean", values.mean()),
                ("std", values.std(ddof=1)),
                ("median", values.median()),
                ("iqr", values.quantile(0.75) - values.quantile(0.25)),
                ("min", values.min()),
                ("max", values.max()),
            ):
                row[f"{metric}_{name}"] = float(value) if len(values) else math.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(experiment_dir / "ga_convergence_summary.csv", index=False)
    generate_convergence_figures(experiment_dir, updated)
    return updated


def generate_convergence_figures(experiment_dir: Path, results: pd.DataFrame) -> list[Path]:
    """Render the common-target curves without replacing their numeric CSVs.

    The first lot is the only portion in which target-by-target comparisons
    across root seeds are meaningful.  Adaptive targets deliberately diverge,
    so their AUCs are shown as distributions and remain stratified in
    ``ga_convergence_summary.csv``.  Plotting is imported lazily because GA
    workers must not carry Matplotlib state or its memory overhead.
    """

    if results.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    common = results[pd.to_numeric(results["round"], errors="coerce").eq(0)].copy()
    targets = sorted(common["target_id"].dropna().astype(str).unique())

    def common_plot(*, normalized: bool, by_wall: bool) -> None:
        if not targets:
            return
        ncols = min(3, len(targets))
        nrows = int(math.ceil(len(targets) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.9 * nrows), squeeze=False)
        for axis, target_id in zip(axes.flat, targets):
            subset = common[common["target_id"].astype(str) == target_id].sort_values("root_seed")
            for row in subset.itertuples():
                target_dir = experiment_dir / f"seed_{int(row.root_seed)}" / target_id
                if normalized:
                    filename = (
                        "convergence_by_wall_time_normalized.csv"
                        if by_wall
                        else "convergence_by_evaluations_normalized.csv"
                    )
                    curve = _read_curve(target_dir / filename)
                    x_column, y_column = ("x_t", "y") if by_wall else ("x_e", "y")
                else:
                    curve = _read_curve(target_dir / "convergence_raw.csv")
                    x_column = "wall_seconds" if by_wall else "evaluation"
                    y_column = "best_so_far_fitness"
                if curve.empty or x_column not in curve or y_column not in curve:
                    continue
                axis.step(
                    pd.to_numeric(curve[x_column], errors="coerce"),
                    pd.to_numeric(curve[y_column], errors="coerce"),
                    where="post",
                    linewidth=1.15,
                    label=str(int(row.root_seed)),
                )
            axis.set_title(target_id)
            axis.set_xlabel(
                "tempo normalizado" if normalized and by_wall else
                "avaliações normalizadas" if normalized else
                "segundos" if by_wall else "avaliações"
            )
            axis.set_ylabel("fitness normalizada" if normalized else "best-so-far fitness")
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(targets):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="seed-raiz",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(5, len(labels)),
            )
        mode = "wall" if by_wall else "evaluations"
        scale = "normalized" if normalized else "raw"
        path = figures_dir / f"convergence_{mode}_{scale}_common_targets.png"
        mode_label = "tempo de parede" if by_wall else "avaliações"
        scale_label = "normalizado" if normalized else "bruto"
        fig.suptitle(f"Convergência do primeiro lote comum: {mode_label} ({scale_label})", y=1.01)
        fig.tight_layout(rect=(0, 0.1 if handles else 0, 1, 1))
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    for normalized in (False, True):
        for by_wall in (False, True):
            common_plot(normalized=normalized, by_wall=by_wall)

    auc_columns = [("auc_eval", "AUC por avaliações"), ("auc_wall", "AUC por tempo de parede")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (column, title) in zip(axes, auc_columns):
        groups: list[np.ndarray] = []
        labels: list[str] = []
        for seed, group in results.groupby("root_seed", sort=True):
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                groups.append(values)
                labels.append(str(int(seed)))
        if groups:
            axis.boxplot(groups, tick_labels=labels, showmeans=True)
        axis.set_title(title)
        axis.set_xlabel("seed-raiz")
        axis.set_ylabel(column)
        axis.grid(axis="y", alpha=0.25)
    auc_path = figures_dir / "auc_distributions_by_root_seed.png"
    fig.tight_layout()
    fig.savefig(auc_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(auc_path)

    write_json(
        figures_dir / "figure_manifest.json",
        {
            "figures": [path.name for path in created],
            "common_targets_only_for_paired_curves": targets,
            "adaptive_targets": "AUC distributions only; numeric comparisons remain stratified by round and radius_class",
            "raw_numeric_source": "one convergence_raw.csv per target and seed",
            "result_provenance": "measured",
        },
    )
    return created


def _read_curve(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def generate_reach_figures(
    experiment_dir: Path,
    baseline_coordinates: pd.DataFrame,
    results: pd.DataFrame,
    *,
    sobol_seed: int = 20260612,
    sobol_points: int = 10_000,
    cloister: Polygon | None = None,
) -> list[Path]:
    """Visualize the accepted internal population and its coverage in PILOT."""

    if baseline_coordinates.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    baseline = baseline_coordinates[["z1", "z2"]].to_numpy(dtype=float)
    hull = MultiPoint(baseline).convex_hull
    accepted = (
        results[results["accepted"].fillna(False).astype(bool)].copy()
        if not results.empty and "accepted" in results
        else pd.DataFrame()
    )

    fig, axis = plt.subplots(figsize=(8.6, 7.2))
    axis.scatter(baseline[:, 0], baseline[:, 1], c="0.7", edgecolors="0.35", s=34, label="baseline (55)", zorder=3)
    if hasattr(hull, "exterior"):
        hx, hy = hull.exterior.xy
        axis.plot(hx, hy, color="black", linewidth=1.2, linestyle="--", label="fecho empírico")
    for seed, group in accepted.groupby("root_seed", sort=True) if not accepted.empty else []:
        axis.scatter(
            pd.to_numeric(group["exact_z1"], errors="coerce"),
            pd.to_numeric(group["exact_z2"], errors="coerce"),
            s=42,
            alpha=0.78,
            label=f"aceitas seed {int(seed)}",
        )
    axis.set_xlabel("PILOT z1")
    axis.set_ylabel("PILOT z2")
    axis.set_title("Povoamento interno aceito")
    axis.grid(alpha=0.22)
    axis.legend(loc="best", fontsize=8)
    geometry_path = figures_dir / "internal_coverage_pilot.png"
    fig.tight_layout()
    fig.savefig(geometry_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(geometry_path)

    if cloister is not None:
        allowed = hull.intersection(cloister)
        excluded = hull.difference(allowed)

        def polygon_parts(geometry: Any) -> list[Polygon]:
            if geometry is None or geometry.is_empty:
                return []
            if isinstance(geometry, Polygon):
                return [geometry]
            return [part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon)]

        seed_colors = {
            int(seed): plt.get_cmap("tab10")(index)
            for index, seed in enumerate(sorted(accepted["root_seed"].dropna().astype(int).unique()))
        } if not accepted.empty else {}

        fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.2))
        for axis in axes:
            for index, polygon in enumerate(polygon_parts(allowed)):
                x, y = polygon.exterior.xy
                axis.fill(
                    x,
                    y,
                    color="#55a868",
                    alpha=0.22,
                    label="domínio permitido (interseção)" if index == 0 else None,
                    zorder=1,
                )
            hx, hy = hull.exterior.xy
            axis.plot(hx, hy, color="black", linewidth=1.4, linestyle="--", label="fecho empírico", zorder=2)
            cx, cy = cloister.exterior.xy
            axis.plot(cx, cy, color="#7b3294", linewidth=1.5, linestyle="-.", label="CLOISTER", zorder=2)
            axis.scatter(
                baseline[:, 0],
                baseline[:, 1],
                c="0.72",
                edgecolors="0.32",
                s=32,
                label="baseline (55)",
                zorder=3,
            )
            for seed, group in accepted.groupby("root_seed", sort=True) if not accepted.empty else []:
                seed_value = int(seed)
                axis.scatter(
                    pd.to_numeric(group["exact_z1"], errors="coerce"),
                    pd.to_numeric(group["exact_z2"], errors="coerce"),
                    color=seed_colors[seed_value],
                    edgecolors="white",
                    linewidths=0.45,
                    s=46,
                    alpha=0.9,
                    label=f"aceitas seed {seed_value}",
                    zorder=4,
                )
            axis.set_xlabel("PILOT z1 (adimensional)")
            axis.set_ylabel("PILOT z2 (adimensional)")
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="box")

        axes[0].set_title("Contexto completo das fronteiras")

        for index, polygon in enumerate(polygon_parts(excluded)):
            x, y = polygon.exterior.xy
            axes[1].fill(
                x,
                y,
                facecolor="#c44e52",
                edgecolor="#c44e52",
                alpha=0.12,
                hatch="///",
                label="dentro do fecho, excluído pelo CLOISTER" if index == 0 else None,
                zorder=1.5,
            )
        if not results.empty and "domain_membership_confirmed_exact" in results:
            rejected = results[~results["domain_membership_confirmed_exact"].fillna(False).astype(bool)].copy()
            rejected_x = pd.to_numeric(rejected.get("exact_z1"), errors="coerce")
            rejected_y = pd.to_numeric(rejected.get("exact_z2"), errors="coerce")
            valid_rejected = rejected_x.notna() & rejected_y.notna()
            if valid_rejected.any():
                axes[1].scatter(
                    rejected_x[valid_rejected],
                    rejected_y[valid_rejected],
                    marker="x",
                    color="#c44e52",
                    linewidths=1.4,
                    s=52,
                    label="finalistas rejeitadas pelo domínio",
                    zorder=5,
                )
        minx, miny, maxx, maxy = hull.bounds
        padding_x = max((maxx - minx) * 0.06, 0.05)
        padding_y = max((maxy - miny) * 0.06, 0.05)
        axes[1].set_xlim(minx - padding_x, maxx + padding_x)
        axes[1].set_ylim(miny - padding_y, maxy + padding_y)
        axes[1].set_title("Zoom do domínio efetivamente permitido")

        hull_area = float(hull.area)
        allowed_area = float(allowed.area)
        retained_fraction = allowed_area / hull_area if hull_area > 0 else math.nan
        fig.suptitle("Domínio interno após CLOISTER: fecho empírico ∩ fronteira correlacional", y=0.99)
        fig.text(
            0.5,
            0.945,
            "z1 (horizontal) ↑: betweenness q90 e chegada mediana/horizonte; ↓: recursos/decisão, "
            "tempo máximo de arco e última liberação/horizonte.\n"
            "z2 (vertical) ↑: tempos mínimo/máximo dos arcos e harmônica de entrada q90; ↓: "
            "betweenness q90, número de vértices e chegada mediana/horizonte. "
            "Leitura dos maiores pesos após PRELIM; eixos adimensionais.",
            ha="center",
            va="top",
            fontsize=9,
            color="0.25",
        )
        fig.text(
            0.5,
            0.025,
            f"Área permitida: {allowed_area:.4f} de {hull_area:.4f} "
            f"({retained_fraction:.1%} do fecho empírico)",
            ha="center",
            fontsize=10,
        )
        handles, labels = axes[1].get_legend_handles_labels()
        unique = OrderedDict(zip(labels, handles))
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=3,
            fontsize=8,
        )
        cloister_path = figures_dir / "internal_domain_cloister.png"
        fig.tight_layout(rect=(0, 0.13, 1, 0.90))
        fig.savefig(cloister_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(cloister_path)

    history_rows: list[dict[str, Any]] = []
    for path in sorted(experiment_dir.glob("seed_*/coverage_history.jsonl")):
        try:
            history_rows.extend(
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        except (OSError, json.JSONDecodeError):
            continue
    history = pd.DataFrame(history_rows)
    if not history.empty:
        baseline_metrics_path = experiment_dir / "baseline_coverage_metrics.json"
        try:
            baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            baseline_metrics = coverage_metrics(
                baseline_coordinates,
                np.empty((0, 2)),
                sobol_seed=sobol_seed,
                sobol_points=sobol_points,
            )
        metrics = (
            ("fill_distance", "Fill distance", "menor é melhor"),
            ("undercovered_fraction", "Fração subcoberta", "menor é melhor"),
            ("occupancy_10x10", "Ocupação 10×10", "maior é melhor"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
        for axis, (metric, title, direction) in zip(axes, metrics):
            for seed, group in history.groupby("root_seed", sort=True):
                ordered = group.sort_values("accepted_instances")
                x = np.concatenate([[0.0], pd.to_numeric(ordered["accepted_instances"], errors="coerce")])
                y = np.concatenate([[float(baseline_metrics[metric])], pd.to_numeric(ordered[metric], errors="coerce")])
                axis.plot(x, y, marker="o", linewidth=1.3, markersize=3.5, label=str(int(seed)))
            axis.set_title(f"{title}\n({direction})")
            axis.set_xlabel("instâncias aceitas acumuladas")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="seed-raiz",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(5, len(labels)),
            )
        coverage_path = figures_dir / "coverage_trajectories.png"
        fig.tight_layout(rect=(0, 0.13 if handles else 0, 1, 1))
        fig.savefig(coverage_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(coverage_path)

    write_json(
        figures_dir / "reach_figure_manifest.json",
        {
            "figures": [path.name for path in created],
            "scope": "internal_gap_population",
            "accepted_coordinates_are_exact": True,
            "baseline_points": len(baseline),
            "domain_geometry": "empirical_hull_intersection_cloister" if cloister is not None else "empirical_hull",
            "axis_semantics": {
                "z1": {
                    "increases_with": ["betweenness_unweighted_q90", "arrival_q50_over_h"],
                    "decreases_with": ["resources_per_decision", "arc_time_max", "last_release_over_h"],
                },
                "z2": {
                    "increases_with": ["arc_time_max", "arc_time_min", "harmonic_in_distance_q90"],
                    "decreases_with": ["betweenness_unweighted_q90", "n_vertices", "arrival_q50_over_h"],
                },
                "interpretation": "PILOT coordinates, not isolated physical attributes",
                "basis": "largest absolute projection weights after PRELIM and SIFTED",
            },
            "empirical_hull_area": float(hull.area),
            "allowed_domain_area": float(hull.intersection(cloister).area) if cloister is not None else float(hull.area),
            "result_provenance": "measured",
        },
    )
    return created


def summarize_reach(
    experiment_dir: Path,
    baseline_coordinates: pd.DataFrame,
    results: pd.DataFrame,
    *,
    reference_seed: int = 20260612,
    sobol_seed: int = 20260612,
    sobol_points: int = 10_000,
    cloister: Polygon | None = None,
) -> dict[str, Any]:
    accepted = (
        results[results["accepted"].fillna(False).astype(bool)].copy()
        if not results.empty and "accepted" in results
        else pd.DataFrame()
    )
    deduplication_field = "best_exact_instance_sha256" if "best_exact_instance_sha256" in accepted else "best_exact_candidate_id"
    union_ids = set(accepted.get(deduplication_field, pd.Series(dtype=str)).dropna().astype(str))
    reference = accepted[accepted.get("root_seed", pd.Series(dtype=int)) == reference_seed] if not accepted.empty else accepted
    summary: dict[str, Any] = {
        "reference_seed": reference_seed,
        "reference_accepted": len(reference),
        "root_seed_count": int(results["root_seed"].nunique()) if not results.empty and "root_seed" in results else 0,
        "deduplicated_seed_union": len(union_ids),
        "deduplication_field": deduplication_field,
        "accepted_by_seed": accepted.groupby("root_seed").size().to_dict() if not accepted.empty else {},
        "result_provenance": "measured",
    }
    baseline_coverage_path = experiment_dir / "baseline_coverage_metrics.json"
    if cloister is not None or not baseline_coverage_path.is_file():
        baseline_coverage = coverage_metrics(
            baseline_coordinates,
            np.empty((0, 2)),
            sobol_seed=sobol_seed,
            sobol_points=sobol_points,
            cloister=cloister,
        )
        write_json(baseline_coverage_path, baseline_coverage)
    else:
        baseline_coverage = json.loads(baseline_coverage_path.read_text(encoding="utf-8"))
    summary["baseline_coverage"] = baseline_coverage
    generate_reach_figures(
        experiment_dir,
        baseline_coordinates,
        results,
        sobol_seed=sobol_seed,
        sobol_points=sobol_points,
        cloister=cloister,
    )
    write_json(experiment_dir / "collection_reach_summary.json", summary)
    return summary
