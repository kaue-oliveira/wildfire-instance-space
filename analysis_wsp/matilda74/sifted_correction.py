"""Versioned, opt-in corrections for the pinned pyInstanceSpace SIFTED.

The installed dependency is not edited. All non-SIFTED stages and GA options
remain those of revision 289a940. The context restores upstream methods.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pygad
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from instancespace.data.options import PilotOptions
from instancespace.stages.pilot import PilotStage
from instancespace.stages.sifted import SiftedStage

REVISION = "sifted-corrected-v1"
UPSTREAM_SHA256 = "4ba3a3cf49f46946600ca2acf54a911b9cc359e38d28f5b00f66d1cf8b3d2564"
_TRACE: dict | None = None


def identity() -> dict:
    source = Path(inspect.getfile(SiftedStage))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != UPSTREAM_SHA256:
        raise RuntimeError("SIFTED upstream differs from the audited pinned source")
    return {
        "revision": REVISION,
        "upstream_commit": "289a940a79cee2001bd8c54621465ddc3e30a094",
        "upstream_sifted_sha256": source_hash,
        "local_correction_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fitness": "negative maximum cross-validated KNN classification MSE across algorithms",
        "direction": "maximize negative loss, equivalent to minimizing worst algorithm loss",
        "correlation_filter": "best feature per algorithm plus any significant absolute correlation >= rho",
        "candidate_mapping": "cluster indices address filtered x and filtered feature names",
        "reference": "https://github.com/andremun/InstanceSpace/blob/master/core/SIFTED.m",
        "scope": "local correction of three confirmed defects, not an upgrade to current upstream",
    }


def select_features_by_performance(self):
    rho, pval = self._compute_correlation(self.x, self.y)
    valid = np.isfinite(rho) & np.isfinite(pval) & (pval <= self.PVAL_THRESHOLD)
    strength = np.where(valid, np.abs(rho), 0.0)
    order = np.argsort(-strength, axis=0)
    selected = np.any(valid & (strength >= self.opts.rho), axis=1)
    selected[np.unique(order[0, :])] = True
    indices = np.flatnonzero(selected)
    if _TRACE is not None:
        _TRACE["performance_filter"] = {
            "input_features": int(self.x.shape[1]),
            "retained_features": len(indices),
            "retained_original_indices": indices.tolist(),
            "retained_labels": np.asarray(self.feat_labels)[indices].tolist(),
            "rho_threshold": float(self.opts.rho),
            "pval_threshold": float(self.PVAL_THRESHOLD),
        }
    return self.x[:, indices], rho, pval, indices


def decode_candidate(clusters, solution):
    mask = np.zeros(clusters.shape[0], dtype=bool)
    for i, value in enumerate(solution):
        members = np.flatnonzero(clusters[:, i])
        if not len(members):
            raise ValueError("Empty SIFTED feature cluster")
        mask[members[int(value) % len(members)]] = True
    return mask


def cost_fcn(instance, solutions, _solution_idx):
    if instance.selfx.shape[1] != instance.clust.shape[0]:
        raise ValueError("SIFTED candidate matrix and cluster indices are misaligned")
    mask = decode_candidate(instance.clust, solutions)
    names = np.asarray(instance.selffeat_labels)
    if len(names) != len(mask):
        raise ValueError("SIFTED candidate labels and columns are misaligned")
    out = PilotStage.pilot(instance.selfx[:, mask], instance.selfy,
                           names[mask].tolist(), PilotOptions.default(), False)
    worst_error = -np.inf
    for i in range(instance.selfy.shape[1]):
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=SiftedStage.K_NEIGHBORS),
            out.z, instance.selfy_bin[:, i].astype(int),
            cv=instance.cv_partition, scoring="neg_mean_squared_error",
            error_score="raise",
        )
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite SIFTED classification loss")
        worst_error = max(worst_error, float(-scores.mean()))
    return -worst_error


def find_best_combination(self, x_aux, clust, selvars, rng):
    selvars = np.asarray(selvars, dtype=int)
    if not np.array_equal(x_aux, self.x[:, selvars]):
        raise ValueError("Filtered matrix differs from its original column mapping")
    cv_partition = KFold(n_splits=SiftedStage.KFOLDS, shuffle=True,
                         random_state=int(rng.integers(1000)))
    ga = pygad.GA(
        fitness_func=cost_fcn,
        num_generations=self.opts.num_generations,
        num_parents_mating=self.opts.num_parents_mating,
        sol_per_pop=self.opts.sol_per_pop, gene_type=int, num_genes=self.opts.k,
        parent_selection_type=self.opts.parent_selection_type,
        K_tournament=self.opts.k_tournament, keep_elitism=self.opts.keep_elitism,
        crossover_type=self.opts.crossover_type,
        crossover_probability=self.opts.cross_over_probability,
        mutation_type=self.opts.mutation_type,
        mutation_probability=self.opts.mutation_probability,
        stop_criteria=self.opts.stop_criteria, random_seed=int(rng.integers(1000)),
        init_range_low=1, init_range_high=x_aux.shape[1], save_solutions=True,
        parallel_processing=(["process", self.parallel_options.n_cores]
                             if self.parallel_options.flag else None),
    )
    ga.selfx = x_aux
    ga.selfy = self.y
    ga.selfy_bin = self.y_bin
    ga.cv_partition = cv_partition
    ga.clust = clust
    ga.selffeat_labels = np.asarray(self.feat_labels)[selvars]
    ga.run()
    best_solution, best_fitness, _ = ga.best_solution()
    decoder = decode_candidate(clust, best_solution)
    decoded = selvars[decoder]
    selected_x = self.x[:, decoded]
    if not np.array_equal(selected_x, ga.selfx[:, decoder]):
        raise AssertionError("Evaluated columns differ from returned selected columns")
    # Reevaluate the winner through the same real PILOT/KNN function, independently
    # of the GA's stored fitness. This does not execute any WSP solver.
    recomputed = float(cost_fcn(ga, best_solution, 0))
    if not np.isclose(recomputed, best_fitness, rtol=0, atol=1e-12):
        raise AssertionError(f"Stored SIFTED fitness differs: {best_fitness} vs {recomputed}")
    if _TRACE is not None:
        _TRACE["genetic_selection"] = {
            "candidate_features": int(x_aux.shape[1]),
            "best_solution": np.asarray(best_solution).astype(int).tolist(),
            "best_fitness": float(best_fitness),
            "worst_classification_error": -float(best_fitness),
            "recomputed_best_fitness": recomputed,
            "generations_completed": int(ga.generations_completed),
            "best_fitness_by_generation": [float(v) for v in ga.best_solutions_fitness],
            "selected_filtered_indices": np.flatnonzero(decoder).tolist(),
            "selected_original_indices": decoded.tolist(),
            "selected_labels": np.asarray(self.feat_labels)[decoded].tolist(),
            "evaluated_and_returned_columns_identical": True,
        }
    print(f"Corrected SIFTED fitness: {best_fitness}; worst CV error: {-best_fitness}", flush=True)
    print(f"-> Keeping {selected_x.shape[1]} out of {x_aux.shape[1]} features (clustering).")
    return selected_x, decoded


@contextmanager
def corrected_sifted(trace_path: Path):
    global _TRACE
    if _TRACE is not None:
        raise RuntimeError("SIFTED correction context is already active")
    _TRACE = {"implementation": identity(), "status": "running"}
    try:
        with patch.object(SiftedStage, "cost_fcn", staticmethod(cost_fcn)), \
             patch.object(SiftedStage, "select_features_by_performance", select_features_by_performance), \
             patch.object(SiftedStage, "_find_best_combination", find_best_combination):
            yield _TRACE
        _TRACE["status"] = "completed"
    except Exception as exc:
        _TRACE["status"] = "failed"
        _TRACE["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace_path.write_text(json.dumps(_TRACE, ensure_ascii=False, indent=2)+"\n")
        _TRACE = None
