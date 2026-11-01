"""Audit four completed MATILDA fits and compare projections on shared cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

from analysis_wsp.extended_analysis.data import ALGORITHMS, numeric_feature_columns
from analysis_wsp.extended_analysis.isa import compare_embeddings, instance_space_input_hash
from analysis_wsp.extended_analysis.provenance import write_json
from analysis_wsp.matilda74.prepare import OUTPUT


MODELS = {
    "all74_control": (OUTPUT, "control", 74),
    "all74_topology": (OUTPUT, "topology", 74),
    "original55_control": (OUTPUT / "comparison_55_v24", "control", 55),
    "original55_topology": (OUTPUT / "comparison_55_v24", "topology", 55),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model(name: str, scope: Path, variant: str, n: int) -> dict:
    output = scope / variant
    inputs = scope / "inputs"
    suffix = "74" if n == 74 else "55"
    feature_path = inputs / f"{'original' if variant == 'control' else 'topology'}_{suffix}_features_for_isa.csv"
    performance_path = inputs / f"performance_{suffix}x6.csv"
    features = pd.read_csv(feature_path, float_precision="round_trip")
    performance = pd.read_csv(performance_path, float_precision="round_trip")
    status = load_json(output / "execution_status.json")
    run = load_json(output / "run_status.json")
    selected = load_json(output / "sifted_selected_features.json")["features"]
    coords = pd.read_csv(output / "pilot_coordinates.csv", float_precision="round_trip")
    predictions = pd.read_csv(output / "pythia_predictions.csv", float_precision="round_trip")
    footprint = pd.read_csv(output / "csv" / "footprint_performance.csv", float_precision="round_trip")
    exported_pythia = pd.read_csv(output / "csv" / "svm_table.csv", float_precision="round_trip")
    if status["status"] != run["status"] or status["status"] != "completed":
        raise RuntimeError(f"Modelo não concluído: {name}")
    if len(features) != n or set(features.instance) != set(performance.instance):
        raise RuntimeError(f"Entradas incompletas: {name}")
    if list(performance.columns) != ["instance", *ALGORITHMS]:
        raise RuntimeError(f"Algoritmos alterados: {name}")
    if status["input_feature_sha256"] != sha256(feature_path) or status["input_performance_sha256"] != sha256(performance_path):
        raise RuntimeError(f"Arquivo de entrada alterado após execução: {name}")
    if run["analysis_input_hash"] != instance_space_input_hash(
        features.sort_values("instance", kind="mergesort").reset_index(drop=True),
        performance.sort_values("instance", kind="mergesort").reset_index(drop=True), 0.10, 4,
    ):
        raise RuntimeError(f"Hash da matriz em memória divergente: {name}")
    if len(selected) != 10 or len(coords) != n or len(predictions) != n * 6 or len(footprint) != 6:
        raise RuntimeError(f"Artefatos MATILDA incompletos: {name}")
    if coords.instance.nunique() != n or not np.isfinite(coords[["z1", "z2"]].to_numpy()).all():
        raise RuntimeError(f"Mapa não finito ou duplicado: {name}")
    if predictions.duplicated(["instance", "algorithm"]).any() or set(predictions.instance) != set(features.instance):
        raise RuntimeError(f"Predições incompletas ou duplicadas: {name}")
    if not np.isfinite(performance[list(ALGORITHMS)].to_numpy()).all():
        raise RuntimeError(f"Desempenho não finito: {name}")

    # These are training-set diagnostics, not held-out predictive performance.
    in_sample = {}
    label_counts = {}
    export_audit = []
    for algorithm in ALGORITHMS:
        sub = predictions[predictions.algorithm == f"algo_{algorithm.lower()}"].set_index("instance")
        truth = (performance.set_index("instance")[algorithm] <= 0.10).astype(bool)
        pred = sub.loc[truth.index, "predicted_good"].astype(bool)
        label_counts[algorithm] = {"good": int(truth.sum()), "not_good": int((~truth).sum())}
        in_sample[algorithm] = {
            "accuracy": float((truth == pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "precision": float(precision_score(truth, pred, zero_division=0)),
            "recall": float(recall_score(truth, pred, zero_division=0)),
            "errors": int((truth != pred).sum()),
        }
        exported = exported_pythia[exported_pythia.Row == f"algo_{algorithm.lower()}"]
        if len(exported) != 1:
            raise RuntimeError(f"Resumo PYTHIA ausente: {name}/{algorithm}")
        row = exported.iloc[0]
        # The pinned exporter swaps the names of accuracy and precision.
        error_precision = abs(float(row.CV_model_accuracy) - 100 * in_sample[algorithm]["precision"])
        error_accuracy = abs(float(row.CV_model_precision) - 100 * in_sample[algorithm]["accuracy"])
        error_recall = abs(float(row.CV_model_recall) - 100 * in_sample[algorithm]["recall"])
        if max(error_precision, error_accuracy, error_recall) > 0.001:
            raise RuntimeError(f"Rótulos PYTHIA não seguem a troca conhecida: {name}/{algorithm}")
        export_audit.append({"algorithm": algorithm,
                             "export_named_accuracy_is_precision": True,
                             "export_named_precision_is_accuracy": True,
                             "export_accuracy_field": float(row.CV_model_accuracy),
                             "recomputed_precision_pct": 100 * in_sample[algorithm]["precision"],
                             "export_precision_field": float(row.CV_model_precision),
                             "recomputed_accuracy_pct": 100 * in_sample[algorithm]["accuracy"],
                             "max_rounding_error_pct_points": max(error_precision, error_accuracy, error_recall)})
    pd.DataFrame(export_audit).to_csv(output / "pythia_export_label_audit.csv", index=False)
    result = {
        "status": "passed",
        "name": name,
        "instances": n,
        "algorithms": 6,
        "epsilon": 0.10,
        "cv_folds": 4,
        "input_features": len(numeric_feature_columns(features)),
        "selected_features": selected,
        "selected_topology_features": [x for x in selected if x.startswith("feature_topo_")],
        "matilda_build_seconds": run["elapsed_seconds"],
        "full_model_seconds": status["elapsed_seconds_total"],
        "analysis_input_hash": run["analysis_input_hash"],
        "label_counts": label_counts,
        "in_sample_diagnostic_only": in_sample,
        "pythia_repeated_cv_is_not_nested": True,
        "footprint_areas_between_separate_maps_comparable": False,
    }
    write_json(output / "scientific_audit.json", result)
    return result


def main() -> None:
    summaries = {name: summarize_model(name, *spec) for name, spec in MODELS.items()}
    comparison_dir = OUTPUT / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    paired = {
        "55_control_vs_74_control": (
            OUTPUT / "comparison_55_v24" / "control", OUTPUT / "control"
        ),
        "55_topology_vs_74_topology": (
            OUTPUT / "comparison_55_v24" / "topology", OUTPUT / "topology"
        ),
        "74_control_vs_74_topology": (OUTPUT / "control", OUTPUT / "topology"),
        "55_control_vs_55_topology": (
            OUTPUT / "comparison_55_v24" / "control", OUTPUT / "comparison_55_v24" / "topology"
        ),
    }
    comparisons = {}
    for label, (reference, candidate) in paired.items():
        ref_coords = pd.read_csv(reference / "pilot_coordinates.csv", float_precision="round_trip")
        cand_coords = pd.read_csv(candidate / "pilot_coordinates.csv", float_precision="round_trip")
        geometry = compare_embeddings(ref_coords, cand_coords, comparison_dir, label)
        ref_features = set(load_json(reference / "sifted_selected_features.json")["features"])
        cand_features = set(load_json(candidate / "sifted_selected_features.json")["features"])
        ref_pred = pd.read_csv(reference / "pythia_predictions.csv")
        cand_pred = pd.read_csv(candidate / "pythia_predictions.csv")
        joined = ref_pred.merge(cand_pred, on=["instance", "algorithm"], suffixes=("_ref", "_candidate"), validate="one_to_one")
        if len(joined) != geometry["n_instances"] * 6:
            raise RuntimeError(f"Predições não alinham na comparação {label}")
        comparisons[label] = {
            **geometry,
            "selected_feature_intersection": sorted(ref_features & cand_features),
            "selected_feature_intersection_count": len(ref_features & cand_features),
            "selected_feature_jaccard": len(ref_features & cand_features) / len(ref_features | cand_features),
            "pythia_in_sample_prediction_agreement_on_shared_cases": float(
                (joined.predicted_good_ref == joined.predicted_good_candidate).mean()
            ),
            "prediction_cases_compared": len(joined),
        }
    write_json(comparison_dir / "scientific_comparison.json", {
        "status": "passed",
        "models": summaries,
        "comparisons": comparisons,
        "interpretation_limits": [
            "The 55-v2.4 reference and 74-v2.4 fit share the same benchmark protocol.",
            "The historical 55 fit used a different performance campaign and is contextual.",
            "Projection axes are refitted; geometry is compared only after common-instance alignment.",
            "PYTHIA agreement and in-sample diagnostics are not external predictive validation.",
            "The 19 GA cases were selected from the historical map, not sampled independently.",
        ],
    })
    print(json.dumps({"status": "passed", "models": list(summaries),
                      "comparisons": {k: {"shared_instances": v["n_instances"],
                                           "neighbor_overlap": v["mean_neighbor_overlap"],
                                           "feature_overlap": v["selected_feature_intersection_count"]}
                                      for k, v in comparisons.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
