"""Prepare a measured 74 × 6 MATILDA input without rerunning WSP solvers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_wsp.extended_analysis.data import ALGORITHMS, base_features, instance_content_hash, numeric_feature_columns
from analysis_wsp.extended_analysis.isa import audit_feature_table, label_audit, threshold_fold
from analysis_wsp.extended_analysis.provenance import write_json


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis_wsp"
INTEGRATED = ROOT / "results" / "benchmark"
HISTORICAL = ROOT / "results" / "generation"
AUDITED = ROOT / "results" / "audit" / "mip"
OUTPUT = ROOT / "experiments" / "isa_preparation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = OUTPUT / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = INTEGRATED / "shards" / "lab-01" / "instances_74.csv"
    historical_features_path = ANALYSIS / "data" / "features_instances.csv"
    performance_path = AUDITED / "per_instance_algorithm_auditado.csv"
    historical = pd.read_csv(historical_features_path).set_index("instance")
    manifest = pd.read_csv(manifest_path)
    measured = pd.read_csv(performance_path)
    assert len(manifest) == 74 and manifest["instance"].nunique() == 74
    assert manifest["collection"].value_counts().to_dict() == {"original_55": 55, "supplementary_19": 19}
    assert len(measured) == 444 and not measured.duplicated(["instance", "algorithm"]).any()
    assert set(measured["instance"]) == set(manifest["instance"])
    assert set(measured["algorithm"].str.upper()) == set(ALGORITHMS)
    assert measured["runs"].sum() == 3936 and measured["failures"].sum() == 2

    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    baseline_max_error = 0.0
    baseline_worst: tuple[str, str] | None = None
    baseline_numeric = numeric_feature_columns(historical.reset_index())
    for item in manifest.sort_values("instance", kind="mergesort").itertuples(index=False):
        name = str(item.instance)
        path = ROOT / "data" / "instances" / "native" / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        semantic_hash = instance_content_hash(data)
        if semantic_hash != item.instance_sha256:
            raise RuntimeError(f"Hash semântico divergente: {name}")
        if item.collection == "original_55":
            if name not in historical.index:
                raise RuntimeError(f"Instância histórica ausente: {name}")
            old = historical.loc[name]
            group, experiment_group, source_zip = (str(old[field]) for field in ("group", "experiment_group", "source_zip"))
        else:
            group, experiment_group, source_zip = "generated_ga", "supplementary_19", "none"
        record = {
            "instance": name,
            "group": group,
            "experiment_group": experiment_group,
            "source_zip": source_zip,
            "data": data,
        }
        row = base_features(record)
        if item.collection == "original_55":
            for field in baseline_numeric:
                actual = float(row[field])
                expected = float(old[field])
                error = abs(actual - expected)
                if error > baseline_max_error:
                    baseline_max_error, baseline_worst = error, (name, field)
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
                    raise RuntimeError(f"A feature histórica mudou: {name}/{field}: {actual} != {expected}")
        rows.append(row)
        source_rows.append({
            "instance": name,
            "collection": item.collection,
            "native_path": str(path.relative_to(ROOT)),
            "native_sha256_bytes": sha256(path),
            "instance_sha256_semantic": semantic_hash,
            "n_vertices": int(data["|V|"]),
        })

    original = pd.DataFrame(rows)
    original["result_provenance"] = "measured"
    assert len(original) == 74 and original["instance"].nunique() == 74
    assert len(numeric_feature_columns(original)) == len(baseline_numeric) == 36
    original.to_csv(output / "base_features_74.csv", index=False)
    prepared, original_audit = audit_feature_table(original, output, prefix="original_74")

    measured = measured.assign(algorithm=measured["algorithm"].str.upper())
    measured["mean_gap_fraction"] = measured["gap_mean_pct"].astype(float) / 100.0
    if not np.isfinite(measured["mean_gap_fraction"]).all() or (measured["mean_gap_fraction"] < -1e-9).any():
        raise RuntimeError("Gap médio não finito ou negativo")
    performance = measured.pivot(index="instance", columns="algorithm", values="mean_gap_fraction")
    performance = performance[list(ALGORITHMS)].reset_index().sort_values("instance", kind="mergesort")
    if len(performance) != 74 or performance[list(ALGORITHMS)].isna().any().any():
        raise RuntimeError("A matriz 74 × 6 contém linhas ou valores ausentes")
    performance.to_csv(output / "performance_74x6.csv", index=False)
    measured.sort_values(["instance", "algorithm"], kind="mergesort").to_csv(output / "performance_source_444.csv", index=False)
    labels = label_audit(performance, [0.10])
    labels.to_csv(output / "label_counts_10pct.csv", index=False)
    folds = threshold_fold(labels, 0.10)
    if folds is None:
        raise RuntimeError("Pelo menos um algoritmo não tem folds estimáveis com o limiar de 10%")
    pd.DataFrame(source_rows).to_csv(output / "native_source_manifest.csv", index=False)
    write_json(output / "preparation_status.json", {
        "status": "completed",
        "instances": 74,
        "original_instances": 55,
        "supplementary_instances": 19,
        "algorithms": list(ALGORITHMS),
        "performance_cells": 444,
        "benchmark_runs": 3936,
        "ils_failures_explicit": 2,
        "gap_unit": "fraction of audited BKV, not percent",
        "good_performance_threshold": 0.10,
        "cv_folds": folds,
        "base_feature_candidates": len(baseline_numeric),
        "base_features_after_audit": len(numeric_feature_columns(prepared)),
        "baseline_55_max_absolute_feature_error": baseline_max_error,
        "baseline_55_worst": baseline_worst,
        "original_feature_audit_decisions": original_audit["decision"].value_counts().to_dict(),
        "source_file_sha256": {
            str(manifest_path.relative_to(ROOT)): sha256(manifest_path),
            str(historical_features_path.relative_to(ROOT)): sha256(historical_features_path),
            str(performance_path.relative_to(ROOT)): sha256(performance_path),
        },
        "solvers_rerun": False,
        "ga_rerun": False,
    })
    print(json.dumps({"status": "completed", "instances": 74, "performance_cells": 444,
                      "base_features": len(numeric_feature_columns(prepared)), "cv_folds": folds}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
