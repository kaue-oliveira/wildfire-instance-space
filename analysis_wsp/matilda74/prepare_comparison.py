"""Prepare a 55-instance v2.4 control to isolate the addition of 19 cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis_wsp.extended_analysis.data import ALGORITHMS, numeric_feature_columns
from analysis_wsp.extended_analysis.isa import audit_feature_table, label_audit, threshold_fold
from analysis_wsp.extended_analysis.provenance import write_json
from analysis_wsp.extended_analysis.topology import audit_and_prepare_topology
from analysis_wsp.matilda74.prepare import HISTORICAL, OUTPUT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    full_inputs = OUTPUT / "inputs"
    output = OUTPUT / "comparison_55_v24" / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = pd.read_csv(full_inputs / "native_source_manifest.csv")
    names = set(source_manifest.loc[source_manifest["collection"] == "original_55", "instance"])
    if len(names) != 55:
        raise RuntimeError("A coleção original não contém 55 instâncias")
    base = pd.read_csv(full_inputs / "base_features_74.csv", float_precision="round_trip")
    base = base[base["instance"].isin(names)].sort_values("instance", kind="mergesort")
    if len(base) != 55:
        raise RuntimeError("Características de 55 instâncias incompletas")
    base.to_csv(output / "base_features_55.csv", index=False)
    original, original_audit = audit_feature_table(base, output, prefix="original_55")

    performance = pd.read_csv(full_inputs / "performance_74x6.csv", float_precision="round_trip")
    performance = performance[performance["instance"].isin(names)].sort_values("instance", kind="mergesort")
    if len(performance) != 55 or list(performance.columns) != ["instance", *ALGORITHMS]:
        raise RuntimeError("Desempenho v2.4 das 55 instâncias incompleto")
    performance.to_csv(output / "performance_55x6.csv", index=False)
    labels = label_audit(performance, [0.10])
    labels.to_csv(output / "label_counts_10pct.csv", index=False)
    folds = threshold_fold(labels, 0.10)
    if folds is None:
        raise RuntimeError("Nenhum número comum de folds estimável no controle 55-v2.4")

    old_dir = HISTORICAL / "02_instancespace_topology"
    candidates_path = old_dir / "topology_candidates.csv"
    diagnostics_path = old_dir / "topology_computation_diagnostics.csv"
    candidates = pd.read_csv(candidates_path, float_precision="round_trip")
    diagnostics = pd.read_csv(diagnostics_path, float_precision="round_trip")
    if len(candidates) != 55 or set(candidates["instance"]) != names or set(diagnostics["instance"]) != names:
        raise RuntimeError("Topologia histórica não corresponde ao controle 55-v2.4")
    selected, topology_audit, _ = audit_and_prepare_topology(candidates, diagnostics, output)
    topocols = [column for column in selected if column.startswith("topo_")]
    augmented = base.merge(selected[["instance", *topocols]], on="instance", validate="one_to_one")
    augmented.to_csv(output / "augmented_candidates_55.csv", index=False)
    augmented_prepared, combined_audit = audit_feature_table(augmented, output, prefix="topology_55")
    write_json(output / "preparation_status.json", {
        "status": "completed",
        "instances": 55,
        "performance_protocol": "benchmark74_v2_4_integrated_final",
        "historical_matilda_performance_reused": False,
        "good_performance_threshold": 0.10,
        "cv_folds": folds,
        "algorithms": list(ALGORITHMS),
        "base_features_after_audit": len(numeric_feature_columns(original)),
        "topology_candidates": len([column for column in candidates if column.startswith("topo_")]),
        "topology_features_after_audit": len(topocols),
        "combined_features_after_audit": len(numeric_feature_columns(augmented_prepared)),
        "original_audit_decisions": original_audit["decision"].value_counts().to_dict(),
        "topology_audit_decisions": topology_audit["decision"].value_counts().to_dict(),
        "combined_audit_decisions": combined_audit["decision"].value_counts().to_dict(),
        "source_sha256": {
            "base_features_74.csv": sha256(full_inputs / "base_features_74.csv"),
            "performance_74x6.csv": sha256(full_inputs / "performance_74x6.csv"),
            "historical_topology_candidates.csv": sha256(candidates_path),
            "historical_topology_diagnostics.csv": sha256(diagnostics_path),
        },
        "solvers_rerun": False,
    })
    print(json.dumps({"status": "completed", "scope": "original55_v24", "instances": 55,
                      "folds": folds, "control_features": len(numeric_feature_columns(original)),
                      "topology_features": len(numeric_feature_columns(augmented_prepared))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
