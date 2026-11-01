"""Run one auditable MATILDA74 model from the prepared, measured matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis_wsp.extended_analysis.data import ALGORITHMS, numeric_feature_columns
from analysis_wsp.extended_analysis.isa import instance_space_input_hash, run_instance_space
from analysis_wsp.extended_analysis.provenance import write_json
from analysis_wsp.matilda74.prepare import OUTPUT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("control", "topology"))
    parser.add_argument("--scope", choices=("all74", "original55_v24"), default="all74")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2] / "experiments" / "isa_corrected")
    parser.add_argument("--sifted-revision", choices=("upstream", "sifted-corrected-v1"), default="sifted-corrected-v1")
    args = parser.parse_args()
    if args.sifted_revision != "upstream" and args.output_root.resolve() == OUTPUT.resolve():
        raise RuntimeError("Corrected fits must use a new output root, preserving the v1 archive")
    scope_root = args.output_root if args.scope == "all74" else args.output_root / "comparison_55_v24"
    expected_instances = 74 if args.scope == "all74" else 55
    output = scope_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    inputs = scope_root / "inputs"
    preparation = json.loads((inputs / "preparation_status.json").read_text(encoding="utf-8"))
    if preparation.get("status") != "completed":
        raise RuntimeError("Preparação de entrada não foi concluída")
    suffix = "74" if args.scope == "all74" else "55"
    feature_path = inputs / (f"original_{suffix}_features_for_isa.csv" if args.model == "control" else f"topology_{suffix}_features_for_isa.csv")
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    performance_path = inputs / f"performance_{suffix}x6.csv"
    labels_path = inputs / "label_counts_10pct.csv"
    features = pd.read_csv(feature_path, float_precision="round_trip").sort_values("instance", kind="mergesort").reset_index(drop=True)
    performance = pd.read_csv(performance_path, float_precision="round_trip").sort_values("instance", kind="mergesort").reset_index(drop=True)
    labels = pd.read_csv(labels_path)
    if len(features) != expected_instances or len(performance) != expected_instances or set(features["instance"]) != set(performance["instance"]):
        raise RuntimeError("As linhas de características e desempenho não coincidem")
    if list(performance.columns) != ["instance", *ALGORITHMS]:
        raise RuntimeError("A ordem dos algoritmos mudou")
    folds = int(preparation["cv_folds"])
    data_hash = instance_space_input_hash(features, performance, 0.10, folds)
    started = datetime.now(timezone.utc).isoformat()
    status_path = output / "execution_status.json"
    write_json(status_path, {
        "status": "running",
        "model": args.model,
        "scope": args.scope,
        "started_utc": started,
        "input_feature_sha256": sha256(feature_path),
        "input_performance_sha256": sha256(performance_path),
        "input_label_sha256": sha256(labels_path),
        "analysis_input_hash": data_hash,
        "feature_count": len(numeric_feature_columns(features)),
        "instances": expected_instances,
        "algorithms": list(ALGORITHMS),
        "epsilon": 0.10,
        "folds": folds,
        "workers": args.workers,
        "wsp_solvers_rerun": False,
        "ga_rerun": False,
        "instance_generation_ga_rerun": False,
        "sifted_revision": args.sifted_revision,
    })
    print(f"MATILDA {args.scope} {args.model} started {started}; features={len(numeric_feature_columns(features))}; folds={folds}", flush=True)
    clock = time.perf_counter()
    try:
        result = run_instance_space(
            features, performance, output, epsilon=0.10, folds=folds, workers=args.workers,
            label_counts=labels, input_data_hash=data_hash,
            performance_provenance="audited_benchmark74", feature_provenance="measured",
            sifted_revision=args.sifted_revision,
        )
    except Exception as exc:
        current = json.loads(status_path.read_text(encoding="utf-8"))
        current.update({"status": "failed", "ended_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.perf_counter() - clock, "error": f"{type(exc).__name__}: {exc}"})
        write_json(status_path, current)
        raise
    current = json.loads(status_path.read_text(encoding="utf-8"))
    current.update({"status": "completed", "ended_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds_total": time.perf_counter() - clock,
                    "elapsed_seconds_matilda_build": result["status"]["elapsed_seconds"],
                    "selected_features": result["status"]["n_features_sifted"]})
    write_json(status_path, current)
    print(json.dumps({"status": "completed", "model": args.model,
                      "elapsed_seconds": current["elapsed_seconds_total"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
