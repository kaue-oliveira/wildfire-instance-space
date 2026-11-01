"""Compute exact topology for 19 new instances and assemble 74 ISA rows."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_wsp.extended_analysis.isa import audit_feature_table
from analysis_wsp.extended_analysis.provenance import write_json
from analysis_wsp.extended_analysis.topology import (
    TOPOLOGY_BACKEND_ID,
    audit_and_prepare_topology,
    compute_topology,
    graph_fingerprint,
)
from analysis_wsp.matilda74.prepare import HISTORICAL, INTEGRATED, OUTPUT, ROOT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_one(data: dict[str, Any]) -> dict[str, Any]:
    result = compute_topology(data, exact_betweenness=True)
    result.diagnostics["approximation_validation_status"] = "not_applicable_exact"
    return {"features": result.features, "diagnostics": result.diagnostics}


def append_progress(path: Path, *, instance: str, status: str, **fields: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                 "instance": instance, "status": status, **fields}, ensure_ascii=False) + "\n")


def main() -> None:
    started = time.perf_counter()
    top_output = OUTPUT / "topology_input"
    cache = top_output / "exact_cache"
    cache.mkdir(parents=True, exist_ok=True)
    progress = top_output / "progress.jsonl"
    manifest = pd.read_csv(INTEGRATED / "shards" / "lab-01" / "instances_74.csv")
    old_dir = HISTORICAL / "02_instancespace_topology"
    old_candidates_path = old_dir / "topology_candidates.csv"
    old_diagnostics_path = old_dir / "topology_computation_diagnostics.csv"
    old_candidates = pd.read_csv(old_candidates_path, float_precision="round_trip")
    old_diagnostics = pd.read_csv(old_diagnostics_path, float_precision="round_trip")
    assert len(old_candidates) == len(old_diagnostics) == 55
    old_by_name = old_candidates.set_index("instance")
    old_features = {column for column in old_candidates if column.startswith("topo_")}
    assert len(old_features) == 136

    pending: dict[Any, tuple[str, str, Path]] = {}
    new_data: dict[str, tuple[dict[str, Any], str, Path]] = {}
    for item in manifest.sort_values("instance", kind="mergesort").itertuples(index=False):
        name = str(item.instance)
        path = ROOT / "data" / "instances" / "native" / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = graph_fingerprint(data)
        if item.collection == "original_55":
            if name not in old_by_name.index or fingerprint != old_by_name.loc[name, "graph_sha256"]:
                raise RuntimeError(f"Topologia histórica não corresponde à entrada do Benchmark74: {name}")
        else:
            new_data[name] = (data, fingerprint, cache / f"{fingerprint}_exact.json")
    assert len(new_data) == 19
    print("Topologia: 55 grafos históricos conferidos; calculando 19 grafos novos exatamente", flush=True)

    # One worker keeps peak memory bounded on the 8 GB host while the nine
    # 6400-node graphs are processed; each completed graph is cached.
    with ProcessPoolExecutor(max_workers=1) as executor:
        for name, (data, fingerprint, path) in new_data.items():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict) and cached.get("topology_backend_id") == TOPOLOGY_BACKEND_ID:
                append_progress(progress, instance=name, status="cache_reused", graph_sha256=fingerprint)
                continue
            pending[executor.submit(compute_one, data)] = (name, fingerprint, path)
            append_progress(progress, instance=name, status="scheduled", graph_sha256=fingerprint,
                            n_vertices=int(data["|V|"]))
        for future in as_completed(pending):
            name, fingerprint, path = pending[future]
            payload = future.result()
            if set(payload["features"]) != old_features:
                raise RuntimeError(f"As 136 definições topológicas mudaram: {name}")
            write_json(path, {"topology_backend_id": TOPOLOGY_BACKEND_ID,
                              "graph_sha256": fingerprint, **payload})
            append_progress(progress, instance=name, status="completed", graph_sha256=fingerprint,
                            elapsed_seconds=payload["diagnostics"]["elapsed_seconds"])
            print(f"Topologia exata {name}: {payload['diagnostics']['elapsed_seconds']:.1f} s", flush=True)

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for name, (_, fingerprint, path) in sorted(new_data.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["topology_backend_id"] != TOPOLOGY_BACKEND_ID or payload["graph_sha256"] != fingerprint:
            raise RuntimeError(f"Cache topológico inválido: {name}")
        if set(payload["features"]) != old_features:
            raise RuntimeError(f"Features topológicas incompletas: {name}")
        rows.append({"instance": name, "group": "generated_ga", "experiment_group": "supplementary_19",
                     "source_zip": "none", "graph_sha256": fingerprint, "result_provenance": "measured",
                     **payload["features"]})
        diagnostics.append({"instance": name, "graph_sha256": fingerprint, "cache_hit": False,
                            **payload["diagnostics"]})
    combined = pd.concat([old_candidates, pd.DataFrame(rows)], ignore_index=True).sort_values("instance", kind="mergesort")
    combined_diagnostics = pd.concat([old_diagnostics, pd.DataFrame(diagnostics)], ignore_index=True).sort_values("instance", kind="mergesort")
    if len(combined) != 74 or combined["instance"].nunique() != 74:
        raise RuntimeError("Tabela topológica não tem 74 instâncias únicas")
    combined.to_csv(top_output / "topology_candidates_74.csv", index=False)
    combined_diagnostics.to_csv(top_output / "topology_diagnostics_74.csv", index=False)
    selected, audit, _ = audit_and_prepare_topology(combined, combined_diagnostics, top_output)
    topocols = [column for column in selected if column.startswith("topo_")]
    base = pd.read_csv(OUTPUT / "inputs" / "base_features_74.csv", float_precision="round_trip")
    augmented = base.merge(selected[["instance", *topocols]], on="instance", validate="one_to_one")
    if len(augmented) != 74:
        raise RuntimeError("Junção das características originais e topológicas falhou")
    augmented.to_csv(OUTPUT / "inputs" / "augmented_candidates_74.csv", index=False)
    prepared, final_audit = audit_feature_table(augmented, OUTPUT / "inputs", prefix="topology_74")
    write_json(top_output / "preparation_status.json", {
        "status": "completed",
        "old_graphs_reused_and_verified": 55,
        "new_graphs_exact": 19,
        "new_size_classes": {"Huge": 9, "Large": 6, "Medium": 3, "Small": 1},
        "topology_backend_id": TOPOLOGY_BACKEND_ID,
        "topology_candidate_features": len(old_features),
        "topology_features_after_audit": len(topocols),
        "combined_features_after_audit": len([x for x in prepared if x.startswith("topo_") or x in base.select_dtypes("number").columns]),
        "topology_audit_decisions": audit["decision"].value_counts().to_dict(),
        "combined_audit_decisions": final_audit["decision"].value_counts().to_dict(),
        "historical_table_sha256": sha256(old_candidates_path),
        "historical_diagnostics_sha256": sha256(old_diagnostics_path),
        "elapsed_seconds": time.perf_counter() - started,
        "approximation_used_for_new_graphs": False,
    })
    print(json.dumps({"status": "completed", "topology_features": len(topocols),
                      "augmented_features": len([x for x in prepared if x.startswith("topo_") or x in base.select_dtypes("number").columns])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
