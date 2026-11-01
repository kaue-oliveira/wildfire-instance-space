from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .common import (
    BASE_REMOTE_REVISION,
    EXPECTED_BRANCH,
    SEEDS,
    TIME_SECONDS_PER_VERTEX,
    accepted_rows,
    atomic_write_json,
    build_protocol,
    canonical_json_bytes,
    content_sha256,
    exact_candidate_path,
    file_sha256,
    git_state,
    machine_metadata,
    native_to_harris,
    project_root,
    protocol_hash,
    raw_instance_path,
    result_root,
    validate_and_repair_native,
    write_csv,
)


def canonical_harris(data: dict[str, Any]) -> dict[str, Any]:
    import ast

    return {
        "nodes": [tuple(node) for node in data["Nodes"]],
        "arcs": {ast.literal_eval(edge): float(weight) for edge, weight in data["Arcs"].items()},
        "ignitions": [tuple(node) for node in data["Ignitions"]],
        "delay": float(data["Delay"]),
        "horizon": float(data["ArrivalTimeTarget"]),
        "resources": {float(time): int(capacity) for time, capacity in data["ResAtTime"].items()},
    }


def validate_historical_conversion() -> list[dict[str, Any]]:
    import zipfile

    root = project_root()
    archive_path = root / "third_party" / "WSP" / "instances" / "literature.zip"
    legacy_dir = root / "third_party" / "Iterated-Beam-Search" / "instances"
    checks: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for group, suffix in (("LA", "a"), ("LB", "b")):
            for index in range(8):
                name = f"{group}{index}"
                native = json.loads(archive.read(f"literature/{name}.json"))
                repaired, audit = validate_and_repair_native(native)
                converted = canonical_harris(native_to_harris(repaired))
                expected = canonical_harris(json.loads((legacy_dir / f"L{index}_{suffix}.json").read_text(encoding="utf-8")))
                fields = {
                    "nodes": converted["nodes"] == expected["nodes"],
                    "arcs": converted["arcs"] == expected["arcs"],
                    "ignitions": converted["ignitions"] == expected["ignitions"],
                    "delay": converted["delay"] == expected["delay"],
                    "horizon": converted["horizon"] == expected["horizon"],
                    "resources": converted["resources"] == expected["resources"],
                }
                checks.append({"instance": name, **fields, "all_equal": all(fields.values()), **audit})
    if len(checks) != 16 or not all(row["all_equal"] for row in checks):
        raise ValueError("Native-to-Harris conversion did not exactly reproduce all 16 historical benchmarks")
    return checks


def validate_git_base() -> dict[str, Any]:
    state = git_state()
    if state["head"] != BASE_REMOTE_REVISION:
        raise RuntimeError(f"HEAD {state['head']} differs from audited remote base {BASE_REMOTE_REVISION}")
    if state["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"Branch {state['branch']} differs from {EXPECTED_BRANCH}")
    if not state["base_revision_exists_on_remote"]:
        raise RuntimeError("The base revision is not contained in the audited remote branch")
    return state


def build_source_manifest() -> dict[str, Any]:
    files = sorted((project_root() / "analysis_wsp" / "caminho_c").glob("*.py"))
    ibs_root = project_root() / "third_party" / "Iterated-Beam-Search"
    files += sorted((ibs_root / "src").rglob("*.hpp"))
    files += sorted((ibs_root / "src").rglob("*.cpp"))
    files += [ibs_root / "CMakeLists.txt"]
    entries = [{"path": str(path.relative_to(project_root())), "sha256": file_sha256(path)} for path in files]
    return {"files": entries, "bundle_sha256": content_sha256(entries)}


def prepare(workers: int | None = None) -> dict[str, Any]:
    git_before = validate_git_base()
    output = result_root()
    runs_dir = output / "runs"
    if runs_dir.exists() and any(runs_dir.rglob("result.json")):
        raise RuntimeError("Measured run artifacts already exist; preparation refuses to overwrite them")
    output.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(workers)
    p_hash = protocol_hash(protocol)
    protocol["protocol_sha256"] = p_hash
    atomic_write_json(output / "protocol.json", protocol)
    atomic_write_json(output / "git_state_before_prepare.json", git_before)
    atomic_write_json(output / "source_manifest.json", build_source_manifest())
    atomic_write_json(output / "environment_manifest.json", machine_metadata())

    historical = validate_historical_conversion()
    write_csv(output / "historical_conversion_validation.csv", historical)
    native_dir = output / "instances" / "native"
    ibs_dir = output / "instances" / "ibs"
    native_dir.mkdir(parents=True, exist_ok=True)
    ibs_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    instance_records: list[tuple[str, int]] = []
    for row in accepted_rows():
        identifier = row["best_exact_candidate_id"]
        raw_path = raw_instance_path(row)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        observed_hash = content_sha256(raw)
        if observed_hash != row["best_exact_instance_sha256"]:
            raise ValueError(f"Scientific instance hash mismatch: {identifier}")
        repaired, audit = validate_and_repair_native(raw)
        converted = native_to_harris(repaired)
        native_path = native_dir / f"{identifier}.json"
        ibs_path = ibs_dir / f"{identifier}.json"
        atomic_write_json(native_path, repaired)
        atomic_write_json(ibs_path, converted)
        exact_payload = json.loads(exact_candidate_path(row).read_text(encoding="utf-8"))
        exact_coordinates = exact_payload.get("pilot_coordinates")
        if not isinstance(exact_coordinates, list) or len(exact_coordinates) != 2:
            raise ValueError(f"Missing fixed no-refit ISA projection for {identifier}")
        target_coordinates = [float(row["exact_z1"]), float(row["exact_z2"])]
        projection_error = math.dist(list(map(float, exact_coordinates)), target_coordinates)
        if projection_error > 1e-10:
            raise ValueError(f"No-refit ISA projection mismatch for {identifier}: {projection_error}")
        manifest.append(
            {
                "instance": identifier,
                "root_seed": int(row["root_seed"]),
                "target_id": row["target_id"],
                "region": row["region"],
                "empty_radius_original": float(row["empty_radius_original"]),
                "hit_threshold": float(row["hit_threshold"]),
                "raw_relative_path": str(raw_path.relative_to(project_root())),
                "raw_file_sha256": file_sha256(raw_path),
                "raw_content_sha256": observed_hash,
                "expected_ga_content_sha256": row["best_exact_instance_sha256"],
                "repaired_content_sha256": content_sha256(repaired),
                "ibs_content_sha256": content_sha256(converted),
                "isa_z1_no_refit": float(exact_coordinates[0]),
                "isa_z2_no_refit": float(exact_coordinates[1]),
                "isa_projection_validation_error": projection_error,
                **audit,
                "result_provenance": "measured_and_audited",
            }
        )
        instance_records.append((identifier, audit["n_vertices"]))
    if len(manifest) != 19 or len({row["raw_content_sha256"] for row in manifest}) != 19:
        raise RuntimeError("The supplementary set is not 19 content-distinct instances")
    write_csv(output / "instance_manifest.csv", manifest)

    jobs: list[dict[str, Any]] = []
    number = 0
    for seed in SEEDS:
        for identifier, n_vertices in sorted(instance_records, key=lambda item: (item[1], item[0])):
            number += 1
            jobs.append(
                {
                    "job_id": f"run_{number:03d}",
                    "instance": identifier,
                    "seed": seed,
                    "n_vertices": n_vertices,
                    "size_class": {400: "Small", 900: "Medium", 1600: "Large", 6400: "Huge"}.get(n_vertices, "Other"),
                    "time_limit_seconds": int(round(TIME_SECONDS_PER_VERTEX * n_vertices)),
                    "ibs_instance": str((ibs_dir / f"{identifier}.json").resolve()),
                    "protocol_sha256": p_hash,
                }
            )
    if len(jobs) != 190 or len({(row["instance"], row["seed"]) for row in jobs}) != 190:
        raise RuntimeError("Experimental plan must contain exactly 190 unique instance-seed runs")
    write_csv(output / "jobs.csv", jobs)
    nominal_seconds = sum(int(row["time_limit_seconds"]) for row in jobs)
    expected_nominal = 1_054_500
    if nominal_seconds != expected_nominal:
        raise RuntimeError(f"Published nominal budget mismatch: {nominal_seconds} != {expected_nominal}")
    summary = {
        "status": "prepared",
        "instances_validated": 19,
        "distinct_instance_hashes": 19,
        "historical_conversion_cases": len(historical),
        "historical_conversion_exact": all(row["all_equal"] for row in historical),
        "isa_projection_no_refit_validated": True,
        "isa_projection_max_error": max(row["isa_projection_validation_error"] for row in manifest),
        "runs": len(jobs),
        "workers": protocol["execution_workers"],
        "nominal_cpu_seconds": nominal_seconds,
        "nominal_parallel_lower_bound_seconds": nominal_seconds / protocol["execution_workers"],
        "protocol_sha256": p_hash,
        "note": "The branch is intentionally local and orchestration source is uncommitted per user instruction; source_manifest.json content-addresses the execution code.",
    }
    atomic_write_json(output / "preparation_summary.json", summary)
    atomic_write_json(
        output / "execution_status.json",
        {"status": "prepared", "total": 190, "completed": 0, "failed": 0, "running": 0, "pending": 190, "workers": protocol["execution_workers"]},
    )
    return summary
