from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis_wsp.caminho_c.prepare import validate_historical_conversion

from .common import (
    ALGORITHMS,
    SEEDS,
    atomic_write_json,
    balanced_shards,
    benchmark_seeds,
    build_protocol,
    content_sha256,
    load_all_instances,
    native_to_harris,
    result_root,
    runtime_manifest,
    size_class,
    source_manifest,
    time_limit_seconds,
    validate_license_path,
    write_csv,
)


def prepare(*, profile: str, shard_count: int, license_path: Path, official: bool) -> dict[str, Any]:
    output = result_root(profile)
    if (output / "runs").exists() and any((output / "runs").rglob("result.json")):
        raise RuntimeError("Measured artifacts already exist; preparation refuses to overwrite them")
    output.mkdir(parents=True, exist_ok=True)
    native_dir = output / "inputs" / "native"
    harris_dir = output / "inputs" / "harris"
    native_dir.mkdir(parents=True, exist_ok=True)
    harris_dir.mkdir(parents=True, exist_ok=True)

    protocol = build_protocol(profile=profile, shard_count=shard_count, official=official)
    protocol["license"] = validate_license_path(license_path)
    atomic_write_json(output / "protocol.json", protocol)
    atomic_write_json(output / "source_manifest.json", source_manifest())
    atomic_write_json(output / "environment_manifest.json", runtime_manifest(license_path))

    manifest: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    supplementary: list[str] = []
    for record in load_all_instances():
        identifier = record["instance"]
        native_path = native_dir / f"{identifier}.json"
        harris_path = harris_dir / f"{identifier}.json"
        atomic_write_json(native_path, record["data"])
        atomic_write_json(harris_path, native_to_harris(record["data"]))
        n_vertices = int(record["audit"]["n_vertices"])
        if record["collection"] == "supplementary_19":
            supplementary.append(identifier)
        manifest.append(
            {
                "instance": identifier,
                "collection": record["collection"],
                "source": record["source"],
                "instance_sha256": record["instance_sha256"],
                "n_vertices": n_vertices,
                "size_class": size_class(n_vertices),
                "time_limit_seconds": time_limit_seconds(n_vertices),
                "native_path": str(native_path.resolve()),
                "harris_path": str(harris_path.resolve()),
            }
        )
        for algorithm in ALGORITHMS:
            for seed in benchmark_seeds(n_vertices):
                jobs.append(
                    {
                        "phase": "benchmark",
                        "algorithm": algorithm,
                        "engine": "unified_cpp" if algorithm in {"ibs", "mip", "cuth", "rs"} else "legacy_python",
                        "instance": identifier,
                        "collection": record["collection"],
                        "instance_sha256": record["instance_sha256"],
                        "seed": seed,
                        "n_vertices": n_vertices,
                        "size_class": size_class(n_vertices),
                        "time_limit_seconds": time_limit_seconds(n_vertices),
                        "native_path": str(native_path.resolve()),
                        "harris_path": str(harris_path.resolve()),
                        "protocol_sha256": protocol["protocol_sha256"],
                    }
                )
    for identifier in sorted(supplementary):
        row = next(item for item in manifest if item["instance"] == identifier)
        for seed in SEEDS:
            jobs.append(
                {
                    "phase": "trajectory",
                    "algorithm": "ibs_stn",
                    "engine": "ibs_stn_cpp",
                    "instance": identifier,
                    "collection": "supplementary_19",
                    "instance_sha256": row["instance_sha256"],
                    "seed": seed,
                    "n_vertices": row["n_vertices"],
                    "size_class": row["size_class"],
                    "time_limit_seconds": row["time_limit_seconds"],
                    "native_path": row["native_path"],
                    "harris_path": row["harris_path"],
                    "protocol_sha256": protocol["protocol_sha256"],
                }
            )

    jobs = balanced_shards(jobs, shard_count)
    for number, job in enumerate(jobs, start=1):
        job["job_id"] = f"job_{number:05d}"
    benchmark_jobs = [job for job in jobs if job["phase"] == "benchmark"]
    trajectory_jobs = [job for job in jobs if job["phase"] == "trajectory"]
    if len(manifest) != 74 or len(benchmark_jobs) != 3936 or len(trajectory_jobs) != 190:
        raise RuntimeError(
            f"Unexpected plan dimensions: instances={len(manifest)}, benchmark={len(benchmark_jobs)}, trajectory={len(trajectory_jobs)}"
        )
    write_csv(output / "instances_74.csv", manifest)
    write_csv(output / "jobs.csv", jobs)
    conversion = validate_historical_conversion()
    atomic_write_json(output / "historical_conversion_validation.json", conversion)
    by_shard = {
        str(shard): {
            "jobs": sum(job["shard_index"] == shard for job in jobs),
            "nominal_cpu_seconds": sum(int(job["time_limit_seconds"]) for job in jobs if job["shard_index"] == shard),
        }
        for shard in range(shard_count)
    }
    summary = {
        "status": "prepared",
        "profile": profile,
        "official": official,
        "instances": len(manifest),
        "benchmark_jobs": len(benchmark_jobs),
        "trajectory_jobs": len(trajectory_jobs),
        "total_jobs": len(jobs),
        "shards": by_shard,
        "protocol_sha256": protocol["protocol_sha256"],
        "historical_conversion_exact": all(row["all_equal"] for row in conversion),
    }
    atomic_write_json(output / "preparation_summary.json", summary)
    return summary
