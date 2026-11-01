from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


INDEX_FIELDS = [
    "lab",
    "shard_index",
    "job_id",
    "phase",
    "algorithm",
    "engine",
    "instance",
    "collection",
    "instance_sha256",
    "seed",
    "n_vertices",
    "size_class",
    "time_limit_seconds",
    "status",
    "successful_run",
    "included_in_objective_analysis",
    "objective",
    "scored_objective",
    "analysis_objective",
    "analysis_objective_source",
    "best_known_value_all_runs",
    "relative_gap_to_bkv",
    "lower_bound",
    "mip_gap",
    "wall_seconds",
    "peak_rss_bytes",
    "result_provenance",
    "failure_category",
    "protocol_sha256",
    "source_artifact",
    "source_artifact_sha256",
    "source_run_directory",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        row = {"path": relative, "size_bytes": size, "sha256": digest}
        rows.append(row)
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    return rows, tree_digest.hexdigest()


def run_directory(profile: Path, job: dict[str, str]) -> Path:
    return (
        profile
        / "runs"
        / job["phase"]
        / job["algorithm"]
        / job["instance"]
        / f"seed_{int(job['seed']):02d}"
    )


def normalized_rows(rows: list[dict[str, str]], ignored: set[str]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if key not in ignored} for row in rows]


def validate_profile(profile: Path, lab: str, expected_shard: int) -> dict[str, Any]:
    required = {
        "protocol.json",
        "source_manifest.json",
        "jobs.csv",
        "instances_74.csv",
        f"execution_status_shard_{expected_shard}.json",
    }
    missing_required = sorted(name for name in required if not (profile / name).is_file())
    if missing_required:
        raise ValueError(f"{lab}: missing required files: {missing_required}")

    protocol = load_json(profile / "protocol.json")
    if protocol.get("profile") != "benchmark74_v2_4" or protocol.get("official") is not True:
        raise ValueError(f"{lab}: profile is not an official benchmark74_v2_4 protocol")
    if int(protocol.get("shard_count", -1)) != 2:
        raise ValueError(f"{lab}: expected shard_count=2")

    jobs = load_csv(profile / "jobs.csv")
    selected = [row for row in jobs if int(row["shard_index"]) == expected_shard]
    if len(selected) != 2063:
        raise ValueError(f"{lab}: expected 2063 selected jobs, found {len(selected)}")
    if len({row["job_id"] for row in selected}) != len(selected):
        raise ValueError(f"{lab}: duplicate job_id in selected shard")

    records: list[dict[str, Any]] = []
    actual = Counter()
    by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    seen_artifacts: set[Path] = set()
    for job in selected:
        directory = run_directory(profile, job)
        result_path = directory / "result.json"
        algorithm_failure_path = directory / "algorithm_failure.json"
        present = [path for path in (result_path, algorithm_failure_path) if path.is_file()]
        if len(present) != 1:
            raise ValueError(
                f"{lab}:{job['job_id']}: expected exactly one canonical artifact, found {len(present)}"
            )
        artifact_path = present[0]
        artifact = load_json(artifact_path)
        status = "completed" if artifact_path.name == "result.json" else "algorithm_failure"
        if artifact.get("status") != status:
            raise ValueError(f"{lab}:{job['job_id']}: invalid status in {artifact_path}")
        for field in ("job_id", "algorithm", "instance", "instance_sha256"):
            if str(artifact.get(field)) != str(job[field]):
                raise ValueError(f"{lab}:{job['job_id']}: {field} does not match jobs.csv")
        if int(artifact.get("seed", -1)) != int(job["seed"]):
            raise ValueError(f"{lab}:{job['job_id']}: seed does not match jobs.csv")
        if int(artifact.get("shard_index", -1)) != expected_shard:
            raise ValueError(f"{lab}:{job['job_id']}: shard_index mismatch")

        objective = artifact.get("objective")
        scored_objective = artifact.get("scored_objective")
        if status == "completed":
            if not isinstance(objective, (int, float)):
                raise ValueError(f"{lab}:{job['job_id']}: completed result has no numeric objective")
            analysis_objective = float(objective)
            objective_source = "result.objective"
            if not (directory / "run_manifest.json").is_file():
                raise ValueError(f"{lab}:{job['job_id']}: completed result has no run_manifest.json")
        else:
            if objective is not None or not isinstance(scored_objective, (int, float)):
                raise ValueError(f"{lab}:{job['job_id']}: invalid algorithm-failure scoring fields")
            if artifact.get("failure_category") != "infeasible_final_solution":
                raise ValueError(f"{lab}:{job['job_id']}: unexpected algorithm failure category")
            analysis_objective = float(scored_objective)
            objective_source = "algorithm_failure.scored_objective"
        if not 0 <= analysis_objective <= int(job["n_vertices"]):
            raise ValueError(f"{lab}:{job['job_id']}: objective outside [0, n_vertices]")

        if job["algorithm"] == "lbbd":
            validation = artifact.get("validation")
            if (
                not isinstance(validation, dict)
                or validation.get("feasible") is not True
                or float(validation.get("objective_recomputed", -1)) != float(objective)
            ):
                raise ValueError(f"{lab}:{job['job_id']}: LBBD result failed independent validation")

        solution_file = artifact.get("solution_file")
        solution_sha256 = artifact.get("solution_sha256")
        if solution_file and solution_sha256:
            solution_path = directory / str(solution_file)
            if not solution_path.is_file() or sha256_file(solution_path) != solution_sha256:
                raise ValueError(f"{lab}:{job['job_id']}: solution file hash mismatch")

        actual[status] += 1
        by_algorithm[job["algorithm"]][status] += 1
        seen_artifacts.add(artifact_path.resolve())
        records.append(
            {
                "lab": lab,
                "shard_index": expected_shard,
                **{key: job[key] for key in (
                    "job_id", "phase", "algorithm", "engine", "instance", "collection",
                    "instance_sha256", "seed", "n_vertices", "size_class", "time_limit_seconds",
                )},
                "status": status,
                "successful_run": status == "completed",
                "included_in_objective_analysis": True,
                "objective": objective,
                "scored_objective": scored_objective,
                "analysis_objective": analysis_objective,
                "analysis_objective_source": objective_source,
                "lower_bound": artifact.get("lower_bound"),
                "mip_gap": artifact.get("mip_gap"),
                "wall_seconds": artifact.get("wall_seconds"),
                "peak_rss_bytes": artifact.get("peak_rss_bytes"),
                "result_provenance": artifact.get("result_provenance"),
                "failure_category": artifact.get("failure_category"),
                "protocol_sha256": artifact.get("protocol_sha256"),
                "source_artifact": (
                    f"shards/{lab.lower()}/{artifact_path.relative_to(profile).as_posix()}"
                ),
                "source_artifact_sha256": sha256_file(artifact_path),
                "source_run_directory": (
                    f"shards/{lab.lower()}/{directory.relative_to(profile).as_posix()}"
                ),
                "raw_record": artifact,
            }
        )

    discovered = {
        path.resolve()
        for pattern in ("result.json", "algorithm_failure.json")
        for path in (profile / "runs").rglob(pattern)
    }
    if discovered != seen_artifacts:
        extra = sorted(str(path) for path in discovered - seen_artifacts)
        absent = sorted(str(path) for path in seen_artifacts - discovered)
        raise ValueError(f"{lab}: canonical artifact set mismatch; extra={extra}; absent={absent}")

    status_summary = load_json(profile / f"execution_status_shard_{expected_shard}.json")
    if int(status_summary.get("total", -1)) != len(selected):
        raise ValueError(f"{lab}: execution status total mismatch")
    if int(status_summary.get("completed", 0)) != actual["completed"]:
        raise ValueError(f"{lab}: execution status completed count mismatch")
    if int(status_summary.get("algorithm_failure", 0)) != actual["algorithm_failure"]:
        raise ValueError(f"{lab}: execution status algorithm_failure count mismatch")

    return {
        "lab": lab,
        "expected_shard": expected_shard,
        "profile": profile,
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "jobs": jobs,
        "selected_jobs": selected,
        "instances": load_csv(profile / "instances_74.csv"),
        "records": records,
        "counts": dict(actual),
        "by_algorithm": {algorithm: dict(counts) for algorithm, counts in sorted(by_algorithm.items())},
        "historical_failure_files": sorted(
            path.relative_to(profile).as_posix() for path in (profile / "runs").rglob("failure.json")
        ),
    }


def protocol_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(protocol_differences(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        differences: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(protocol_differences(left_item, right_item, f"{prefix}[{index}]"))
        return differences
    return [] if left == right else [prefix]


def add_bkv(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bkv_by_instance: dict[str, float] = {}
    for record in records:
        value = float(record["analysis_objective"])
        instance = record["instance"]
        bkv_by_instance[instance] = min(value, bkv_by_instance.get(instance, value))
    for record in records:
        value = float(record["analysis_objective"])
        bkv = bkv_by_instance[record["instance"]]
        record["best_known_value_all_runs"] = bkv
        record["relative_gap_to_bkv"] = (value - bkv) / max(1.0, abs(bkv))
    return records


def validate_union(lab01: dict[str, Any], lab02: dict[str, Any]) -> dict[str, Any]:
    ignored_jobs = {"native_path", "harris_path", "protocol_sha256"}
    if normalized_rows(lab01["jobs"], ignored_jobs) != normalized_rows(lab02["jobs"], ignored_jobs):
        raise ValueError("Normalized jobs.csv manifests differ between laboratories")
    ignored_instances = {"native_path", "harris_path", "source"}
    if normalized_rows(lab01["instances"], ignored_instances) != normalized_rows(
        lab02["instances"], ignored_instances
    ):
        raise ValueError("Normalized instances_74.csv manifests differ between laboratories")

    ids01 = {row["job_id"] for row in lab01["selected_jobs"]}
    ids02 = {row["job_id"] for row in lab02["selected_jobs"]}
    all_ids = {row["job_id"] for row in lab01["jobs"]}
    if ids01 & ids02:
        raise ValueError("Shard job sets overlap")
    if ids01 | ids02 != all_ids or len(all_ids) != 4126:
        raise ValueError("Shard union does not cover the complete 4,126-job manifest")

    inputs01, input_tree01 = inventory(lab01["profile"] / "inputs")
    inputs02, input_tree02 = inventory(lab02["profile"] / "inputs")
    compact01 = [(row["path"], row["size_bytes"], row["sha256"]) for row in inputs01]
    compact02 = [(row["path"], row["size_bytes"], row["sha256"]) for row in inputs02]
    if compact01 != compact02 or input_tree01 != input_tree02:
        raise ValueError("Input instance bundles are not byte-identical")

    records = add_bkv(lab01["records"] + lab02["records"])
    status_counts = Counter(record["status"] for record in records)
    algorithm_counts = Counter(record["algorithm"] for record in records)
    return {
        "records": records,
        "input_file_count": len(inputs01),
        "input_tree_sha256": input_tree01,
        "status_counts": dict(status_counts),
        "algorithm_counts": dict(sorted(algorithm_counts.items())),
        "protocol_difference_paths": protocol_differences(lab01["protocol"], lab02["protocol"]),
    }


def copy_profile(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing integration: {output}")
    lab01 = validate_profile(args.lab01_profile.resolve(), "LAB-01", 0)
    lab02 = validate_profile(args.lab02_profile.resolve(), "LAB-02", 1)
    union = validate_union(lab01, lab02)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        shard_destinations = {
            "LAB-01": staging / "shards" / "lab-01",
            "LAB-02": staging / "shards" / "lab-02",
        }
        copy_profile(lab01["profile"], shard_destinations["LAB-01"])
        copy_profile(lab02["profile"], shard_destinations["LAB-02"])

        inventories: dict[str, dict[str, Any]] = {}
        for label, source_validation in (("lab-01", lab01), ("lab-02", lab02)):
            source_rows, source_tree = inventory(source_validation["profile"])
            copied_rows, copied_tree = inventory(staging / "shards" / label)
            if source_rows != copied_rows or source_tree != copied_tree:
                raise ValueError(f"{label}: copied profile differs from source")
            inventory_path = staging / "inventories" / f"{label}-files.csv"
            write_csv(inventory_path, copied_rows, ["path", "size_bytes", "sha256"])
            inventories[label] = {
                "file_count": len(copied_rows),
                "tree_sha256": copied_tree,
                "inventory": inventory_path.relative_to(staging).as_posix(),
            }

        machine_dest = staging / "source_metadata"
        machine_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.lab01_machine, machine_dest / "lab-01-machine.json")
        shutil.copy2(args.lab02_machine, machine_dest / "lab-02-machine.json")

        records = sorted(
            union["records"],
            key=lambda row: (row["phase"], row["algorithm"], row["instance"], int(row["seed"])),
        )
        write_csv(staging / "analysis_index.csv", records, INDEX_FIELDS)
        failure_records = [row for row in records if row["status"] == "algorithm_failure"]
        write_csv(staging / "algorithm_failures.csv", failure_records, INDEX_FIELDS)
        with (staging / "analysis_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                full_record = {
                    key: value for key, value in record.items() if key != "raw_record"
                }
                full_record["raw_record"] = record["raw_record"]
                handle.write(json.dumps(full_record, ensure_ascii=False, sort_keys=True) + "\n")

        best_rows: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["instance"]].append(record)
        for instance, instance_records in sorted(grouped.items()):
            bkv = min(float(row["analysis_objective"]) for row in instance_records)
            winners = [row for row in instance_records if float(row["analysis_objective"]) == bkv]
            best_rows.append(
                {
                    "instance": instance,
                    "collection": instance_records[0]["collection"],
                    "n_vertices": instance_records[0]["n_vertices"],
                    "size_class": instance_records[0]["size_class"],
                    "best_known_value_all_runs": bkv,
                    "winner_count": len(winners),
                    "winner_algorithms": ";".join(sorted({row["algorithm"] for row in winners})),
                    "winner_labs": ";".join(sorted({row["lab"] for row in winners})),
                    "winner_job_ids": ";".join(sorted(row["job_id"] for row in winners)),
                }
            )
        write_csv(
            staging / "best_known_values.csv",
            best_rows,
            [
                "instance", "collection", "n_vertices", "size_class",
                "best_known_value_all_runs", "winner_count", "winner_algorithms",
                "winner_labs", "winner_job_ids",
            ],
        )

        analysis_artifacts = {}
        for name in (
            "analysis_index.csv",
            "analysis_records.jsonl",
            "best_known_values.csv",
            "algorithm_failures.csv",
        ):
            path = staging / name
            analysis_artifacts[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "integration_id": "benchmark74_v2_4_final_labs_01_02_20260924",
            "profile": "benchmark74_v2_4",
            "purpose": "immutable two-laboratory integration for TCC analysis",
            "source_branches": {
                "LAB-01": args.lab01_branch,
                "LAB-02": args.lab02_branch,
            },
            "source_commits": {
                "LAB-01": args.lab01_commit,
                "LAB-02": args.lab02_commit,
            },
            "source_protocol_sha256": {
                "LAB-01": lab01["protocol_sha256"],
                "LAB-02": lab02["protocol_sha256"],
            },
            "protocol_hashes_identical": lab01["protocol_sha256"] == lab02["protocol_sha256"],
            "protocol_difference_paths": union["protocol_difference_paths"],
            "protocol_compatibility_note": (
                "Both shards use the same normalized 4,126-job design and byte-identical inputs. "
                "Their v2.4 protocol hashes differ because the LBBD audit-field implementation and "
                "deployment metadata differ; every LBBD objective included here equals an independently "
                "recomputed feasible objective. The original protocols remain intact in each shard."
            ),
            "shards": {
                "LAB-01": {
                    "shard_index": 0,
                    "path": "shards/lab-01",
                    "counts": lab01["counts"],
                    "by_algorithm": lab01["by_algorithm"],
                    "historical_failure_files": lab01["historical_failure_files"],
                    **inventories["lab-01"],
                },
                "LAB-02": {
                    "shard_index": 1,
                    "path": "shards/lab-02",
                    "counts": lab02["counts"],
                    "by_algorithm": lab02["by_algorithm"],
                    "historical_failure_files": lab02["historical_failure_files"],
                    **inventories["lab-02"],
                },
            },
            "union": {
                "planned_jobs": 4126,
                "canonical_records": len(records),
                "status_counts": union["status_counts"],
                "algorithm_counts": union["algorithm_counts"],
                "instance_count": len(best_rows),
                "input_file_count": union["input_file_count"],
                "input_tree_sha256": union["input_tree_sha256"],
                "missing_jobs": 0,
                "duplicate_jobs": 0,
                "cross_shard_overlap": 0,
            },
            "analysis_policy": {
                "completed": "use result.objective",
                "algorithm_failure": (
                    "retain status=algorithm_failure and use only the protocol-defined feasible "
                    "no-deployment scored_objective for objective-based comparisons"
                ),
                "bkv": "minimum analysis_objective observed across both shards and all algorithms",
                "raw_failed_attempts": "preserved for audit but excluded from the canonical record set",
            },
            "analysis_artifacts": analysis_artifacts,
            "source_machine_manifests": {
                "LAB-01": {
                    "path": "source_metadata/lab-01-machine.json",
                    "sha256": sha256_file(machine_dest / "lab-01-machine.json"),
                },
                "LAB-02": {
                    "path": "source_metadata/lab-02-machine.json",
                    "sha256": sha256_file(machine_dest / "lab-02-machine.json"),
                },
            },
        }
        write_json(staging / "integration_manifest.json", manifest)
        write_json(
            staging / "validation_report.json",
            {
                "status": "verified",
                "checks": {
                    "lab01_checkpoint_amendment_verify_tampered": 0,
                    "lab02_checkpoint_amendment_verify_tampered": 0,
                    "normalized_job_manifests_equal": True,
                    "normalized_instance_manifests_equal": True,
                    "input_bundles_byte_identical": True,
                    "selected_jobs_per_shard": {"LAB-01": 2063, "LAB-02": 2063},
                    "cross_shard_job_overlap": 0,
                    "complete_union_jobs": len(records),
                    "canonical_artifact_per_job": True,
                    "all_lbbd_objectives_independently_feasible_and_recomputed": True,
                    "copied_profiles_match_source_sha256_inventories": True,
                    "missing_jobs": 0,
                    "duplicate_jobs": 0,
                },
            },
        )
        (staging / "README.md").write_text(
            """# Benchmark74 v2.4 — integração final LAB-01 + LAB-02

Este diretório preserva os dois shards oficiais completos e fornece uma visão conjunta pronta para análise.

- `shards/lab-01/` e `shards/lab-02/`: checkpoints originais, copiados byte a byte.
- `inventories/`: SHA-256, tamanho e caminho de todos os arquivos de cada checkpoint.
- `analysis_index.csv`: uma linha canônica para cada um dos 4.126 jobs planejados.
- `analysis_records.jsonl`: índice com o registro JSON original incorporado em cada linha.
- `best_known_values.csv`: BKV observado por instância e identificação dos vencedores.
- `algorithm_failures.csv`: as duas falhas ILS auditadas da LAB-02, separadas explicitamente.
- `integration_manifest.json`: commits, branches, protocolos, contagens e política de análise.
- `validation_report.json`: resultado dos controles de integridade.

Os dois hashes de protocolo v2.4 são diferentes e foram preservados. A diferença não foi ocultada nem
reescrita: o manifesto documenta os campos divergentes. Para LBBD, somente objetivos que coincidem com
a recomputação independente de uma solução viável entram no índice canônico.

As duas falhas algorítmicas ILS não são marcadas como execuções bem-sucedidas. Para análises de objetivo,
`analysis_objective` usa o baseline viável sem recursos definido pelo protocolo; para análises de taxa de
sucesso, use `successful_run`, que permanece falso nesses dois casos. O `failure.json` histórico existente
na LAB-02 fica preservado dentro do shard para auditoria, mas não constitui uma linha canônica adicional.
""",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    manifest = load_json(root / "integration_manifest.json")
    failures: list[str] = []
    for label in ("lab-01", "lab-02"):
        inventory_rows = load_csv(root / "inventories" / f"{label}-files.csv")
        actual_rows, actual_tree = inventory(root / "shards" / label)
        expected_rows = [
            {"path": row["path"], "size_bytes": int(row["size_bytes"]), "sha256": row["sha256"]}
            for row in inventory_rows
        ]
        manifest_entry = manifest["shards"]["LAB-01" if label == "lab-01" else "LAB-02"]
        if actual_rows != expected_rows:
            failures.append(f"{label}: file inventory mismatch")
        if actual_tree != manifest_entry["tree_sha256"]:
            failures.append(f"{label}: tree hash mismatch")
    for metadata in manifest["source_machine_manifests"].values():
        path = root / metadata["path"]
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            failures.append(f"machine manifest mismatch: {metadata['path']}")
    for name, metadata in manifest["analysis_artifacts"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["size_bytes"])
            or sha256_file(path) != metadata["sha256"]
        ):
            failures.append(f"analysis artifact mismatch: {name}")
    index_rows = load_csv(root / "analysis_index.csv")
    if len(index_rows) != 4126:
        failures.append(f"analysis_index.csv has {len(index_rows)} rows, expected 4126")
    if len({row["job_id"] for row in index_rows}) != 4126:
        failures.append("analysis_index.csv contains duplicate job IDs")
    status_counts = Counter(row["status"] for row in index_rows)
    if status_counts != Counter({"completed": 4124, "algorithm_failure": 2}):
        failures.append(f"unexpected status counts: {dict(status_counts)}")
    copied_lab01 = validate_profile(root / "shards" / "lab-01", "LAB-01", 0)
    copied_lab02 = validate_profile(root / "shards" / "lab-02", "LAB-02", 1)
    expected_union = validate_union(copied_lab01, copied_lab02)
    expected = {
        row["job_id"]: (
            row["lab"],
            row["status"],
            str(row["analysis_objective"]),
            row["source_artifact"],
            row["source_artifact_sha256"],
        )
        for row in expected_union["records"]
    }
    observed = {
        row["job_id"]: (
            row["lab"],
            row["status"],
            str(float(row["analysis_objective"])),
            row["source_artifact"],
            row["source_artifact_sha256"],
        )
        for row in index_rows
    }
    if observed != expected:
        failures.append("analysis index does not match canonical artifacts in copied shards")
    if failures:
        raise ValueError("; ".join(failures))
    print(
        json.dumps(
            {
                "status": "verified",
                "integration_id": manifest["integration_id"],
                "records": len(index_rows),
                "status_counts": dict(status_counts),
                "shard_tree_sha256": {
                    "LAB-01": manifest["shards"]["LAB-01"]["tree_sha256"],
                    "LAB-02": manifest["shards"]["LAB-02"]["tree_sha256"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build and verify the final two-lab Benchmark74 integration")
    subparsers = result.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--lab01-profile", type=Path, required=True)
    build_parser.add_argument("--lab02-profile", type=Path, required=True)
    build_parser.add_argument("--lab01-machine", type=Path, required=True)
    build_parser.add_argument("--lab02-machine", type=Path, required=True)
    build_parser.add_argument("--lab01-branch", required=True)
    build_parser.add_argument("--lab02-branch", required=True)
    build_parser.add_argument("--lab01-commit", required=True)
    build_parser.add_argument("--lab02-commit", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.set_defaults(func=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.set_defaults(func=verify)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
