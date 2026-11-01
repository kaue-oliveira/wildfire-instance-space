from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json,
    content_sha256,
    file_sha256,
    git_state,
    read_csv,
    result_root,
    utc_now,
)


ALLOWED_ALGORITHMS = frozenset({"ibs", "rs"})
FEASIBILITY_FORWARD_ALGORITHMS = frozenset({"ibs", "rs", "ils", "cuth", "mip", "lbbd", "ibs_stn"})
IDENTITY_FIELDS = (
    "phase", "algorithm", "engine", "instance", "collection",
    "instance_sha256", "seed", "n_vertices", "size_class",
    "time_limit_seconds", "shard_index", "job_id",
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def run_relative(job: dict[str, Any]) -> Path:
    return Path("runs") / job["phase"] / job["algorithm"] / job["instance"] / f"seed_{int(job['seed']):02d}"


def job_identity(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in IDENTITY_FIELDS}


def file_manifest(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(directory)),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def protocol_payload_hash(protocol: dict[str, Any]) -> str:
    ignored = {"profile", "official", "deployment_git", "license", "protocol_sha256"}
    return content_sha256({key: value for key, value in protocol.items() if key not in ignored})


def recorded_project_root(job: dict[str, Any]) -> Path:
    profile_root = Path(job["native_path"]).parents[2]
    return profile_root.parents[2]


def expected_command(job: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    project = recorded_project_root(job)
    profile_root = Path(job["native_path"]).parents[2]
    directory = profile_root / run_relative(job)
    limit = int(job["time_limit_seconds"])
    if job["engine"] == "unified_cpp":
        command = [
            str(project / "third_party" / "Wildfire-Suppression" / "build" / "fire"),
            "--instance", job["native_path"], "--alg", job["algorithm"],
            "--seed", str(job["seed"]), "--timelimit", str(limit),
            "--output", str((directory / "statistics.csv").resolve()),
            "--save_solution",
        ]
        if job["algorithm"] == "ibs":
            command += [
                "--ibsBeta", "50", "--ibsEta", "70", "--ibsC", "30",
                "--ibsP", "0.5", "--ibsPhat", "1.0", "--ibsZmax", "3",
            ]
    else:
        raise ValueError(f"Carry-forward engine not allowed: {job['engine']}")
    worker = manifest.get("worker", {})
    return ["taskset", "--cpu-list", str(worker.get("logical_cpu")), *command]


def validate_v1_run(root: Path, job: dict[str, Any], protocol: dict[str, Any]) -> tuple[bool, str]:
    directory = root / run_relative(job)
    result = load_json(directory / "result.json")
    manifest = load_json(directory / "run_manifest.json")
    if not result or not manifest:
        return False, "missing_result_or_manifest"
    if result.get("status") != "completed" or result.get("pilot"):
        return False, "not_completed_measured"
    if result.get("result_provenance") != "measured":
        return False, "invalid_provenance"
    if result.get("protocol_sha256") != protocol.get("protocol_sha256"):
        return False, "protocol_mismatch"
    if job.get("protocol_sha256") != protocol.get("protocol_sha256"):
        return False, "job_protocol_mismatch"
    if result.get("instance_sha256") != job.get("instance_sha256"):
        return False, "instance_mismatch"
    if result.get("algorithm") != job.get("algorithm") or int(result.get("seed", -1)) != int(job["seed"]):
        return False, "run_identity_mismatch"
    if manifest.get("job") != job or int(manifest.get("shard_index", -1)) != int(job["shard_index"]):
        return False, "manifest_job_mismatch"
    if int(manifest.get("effective_time_limit_seconds", -1)) != int(job["time_limit_seconds"]):
        return False, "time_limit_mismatch"
    if manifest.get("pilot_limit_seconds") is not None:
        return False, "pilot_manifest"
    if manifest.get("command") != expected_command(job, manifest):
        return False, "command_mismatch"
    if not isinstance(result.get("objective"), (int, float)):
        return False, "invalid_objective"
    if not 0 <= float(result["objective"]) <= int(job["n_vertices"]):
        return False, "objective_out_of_range"
    for name in ("stdout", "stderr"):
        path = directory / f"{name}.log"
        if not path.is_file() or result.get(f"{name}_sha256") != file_sha256(path):
            return False, f"{name}_hash_mismatch"
    solution = directory / str(result.get("solution_file", ""))
    if not solution.is_file() or result.get("solution_sha256") != file_sha256(solution):
        return False, "solution_hash_mismatch"
    return True, "accepted"


def find_record(amendment: dict[str, Any], job: dict[str, Any]) -> dict[str, Any] | None:
    identity = job_identity(job)
    for record in amendment.get("runs", []):
        if record.get("identity") == identity:
            return record
    return None


def valid_carried_forward(profile: str, job: dict[str, Any], protocol: dict[str, Any]) -> bool:
    root = result_root(profile)
    amendment = load_json(root / "amendment_manifest.json")
    if not amendment or amendment.get("to_profile") != profile:
        return False
    allowed = set(amendment.get("allowed_algorithms", []))
    if job.get("algorithm") not in allowed:
        return False
    if amendment.get("to_protocol_sha256") != protocol.get("protocol_sha256"):
        return False
    record = find_record(amendment, job)
    if not record or record.get("destination_job_sha256") != content_sha256(job):
        return False
    directory = root / record["destination_run"]
    files = file_manifest(directory) if directory.is_dir() else []
    return files == record.get("files") and content_sha256(files) == record.get("run_tree_sha256")


def carry_forward(from_profile: str, to_profile: str, algorithms: list[str], shard_index: int) -> dict[str, Any]:
    requested = set(algorithms)
    if not requested or not requested <= ALLOWED_ALGORITHMS:
        raise ValueError("Carry-forward permits exactly IBS/RS selections; ILS and all other algorithms are forbidden")
    source = result_root(from_profile)
    destination = result_root(to_profile)
    source_protocol = load_json(source / "protocol.json")
    destination_protocol = load_json(destination / "protocol.json")
    if not source_protocol or not destination_protocol:
        raise FileNotFoundError("Both source and destination profiles must be prepared")
    if source_protocol.get("profile") != "benchmark74_v1" or destination_protocol.get("profile") != "benchmark74_v2":
        raise ValueError("Carry-forward requires benchmark74_v1 -> benchmark74_v2")
    if protocol_payload_hash(source_protocol) != source_protocol.get("protocol_sha256"):
        raise ValueError("The v1 protocol hash is not self-consistent")
    if protocol_payload_hash(destination_protocol) != destination_protocol.get("protocol_sha256"):
        raise ValueError("The v2 protocol hash is not self-consistent")
    if (destination / "amendment_manifest.json").exists() or any(destination.rglob("result.json")):
        raise RuntimeError("The v2 profile already contains results or an amendment")
    if (destination / "runs").exists():
        raise RuntimeError("The v2 runs directory must not exist before carry-forward")

    source_jobs = read_csv(source / "jobs.csv")
    destination_jobs = read_csv(destination / "jobs.csv")
    destination_by_identity = {content_sha256(job_identity(job)): job for job in destination_jobs}
    selected = [
        job for job in source_jobs
        if job["algorithm"] in requested and int(job["shard_index"]) == shard_index
    ]
    accepted: list[tuple[dict[str, Any], dict[str, Any], Path, list[dict[str, Any]]]] = []
    rejected: list[dict[str, Any]] = []
    for source_job in selected:
        valid, reason = validate_v1_run(source, source_job, source_protocol)
        destination_job = destination_by_identity.get(content_sha256(job_identity(source_job)))
        if not valid or destination_job is None:
            rejected.append({"identity": job_identity(source_job), "reason": reason if not valid else "missing_v2_job"})
            continue
        relative = run_relative(source_job)
        files = file_manifest(source / relative)
        accepted.append((source_job, destination_job, relative, files))

    with tempfile.TemporaryDirectory(prefix="carry_forward_", dir=destination) as temporary:
        staged_runs = Path(temporary) / "runs"
        for _source_job, _destination_job, relative, files in accepted:
            staged = staged_runs / relative.relative_to("runs")
            shutil.copytree(source / relative, staged, copy_function=shutil.copy2)
            if file_manifest(staged) != files:
                raise RuntimeError(f"Byte verification failed while staging {relative}")
        staged_runs.replace(destination / "runs")

    machine = load_json(Path(__file__).with_name("lab_machine.json")) or {}
    records = []
    for source_job, destination_job, relative, files in accepted:
        records.append({
            "identity": job_identity(destination_job),
            "source_job_sha256": content_sha256(source_job),
            "destination_job_sha256": content_sha256(destination_job),
            "source_run": str(relative),
            "destination_run": str(relative),
            "files": files,
            "run_tree_sha256": content_sha256(files),
            "provenance": "carried_forward_v1",
        })
    counts = Counter(record["identity"]["algorithm"] for record in records)
    amendment = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "scientific_reason": "Strict ILS deadline amendment; preserve unaffected completed IBS/RS measurements byte for byte and rerun every ILS run",
        "from_profile": from_profile,
        "to_profile": to_profile,
        "from_protocol_sha256": source_protocol["protocol_sha256"],
        "to_protocol_sha256": destination_protocol["protocol_sha256"],
        "deployment_git": git_state(),
        "machine": machine,
        "shard_index": shard_index,
        "allowed_algorithms": sorted(ALLOWED_ALGORITHMS),
        "requested_algorithms": sorted(requested),
        "accepted": len(records),
        "accepted_by_algorithm": dict(sorted(counts.items())),
        "rejected": len(rejected),
        "rejections": rejected,
        "runs": records,
        "carried_tree_sha256": content_sha256([
            {"destination_run": row["destination_run"], "run_tree_sha256": row["run_tree_sha256"]}
            for row in records
        ]),
    }
    atomic_write_json(destination / "amendment_manifest.json", amendment)
    return verify(to_profile, shard_index)


def no_deployment_score(job: dict[str, Any]) -> int:
    from .legacy_worker import load_instance, validate_solution

    values = load_instance(Path(job["harris_path"]))
    validation = validate_solution(*values, [])
    return int(validation["objective_recomputed"])


def feasibility_forward(from_profile: str, to_profile: str, shard_index: int) -> dict[str, Any]:
    """Carry valid results and audited algorithm failures to the next protocol."""
    from .execute import valid_completed

    source = result_root(from_profile)
    destination = result_root(to_profile)
    source_protocol = load_json(source / "protocol.json")
    destination_protocol = load_json(destination / "protocol.json")
    if not source_protocol or not destination_protocol:
        raise FileNotFoundError("Both source and destination profiles must be prepared")
    pair = (source_protocol.get("profile"), destination_protocol.get("profile"))
    if pair not in {
        ("benchmark74_v2", "benchmark74_v2_1"),
        ("benchmark74_v2_1", "benchmark74_v2_2"),
        ("benchmark74_v2_2", "benchmark74_v2_3"),
        ("benchmark74_v2_3", "benchmark74_v2_4"),
    }:
        raise ValueError("Protocol forward requires one supported consecutive protocol transition")
    if protocol_payload_hash(source_protocol) != source_protocol.get("protocol_sha256"):
        raise ValueError("The v2 protocol hash is not self-consistent")
    if protocol_payload_hash(destination_protocol) != destination_protocol.get("protocol_sha256"):
        raise ValueError("The v2.1 protocol hash is not self-consistent")
    if (destination / "amendment_manifest.json").exists() or any(destination.rglob("result.json")):
        raise RuntimeError("The v2.1 profile already contains results or an amendment")
    if (destination / "runs").exists():
        raise RuntimeError("The v2.1 runs directory must not exist before feasibility forward")

    source_jobs = read_csv(source / "jobs.csv")
    destination_jobs = read_csv(destination / "jobs.csv")
    destination_by_identity = {content_sha256(job_identity(job)): job for job in destination_jobs}
    selected = [
        job for job in source_jobs
        if job["algorithm"] in FEASIBILITY_FORWARD_ALGORITHMS and int(job["shard_index"]) == shard_index
    ]
    accepted: list[tuple[dict[str, Any], dict[str, Any], Path, list[dict[str, Any]]]] = []
    resolved: list[tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for source_job in selected:
        destination_job = destination_by_identity.get(content_sha256(job_identity(source_job)))
        if destination_job is None:
            rejected.append({"identity": job_identity(source_job), "reason": "missing_v2_1_job"})
            continue
        relative = run_relative(source_job)
        if valid_completed(from_profile, source_job, source_protocol, accept_any_mode=True):
            files = file_manifest(source / relative)
            accepted.append((source_job, destination_job, relative, files))
            continue
        prior_algorithm_failure = load_json(source / relative / "algorithm_failure.json")
        if prior_algorithm_failure and prior_algorithm_failure.get("failure_category") == "infeasible_final_solution":
            algorithm_failure = {
                **prior_algorithm_failure,
                **destination_job,
                "algorithm": destination_job["algorithm"],
                "seed": int(destination_job["seed"]),
                "shard_index": int(destination_job["shard_index"]),
                "protocol_sha256": destination_protocol["protocol_sha256"],
                "result_provenance": f"carried_algorithm_failure_{from_profile}",
                "source_profile": from_profile,
                "source_protocol_sha256": source_protocol["protocol_sha256"],
                "source_algorithm_failure_sha256": file_sha256(source / relative / "algorithm_failure.json"),
                "recorded_utc": utc_now(),
            }
            resolved.append((source_job, destination_job, relative, algorithm_failure))
            continue
        failure = load_json(source / relative / "failure.json")
        stderr_path = source / relative / "stderr.log"
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
        recognized = (
            source_job["algorithm"] == "ils"
            and failure is not None
            and "ValueError: A node received more than one resource" in stderr
        )
        if recognized:
            algorithm_failure = {
                **destination_job,
                "algorithm": destination_job["algorithm"],
                "seed": int(destination_job["seed"]),
                "shard_index": int(destination_job["shard_index"]),
                "status": "algorithm_failure",
                "failure_category": "infeasible_final_solution",
                "validation_error": "A node received more than one resource",
                "objective": None,
                "scored_objective": no_deployment_score(destination_job),
                "fallback_policy": "discard_invalid_and_score_no_deployment_baseline",
                "protocol_sha256": destination_protocol["protocol_sha256"],
                "instance_sha256": destination_job["instance_sha256"],
                "result_provenance": "carried_algorithm_failure_v2",
                "source_profile": from_profile,
                "source_protocol_sha256": source_protocol["protocol_sha256"],
                "source_failure_sha256": file_sha256(source / relative / "failure.json"),
                "source_stderr_sha256": file_sha256(stderr_path),
                "recorded_utc": utc_now(),
            }
            resolved.append((source_job, destination_job, relative, algorithm_failure))

    with tempfile.TemporaryDirectory(prefix="feasibility_forward_", dir=destination) as temporary:
        staged_runs = Path(temporary) / "runs"
        for _source_job, _destination_job, relative, files in accepted:
            staged = staged_runs / relative.relative_to("runs")
            shutil.copytree(source / relative, staged, copy_function=shutil.copy2)
            if file_manifest(staged) != files:
                raise RuntimeError(f"Byte verification failed while staging {relative}")
        for _source_job, _destination_job, relative, algorithm_failure in resolved:
            staged = staged_runs / relative.relative_to("runs")
            shutil.copytree(source / relative, staged, copy_function=shutil.copy2)
            atomic_write_json(staged / "algorithm_failure.json", algorithm_failure)
        staged_runs.replace(destination / "runs")

    records = []
    for source_job, destination_job, relative, files in accepted:
        records.append({
            "identity": job_identity(destination_job),
            "source_job_sha256": content_sha256(source_job),
            "destination_job_sha256": content_sha256(destination_job),
            "source_run": str(relative),
            "destination_run": str(relative),
            "files": files,
            "run_tree_sha256": content_sha256(files),
            "provenance": f"carried_forward_{from_profile}",
        })
    resolved_records = [
        {
            "identity": job_identity(destination_job),
            "source_job_sha256": content_sha256(source_job),
            "destination_job_sha256": content_sha256(destination_job),
            "source_run": str(relative),
            "destination_run": str(relative),
            "algorithm_failure_sha256": file_sha256(destination / relative / "algorithm_failure.json"),
            "provenance": f"carried_algorithm_failure_{from_profile}",
        }
        for source_job, destination_job, relative, _algorithm_failure in resolved
    ]
    counts = Counter(record["identity"]["algorithm"] for record in records)
    amendment = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "scientific_reason": "Versioned protocol amendment: preserve valid parent measurements and audited algorithm failures byte for byte where applicable, then run only work affected by the new rule",
        "from_profile": from_profile,
        "to_profile": to_profile,
        "from_protocol_sha256": source_protocol["protocol_sha256"],
        "to_protocol_sha256": destination_protocol["protocol_sha256"],
        "deployment_git": git_state(),
        "machine": load_json(Path(__file__).with_name("lab_machine.json")) or {},
        "shard_index": shard_index,
        "allowed_algorithms": sorted(FEASIBILITY_FORWARD_ALGORITHMS),
        "accepted": len(records),
        "accepted_by_algorithm": dict(sorted(counts.items())),
        "resolved_algorithm_failures": resolved_records,
        "rejected": len(rejected),
        "rejections": rejected,
        "runs": records,
        "carried_tree_sha256": content_sha256([
            {"destination_run": row["destination_run"], "run_tree_sha256": row["run_tree_sha256"]}
            for row in records
        ]),
    }
    atomic_write_json(destination / "amendment_manifest.json", amendment)
    return verify(to_profile, shard_index)


def verify(profile: str, shard_index: int) -> dict[str, Any]:
    root = result_root(profile)
    amendment = load_json(root / "amendment_manifest.json")
    protocol = load_json(root / "protocol.json")
    if not amendment or not protocol:
        raise FileNotFoundError("Protocol or amendment manifest is absent")
    if amendment.get("to_profile") != profile or int(amendment.get("shard_index", -1)) != shard_index:
        raise ValueError("Amendment profile/shard identity mismatch")
    if amendment.get("to_protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("Amendment destination protocol mismatch")
    source = result_root(amendment["from_profile"])
    tampered: list[str] = []
    algorithms = Counter()
    allowed = set(amendment.get("allowed_algorithms", []))
    for record in amendment.get("runs", []):
        algorithm = record.get("identity", {}).get("algorithm")
        algorithms[algorithm] += 1
        if algorithm not in allowed:
            tampered.append(f"forbidden_algorithm:{algorithm}")
            continue
        source_dir = source / record["source_run"]
        destination_dir = root / record["destination_run"]
        source_files = file_manifest(source_dir) if source_dir.is_dir() else []
        destination_files = file_manifest(destination_dir) if destination_dir.is_dir() else []
        if source_files != record.get("files") or destination_files != record.get("files"):
            tampered.append(record["destination_run"])
    resolved = amendment.get("resolved_algorithm_failures", [])
    for record in resolved:
        destination = root / record["destination_run"] / "algorithm_failure.json"
        failure = load_json(destination)
        if (
            not failure
            or failure.get("status") != "algorithm_failure"
            or failure.get("failure_category") != "infeasible_final_solution"
            or failure.get("protocol_sha256") != protocol.get("protocol_sha256")
            or file_sha256(destination) != record.get("algorithm_failure_sha256")
        ):
            tampered.append(f"algorithm_failure:{record.get('destination_run')}")
    if tampered:
        raise RuntimeError(f"Amendment verification found {len(tampered)} altered or forbidden runs: {tampered[:5]}")
    return {
        "status": "verified",
        "profile": profile,
        "shard_index": shard_index,
        "carried_forward": len(amendment.get("runs", [])),
        "by_algorithm": dict(sorted(algorithms.items())),
        "carried_ils": algorithms.get("ils", 0),
        "algorithm_failures": len(resolved),
        "tampered": 0,
        "from_protocol_sha256": amendment["from_protocol_sha256"],
        "to_protocol_sha256": amendment["to_protocol_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable benchmark74 protocol amendments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    carry = subparsers.add_parser("carry-forward")
    carry.add_argument("--from-profile", required=True)
    carry.add_argument("--to-profile", required=True)
    carry.add_argument("--algorithms", nargs="+", required=True)
    carry.add_argument("--shard-index", type=int, required=True)
    feasibility = subparsers.add_parser("feasibility-forward")
    feasibility.add_argument("--from-profile", required=True)
    feasibility.add_argument("--to-profile", required=True)
    feasibility.add_argument("--shard-index", type=int, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--profile", required=True)
    verify_parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "carry-forward":
        value = carry_forward(args.from_profile, args.to_profile, args.algorithms, args.shard_index)
    elif args.command == "feasibility-forward":
        value = feasibility_forward(args.from_profile, args.to_profile, args.shard_index)
    else:
        value = verify(args.profile, args.shard_index)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
