from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from analysis_wsp.caminho_c.common import (
    accepted_rows,
    native_to_harris,
    physical_core_slots,
    raw_instance_path,
    thermal_snapshot,
    validate_and_repair_native,
)


PROFILE = "benchmark74_v2_4"
PROFILE_V1 = "benchmark74_v1"
PROFILE_V2 = "benchmark74_v2"
PROFILE_V2_1 = "benchmark74_v2_1"
PROFILE_V2_2 = "benchmark74_v2_2"
PROFILE_V2_3 = "benchmark74_v2_3"
PROFILE_V2_4 = "benchmark74_v2_4"
ALGORITHMS = ("ibs", "mip", "cuth", "rs", "ils", "lbbd")
UNIFIED_ALGORITHMS = frozenset({"ibs", "mip", "cuth", "rs"})
LEGACY_ALGORITHMS = frozenset({"ils", "lbbd"})
SEEDS = tuple(range(1, 11))
HUGE_REPLICATIONS = 3
SHARD_COUNT = 2
IBS_PARAMETERS = {"beta": 50, "eta": 70, "c": 30, "p": 0.5, "phat": 1.0, "zmax": 3}
ILS_PARAMETERS = {
    "MultiStarts": 50,
    "MaxCandidates": 5,
    "p1": 0.075,
    "p2": 0.025,
    "MaxNeighbours": 20,
    "MaxModifications": 5,
    "MaxFailures": 100,
    "MaxNoImprovements": 50,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def result_root(profile: str = PROFILE) -> Path:
    return project_root() / "experiments" / "benchmark" / profile


def profile_generation(profile: str) -> int:
    if profile == PROFILE_V1 or profile.endswith("_v1"):
        return 1
    if profile in {PROFILE_V2, PROFILE_V2_1, PROFILE_V2_2, PROFILE_V2_3, PROFILE_V2_4} or profile.endswith(("_v2", "_v2_1", "_v2_2", "_v2_3", "_v2_4")):
        return 2
    raise ValueError(f"Unsupported benchmark74 profile generation: {profile}")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def size_class(n_vertices: int) -> str:
    if n_vertices <= 400:
        return "Small"
    return {900: "Medium", 1600: "Large", 6400: "Huge"}.get(n_vertices, "Other")


def time_limit_seconds(n_vertices: int) -> int:
    klass = size_class(n_vertices)
    return {"Small": 600, "Medium": 1350, "Large": 2400, "Huge": 9600}[klass]


def benchmark_seeds(n_vertices: int) -> tuple[int, ...]:
    return SEEDS[:HUGE_REPLICATIONS] if size_class(n_vertices) == "Huge" else SEEDS


def load_original_instances() -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    instance_dir = project_root() / "third_party" / "Wildfire-Suppression" / "instances"
    for archive_path in sorted(instance_dir.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json"):
                    continue
                data = json.loads(archive.read(member).decode("utf-8"))
                identifier = Path(member).stem
                records.setdefault(
                    identifier,
                    {
                        "instance": identifier,
                        "collection": "original_55",
                        "source": f"{archive_path.name}:{member}",
                        "data": data,
                    },
                )
    if len(records) != 55:
        raise ValueError(f"Expected 55 original instances; found {len(records)}")
    return [records[key] for key in sorted(records)]


def load_supplementary_instances() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in accepted_rows():
        path = raw_instance_path(row)
        records.append(
            {
                "instance": row["best_exact_candidate_id"],
                "collection": "supplementary_19",
                "source": str(path.relative_to(project_root())),
                "data": json.loads(path.read_text(encoding="utf-8")),
            }
        )
    if len(records) != 19:
        raise ValueError(f"Expected 19 supplementary instances; found {len(records)}")
    return sorted(records, key=lambda row: row["instance"])


def load_all_instances() -> list[dict[str, Any]]:
    records = load_original_instances() + load_supplementary_instances()
    hashes: dict[str, str] = {}
    for record in records:
        repaired, audit = validate_and_repair_native(record["data"])
        record["data"] = repaired
        record["audit"] = audit
        record["instance_sha256"] = content_sha256(repaired)
        if record["instance"] in hashes:
            raise ValueError(f"Duplicate instance identifier: {record['instance']}")
        hashes[record["instance"]] = record["instance_sha256"]
    if len(records) != 74 or len(set(hashes.values())) != 74:
        raise ValueError("The final collection must contain 74 distinct identifiers and contents")
    return records


def git_state() -> dict[str, Any]:
    root = project_root().parent
    def output(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()
    return {
        "head": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "status": output("status", "--porcelain=v1").splitlines(),
    }


def source_manifest() -> dict[str, Any]:
    root = project_root()
    files = sorted((root / "analysis_wsp" / "benchmark74").glob("*.py"))
    files += sorted((root / "analysis_wsp" / "benchmark74").glob("*.template"))
    files += sorted((root / "analysis_wsp" / "caminho_c").glob("*.py"))
    for relative in (
        "third_party/Wildfire-Suppression/src",
        "third_party/Iterated-Beam-Search/src",
        "third_party/Logic-based-Benders-decomposition-for-wildfire-suppression",
    ):
        base = root / relative
        files += sorted(path for path in base.rglob("*") if path.is_file() and path.suffix in {".py", ".cpp", ".hpp"})
    rows = [{"path": str(path.relative_to(root)), "sha256": file_sha256(path)} for path in sorted(set(files))]
    return {"files": rows, "bundle_sha256": content_sha256(rows)}


def machine_metadata() -> dict[str, Any]:
    meminfo: dict[str, str] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[key] = value.strip()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "memory": meminfo,
        "logical_cpus": os.cpu_count(),
        "physical_core_slots": physical_core_slots(),
        "load_average": os.getloadavg(),
        "captured_utc": utc_now(),
    }


def runtime_manifest(license_path: Path) -> dict[str, Any]:
    root = project_root()
    isa_python = root / ".venv-isa" / "bin" / "python"
    gurobi_python = root / ".venv-gurobi" / "bin" / "python"
    binaries = {
        "unified_fire": root / "third_party" / "Wildfire-Suppression" / "build" / "fire",
        "ibs_stn_fire": root / "third_party" / "Iterated-Beam-Search" / "build" / "fire",
    }
    missing = [str(path) for path in (isa_python, gurobi_python, *binaries.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Runtime artifacts are absent: {missing}")

    gurobi_home = discover_gurobi_home()
    runtime_environment = os.environ.copy()
    runtime_environment.update(
        {
            "GUROBI_HOME": str(gurobi_home),
            "GRB_LICENSE_FILE": str(license_path),
            "LD_LIBRARY_PATH": f"{gurobi_home / 'lib'}:{runtime_environment.get('LD_LIBRARY_PATH', '')}",
        }
    )

    def capture(command: list[str]) -> str:
        completed = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True, env=runtime_environment)
        return (completed.stdout or completed.stderr).strip()

    return {
        "captured_utc": utc_now(),
        "machine": machine_metadata(),
        "git": git_state(),
        "license": validate_license_path(license_path),
        "gurobi_home": str(gurobi_home),
        "gurobi_version": capture([str(gurobi_python), "-c", "import gurobipy as gp; print(gp.gurobi.version())"]),
        "python_versions": {
            "isa": capture([str(isa_python), "--version"]),
            "gurobi": capture([str(gurobi_python), "--version"]),
        },
        "pip_freeze": {
            "isa": capture([str(isa_python), "-m", "pip", "freeze"]).splitlines(),
            "gurobi": capture([str(gurobi_python), "-m", "pip", "freeze"]).splitlines(),
        },
        "binaries": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in binaries.items()},
    }


def build_protocol(*, profile: str = PROFILE, shard_count: int = SHARD_COUNT, official: bool = False) -> dict[str, Any]:
    generation = profile_generation(profile)
    sources = source_manifest()
    protocol = {
        "profile": profile,
        "scope": "six algorithms measured on all 74 instances; IBS trajectories on the 19 supplementary instances",
        "collections": {"original": 55, "supplementary": 19, "total": 74},
        "algorithms": list(ALGORITHMS),
        "benchmark_replications": {"Small": 10, "Medium": 10, "Large": 10, "Huge": 3},
        "trajectory_replications": {"algorithm": "IBS", "supplementary_instances": 19, "seeds": list(SEEDS)},
        "time_limits_seconds": {"Small": 600, "Medium": 1350, "Large": 2400, "Huge": 9600},
        "time_limit_semantics": (
            "wall-clock limit checked between complete algorithm operations; ILS Huge may finish the current incumbent construction after the nominal limit, with up to 3600 seconds of external safety grace"
            if generation == 1
            else "strict monotonic ILS deadline: incomplete candidates and work evaluated after the deadline are discarded; an empty feasible incumbent is always available; external grace is serialization/watchdog time only"
        ),
        "ibs_parameters": IBS_PARAMETERS,
        "ils_parameters": ILS_PARAMETERS,
        "mip_parameters": {"model": "basic", "threads": 1, "gurobi_seed_matches_run_seed": True},
        "lbbd_parameters": {"threads": 1, "greedy_warm_start": True, "gurobi_seed_matches_run_seed": True},
        "cuth_parameters": {"epsilon": 0.01, "solver_time_seconds": 300, "threads": 1},
        "random_search_parameters": {"threads": 1},
        "shard_count": shard_count,
        "sharding": "deterministic longest-processing-time balance independently within each algorithm/phase",
        "hardware_policy": "official shards run only on the two identical LAB machines; local notebook artifacts are pilots and cannot be merged",
        "resume_policy": "reuse only completed results with exact protocol, source bundle, instance hash, command and shard",
        "performance_metric": "relative gap to proven optimum when available, otherwise final BKV computed across measured runs",
        "matilda_policy": "refit PRELIM, SIFTED, PILOT, PYTHIA and footprints on the complete measured 74x6 matrix; preserve 55-instance analysis as baseline",
        "official": official,
        "deployment_git": git_state(),
        "source_manifest": sources,
    }
    if generation == 2:
        protocol["amendment"] = {
            "reason": "Correct ILS deadline bias while preserving byte-identical completed IBS/RS v1 measurements",
            "carry_forward_algorithms": ["ibs", "rs"],
            "excluded_v1_algorithms": ["ils", "cuth", "mip", "lbbd", "ibs_stn"],
            "ils_repetition_policy": "rerun_all_ils",
        }
        if profile == PROFILE_V2_1 or profile.endswith("_v2_1"):
            protocol["amendment"] = {
                "reason": "Enforce an explicit final-feasibility postcondition without changing the strict-deadline ILS search",
                "parent_profile": PROFILE_V2,
                "carry_forward_algorithms": ["ibs", "rs", "ils", "cuth", "mip", "lbbd", "ibs_stn"],
                "carry_forward_policy": "preserve every valid completed parent result byte for byte, regardless of algorithm",
                "ils_repetition_policy": "preserve_prior_valid_runs_and_continue_pending_runs",
                "recognized_prior_invalid_policy": "record_algorithm_failure_without_rerun",
                "final_feasibility_policy": {
                    "valid_output": "accept unchanged",
                    "invalid_output": "record algorithm_failure, discard the candidate, and score the deterministic no-deployment feasible baseline",
                    "search_algorithm_changed": False,
                    "prior_valid_results": "eligible for byte-identical carry-forward because this postcondition leaves them unchanged",
                },
            }
        elif profile == PROFILE_V2_2 or profile.endswith("_v2_2"):
            protocol["amendment"] = {
                "reason": "Enforce one global wall-clock budget across every LBBD heuristic and exact optimize call",
                "parent_profile": PROFILE_V2_1,
                "carry_forward_policy": "preserve every valid parent result and prior algorithm failure; rerun no completed LBBD because neither lab completed one",
                "lbbd_budget_policy": "divide the current global time remaining among the remaining greedy stages; exact solve receives only time left after greedy",
                "lbbd_repetition_policy": "run_all_lbbd_from_scratch",
            }
        elif profile == PROFILE_V2_3 or profile.endswith("_v2_3"):
            protocol["amendment"] = {
                "reason": "Guarantee a deterministic feasible LBBD incumbent and never access solution attributes when SolCount is zero",
                "parent_profile": PROFILE_V2_2,
                "carry_forward_policy": "preserve all valid parent results and audited algorithm failures; discard failed/partial LBBD attempts",
                "lbbd_budget_policy": "one global wall-clock budget divided among remaining greedy stages; exact receives only leftover time",
                "lbbd_no_incumbent_policy": "inject no-deployment MIP start; if no incumbent is returned, emit the validated no-deployment baseline instead of failing",
            }
        elif profile == PROFILE_V2_4 or profile.endswith("_v2_4"):
            protocol["amendment"] = {
                "reason": "Score every feasible LBBD deployment by deterministic independent recomputation, while retaining the Gurobi master objective for audit",
                "parent_profile": PROFILE_V2_3,
                "carry_forward_policy": "preserve all valid parent results and audited algorithm failures; discard failed/partial LBBD attempts",
                "lbbd_budget_policy": "one global wall-clock budget divided among remaining greedy stages; exact receives only leftover time",
                "lbbd_no_incumbent_policy": "inject no-deployment MIP start; if no incumbent is returned, emit the validated no-deployment baseline instead of failing",
                "lbbd_objective_policy": "validated recomputation is canonical for scoring; model objective, delta, and model-reported gap are retained as audit telemetry",
            }
    scientific_payload = {
        key: value
        for key, value in protocol.items()
        if key not in {"profile", "official", "deployment_git"}
    }
    protocol["protocol_sha256"] = content_sha256(scientific_payload)
    return protocol


def balanced_shards(jobs: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    loads = [0] * count
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault((job["phase"], job["algorithm"]), []).append(job)
    assigned: list[dict[str, Any]] = []
    for key in sorted(grouped):
        phase_loads = [0] * count
        for job in sorted(grouped[key], key=lambda row: (-int(row["time_limit_seconds"]), row["instance"], int(row["seed"]))):
            shard = min(range(count), key=lambda index: (phase_loads[index], loads[index], index))
            copy = dict(job)
            copy["shard_index"] = shard
            assigned.append(copy)
            phase_loads[shard] += int(job["time_limit_seconds"])
            loads[shard] += int(job["time_limit_seconds"])
    return sorted(assigned, key=lambda row: (row["phase"], row["algorithm"], row["shard_index"], -int(row["time_limit_seconds"]), row["instance"], int(row["seed"])))


def validate_license_path(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Gurobi license not found: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError("gurobi.lic must not be readable by group or others")
    return {"path": str(path), "sha256": file_sha256(path), "mode": oct(mode), "size_bytes": path.stat().st_size}


def discover_gurobi_home() -> Path:
    configured = os.environ.get("GUROBI_HOME")
    candidates = [Path(configured)] if configured else []
    candidates += sorted((Path.home() / ".local" / "opt").glob("gurobi*/linux64"), reverse=True)
    candidates += sorted(Path("/opt").glob("gurobi*/linux64"), reverse=True)
    for candidate in candidates:
        if (candidate / "lib").is_dir() and (candidate / "include" / "gurobi_c++.h").is_file():
            return candidate.resolve()
    raise FileNotFoundError("GUROBI_HOME was not configured and no Gurobi linux64 installation was found")
