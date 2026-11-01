from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


PROFILE = "caminho_c_v1"
BASE_REMOTE_REVISION = "ae45999cabef7873805064563c8b49bc79fcaa9f"
EXPECTED_BRANCH = "feat/caminho-C"
TIME_SECONDS_PER_VERTEX = 1.5
SEEDS = tuple(range(1, 11))
IBS_PARAMETERS = {
    "beta": 50,
    "eta": 70,
    "c": 30,
    "p": 0.5,
    "phat": 1.0,
    "zmax": 3,
    "termination": "wall_time_only",
    "objective": "minimize_burned_vertices",
    "maxiter_explicit": False,
    "budget_explicit": False,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def result_root() -> Path:
    return project_root() / "analysis_wsp" / "results_caminho_c" / PROFILE


def accepted_table() -> Path:
    return project_root() / "results" / "generation" / "04a_ga_internal" / "ga_target_results.csv"


def accepted_rows() -> list[dict[str, str]]:
    with accepted_table().open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("accepted", "").strip().lower() == "true"]
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row["best_exact_candidate_id"]
        expected_hash = row["best_exact_instance_sha256"]
        if identifier in unique and unique[identifier]["best_exact_instance_sha256"] != expected_hash:
            raise ValueError(f"Accepted identifier with conflicting hashes: {identifier}")
        unique[identifier] = row
    if len(unique) != 19:
        raise ValueError(f"Expected 19 distinct accepted instances; found {len(unique)}")
    return sorted(unique.values(), key=lambda row: (int(row["root_seed"]), row["target_id"]))


def raw_instance_path(row: dict[str, str]) -> Path:
    return (
        accepted_table().parent
        / f"seed_{int(row['root_seed'])}"
        / "generated_candidates"
        / f"{row['best_exact_candidate_id']}.instance.json"
    )


def exact_candidate_path(row: dict[str, str]) -> Path:
    return raw_instance_path(row).with_name(f"{row['best_exact_candidate_id']}_exact.json")


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


def normalize_number(value: Any) -> int | float:
    number = float(value)
    return int(round(number)) if math.isclose(number, round(number), abs_tol=1e-12) else number


def validate_and_repair_native(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"H", "|V|", "|R|", "I", "t", "c", "delta", "arcs", "distance"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Missing WSP fields: {missing}")
    repaired = json.loads(json.dumps(data))
    lengths = {name: len(repaired[name]) for name in ("t", "c", "delta")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Misaligned resource vectors: {lengths}")
    actual_resources = lengths["t"]
    declared_resources = int(repaired["|R|"])
    repaired["|R|"] = actual_resources
    n = int(repaired["|V|"])
    coordinates = repaired["distance"]["coordinates"]
    if n != len(coordinates):
        raise ValueError(f"|V|={n}, but there are {len(coordinates)} coordinates")
    if len({tuple(map(float, coordinate[:2])) for coordinate in coordinates}) != n:
        raise ValueError("Two-dimensional vertex coordinates are not unique")
    if not repaired["I"] or any(not 0 <= int(vertex) < n for vertex in repaired["I"]):
        raise ValueError("Invalid ignition set")
    seen_arcs: set[tuple[int, int]] = set()
    for raw_u, raw_v, raw_weight in repaired["arcs"]:
        u, v, weight = int(raw_u), int(raw_v), float(raw_weight)
        if not 0 <= u < n or not 0 <= v < n or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid arc: {(raw_u, raw_v, raw_weight)}")
        if (u, v) in seen_arcs:
            raise ValueError(f"Duplicate arc: {(u, v)}")
        seen_arcs.add((u, v))
    if len(set(map(float, repaired["delta"]))) != 1:
        raise ValueError("The Harris/IBS adapter requires uniform delta within an instance")
    audit = {
        "declared_resource_points_raw": declared_resources,
        "serialized_resource_points": actual_resources,
        "resource_count_repaired": declared_resources != actual_resources,
        "n_vertices": n,
        "n_arcs": len(repaired["arcs"]),
        "horizon": float(repaired["H"]),
        "total_resources": int(sum(map(int, repaired["c"]))),
    }
    return repaired, audit


def native_to_harris(data: dict[str, Any]) -> dict[str, Any]:
    physical = [tuple(normalize_number(value) for value in row[:2]) for row in data["distance"]["coordinates"]]

    def unit_axis(values: set[int | float]) -> dict[int | float, int | float]:
        ordered = sorted(values)
        positive_steps = [float(right) - float(left) for left, right in zip(ordered, ordered[1:]) if right > left]
        step = min(positive_steps) if positive_steps else 1.0
        if math.isclose(step, 1.0, abs_tol=1e-12):
            return {value: value for value in ordered}
        if any(
            not math.isclose((float(value) - float(ordered[0])) / step, round((float(value) - float(ordered[0])) / step), abs_tol=1e-8)
            for value in ordered
        ):
            raise ValueError("Coordinates are not a regular grid compatible with IBS")
        return {value: index + 1 for index, value in enumerate(ordered)}

    x_rank = unit_axis({node[0] for node in physical})
    y_rank = unit_axis({node[1] for node in physical})
    coordinates = [(x_rank[node[0]], y_rank[node[1]]) for node in physical]
    arcs: dict[str, float] = {}
    for raw_u, raw_v, raw_weight in data["arcs"]:
        arcs[str((coordinates[int(raw_u)], coordinates[int(raw_v)]))] = float(raw_weight)
    resources: dict[str, int] = {}
    for raw_time, raw_capacity in zip(data["t"], data["c"]):
        key = str(normalize_number(raw_time))
        resources[key] = resources.get(key, 0) + int(raw_capacity)
    return {
        "Delay": normalize_number(data["delta"][0]),
        "ArrivalTimeTarget": normalize_number(data["H"]),
        "ResAtTime": resources,
        "Ignitions": [list(coordinates[int(vertex)]) for vertex in data["I"]],
        "Nodes": [list(node) for node in coordinates],
        "Arcs": arcs,
    }


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=project_root().parent, check=True, text=True, capture_output=True).stdout.strip()


def git_state() -> dict[str, Any]:
    status = git_output("status", "--porcelain=v1", "--branch")
    head = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    remote_contains = subprocess.run(
        ["git", "branch", "-r", "--contains", head], cwd=project_root().parent, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    return {
        "head": head,
        "branch": branch,
        "status_porcelain": status.splitlines(),
        "remote_refs_containing_head": [line.strip() for line in remote_contains],
        "base_revision_exists_on_remote": "origin/docs/tcc-escopo-essencial" in {line.strip() for line in remote_contains},
    }


def physical_core_slots() -> list[dict[str, Any]]:
    available = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    stable_locale = dict(os.environ)
    stable_locale.update({"LC_ALL": "C", "LANG": "C"})
    completed = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,MAXMHZ"],
        check=True,
        text=True,
        capture_output=True,
        env=stable_locale,
    )
    grouped: dict[tuple[int, int], list[tuple[int, float | None]]] = {}
    for line in completed.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        raw_cpu, raw_core, raw_socket, raw_mhz = line.split(",")
        cpu = int(raw_cpu)
        if cpu not in available:
            continue
        mhz = float(raw_mhz) if raw_mhz.strip() not in {"", "-"} else None
        if mhz is not None and (not math.isfinite(mhz) or mhz <= 0):
            mhz = None
        grouped.setdefault((int(raw_socket), int(raw_core)), []).append((cpu, mhz))
    frequencies_known = all(mhz is not None for siblings in grouped.values() for _cpu, mhz in siblings)
    maxima = sorted({round(mhz, -2) for siblings in grouped.values() for _cpu, mhz in siblings if mhz is not None})
    hybrid = frequencies_known and len(maxima) > 1 and max(maxima) - min(maxima) >= 500
    threshold = (max(maxima) + min(maxima)) / 2 if hybrid else 0.0
    slots: list[dict[str, Any]] = []
    for worker_id, ((socket, core), siblings) in enumerate(sorted(grouped.items())):
        siblings = sorted(siblings)
        max_mhz = max((mhz for _cpu, mhz in siblings if mhz is not None), default=None)
        slots.append(
            {
                "worker_id": worker_id,
                "logical_cpu": siblings[0][0],
                "sibling_logical_cpus": [cpu for cpu, _mhz in siblings],
                "socket": socket,
                "physical_core": core,
                "core_type": ("unknown" if not frequencies_known else
                              "P" if hybrid and max_mhz >= threshold else
                              "E" if hybrid else "homogeneous"),
                "max_mhz": max_mhz,
            }
        )
    if not slots:
        raise RuntimeError("No available physical CPU cores detected")
    return slots


def machine_metadata() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[key] = value.strip()
    configured_rscript = os.environ.get("TCC_RSCRIPT")
    rscript = configured_rscript if configured_rscript and Path(configured_rscript).is_file() else shutil_which("Rscript")
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": command("lscpu"),
        "numa": command("numactl", "--hardware") if shutil_which("numactl") else "numactl unavailable; lscpu reports one NUMA node",
        "memory": meminfo,
        "load_average": os.getloadavg(),
        "logical_cpus": os.cpu_count(),
        "physical_core_slots": physical_core_slots(),
        "python": platform.python_version(),
        "gcc": command("g++", "--version").splitlines()[0],
        "cmake": command("cmake", "--version").splitlines()[0],
        "rscript_path": rscript or "unavailable",
        "rscript": command(rscript, "--version") if rscript else "unavailable",
        "captured_utc": utc_now(),
    }


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_protocol(workers: int | None = None) -> dict[str, Any]:
    slots = physical_core_slots()
    selected = slots if workers is None else slots[:workers]
    if not selected or len(selected) != (workers or len(slots)):
        raise ValueError(f"Requested {workers} workers, but {len(slots)} physical cores are available")
    return {
        "profile": PROFILE,
        "scientific_scope": "IBS only on 19 supplementary instances; no supervised ISA refit",
        "instances": 19,
        "seeds": list(SEEDS),
        "runs": 19 * len(SEEDS),
        "ibs_parameters": IBS_PARAMETERS,
        "time_limit_rule": {"seconds_per_vertex": TIME_SECONDS_PER_VERTEX, "published_limits": {"400": 600, "900": 1350, "1600": 2400, "6400": 9600}},
        "execution_workers": len(selected),
        "affinity_policy": "one monothread IBS process per detected physical core; one SMT sibling selected on P-cores; dynamic queue balances heterogeneous P/E cores",
        "worker_slots": selected,
        "hardware_uniformity": "all 190 measured runs are executed on this machine with the same worker pool",
        "resume_policy": "reuse only completed run with valid result, raw trajectory, logs, and exact protocol hash; otherwise quarantine and restart",
        "base_remote_revision": BASE_REMOTE_REVISION,
        "branch": EXPECTED_BRANCH,
        "analysis_constraints": {
            "no_supervised_refit": True,
            "no_bkv_or_optimum_inference": True,
            "no_optimality_gap": True,
            "no_cross_algorithm_claims_on_supplementary_instances": True,
            "association_not_causation": True,
        },
    }


def protocol_hash(protocol: dict[str, Any]) -> str:
    return content_sha256(protocol)


def load_jobs() -> list[dict[str, str]]:
    with (result_root() / "jobs.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def thermal_snapshot() -> dict[str, Any]:
    temperatures: dict[str, float] = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            label = (zone / "type").read_text().strip()
            value = float((zone / "temp").read_text().strip()) / 1000.0
            temperatures[f"{zone.name}:{label}"] = value
        except (OSError, ValueError):
            continue
    frequencies: dict[str, float] = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        try:
            frequencies[cpu.name] = float((cpu / "cpufreq" / "scaling_cur_freq").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return {"utc": utc_now(), "temperatures_c": temperatures, "frequencies_mhz": frequencies}
