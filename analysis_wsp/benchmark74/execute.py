from __future__ import annotations

import ast
import csv
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil

from .common import (
    IBS_PARAMETERS,
    atomic_write_json,
    discover_gurobi_home,
    file_sha256,
    physical_core_slots,
    project_root,
    read_csv,
    result_root,
    source_manifest,
    thermal_snapshot,
    utc_now,
    validate_license_path,
)


PHASE_ORDER = (
    ("benchmark", "ibs"),
    ("benchmark", "rs"),
    ("benchmark", "ils"),
    ("benchmark", "cuth"),
    ("benchmark", "mip"),
    ("benchmark", "lbbd"),
    ("trajectory", "ibs_stn"),
)


def effective_time_limit(job: dict[str, Any], pilot_limit: int | None) -> int:
    official_limit = int(job["time_limit_seconds"])
    if pilot_limit is None:
        return official_limit
    return min(official_limit, pilot_limit)


def operational_guard_seconds(job: dict[str, Any], effective_limit: int) -> int:
    grace = max(300, int(effective_limit * 0.15))
    # This is watchdog/serialization grace only. Strict FireILS instrumentation
    # prevents any work completed after its scientific deadline from improving
    # the incumbent.
    return effective_limit + grace


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def run_directory(profile: str, job: dict[str, Any]) -> Path:
    return result_root(profile) / "runs" / job["phase"] / job["algorithm"] / job["instance"] / f"seed_{int(job['seed']):02d}"


def valid_completed(
    profile: str,
    job: dict[str, Any],
    protocol: dict[str, Any],
    pilot_limit: int | None = None,
    *,
    accept_any_mode: bool = False,
) -> bool:
    from .amend import valid_carried_forward

    if valid_carried_forward(profile, job, protocol):
        return True
    directory = run_directory(profile, job)
    result = load_json(directory / "result.json")
    manifest = load_json(directory / "run_manifest.json")
    if not result or result.get("status") != "completed" or not manifest:
        return False
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        return False
    if result.get("instance_sha256") != job["instance_sha256"]:
        return False
    if int(result.get("seed", -1)) != int(job["seed"]) or result.get("algorithm") != job["algorithm"]:
        return False
    if int(manifest.get("shard_index", -1)) != int(job["shard_index"]):
        return False
    if manifest.get("job") != job:
        return False
    is_pilot = bool(result.get("pilot"))
    if protocol.get("official"):
        if is_pilot:
            return False
        expected_limit = effective_time_limit(job, None)
        if int(manifest.get("effective_time_limit_seconds", -1)) != expected_limit:
            return False
        if manifest.get("pilot_limit_seconds") is not None:
            return False
    elif not accept_any_mode:
        if is_pilot != (pilot_limit is not None):
            return False
        expected_limit = effective_time_limit(job, pilot_limit)
        if int(manifest.get("effective_time_limit_seconds", -1)) != expected_limit:
            return False
        if manifest.get("pilot_limit_seconds") != pilot_limit:
            return False
    if protocol.get("official") or not accept_any_mode:
        directory = run_directory(profile, job)
        if job["engine"] == "unified_cpp":
            expected_command = unified_command(job, directory, expected_limit)
        elif job["engine"] == "ibs_stn_cpp":
            expected_command = stn_command(job, directory, expected_limit)
        else:
            expected_command = legacy_command(job, directory, expected_limit)
        if manifest.get("command", [])[3:] != expected_command:
            return False
    objective = result.get("objective")
    return isinstance(objective, (int, float)) and 0 <= float(objective) <= int(job["n_vertices"])


def valid_algorithm_failure(profile: str, job: dict[str, Any], protocol: dict[str, Any]) -> bool:
    failure = load_json(run_directory(profile, job) / "algorithm_failure.json")
    if not failure or failure.get("status") != "algorithm_failure":
        return False
    if failure.get("protocol_sha256") != protocol.get("protocol_sha256"):
        return False
    if failure.get("instance_sha256") != job.get("instance_sha256"):
        return False
    if failure.get("algorithm") != job.get("algorithm") or int(failure.get("seed", -1)) != int(job["seed"]):
        return False
    if int(failure.get("shard_index", -1)) != int(job["shard_index"]):
        return False
    if failure.get("failure_category") != "infeasible_final_solution":
        return False
    score = failure.get("scored_objective")
    return isinstance(score, (int, float)) and 0 <= float(score) <= int(job["n_vertices"])


def quarantine(profile: str, job: dict[str, Any]) -> None:
    directory = run_directory(profile, job)
    if not directory.exists() or not any(directory.iterdir()):
        return
    target = result_root(profile) / "_discarded_attempts" / f"{job['job_id']}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(target))


def unified_command(job: dict[str, Any], directory: Path, limit: int) -> list[str]:
    binary = project_root() / "third_party" / "Wildfire-Suppression" / "build" / "fire"
    command = [
        str(binary),
        "--instance", job["native_path"],
        "--alg", job["algorithm"],
        "--seed", str(job["seed"]),
        "--timelimit", str(limit),
        "--output", str((directory / "statistics.csv").resolve()),
        "--save_solution",
    ]
    if job["algorithm"] == "ibs":
        command += [
            "--ibsBeta", str(IBS_PARAMETERS["beta"]),
            "--ibsEta", str(IBS_PARAMETERS["eta"]),
            "--ibsC", str(IBS_PARAMETERS["c"]),
            "--ibsP", str(IBS_PARAMETERS["p"]),
            "--ibsPhat", str(IBS_PARAMETERS["phat"]),
            "--ibsZmax", str(IBS_PARAMETERS["zmax"]),
        ]
    elif job["algorithm"] == "cuth":
        command += ["--cuthEpsilon", "0.01", "--cuthSolverTime", str(min(300, limit))]
    elif job["algorithm"] == "mip":
        command += ["--mipModel", "basic"]
    if "--maxiter" in command or "--ibsMaxBudget" in command:
        raise AssertionError("The benchmark command must be time-limited only")
    return command


def stn_command(job: dict[str, Any], directory: Path, limit: int) -> list[str]:
    binary = project_root() / "third_party" / "Iterated-Beam-Search" / "build" / "fire"
    return [
        str(binary), "--instance", job["harris_path"], "--seed", str(job["seed"]),
        "--beta", str(IBS_PARAMETERS["beta"]), "--eta", str(IBS_PARAMETERS["eta"]),
        "--c", str(IBS_PARAMETERS["c"]), "--p", str(IBS_PARAMETERS["p"]),
        "--phat", str(IBS_PARAMETERS["phat"]), "--zmax", str(IBS_PARAMETERS["zmax"]),
        "--timelimit", str(limit), "--stn_output", str((directory / "trajectory_raw.csv").resolve()),
        "--stn_run", str(job["seed"]),
    ]


def legacy_command(job: dict[str, Any], directory: Path, limit: int) -> list[str]:
    python = project_root() / ".venv-gurobi" / "bin" / "python"
    return [
        str(python), "-m", "analysis_wsp.benchmark74.legacy_worker",
        "--algorithm", job["algorithm"], "--instance", job["harris_path"],
        "--seed", str(job["seed"]), "--timelimit", str(limit),
        "--output", str((directory / "legacy_result.json").resolve()),
    ]


def parse_unified(job: dict[str, Any], directory: Path) -> dict[str, Any]:
    with (directory / "statistics.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one statistics row; found {len(rows)}")
    row = rows[0]
    # For MIP, column 2 is the model's reported objective, while column 5 is
    # B.burnt_vertices after the saved solution has been updated and checked.
    # They differ in one archived run; quality analysis must use the latter.
    if job["algorithm"] == "ibs":
        objective_index = 8
    elif job["algorithm"] == "mip":
        objective_index = 5
    else:
        objective_index = 2
    if len(row) <= objective_index or row[objective_index] in {"", "-"}:
        raise ValueError("The unified solver did not report a feasible objective")
    objective = int(round(float(row[objective_index])))
    lower_bound = None
    if job["algorithm"] == "mip" and len(row) > 6 and row[6] not in {"", "-"}:
        lower_bound = float(row[6])
    trajectories: Any = None
    if row and row[-1].startswith("["):
        trajectories = [list(item) for item in ast.literal_eval(row[-1])]
    solutions = sorted(directory.glob(f"sol_{job['algorithm']}_*.json"))
    if not solutions:
        raise ValueError("The unified solver did not persist a feasible solution")
    with solutions[0].open(encoding="utf-8") as handle:
        saved_objective = float(json.load(handle)["objv"])
    if saved_objective != objective:
        raise ValueError(f"Statistics objective {objective} differs from saved solution {saved_objective}")
    result = {
        "objective": objective,
        "lower_bound": lower_bound,
        "statistics_fields": row,
        "improvement_trajectory": trajectories,
        "solution_file": solutions[0].name,
        "solution_sha256": file_sha256(solutions[0]),
    }
    if job["algorithm"] == "mip":
        result["model_objective"] = float(row[2])
    return result


def parse_stn(job: dict[str, Any], directory: Path) -> dict[str, Any]:
    stdout = directory / "stdout.log"
    with stdout.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) != 1 or len(rows[0]) != 15:
        raise ValueError("Unexpected IBS-STN statistics output")
    row = rows[0]
    objective = int(float(row[8]))
    trajectory = directory / "trajectory_raw.csv"
    if not trajectory.is_file() or trajectory.stat().st_size <= 25:
        raise ValueError("IBS-STN raw trajectory is absent or empty")
    return {
        "objective": objective,
        "lower_bound": None,
        "statistics_fields": row,
        "trajectory_raw_sha256": file_sha256(trajectory),
    }


def execute_one(
    profile: str,
    job: dict[str, Any],
    protocol: dict[str, Any],
    slot: dict[str, Any],
    stop: threading.Event,
    license_path: Path,
    pilot_limit: int | None,
) -> dict[str, Any]:
    if valid_completed(profile, job, protocol, pilot_limit):
        return load_json(run_directory(profile, job) / "result.json") or {}
    if valid_algorithm_failure(profile, job, protocol):
        return load_json(run_directory(profile, job) / "algorithm_failure.json") or {}
    quarantine(profile, job)
    directory = run_directory(profile, job)
    directory.mkdir(parents=True, exist_ok=True)
    limit = effective_time_limit(job, pilot_limit)
    if job["engine"] == "unified_cpp":
        command = unified_command(job, directory, limit)
    elif job["engine"] == "ibs_stn_cpp":
        command = stn_command(job, directory, limit)
    else:
        command = legacy_command(job, directory, limit)
    command = ["taskset", "--cpu-list", str(slot["logical_cpu"]), *command]
    manifest = {
        "job": job,
        "command": command,
        "worker": slot,
        "shard_index": int(job["shard_index"]),
        "pilot_limit_seconds": pilot_limit,
        "effective_time_limit_seconds": limit,
        "started_utc": utc_now(),
        "thermal_before": thermal_snapshot(),
    }
    atomic_write_json(directory / "run_manifest.json", manifest)
    environment = os.environ.copy()
    gurobi_home = discover_gurobi_home()
    environment.update(
        {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "GRB_LICENSE_FILE": str(license_path),
            "GUROBI_HOME": str(gurobi_home),
            "LD_LIBRARY_PATH": f"{gurobi_home / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}",
            "PYTHONPATH": f"{project_root()}:{environment.get('PYTHONPATH', '')}",
        }
    )
    started = time.monotonic()
    peak_rss = 0
    interrupted = False
    with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=directory, stdout=stdout, stderr=stderr, text=True, env=environment, start_new_session=True)
        child = psutil.Process(process.pid)
        guard = started + operational_guard_seconds(job, limit)
        while process.poll() is None:
            if stop.wait(0.5):
                interrupted = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            try:
                rss = child.memory_info().rss + sum(grandchild.memory_info().rss for grandchild in child.children(recursive=True))
                peak_rss = max(peak_rss, rss)
            except (psutil.Error, ProcessLookupError):
                pass
            if time.monotonic() > guard:
                os.killpg(process.pid, signal.SIGTERM)
                raise TimeoutError("Operational guard exceeded")
        returncode = process.wait(timeout=30)
    if interrupted:
        raise InterruptedError("Campaign stop requested")
    if returncode != 0:
        raise RuntimeError(f"Process exited with code {returncode}")
    if job["engine"] == "unified_cpp":
        parsed = parse_unified(job, directory)
    elif job["engine"] == "ibs_stn_cpp":
        parsed = parse_stn(job, directory)
    else:
        parsed = load_json(directory / "legacy_result.json") or {}
        if parsed.get("status") == "algorithm_failure":
            failure = {
                **job,
                **parsed,
                "algorithm": job["algorithm"],
                "seed": int(job["seed"]),
                "status": "algorithm_failure",
                "protocol_sha256": protocol["protocol_sha256"],
                "instance_sha256": job["instance_sha256"],
                "shard_index": int(job["shard_index"]),
                "peak_rss_bytes": peak_rss,
                "worker": slot,
                "pilot": pilot_limit is not None,
                "stdout_sha256": file_sha256(directory / "stdout.log"),
                "stderr_sha256": file_sha256(directory / "stderr.log"),
                "thermal_after": thermal_snapshot(),
                "recorded_utc": utc_now(),
            }
            atomic_write_json(directory / "algorithm_failure.json", failure)
            return failure
        if parsed.get("status") != "completed":
            raise ValueError("Legacy worker did not produce a completed result")
    result = {
        **job,
        **parsed,
        "algorithm": job["algorithm"],
        "seed": int(job["seed"]),
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "instance_sha256": job["instance_sha256"],
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "worker": slot,
        "pilot": pilot_limit is not None,
        "stdout_sha256": file_sha256(directory / "stdout.log"),
        "stderr_sha256": file_sha256(directory / "stderr.log"),
        "thermal_after": thermal_snapshot(),
        "completed_utc": utc_now(),
        "result_provenance": "pilot" if pilot_limit is not None else "measured",
    }
    atomic_write_json(directory / "result.json", result)
    return result


class PhaseCampaign:
    def __init__(self, profile: str, jobs: list[dict[str, Any]], protocol: dict[str, Any], workers: int, license_path: Path, pilot_limit: int | None) -> None:
        self.profile = profile
        self.jobs = jobs
        self.protocol = protocol
        self.slots = physical_core_slots()[:workers]
        self.license_path = license_path
        self.pilot_limit = pilot_limit
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.running: dict[int, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []

    def worker(self, slot: dict[str, Any], work: queue.Queue[dict[str, Any]]) -> None:
        while not self.stop.is_set():
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            with self.lock:
                self.running[int(slot["worker_id"])] = {"job_id": job["job_id"], "instance": job["instance"], "seed": job["seed"], "started_utc": utc_now()}
            try:
                execute_one(self.profile, job, self.protocol, slot, self.stop, self.license_path, self.pilot_limit)
            except InterruptedError:
                pass
            except Exception as exc:
                failure = {**job, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "failed_utc": utc_now()}
                atomic_write_json(run_directory(self.profile, job) / "failure.json", failure)
                with self.lock:
                    self.errors.append(failure)
                self.stop.set()
            finally:
                with self.lock:
                    self.running.pop(int(slot["worker_id"]), None)
                work.task_done()

    def run(self) -> None:
        work: queue.Queue[dict[str, Any]] = queue.Queue()
        for job in self.jobs:
            if not valid_completed(self.profile, job, self.protocol, self.pilot_limit) and not valid_algorithm_failure(self.profile, job, self.protocol):
                work.put(job)
        threads = [threading.Thread(target=self.worker, args=(slot, work), name=f"worker-{slot['worker_id']}") for slot in self.slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if self.errors:
            raise RuntimeError(f"Phase failed: {self.errors[0]['error']}")


def status(profile: str, shard_index: int) -> dict[str, Any]:
    output = result_root(profile)
    protocol = load_json(output / "protocol.json") or {}
    jobs = [job for job in read_csv(output / "jobs.csv") if int(job["shard_index"]) == shard_index]
    counts: Counter[str] = Counter()
    by_algorithm: dict[str, Counter[str]] = {}
    for job in jobs:
        if valid_completed(profile, job, protocol, accept_any_mode=True):
            state = "completed"
        elif valid_algorithm_failure(profile, job, protocol):
            state = "algorithm_failure"
        else:
            state = "failed" if (run_directory(profile, job) / "failure.json").is_file() else "pending"
        counts[state] += 1
        by_algorithm.setdefault(job["algorithm"], Counter())[state] += 1
    value = {
        "profile": profile,
        "shard_index": shard_index,
        "total": len(jobs),
        **dict(counts),
        "by_algorithm": {key: dict(value) for key, value in sorted(by_algorithm.items())},
        "updated_utc": utc_now(),
    }
    atomic_write_json(output / f"execution_status_shard_{shard_index}.json", value)
    return value


def execute(*, profile: str, shard_index: int, workers: int, solver_workers: int, license_path: Path, resume: bool, pilot_limit: int | None) -> dict[str, Any]:
    output = result_root(profile)
    protocol = load_json(output / "protocol.json")
    if not protocol:
        raise FileNotFoundError("protocol.json is absent; run prepare first")
    current_sources = source_manifest()
    if protocol.get("source_manifest", {}).get("bundle_sha256") != current_sources["bundle_sha256"]:
        raise RuntimeError("Current source bundle differs from prepared protocol; prepare a new profile before executing")
    current_license = validate_license_path(license_path)
    if protocol.get("license", {}).get("sha256") != current_license["sha256"]:
        raise RuntimeError("Gurobi license differs from the license recorded during preparation")
    environment_manifest = load_json(output / "environment_manifest.json")
    if not environment_manifest:
        raise FileNotFoundError("environment_manifest.json is absent; run prepare first")
    for binary in environment_manifest.get("binaries", {}).values():
        path = Path(binary["path"])
        if not path.is_file() or file_sha256(path) != binary["sha256"]:
            raise RuntimeError(f"Runtime binary differs from the prepared environment: {path}")
    if not resume and any((output / "runs").rglob("result.json")):
        raise RuntimeError("Run artifacts exist; pass --resume")
    jobs = [job for job in read_csv(output / "jobs.csv") if int(job["shard_index"]) == shard_index]
    if pilot_limit is not None:
        representatives: dict[tuple[str, str, str], dict[str, Any]] = {}
        for job in jobs:
            key = (job["phase"], job["algorithm"], job["size_class"])
            representatives.setdefault(key, job)
        jobs = list(representatives.values())
    for phase, algorithm in PHASE_ORDER:
        selected = [job for job in jobs if job["phase"] == phase and job["algorithm"] == algorithm]
        if not selected:
            continue
        phase_workers = solver_workers if algorithm in {"cuth", "mip", "lbbd"} else workers
        PhaseCampaign(profile, selected, protocol, phase_workers, license_path, pilot_limit).run()
        status(profile, shard_index)
    return status(profile, shard_index)
