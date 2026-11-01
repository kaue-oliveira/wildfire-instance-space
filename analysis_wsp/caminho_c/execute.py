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
    file_sha256,
    load_jobs,
    physical_core_slots,
    project_root,
    protocol_hash,
    result_root,
    thermal_snapshot,
    utc_now,
    write_csv,
)


STDOUT_FIELDS = [
    "instance",
    "seed",
    "p",
    "phat",
    "beta",
    "eta",
    "zmax",
    "c",
    "objective",
    "time_to_best_seconds",
    "iteration_to_best",
    "total_iterations",
    "solver_elapsed_seconds",
    "budget",
    "improvement_trajectory",
]


def run_dir(job: dict[str, str]) -> Path:
    return result_root() / "runs" / job["instance"] / f"seed_{int(job['seed']):02d}"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def current_protocol() -> dict[str, Any]:
    protocol = load_json(result_root() / "protocol.json")
    if not protocol:
        raise FileNotFoundError("protocol.json is absent; run prepare first")
    return protocol


def valid_completed_run(job: dict[str, str], protocol: dict[str, Any]) -> bool:
    directory = run_dir(job)
    result = load_json(directory / "result.json")
    run_protocol = load_json(directory / "protocol.json")
    trajectory = directory / "trajectory_raw.csv"
    best = directory / "trajectory_best_so_far.csv"
    if not result or result.get("status") != "completed":
        return False
    if run_protocol != protocol or result.get("protocol_sha256") != protocol["protocol_sha256"]:
        return False
    required = [trajectory, best, directory / "stdout.log", directory / "stderr.log"]
    if any(not path.is_file() for path in required) or trajectory.stat().st_size <= 25:
        return False
    if int(result.get("seed", -1)) != int(job["seed"]) or result.get("instance") != job["instance"]:
        return False
    try:
        objective = float(result["objective"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= objective <= int(job["n_vertices"])


def quarantine_partial(job: dict[str, str]) -> None:
    directory = run_dir(job)
    if not directory.exists() or not any(directory.iterdir()):
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = result_root() / "_discarded_attempts" / f"{job['job_id']}_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(target))


def build_command(job: dict[str, str], slot: dict[str, Any], trajectory: Path) -> list[str]:
    binary = project_root() / "third_party" / "Iterated-Beam-Search" / "build" / "fire"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"IBS Release binary is absent: {binary}")
    command = [
        "taskset",
        "--cpu-list",
        str(slot["logical_cpu"]),
        str(binary),
        "--instance",
        job["ibs_instance"],
        "--seed",
        job["seed"],
        "--beta",
        str(IBS_PARAMETERS["beta"]),
        "--eta",
        str(IBS_PARAMETERS["eta"]),
        "--c",
        str(IBS_PARAMETERS["c"]),
        "--p",
        str(IBS_PARAMETERS["p"]),
        "--phat",
        str(IBS_PARAMETERS["phat"]),
        "--zmax",
        str(IBS_PARAMETERS["zmax"]),
        "--timelimit",
        job["time_limit_seconds"],
        "--stn_output",
        str(trajectory),
        "--stn_run",
        job["seed"],
    ]
    if "--maxiter" in command or "--budget" in command:
        raise AssertionError("Scientific command must not set maxiter or evaluation budget")
    return command


def parse_stdout(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) != 1 or len(rows[0]) != len(STDOUT_FIELDS):
        raise ValueError(f"IBS stdout must contain one {len(STDOUT_FIELDS)}-field CSV row; found {len(rows)} rows")
    raw = dict(zip(STDOUT_FIELDS, rows[0]))
    integers = {"seed", "beta", "eta", "zmax", "c", "objective", "time_to_best_seconds", "iteration_to_best", "total_iterations", "solver_elapsed_seconds", "budget"}
    floats = {"p", "phat"}
    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if key in integers:
            parsed[key] = int(float(value))
        elif key in floats:
            parsed[key] = float(value)
        elif key == "improvement_trajectory":
            parsed[key] = [list(item) for item in ast.literal_eval(value)]
        else:
            parsed[key] = value
    expected = IBS_PARAMETERS
    for key in ("beta", "eta", "c", "zmax"):
        if parsed[key] != expected[key]:
            raise ValueError(f"IBS reported {key}={parsed[key]}, expected {expected[key]}")
    for key in ("p", "phat"):
        if abs(parsed[key] - expected[key]) > 1e-12:
            raise ValueError(f"IBS reported {key}={parsed[key]}, expected {expected[key]}")
    return parsed


def derive_best_so_far(raw_path: Path, output_path: Path) -> dict[str, Any]:
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Raw trajectory has no steps")
    derived: list[dict[str, Any]] = []
    best = float("inf")
    visited: set[str] = set()
    for step, row in enumerate(rows, start=1):
        fitness = int(float(row["Fitness"]))
        improved = fitness < best
        best = min(best, fitness)
        visited.add(row["Solution"])
        derived.append(
            {
                "run": int(row["Run"]),
                "step": step,
                "fitness": fitness,
                "best_so_far": int(best),
                "solution": row["Solution"],
                "is_improvement": improved,
            }
        )
    write_csv(output_path, derived)
    return {"trajectory_steps": len(rows), "trajectory_unique_locations": len(visited), "trajectory_best_fitness": int(best)}


def execute_one(job: dict[str, str], slot: dict[str, Any], protocol: dict[str, Any], stop_event: threading.Event) -> dict[str, Any]:
    if valid_completed_run(job, protocol):
        return load_json(run_dir(job) / "result.json") or {}
    quarantine_partial(job)
    directory = run_dir(job)
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = directory / "trajectory_raw.csv"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    atomic_write_json(directory / "protocol.json", protocol)
    command = build_command(job, slot, trajectory)
    atomic_write_json(
        directory / "run_manifest.json",
        {
            "job": job,
            "worker": slot,
            "command": command,
            "scientific_command_has_no_maxiter": "--maxiter" not in command,
            "scientific_command_has_no_budget": "--budget" not in command,
            "started_utc": utc_now(),
            "thermal_before": thermal_snapshot(),
        },
    )
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    started = time.monotonic()
    deadline = started + int(job["time_limit_seconds"]) + max(1800, int(int(job["time_limit_seconds"]) * 0.25))
    peak_rss = 0
    interrupted = False
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=project_root(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=environment,
            start_new_session=True,
        )
        child = psutil.Process(process.pid)
        while process.poll() is None:
            if stop_event.wait(1.0):
                interrupted = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            try:
                peak_rss = max(peak_rss, child.memory_info().rss)
            except (psutil.Error, ProcessLookupError):
                pass
            if time.monotonic() > deadline:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(2)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                raise TimeoutError(f"Operational guard exceeded at {deadline - started:.0f} seconds")
        returncode = process.wait(timeout=30)
    wall = time.monotonic() - started
    if interrupted:
        raise InterruptedError("Campaign stopped after another run failed or the supervisor received a signal")
    if returncode != 0:
        raise RuntimeError(f"IBS exited with code {returncode}; inspect stderr.log")
    parsed = parse_stdout(stdout_path)
    trajectory_stats = derive_best_so_far(trajectory, directory / "trajectory_best_so_far.csv")
    if parsed["objective"] != trajectory_stats["trajectory_best_fitness"]:
        raise ValueError(f"Final objective {parsed['objective']} differs from raw trajectory best {trajectory_stats['trajectory_best_fitness']}")
    result = {
        **job,
        **parsed,
        **trajectory_stats,
        "status": "completed",
        "wall_seconds": wall,
        "wall_overrun_seconds": wall - int(job["time_limit_seconds"]),
        "peak_rss_bytes": peak_rss,
        "worker": slot,
        "protocol_sha256": protocol["protocol_sha256"],
        "trajectory_raw_sha256": file_sha256(trajectory),
        "trajectory_best_so_far_sha256": file_sha256(directory / "trajectory_best_so_far.csv"),
        "stdout_sha256": file_sha256(stdout_path),
        "stderr_sha256": file_sha256(stderr_path),
        "thermal_after": thermal_snapshot(),
        "completed_utc": utc_now(),
        "result_provenance": "measured",
    }
    atomic_write_json(directory / "result.json", result)
    return result


class Campaign:
    def __init__(self, jobs: list[dict[str, str]], protocol: dict[str, Any], workers: int) -> None:
        self.jobs = jobs
        self.protocol = protocol
        self.slots = physical_core_slots()[:workers]
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.running: dict[int, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []

    def status(self) -> dict[str, Any]:
        states: Counter[str] = Counter()
        completed_wall = 0.0
        pending_nominal = 0
        by_size: dict[str, Counter[str]] = {}
        for job in self.jobs:
            result = load_json(run_dir(job) / "result.json")
            state = "completed" if valid_completed_run(job, self.protocol) else (str(result.get("status")) if result else "pending")
            states[state] += 1
            by_size.setdefault(job["size_class"], Counter())[state] += 1
            if state == "completed" and result:
                completed_wall += float(result.get("wall_seconds", 0.0))
            else:
                pending_nominal += int(job["time_limit_seconds"])
        running = list(self.running.values())
        return {
            "status": "failed" if self.errors else ("completed" if states["completed"] == len(self.jobs) else ("running" if running else "prepared")),
            "total": len(self.jobs),
            "completed": states["completed"],
            "failed": sum(value for key, value in states.items() if key not in {"completed", "pending"}),
            "running": len(running),
            "pending": states["pending"],
            "states": dict(states),
            "by_size": {size: dict(counts) for size, counts in sorted(by_size.items())},
            "workers": len(self.slots),
            "worker_slots": self.slots,
            "current_runs": running,
            "errors": self.errors,
            "completed_wall_seconds_sum": completed_wall,
            "nominal_remaining_cpu_seconds": pending_nominal,
            "nominal_remaining_parallel_seconds": pending_nominal / len(self.slots),
            "updated_utc": utc_now(),
        }

    def write_status(self) -> None:
        with self.lock:
            atomic_write_json(result_root() / "execution_status.json", self.status())

    def worker(self, slot: dict[str, Any], jobs_queue: queue.Queue[dict[str, str]]) -> None:
        while not self.stop.is_set():
            try:
                job = jobs_queue.get_nowait()
            except queue.Empty:
                return
            with self.lock:
                self.running[int(slot["worker_id"])] = {"job_id": job["job_id"], "instance": job["instance"], "seed": int(job["seed"]), "slot": slot, "started_utc": utc_now()}
            self.write_status()
            try:
                execute_one(job, slot, self.protocol, self.stop)
            except InterruptedError:
                pass
            except Exception as exc:
                failure = {
                    **job,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "worker": slot,
                    "failed_utc": utc_now(),
                }
                atomic_write_json(run_dir(job) / "result.json", failure)
                with self.lock:
                    self.errors.append(failure)
                self.stop.set()
            finally:
                with self.lock:
                    self.running.pop(int(slot["worker_id"]), None)
                jobs_queue.task_done()
                self.write_status()

    def monitor(self) -> None:
        path = result_root() / "environment_samples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        while not self.stop.wait(30):
            sample = thermal_snapshot()
            sample["virtual_memory"] = psutil.virtual_memory()._asdict()
            sample["swap_memory"] = psutil.swap_memory()._asdict()
            sample["load_average"] = os.getloadavg()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")

    def run(self) -> int:
        jobs_queue: queue.Queue[dict[str, str]] = queue.Queue()
        for job in self.jobs:
            if not valid_completed_run(job, self.protocol):
                jobs_queue.put(job)
        if jobs_queue.empty():
            self.write_status()
            return 0
        monitor = threading.Thread(target=self.monitor, name="environment-monitor", daemon=True)
        monitor.start()
        threads = [threading.Thread(target=self.worker, args=(slot, jobs_queue), name=f"worker-{slot['worker_id']}") for slot in self.slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.stop.set()
        monitor.join(timeout=5)
        self.write_status()
        final = self.status()
        return 0 if final["completed"] == len(self.jobs) and not self.errors else 1


def execute(workers: int, resume: bool) -> int:
    protocol = current_protocol()
    if workers != int(protocol["execution_workers"]):
        raise ValueError(f"workers={workers} differs from persisted protocol workers={protocol['execution_workers']}")
    if not resume and any((result_root() / "runs").rglob("result.json")):
        raise RuntimeError("Run artifacts exist; pass --resume to apply strict validation")
    jobs = load_jobs()
    campaign = Campaign(jobs, protocol, workers)

    def request_stop(_signum: int, _frame: Any) -> None:
        campaign.stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return campaign.run()
