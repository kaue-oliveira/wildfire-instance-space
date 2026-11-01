#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator, TextIO

try:  # Package import in tests; direct-script import in the documented CLI.
    from .extended_analysis.config import ExperimentConfig
    from .extended_analysis.pipeline import run_pipeline
    from .extended_analysis.provenance import write_json
except ImportError:  # pragma: no cover - exercised by CLI integration runs
    from extended_analysis.config import ExperimentConfig
    from extended_analysis.pipeline import run_pipeline
    from extended_analysis.provenance import write_json


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class ProfileAlreadyRunningError(RuntimeError):
    """Raised when another process holds the exclusive profile lock."""


class ProfileStateError(RuntimeError):
    """Raised when fresh/resume intent conflicts with persisted profile state."""


@contextmanager
def profile_lock(path: Path, command: list[str]) -> Iterator[None]:
    """Hold an OS-level lock for the complete lifetime of one profile run.

    The lock is inherited by multiprocessing workers. Consequently, an orphaned
    worker also prevents a second runner from writing to the same result tree.
    The JSON stored in the lock file is diagnostic metadata; exclusivity comes
    from ``flock``, not from the file contents.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "metadados do processo indisponíveis"
            raise ProfileAlreadyRunningError(
                f"o perfil já possui uma execução ativa ({owner})"
            ) from exc
        metadata = {
            "status": "active",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": command,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISA topológica e povoamento do espaço WSP para o TCC")
    parser.add_argument("--profile", choices=("tcc",), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-profile",
        help="nome separado para a árvore de resultados; mantém perfis anteriores intactos",
    )
    return parser.parse_args()


def validate_profile_request(result_root: Path, *, resume: bool) -> str:
    state_markers = (
        result_root / "experiment_config_resolved.json",
        result_root / "environment_manifest.json",
        result_root / "pipeline_status.json",
        *(result_root / name for name in ("00_baseline_55", "01_instancespace_original", "02_instancespace_topology", "04a_ga_internal", "05_final_isa")),
    )
    has_state = any(path.exists() for path in state_markers)
    if resume and not has_state:
        raise ProfileStateError("--resume foi solicitado, mas o perfil não contém um checkpoint")
    if not resume and has_state:
        raise ProfileStateError(
            "o perfil já contém resultados; use outro --output-profile ou confirme a retomada com --resume",
        )
    status_path = result_root / "pipeline_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    terminal_path = result_root / "terminal_acceptance_validation.json"
    try:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        terminal = {}
    if resume and status.get("status") == "completed" and terminal.get("status") == "passed":
        return "already_completed"
    return "ready"


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / "analysis_wsp" / "configs" / "tcc_essential.json"
    config = ExperimentConfig.from_file(config_path)
    if args.output_profile:
        config = replace(config, profile=args.output_profile)
    result_root = root / "experiments" / "generation" / config.profile
    command = [sys.executable, *sys.argv]
    try:
        with profile_lock(result_root / ".pipeline.lock", command):
            request_state = validate_profile_request(result_root, resume=args.resume)
            if request_state == "already_completed":
                print(f"Perfil já concluído e validado: {result_root}")
                return 0
            return run_locked(args, root, config_path, config, result_root, command)
    except ProfileAlreadyRunningError as exc:
        print(f"Execução recusada: {exc}", file=sys.stderr)
        return 75
    except ProfileStateError as exc:
        print(f"Execução recusada: {exc}", file=sys.stderr)
        return 64


def run_locked(
    args: argparse.Namespace,
    root: Path,
    config_path: Path,
    config: ExperimentConfig,
    result_root: Path,
    command: list[str],
) -> int:
    logs_dir = result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    stdout_path, stderr_path = logs_dir / f"stdout_{stamp}.log", logs_dir / f"stderr_{stamp}.log"
    started = time.perf_counter()
    exit_code = 1
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(old_stdout, stdout_file), Tee(old_stderr, stderr_file)
        try:
            print(f"Configuração: {config_path}")
            print(f"Resultados: {result_root}")
            run_pipeline(root, config, resume=args.resume, command=command)
            exit_code = 0
        except Exception:
            traceback.print_exc()
            write_json(
                result_root / "pipeline_status.json",
                {
                    "status": "failed",
                    "traceback": traceback.format_exc(),
                    "result_provenance": "missing",
                },
            )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    manifest_path = logs_dir / "commands.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "command": command,
                    "config": str(config_path),
                    "profile": config.profile,
                    "resume": args.resume,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "exit_code": exit_code,
                    "elapsed_seconds": time.perf_counter() - started,
                    "result_provenance": "measured",
                },
                sort_keys=True,
            )
            + "\n",
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
