from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil


# The 74-instance continuation uses benchmark measurements after one
# transparent objective correction in a separate derived analysis layer.
VALID_PROVENANCE = {"measured", "original_source", "audited_benchmark74", "predicted", "missing"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(value):
        value = asdict(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def environment_manifest(root: Path, command: list[str]) -> dict[str, Any]:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        freeze = []
    hashed_inputs = {}
    for label, relative in (
        ("wsp_generator", Path("third_party/Wildfire-Suppression/scripts/instance_generation.py")),
        ("experiment_config", Path("analysis_wsp/configs/tcc_essential.json")),
        ("requirements", Path("analysis_wsp/requirements-isa.txt")),
        ("methodology", Path("docs/metodologia.md")),
    ):
        path = root / relative
        hashed_inputs[label] = {
            "path": str(relative),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(root),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": psutil.virtual_memory().total,
        "root_git_commit": git_commit(root),
        "critical_input_hashes": hashed_inputs,
        "pip_freeze": freeze,
        "result_provenance": "measured",
    }


def file_manifest(paths: Iterable[Path], base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        try:
            name = str(path.relative_to(base))
        except ValueError:
            name = str(path)
        rows.append({"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def validate_provenance(values: Iterable[str]) -> None:
    invalid = sorted(set(values) - VALID_PROVENANCE)
    if invalid:
        raise ValueError(f"Proveniências inválidas: {invalid}")
