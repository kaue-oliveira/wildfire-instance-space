from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import discover_gurobi_home, project_root, validate_license_path


ORIGINAL_ILS_SHA256 = "c283ebb8da75334dade8ab2af553104d5b5bfc331053877240496949d7cfc862"
INSTRUMENTED_ILS_SHA256 = "ca020668ce1e8d915d5f8d12be0ab7d30093bbfe94c08bfc04f4231b7bda44d9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_ils() -> dict[str, str]:
    path = project_root() / "third_party" / "Logic-based-Benders-decomposition-for-wildfire-suppression" / "IteratedLocalSearch.py"
    reviewed = project_root() / "analysis_wsp" / "benchmark74" / "ils_strict_v2.py.template"
    reviewed_hash = sha256(reviewed)
    if reviewed_hash != INSTRUMENTED_ILS_SHA256:
        raise RuntimeError(
            f"Reviewed ILS template hash {reviewed_hash} differs from expected "
            f"{INSTRUMENTED_ILS_SHA256}"
        )
    current = sha256(path)
    if current == INSTRUMENTED_ILS_SHA256:
        return {"status": "already_instrumented", "path": str(path), "sha256": current}
    if current != ORIGINAL_ILS_SHA256:
        raise RuntimeError(f"Unexpected ILS source hash {current}; refusing an unreviewed patch")
    shutil.copyfile(reviewed, path)
    updated = sha256(path)
    if updated != INSTRUMENTED_ILS_SHA256:
        raise RuntimeError(f"Instrumented ILS hash {updated} differs from reviewed hash {INSTRUMENTED_ILS_SHA256}")
    return {"status": "instrumented", "path": str(path), "sha256": updated}


def validate_environment(license_path: Path) -> dict[str, object]:
    root = project_root()
    license_metadata = validate_license_path(license_path)
    required = [
        root / ".venv-isa" / "bin" / "python",
        root / ".venv-gurobi" / "bin" / "python",
        root / "third_party" / "Wildfire-Suppression" / "build" / "fire",
        root / "third_party" / "Iterated-Beam-Search" / "build" / "fire",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing environment artifacts: {missing}")
    environment = os.environ.copy()
    environment["GRB_LICENSE_FILE"] = str(license_path)
    gurobi_home = discover_gurobi_home()
    environment["GUROBI_HOME"] = str(gurobi_home)
    environment["LD_LIBRARY_PATH"] = f"{gurobi_home / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    completed = subprocess.run(
        [str(root / ".venv-gurobi" / "bin" / "python"), "-c", "import gurobipy as gp; m=gp.Model(); m.Params.OutputFlag=0; x=m.addVar(vtype=gp.GRB.BINARY); m.setObjective(x,gp.GRB.MAXIMIZE); m.optimize(); assert m.Status==gp.GRB.OPTIMAL"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(f"Gurobi validation failed with exit code {completed.returncode}{suffix}")
    return {"license": license_metadata, "gurobi_home": str(gurobi_home), "required_artifacts": [str(path) for path in required], "gurobi_validation": "passed"}
