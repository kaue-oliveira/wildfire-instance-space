"""Verify the public research snapshot using Python's standard library only."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def contained(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f"Path leaves repository: {relative}")
    return path

def main():
    manifest = json.loads((ROOT / "data/manifest.sha256.json").read_text())
    for name, expected in manifest["files"].items():
        path = contained(name)
        require(path.is_file(), f"Missing file: {name}")
        require(digest(path) == expected, f"Checksum mismatch: {name}")
    index = rows("results/tables/analysis_index_final.csv")
    require(Counter(r["phase"] for r in index) == {"benchmark": 3936, "trajectory": 190}, "Unexpected phase counts")
    require(len({(r["phase"], r["instance"], r["algorithm"], r["seed"]) for r in index}) == 4126, "Duplicate run")
    bench = [r for r in index if r["phase"] == "benchmark"]
    require(set(Counter(r["algorithm"] for r in bench).values()) == {656}, "Algorithm repetitions differ")
    require(len({r["instance"] for r in bench}) == 74, "Expected 74 benchmark instances")
    require(sum(r["successful_run"].lower() != "true" for r in bench) == 2, "Failure accounting changed")
    for row in index:
        path = contained(row["source_artifact_local"])
        require(digest(path) == row["source_artifact_sha256"], f"Run artifact changed: {row['job_id']}")
        require(contained(row["source_run_directory_local"]).is_dir(), f"Missing run directory: {row['job_id']}")
        payload = json.loads(path.read_text())
        require(str(payload["instance"]) == row["instance"], f"Instance mismatch: {row['job_id']}")
        require(str(payload["seed"]) == row["seed"], f"Seed mismatch: {row['job_id']}")
        for filename, expected in payload.get("artifact_hashes", {}).items():
            artifact = path.parent / filename
            require(artifact.resolve().is_relative_to(ROOT), "Artifact escapes repository")
            require(artifact.is_file(), f"Missing run artifact: {artifact.relative_to(ROOT)}")
            require(digest(artifact) == expected, f"Run checksum mismatch: {artifact.relative_to(ROOT)}")
    for form in ("native", "harris"):
        require(len(list((ROOT / "data/instances" / form).glob("*.json"))) == 74, f"Expected 74 {form} instances")
    cells = rows("results/tables/per_instance_algorithm_decimal.csv")
    require(len(cells) == 444 and len({(r['instance'], r['algorithm']) for r in cells}) == 444, "Expected 444 unique cells")
    require(len(rows("results/tables/stn_descriptors.csv")) == 38, "Expected 38 aggregate STNs")
    require(len(rows("results/tables/stn_per_seed_descriptors.csv")) == 380, "Expected 380 per-seed STNs")
    require(len(rows("results/tables/isa_selected_features_40.csv")) == 40, "Expected 40 feature selections")
    require(len(rows("results/tables/isa_auxiliary_cv_summary_24.csv")) == 24, "Expected 24 CV summaries")
    generation = rows("results/generation/04a_ga_internal/ga_target_results.csv")
    require(len(generation) == 36, "Expected 36 generation searches")
    require(sum(int(r["evaluations_completed"]) for r in generation) == 72000, "Generation evaluation total changed")
    require(len({r["best_exact_candidate_id"] for r in generation if r["accepted"].lower() == "true"}) == 19, "Expected 19 distinct accepted instances")
    print(json.dumps({"status": "passed", "files_hashed": len(manifest['files']), "benchmark_runs": 3936, "trajectory_runs": 190, "instances": 74, "cells": 444}, indent=2))

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
