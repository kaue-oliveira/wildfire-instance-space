from __future__ import annotations

import json
import math
from pathlib import Path

from analysis_wsp.caminho_c.common import (
    IBS_PARAMETERS,
    SEEDS,
    TIME_SECONDS_PER_VERTEX,
    accepted_rows,
    build_protocol,
    content_sha256,
    exact_candidate_path,
    physical_core_slots,
    raw_instance_path,
    validate_and_repair_native,
)
from analysis_wsp.caminho_c.execute import build_command


def test_nineteen_supplementary_instances_are_valid_distinct_and_hash_matched() -> None:
    hashes = set()
    repairs = 0
    for row in accepted_rows():
        payload = json.loads(raw_instance_path(row).read_text(encoding="utf-8"))
        assert content_sha256(payload) == row["best_exact_instance_sha256"]
        repaired, audit = validate_and_repair_native(payload)
        assert repaired["|R|"] == len(repaired["t"]) == len(repaired["c"]) == len(repaired["delta"])
        hashes.add(row["best_exact_instance_sha256"])
        repairs += int(audit["resource_count_repaired"])
    assert len(hashes) == 19
    assert repairs == 4


def test_no_refit_projection_is_persisted_for_all_accepted_instances() -> None:
    for row in accepted_rows():
        exact = json.loads(exact_candidate_path(row).read_text(encoding="utf-8"))
        coordinates = exact["pilot_coordinates"]
        assert len(coordinates) == 2
        assert math.dist(list(map(float, coordinates)), [float(row["exact_z1"]), float(row["exact_z2"])]) <= 1e-10


def test_protocol_is_ibs_only_time_limited_and_has_190_runs() -> None:
    protocol = build_protocol()
    assert protocol["instances"] == 19
    assert protocol["seeds"] == list(range(1, 11)) == list(SEEDS)
    assert protocol["runs"] == 190
    assert protocol["ibs_parameters"] == IBS_PARAMETERS
    assert protocol["ibs_parameters"]["termination"] == "wall_time_only"
    assert protocol["ibs_parameters"]["maxiter_explicit"] is False
    assert protocol["ibs_parameters"]["budget_explicit"] is False
    assert TIME_SECONDS_PER_VERTEX == 1.5
    assert protocol["analysis_constraints"]["no_supervised_refit"] is True
    assert protocol["analysis_constraints"]["no_optimality_gap"] is True


def test_machine_uses_one_logical_cpu_per_physical_core() -> None:
    slots = physical_core_slots()
    # The LAB workers are derived from the physical topology at runtime.  This
    # keeps one process per physical core on both hybrid and homogeneous CPUs;
    # the exact core count is persisted in the campaign protocol.
    assert slots
    assert len({(slot["socket"], slot["physical_core"]) for slot in slots}) == len(slots)
    assert len({slot["logical_cpu"] for slot in slots}) == len(slots)
    assert all(slot["logical_cpu"] in slot["sibling_logical_cpus"] for slot in slots)
    assert {slot["core_type"] for slot in slots} in ({"homogeneous"}, {"P", "E"})


def test_real_command_has_published_parameters_and_no_iteration_cap(tmp_path: Path, monkeypatch) -> None:
    from analysis_wsp.caminho_c import execute
    binary = tmp_path / "third_party" / "Iterated-Beam-Search" / "build" / "fire"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setattr(execute, "project_root", lambda: tmp_path)
    slots = physical_core_slots()
    job = {
        "ibs_instance": str(tmp_path / "instance.json"),
        "seed": "1",
        "time_limit_seconds": "600",
    }
    command = build_command(job, slots[0], tmp_path / "trajectory_raw.csv")
    joined = " ".join(command)
    for option, value in (("--beta", "50"), ("--eta", "70"), ("--c", "30"), ("--p", "0.5"), ("--phat", "1.0"), ("--zmax", "3"), ("--timelimit", "600")):
        assert f"{option} {value}" in joined
    assert "--maxiter" not in command
    assert "--budget" not in command


def test_published_nominal_budget_is_1054500_seconds() -> None:
    counts = {400: 1, 900: 3, 1600: 6, 6400: 9}
    total = sum(vertices * TIME_SECONDS_PER_VERTEX * 10 * count for vertices, count in counts.items())
    assert total == 1_054_500
