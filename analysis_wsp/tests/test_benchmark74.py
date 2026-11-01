from __future__ import annotations

import importlib.util
import subprocess
import sys
import types

import pytest

from analysis_wsp.benchmark74 import bootstrap
from analysis_wsp.benchmark74 import amend
from analysis_wsp.benchmark74 import legacy_worker
from analysis_wsp.benchmark74.common import (
    ALGORITHMS,
    balanced_shards,
    benchmark_seeds,
    load_all_instances,
    profile_generation,
    size_class,
    time_limit_seconds,
)
from analysis_wsp.benchmark74.execute import effective_time_limit, operational_guard_seconds


def test_final_collection_has_74_distinct_instances() -> None:
    records = load_all_instances()
    assert len(records) == 74
    assert len({record["instance"] for record in records}) == 74
    assert len({record["instance_sha256"] for record in records}) == 74


def test_published_time_and_replication_rules() -> None:
    assert time_limit_seconds(282) == 600
    assert time_limit_seconds(400) == 600
    assert time_limit_seconds(900) == 1350
    assert time_limit_seconds(1600) == 2400
    assert time_limit_seconds(6400) == 9600
    assert len(benchmark_seeds(1600)) == 10
    assert len(benchmark_seeds(6400)) == 3


def test_expected_benchmark_dimensions() -> None:
    records = load_all_instances()
    runs_per_algorithm = sum(len(benchmark_seeds(int(record["audit"]["n_vertices"]))) for record in records)
    assert runs_per_algorithm == 656
    assert runs_per_algorithm * len(ALGORITHMS) == 3936


def test_balanced_shards_are_deterministic_and_close() -> None:
    jobs = [
        {"phase": "benchmark", "algorithm": "ibs", "instance": f"i{index}", "seed": 1, "time_limit_seconds": limit}
        for index, limit in enumerate([9600, 9600, 2400, 1350, 600, 600])
    ]
    first = balanced_shards(jobs, 2)
    second = balanced_shards(jobs, 2)
    assert first == second
    loads = [sum(int(job["time_limit_seconds"]) for job in first if job["shard_index"] == shard) for shard in range(2)]
    assert abs(loads[0] - loads[1]) <= 2400


def test_strict_ils_pilot_uses_the_requested_scientific_limit() -> None:
    huge_ils = {"algorithm": "ils", "size_class": "Huge", "time_limit_seconds": 9600}
    medium_ils = {"algorithm": "ils", "size_class": "Medium", "time_limit_seconds": 1350}
    assert effective_time_limit(huge_ils, 10) == 10
    assert effective_time_limit(medium_ils, 10) == 10
    assert effective_time_limit(huge_ils, None) == 9600
    assert operational_guard_seconds(huge_ils, 10) == 310
    assert operational_guard_seconds(medium_ils, 10) == 310


def test_v2_1_is_a_strict_v2_generation_profile() -> None:
    assert profile_generation("benchmark74_v2_1") == 2
    assert profile_generation("benchmark74_pilot_lab-02_v2_1") == 2
    assert amend.FEASIBILITY_FORWARD_ALGORITHMS == {
        "ibs", "rs", "ils", "cuth", "mip", "lbbd", "ibs_stn",
    }
    assert profile_generation("benchmark74_v2_2") == 2
    assert profile_generation("benchmark74_v2_3") == 2
    assert profile_generation("benchmark74_v2_4") == 2


def test_lbbd_stage_limit_shares_only_the_global_time_remaining() -> None:
    assert legacy_worker.lbbd_stage_limit(100.0, 40.0, 3) == 20.0
    assert legacy_worker.lbbd_stage_limit(100.0, 90.0, 1) == 10.0
    assert legacy_worker.lbbd_stage_limit(100.0, 101.0, 4) == 0.0


def test_lbbd_validated_objective_is_canonical_and_audited() -> None:
    audit = legacy_worker.lbbd_objective_audit(886.0, 885, 800.0)
    assert audit == {
        "objective": 885,
        "objective_source": "validated_recomputation",
        "model_objective": 886,
        "objective_delta": -1,
        "objective_consistent": False,
        "mip_gap": 85 / 885,
    }


def test_lbbd_validated_objective_rejects_impossible_bound() -> None:
    with pytest.raises(ValueError, match="exceeds validated feasible objective"):
        legacy_worker.lbbd_objective_audit(886.0, 885, 885.01)


def test_ils_instrumentation_from_clean_checkout_is_reviewed_and_idempotent(tmp_path, monkeypatch) -> None:
    source_repo = (
        bootstrap.project_root()
        / "third_party"
        / "Logic-based-Benders-decomposition-for-wildfire-suppression"
    )
    original = subprocess.check_output(
        ["git", "-C", str(source_repo), "show", "HEAD:IteratedLocalSearch.py"]
    )
    target = (
        tmp_path
        / "third_party"
        / "Logic-based-Benders-decomposition-for-wildfire-suppression"
        / "IteratedLocalSearch.py"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    reviewed = tmp_path / "analysis_wsp" / "benchmark74" / "ils_strict_v2.py.template"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_bytes(
        (bootstrap.project_root() / "analysis_wsp" / "benchmark74" / "ils_strict_v2.py.template").read_bytes()
    )
    monkeypatch.setattr(bootstrap, "project_root", lambda: tmp_path)

    first = bootstrap.instrument_ils()
    second = bootstrap.instrument_ils()

    assert first == {
        "status": "instrumented",
        "path": str(target),
        "sha256": bootstrap.INSTRUMENTED_ILS_SHA256,
    }
    assert second == {
        "status": "already_instrumented",
        "path": str(target),
        "sha256": bootstrap.INSTRUMENTED_ILS_SHA256,
    }


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def load_test_fireils(tmp_path, monkeypatch, durations: list[float]):
    clock = FakeClock()
    pending = iter(durations)
    shortest_paths = types.ModuleType("shortest_paths")

    def shortest(nodes, arcs, in_arcs, out_arcs, protected, ignitions, delay):
        clock.value += next(pending, 0.0)
        arrival = {node: (10.0 if node in protected else 0.0) for node in nodes}
        return arrival, {}, {}

    shortest_paths.ShortestPaths = shortest
    monkeypatch.setitem(sys.modules, "shortest_paths", shortest_paths)
    target = tmp_path / "IteratedLocalSearch.py"
    target.write_bytes((bootstrap.project_root() / "analysis_wsp" / "benchmark74" / "ils_strict_v2.py.template").read_bytes())
    spec = importlib.util.spec_from_file_location("strict_ils_test", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FireILS, clock


def run_tiny_ils(fire_ils, clock, deadline: float):
    return fire_ils(
        [(0, 0)], {}, {0.0: 1}, [(0, 0)], 1.0, 5.0,
        1, 1.0, 0.0, 1, 1, 1, 1, 1,
        Deadline=deadline, Clock=clock,
    )


def test_incomplete_candidate_at_deadline_returns_feasible_base(tmp_path, monkeypatch) -> None:
    fire_ils, clock = load_test_fireils(tmp_path, monkeypatch, [0.0, 2.0])
    solution, objective, telemetry = run_tiny_ils(fire_ils, clock, 1.0)
    assert sum(solution.values()) == 0
    assert objective == 1
    assert telemetry["complete_candidates"] == 0
    assert telemetry["termination_reason"] == "time_limit_no_complete_candidate"


def test_candidate_evaluated_after_deadline_is_discarded(tmp_path, monkeypatch) -> None:
    fire_ils, clock = load_test_fireils(tmp_path, monkeypatch, [0.0, 0.0, 2.0])
    solution, objective, telemetry = run_tiny_ils(fire_ils, clock, 1.0)
    assert sum(solution.values()) == 0
    assert objective == 1
    assert telemetry["complete_candidates"] == 0
    assert telemetry["termination_reason"] == "time_limit_no_complete_candidate"


def test_candidate_accepted_before_deadline_survives_late_local_search(tmp_path, monkeypatch) -> None:
    fire_ils, clock = load_test_fireils(tmp_path, monkeypatch, [0.0, 0.0, 0.5, 1.0])
    solution, objective, telemetry = run_tiny_ils(fire_ils, clock, 1.0)
    assert sum(solution.values()) == 1
    assert objective == 0
    assert telemetry["complete_candidates"] == 1
    assert telemetry["termination_reason"] == "time_limit"


def test_invalid_final_ils_candidate_is_discarded_and_scored_with_baseline(monkeypatch) -> None:
    fake_ils = types.ModuleType("IteratedLocalSearch")

    def fire_ils(*_args, Deadline):
        assert Deadline > 0
        return {
            ((0, 0), 0.0): 1.0,
            ((0, 0), 1.0): 1.0,
        }, 0, {"complete_candidates": 1, "termination_reason": "time_limit"}

    fake_ils.FireILS = fire_ils
    monkeypatch.setitem(sys.modules, "IteratedLocalSearch", fake_ils)
    monkeypatch.setattr(
        legacy_worker,
        "load_instance",
        lambda _path: ([(0, 0)], {}, [(0, 0)], 1.0, 5.0, {0.0: 1, 1.0: 1}),
    )

    def validate(_nodes, _arcs, _ignitions, _delay, _horizon, _resources, assignments):
        if assignments:
            raise ValueError("A node received more than one resource")
        return {"objective_recomputed": 1, "assignments": 0, "feasible": True}

    monkeypatch.setattr(legacy_worker, "validate_solution", validate)
    result = legacy_worker.run_ils(None, 4, 1)
    assert result["_status"] == "algorithm_failure"
    assert result["objective"] is None
    assert result["scored_objective"] == 1
    assert result["failure_category"] == "infeasible_final_solution"
    assert result["baseline_solution"] == []


def test_carry_forward_rejects_ils_before_touching_profiles() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        amend.carry_forward("benchmark74_v1", "benchmark74_v2", ["ibs", "ils"], 1)


def test_carried_forward_validator_rejects_one_byte_change(tmp_path, monkeypatch) -> None:
    profile = "benchmark74_v2"
    root = tmp_path / profile
    job = {
        "phase": "benchmark", "algorithm": "ibs", "engine": "unified_cpp",
        "instance": "tiny", "collection": "test", "instance_sha256": "abc",
        "seed": "1", "n_vertices": "1", "size_class": "Small",
        "time_limit_seconds": "600", "shard_index": "1", "job_id": "job_1",
        "protocol_sha256": "v2", "native_path": "/tmp/native.json",
        "harris_path": "/tmp/harris.json",
    }
    directory = root / amend.run_relative(job)
    directory.mkdir(parents=True)
    artifact = directory / "result.json"
    artifact.write_text("original\n", encoding="utf-8")
    files = amend.file_manifest(directory)
    record = {
        "identity": amend.job_identity(job),
        "destination_job_sha256": amend.content_sha256(job),
        "destination_run": str(amend.run_relative(job)),
        "files": files,
        "run_tree_sha256": amend.content_sha256(files),
    }
    amend.atomic_write_json(root / "amendment_manifest.json", {
        "to_profile": profile, "to_protocol_sha256": "v2",
        "allowed_algorithms": ["ibs", "rs"], "runs": [record]
    })
    monkeypatch.setattr(amend, "result_root", lambda _profile: root)
    assert amend.valid_carried_forward(profile, job, {"protocol_sha256": "v2"})
    artifact.write_text("original!\n", encoding="utf-8")
    assert not amend.valid_carried_forward(profile, job, {"protocol_sha256": "v2"})


def test_carried_forward_validator_accepts_ils_when_amendment_allows_it(tmp_path, monkeypatch) -> None:
    profile = "benchmark74_v2_1"
    root = tmp_path / profile
    job = {
        "phase": "benchmark", "algorithm": "ils", "engine": "legacy_python",
        "instance": "tiny", "collection": "test", "instance_sha256": "abc",
        "seed": "1", "n_vertices": "1", "size_class": "Small",
        "time_limit_seconds": "600", "shard_index": "1", "job_id": "job_1",
        "protocol_sha256": "v2.1", "native_path": "/tmp/native.json",
        "harris_path": "/tmp/harris.json",
    }
    directory = root / amend.run_relative(job)
    directory.mkdir(parents=True)
    (directory / "result.json").write_text("valid measured bytes\n", encoding="utf-8")
    files = amend.file_manifest(directory)
    amend.atomic_write_json(root / "amendment_manifest.json", {
        "to_profile": profile,
        "to_protocol_sha256": "v2.1",
        "allowed_algorithms": ["ibs", "ils", "rs"],
        "runs": [{
            "identity": amend.job_identity(job),
            "destination_job_sha256": amend.content_sha256(job),
            "destination_run": str(amend.run_relative(job)),
            "files": files,
            "run_tree_sha256": amend.content_sha256(files),
        }],
    })
    monkeypatch.setattr(amend, "result_root", lambda _profile: root)
    assert amend.valid_carried_forward(profile, job, {"protocol_sha256": "v2.1"})
