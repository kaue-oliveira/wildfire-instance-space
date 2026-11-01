from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from analysis_wsp.extended_analysis.config import ExperimentConfig
from analysis_wsp.extended_analysis.data import instance_content_hash
from analysis_wsp.extended_analysis.ga import mutate, random_genotype, uniform_crossover
from analysis_wsp.extended_analysis.isa import label_audit
from analysis_wsp.extended_analysis.pipeline import _validate_output_provenance, _validate_terminal
from analysis_wsp.extended_analysis.provenance import write_json
from analysis_wsp.extended_analysis.report import write_report
from analysis_wsp.extended_analysis.topology import (
    AGGREGATES,
    TOPOLOGY_FAMILIES,
    _hits_repeated,
    _igraph_exact_betweenness,
    _igraph_exact_harmonic,
    audit_and_prepare_topology,
    compute_selected_topology_for_search,
    compute_topology,
    validate_exact_topology_recomputation,
)
from analysis_wsp.run_extended_analysis import (
    ProfileAlreadyRunningError,
    ProfileStateError,
    profile_lock,
    validate_profile_request,
)


def small_instance() -> dict:
    return {
        "|V|": 4,
        "arcs": [[0, 1, 1.0], [1, 0, 2.0], [1, 2, 1.0], [2, 1, 1.0], [2, 3, 3.0], [3, 2, 1.0], [0, 2, 2.5]],
    }


def test_tcc_config_is_fixed_to_the_registered_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_file(root / "configs" / "tcc_essential.json")
    assert config.root_seeds == (20260612, 104729, 1299709)
    assert config.thresholds == (0.10,)
    assert config.target_budget == 12
    assert config.ga_evaluation_budget == 2000


def test_profile_lock_rejects_a_concurrent_runner(tmp_path: Path) -> None:
    lock = tmp_path / ".pipeline.lock"
    with profile_lock(lock, ["first"]):
        with pytest.raises(ProfileAlreadyRunningError):
            with profile_lock(lock, ["second"]):
                pass


def test_profile_state_requires_explicit_fresh_or_resume_intent(tmp_path: Path) -> None:
    assert validate_profile_request(tmp_path, resume=False) == "ready"
    with pytest.raises(ProfileStateError):
        validate_profile_request(tmp_path, resume=True)
    write_json(tmp_path / "pipeline_status.json", {"status": "failed"})
    with pytest.raises(ProfileStateError):
        validate_profile_request(tmp_path, resume=False)
    assert validate_profile_request(tmp_path, resume=True) == "ready"
    write_json(tmp_path / "pipeline_status.json", {"status": "completed"})
    write_json(tmp_path / "terminal_acceptance_validation.json", {"status": "passed"})
    assert validate_profile_request(tmp_path, resume=True) == "already_completed"


def test_topology_features_are_directed_and_finite() -> None:
    result = compute_topology(small_instance(), exact_betweenness=True)
    assert len(result.features) == 17 * len(AGGREGATES)
    assert all(np.isfinite(list(result.features.values())))
    assert result.features["topo_degree_imbalance_min"] < 0 < result.features["topo_degree_imbalance_max"]


def test_hits_is_deterministic() -> None:
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 1)])
    first = _hits_repeated(graph, (11, 23, 37, 53, 71))
    second = _hits_repeated(graph, (11, 23, 37, 53, 71))
    assert first[0] == pytest.approx(second[0])
    assert first[1] == pytest.approx(second[1])


def test_exact_backend_matches_networkx_for_directed_centralities() -> None:
    data = small_instance()
    graph = nx.DiGraph()
    graph.add_weighted_edges_from((u, v, weight) for u, v, weight in data["arcs"])
    for _, _, attributes in graph.edges(data=True):
        attributes["distance"] = attributes["weight"]
    expected = nx.betweenness_centrality(graph, normalized=True, weight="distance")
    measured = _igraph_exact_betweenness(data, weighted=True)
    assert measured["topo_betweenness_distance_mean"] == pytest.approx(np.mean(list(expected.values())))
    incoming, outgoing = _igraph_exact_harmonic(data, weighted=True)
    assert incoming == pytest.approx({node: value / 3 for node, value in nx.harmonic_centrality(graph, distance="distance").items()})
    assert outgoing == pytest.approx({node: value / 3 for node, value in nx.harmonic_centrality(graph.reverse(copy=False), distance="distance").items()})


def test_candidate_features_match_the_canonical_baseline() -> None:
    data = small_instance()
    required = ["topo_betweenness_distance_q75", "topo_harmonic_in_distance_q75", "topo_hits_authority_strength_q50"]
    baseline = compute_topology(data, exact_betweenness=True).features
    candidate = compute_selected_topology_for_search(data, required, landmarks=0, seed=0).features
    assert {name: candidate[name] for name in required} == pytest.approx({name: baseline[name] for name in required}, abs=1e-12)


def test_full_topology_recomputation_is_a_gate(tmp_path: Path) -> None:
    features = compute_topology(small_instance(), exact_betweenness=True).features
    result = validate_exact_topology_recomputation({"tiny": {"data": small_instance()}}, pd.DataFrame([{"instance": "tiny", **features}]), tmp_path / "validation.json")
    assert result["status"] == "passed"
    assert result["features_checked"] == len(features)


def test_topology_is_not_manually_transformed_before_matilda(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        {
            "instance": ["a", "b", "c", "d"],
            "result_provenance": ["measured"] * 4,
            "topo_in_degree_mean": [0.0, 1.0, 2.0, 1000.0],
        },
    )
    selected, _, transformations = audit_and_prepare_topology(candidates, pd.DataFrame(), tmp_path)
    assert "topo_in_degree_mean" in selected
    row = transformations.set_index("feature").loc["topo_in_degree_mean"]
    assert row["transformation"] == "identity"
    assert row["matilda_prelim"] == "positive_shift_boxcox_zscore"


def test_labels_define_folds_after_observed_class_counts() -> None:
    performance = pd.DataFrame({"instance": [f"i{x}" for x in range(6)], "A": [0.0, 0.01, 0.02, 0.2, 0.3, 0.4], "B": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]})
    labels = label_audit(performance, [0.1]).set_index("algorithm")
    assert labels.loc["A", "cv_folds"] == 3
    assert labels.loc["B", "status"] == "not_estimable"


def test_genetic_operators_remain_in_generator_domain() -> None:
    rng = np.random.default_rng(123)
    child = mutate(uniform_crossover(random_genotype(rng), random_genotype(rng), rng, 0.8), rng, 1 / 9)
    data = asdict(child)
    assert data["generator_seed"] > 0
    assert data["grid"] in {"Small", "Medium", "Large", "Huge"}


def test_provenance_does_not_impute_missing_evidence(tmp_path: Path) -> None:
    (tmp_path / "opaque.png").write_bytes(b"png")
    pd.DataFrame({"prediction": [0.8]}).to_csv(tmp_path / "pythia_predictions.csv", index=False)
    pd.DataFrame({"value": [1.0], "result_provenance": ["original_source"]}).to_csv(tmp_path / "source.csv", index=False)
    result = _validate_output_provenance(tmp_path)
    manifest = pd.read_csv(tmp_path / "artifact_provenance_manifest.csv").set_index("artifact_path")
    assert result["status"] == "passed"
    assert manifest.loc["source.csv", "content_provenance"] == "original_source"
    assert manifest.loc["pythia_predictions.csv", "content_provenance"] == "predicted"


def _complete_terminal_fixture(result_root: Path, config: ExperimentConfig) -> None:
    baseline = result_root / "00_baseline_55"
    topology = result_root / "02_instancespace_topology"
    topology_run = topology / "threshold_10"
    control_run = result_root / "01_instancespace_original" / "threshold_10"
    final = result_root / "05_final_isa"
    ga_dir = result_root / "04a_ga_internal"
    for directory in (baseline, topology_run, control_run, final, ga_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_json(baseline / "baseline_manifest_before.json", {"observed_unique_instances": 55})
    write_json(baseline / "baseline_manifest_after.json", {"byte_identical": True})
    write_json(baseline / "baseline_feature_recomputation_validation.json", {"status": "passed"})

    topology_features = sorted(
        f"topo_{metric}_{aggregate}"
        for metrics in TOPOLOGY_FAMILIES.values()
        for metric in metrics
        for aggregate in AGGREGATES
    )
    topology_rows = [{"instance": f"i{index:02d}", **{feature: float(index) for feature in topology_features}} for index in range(55)]
    pd.DataFrame(topology_rows).to_csv(topology / "topology_candidates.csv", index=False)
    write_json(topology / "exact_ga_topology_recomputation_validation.json", {"status": "passed"})
    pd.DataFrame(
        {"feature": topology_features, "decision": ["kept"] * len(topology_features)},
    ).to_csv(topology / "augmented_feature_audit.csv", index=False)
    pd.DataFrame({"source_feature": topology_features}).to_csv(
        topology_run / "prelim_feature_parameters.csv",
        index=False,
    )
    write_json(topology_run / "resolved_options.json", {"norm": {"flag": True}, "auto": {"preproc": True}})
    write_json(topology_run / "run_status.json", {"status": "completed"})
    write_json(control_run / "run_status.json", {"status": "completed"})
    write_json(final / "ga_projection_validation.json", {"status": "passed"})
    write_json(final / "embedding_comparison_controle_vs_topologia.json", {"n_instances": 55})
    write_json(
        final / "isa_artifact_comparison_controle_vs_topologia.json",
        {"sifted_reference_count": 10, "sifted_candidate_count": 12},
    )

    ga_rows = []
    for seed in config.root_seeds:
        generated = ga_dir / f"seed_{seed}" / "generated_candidates"
        generated.mkdir(parents=True, exist_ok=True)
        for target in range(config.target_budget):
            identifier = f"candidate_{seed}_{target:02d}"
            payload = {"candidate": identifier, "root_seed": seed, "target": target}
            write_json(generated / f"{identifier}.instance.json", payload)
            ga_rows.append(
                {
                    "root_seed": seed,
                    "target_id": f"target_{target:02d}",
                    "evaluations_completed": config.ga_evaluation_budget,
                    "best_exact_candidate_id": identifier,
                    "best_exact_instance_sha256": instance_content_hash(payload),
                    "accepted": False,
                    "domain_membership_confirmed_exact": False,
                },
            )
    pd.DataFrame(ga_rows).to_csv(ga_dir / "ga_target_results_with_wall_auc.csv", index=False)


def test_terminal_validation_accepts_only_a_complete_experiment(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_file(root / "configs" / "tcc_essential.json")
    _complete_terminal_fixture(tmp_path, config)
    assert _validate_terminal(tmp_path, config)["status"] == "passed"

    results_path = tmp_path / "04a_ga_internal" / "ga_target_results_with_wall_auc.csv"
    results = pd.read_csv(results_path)
    results.loc[0, "evaluations_completed"] -= 1
    results.to_csv(results_path, index=False)
    with pytest.raises(RuntimeError, match="validação terminal"):
        _validate_terminal(tmp_path, config)
    assert pd.read_json(tmp_path / "terminal_acceptance_validation.json", typ="series")["status"] == "failed"


def test_generated_report_describes_only_the_tcc_execution(tmp_path: Path) -> None:
    root = tmp_path
    result_root = root / "analysis_wsp" / "results_extended" / "tcc_essential_test"
    (root / "analysis_wsp" / "reports").mkdir(parents=True)
    result_root.mkdir(parents=True)
    report = write_report(root, result_root, SimpleNamespace(profile="tcc_essential_test"), {})
    text = report.read_text(encoding="utf-8")
    assert "## Objetivo" in text
    assert "incêndios reais" in text
    assert "benchmark" not in text.lower()
