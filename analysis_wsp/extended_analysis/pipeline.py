from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import base_features, instance_content_hash, load_original_features, load_performance_matrix, load_wildfire_instances, numeric_feature_columns
from .ga import CandidateEvaluator, cloister_polygon, consolidate_wall_auc, make_initial_targets, run_ga_experiment, summarize_reach, validate_candidate_projection
from .isa import audit_feature_table, compare_embeddings, compare_isa_artifacts, instance_space_input_hash, label_audit, run_instance_space, threshold_fold
from .provenance import environment_manifest, file_manifest, sha256_file, stable_json_hash, validate_provenance, write_json
from .report import generate_analysis_figures, write_report
from .topology import AGGREGATES, HITS_BACKEND_ID, TOPOLOGY_FAMILIES, audit_and_prepare_topology, build_topology_table, precompute_exact_topology_cache, validate_exact_topology_recomputation


RESULT_DIRECTORIES = (
    "00_baseline_55",
    "01_instancespace_original",
    "02_instancespace_topology",
    "04a_ga_internal",
    "05_final_isa",
)
PRESERVED_REPORTS = ("docs/metodologia.md",)


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _checkpoint(path: Path, status: str, **fields: Any) -> None:
    write_json(path, {"status": status, **fields})


def _freeze_baseline(root: Path, result_root: Path, records: dict[str, dict[str, Any]]) -> dict[str, str]:
    output = result_root / "00_baseline_55"
    output.mkdir(parents=True, exist_ok=True)
    report_hashes = {relative: sha256_file(root / relative) for relative in PRESERVED_REPORTS}
    inputs = [
        root / "analysis_wsp" / "data" / "features_instances.csv",
        root / "analysis_wsp" / "data" / "performance_by_algorithm.csv",
        root / "analysis_wsp" / "configs" / "tcc_essential.json",
        root / "docs" / "metodologia.md",
        *(root / "third_party" / "Wildfire-Suppression" / "instances").glob("*.zip"),
        *(root / relative for relative in PRESERVED_REPORTS),
    ]
    write_json(
        output / "baseline_manifest_before.json",
        {
            "expected_unique_instances": 55,
            "observed_unique_instances": len(records),
            "files": file_manifest(inputs, root),
            "preserved_report_hashes": report_hashes,
            "result_provenance": "original_source",
        },
    )
    shutil.copy2(root / "analysis_wsp" / "data" / "features_instances.csv", output / "features_instances_original.csv")
    shutil.copy2(root / "analysis_wsp" / "data" / "performance_by_algorithm.csv", output / "performance_by_algorithm_original.csv")
    pd.DataFrame([base_features(record) for _, record in sorted(records.items())]).to_csv(
        output / "features_recomputed_measured.csv", index=False,
    )
    return report_hashes


def _validate_baseline_feature_recomputation(
    original: pd.DataFrame,
    recomputed_path: Path,
    output_path: Path,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    recomputed = pd.read_csv(recomputed_path)
    columns = numeric_feature_columns(original)
    missing = sorted(set(columns) - set(recomputed.columns))
    rows: list[dict[str, Any]] = []
    maximum = math.inf if missing else 0.0
    worst_feature: str | None = None
    if not missing:
        joined = original[["instance", *columns]].merge(
            recomputed[["instance", *columns]],
            on="instance",
            suffixes=("_original", "_recomputed"),
            validate="one_to_one",
        )
        if len(joined) != len(original):
            raise RuntimeError("A recomputação não alinhou as 55 instâncias da base")
        for column in columns:
            left = pd.to_numeric(joined[f"{column}_original"], errors="raise").to_numpy(dtype=float)
            right = pd.to_numeric(joined[f"{column}_recomputed"], errors="raise").to_numpy(dtype=float)
            error = float(np.max(np.abs(left - right)))
            rows.append(
                {
                    "feature": column,
                    "max_absolute_error": error,
                    "result_provenance": "measured",
                },
            )
            if error >= maximum:
                maximum = error
                worst_feature = column
    details_path = output_path.with_name(f"{output_path.stem}_by_feature.csv")
    pd.DataFrame(rows).to_csv(details_path, index=False)
    result = {
        "status": "passed" if not missing and len(original) == 55 and maximum <= tolerance else "failed",
        "instances_checked": int(len(original)),
        "features_checked": len(columns),
        "missing_features": missing,
        "max_absolute_error": maximum,
        "worst_feature": worst_feature,
        "tolerance": tolerance,
        "details_csv": details_path.name,
        "result_provenance": "measured",
    }
    write_json(output_path, result)
    if result["status"] != "passed":
        raise RuntimeError(
            "Features originais divergem da recomputação a partir das instâncias: "
            f"feature={worst_feature}; erro={maximum}; tolerância={tolerance}; ausentes={missing}",
        )
    return result


def _verify_baseline_reports(root: Path, before: dict[str, str], output: Path) -> None:
    after = {relative: sha256_file(root / relative) for relative in PRESERVED_REPORTS}
    result = {"before": before, "after": after, "byte_identical": before == after, "result_provenance": "measured"}
    write_json(output / "baseline_manifest_after.json", result)
    if before != after:
        raise RuntimeError("Um relatório-base foi modificado durante a execução")


def _augment_features(original: pd.DataFrame, topology: pd.DataFrame, output: Path) -> pd.DataFrame:
    topology_columns = [column for column in topology if column.startswith("topo_")]
    augmented = original.merge(topology[["instance", *topology_columns]], on="instance", validate="one_to_one")
    augmented["result_provenance"] = "original_source"
    augmented.to_csv(output / "all_augmented_candidates.csv", index=False)
    write_json(
        output / "all_augmented_candidates_feature_provenance.json",
        {
            "feature_provenance": {
                **{column: "original_source" for column in numeric_feature_columns(original)},
                **{column: "measured" for column in topology_columns},
            },
            "result_provenance": "measured",
        },
    )
    prepared, _ = audit_feature_table(augmented, output, prefix="augmented")
    return prepared


def _run_matilda(
    name: str,
    features: pd.DataFrame,
    directory: Path,
    performance: pd.DataFrame,
    labels: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
    epsilon = config.primary_threshold
    folds = threshold_fold(labels, epsilon)
    if folds is None:
        raise RuntimeError("O limiar pré-fixado não possui folds estimáveis")
    run_directory = directory / "threshold_10"
    data_hash = instance_space_input_hash(features, performance, epsilon, folds)
    print(f"[MATILDA] {name}, epsilon={epsilon:.2f}, folds={folds}", flush=True)
    return run_instance_space(
        features,
        performance,
        run_directory,
        epsilon=epsilon,
        folds=folds,
        workers=config.workers,
        label_counts=labels,
        input_data_hash=data_hash,
    )


def _artifact_provenance(path: Path, result_root: Path) -> str:
    relative = str(path.relative_to(result_root)).lower()
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path, usecols=lambda column: column == "result_provenance")
        except (OSError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError):
            frame = pd.DataFrame()
        if "result_provenance" in frame and not frame.empty:
            values = set(frame["result_provenance"].dropna().astype(str))
            validate_provenance(values)
            if values == {"original_source"}:
                return "original_source"
    if any(token in relative for token in ("pythia", "trace", "prediction", "instance_space_model.pkl")):
        return "predicted"
    return "measured"


def _validate_output_provenance(result_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    checked = 0
    for path in sorted(path for path in result_root.rglob("*") if path.is_file()):
        provenance = _artifact_provenance(path, result_root)
        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(path, usecols=lambda column: column == "result_provenance")
            except (OSError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError):
                frame = pd.DataFrame()
            if "result_provenance" in frame:
                validate_provenance(frame["result_provenance"].dropna().astype(str))
                checked += 1
        rows.append({"artifact_path": str(path.relative_to(result_root)), "content_provenance": provenance, "result_provenance": "measured"})
    for name in ("artifact_provenance_manifest.csv", "provenance_validation.json", "terminal_acceptance_validation.json", "final_report_manifest.json", "sha256_manifest.json"):
        rows.append({"artifact_path": name, "content_provenance": "measured", "result_provenance": "measured"})
    manifest = pd.DataFrame(rows).drop_duplicates().sort_values("artifact_path", kind="mergesort")
    manifest.to_csv(result_root / "artifact_provenance_manifest.csv", index=False)
    result = {
        "status": "passed",
        "csv_files_with_provenance_checked": checked,
        "artifact_files_covered": int(manifest["artifact_path"].nunique()),
        "result_provenance": "measured",
    }
    write_json(result_root / "provenance_validation.json", result)
    return result


def _validate_terminal(result_root: Path, config: ExperimentConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: dict[str, Any]) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details, "result_provenance": "measured"})

    before = _json(result_root / "00_baseline_55" / "baseline_manifest_before.json")
    after = _json(result_root / "00_baseline_55" / "baseline_manifest_after.json")
    check("baseline_preserved", before.get("observed_unique_instances") == 55 and after.get("byte_identical") is True, {"instances": before.get("observed_unique_instances")})
    baseline_features = _json(result_root / "00_baseline_55" / "baseline_feature_recomputation_validation.json")
    check("baseline_features_recomputed", baseline_features.get("status") == "passed", baseline_features)

    expected_topology = {
        f"topo_{metric}_{aggregate}"
        for metrics in TOPOLOGY_FAMILIES.values()
        for metric in metrics
        for aggregate in AGGREGATES
    }
    topology_candidates = _csv(result_root / "02_instancespace_topology" / "topology_candidates.csv")
    observed_topology = {column for column in topology_candidates.columns if column.startswith("topo_")}
    missing_topology = sorted(expected_topology - observed_topology)
    check(
        "attachment_metric_catalog_complete",
        len(topology_candidates) == 55
        and topology_candidates.get("instance", pd.Series(dtype=str)).nunique() == 55
        and not missing_topology,
        {
            "instances": len(topology_candidates),
            "unique_instances": int(topology_candidates.get("instance", pd.Series(dtype=str)).nunique()),
            "expected_aggregated_features": len(expected_topology),
            "observed_aggregated_features": len(observed_topology),
            "missing": missing_topology,
        },
    )
    topology_validation = _json(result_root / "02_instancespace_topology" / "exact_ga_topology_recomputation_validation.json")
    check("canonical_topology_recomputed", topology_validation.get("status") == "passed", topology_validation)

    augmented_audit = _csv(result_root / "02_instancespace_topology" / "augmented_feature_audit.csv")
    actual_topology_inputs = set(
        augmented_audit.loc[
            augmented_audit.get("decision", pd.Series(dtype=str)).eq("kept")
            & augmented_audit.get("feature", pd.Series(dtype=str)).astype(str).str.startswith("topo_"),
            "feature",
        ].astype(str)
    ) if not augmented_audit.empty and {"feature", "decision"}.issubset(augmented_audit) else set()
    prelim = _csv(result_root / "02_instancespace_topology" / "threshold_10" / "prelim_feature_parameters.csv")
    prelim_topology = set(
        prelim.loc[prelim.get("source_feature", pd.Series(dtype=str)).astype(str).str.startswith("topo_"), "source_feature"].astype(str)
    ) if not prelim.empty and "source_feature" in prelim else set()
    options = _json(result_root / "02_instancespace_topology" / "threshold_10" / "resolved_options.json")
    normalization_enabled = options.get("norm", {}).get("flag") is True
    automatic_preprocessing = options.get("auto", {}).get("preproc") is True
    check(
        "matilda_boxcox_and_zscore_recorded",
        bool(actual_topology_inputs)
        and actual_topology_inputs == prelim_topology
        and normalization_enabled
        and automatic_preprocessing,
        {
            "topology_inputs": len(actual_topology_inputs),
            "prelim_topology_parameters": len(prelim_topology),
            "missing_prelim_parameters": sorted(actual_topology_inputs - prelim_topology),
            "automatic_preprocessing": automatic_preprocessing,
            "normalization_enabled": normalization_enabled,
        },
    )
    projection = _json(result_root / "05_final_isa" / "ga_projection_validation.json")
    check("candidate_projection_reproduces_isa", projection.get("status") == "passed", projection)
    for name, directory in (("matilda_original", "01_instancespace_original"), ("matilda_topology", "02_instancespace_topology")):
        state = _json(result_root / directory / "threshold_10" / "run_status.json").get("status")
        check(name, state == "completed", {"status": state})
    embedding = _json(result_root / "05_final_isa" / "embedding_comparison_controle_vs_topologia.json")
    isa_artifacts = _json(result_root / "05_final_isa" / "isa_artifact_comparison_controle_vs_topologia.json")
    check(
        "control_vs_topology_compared",
        embedding.get("n_instances") == 55
        and isa_artifacts.get("sifted_reference_count") is not None
        and isa_artifacts.get("sifted_candidate_count") is not None,
        {"embedding": embedding, "isa_artifacts": isa_artifacts},
    )

    ga = _csv(result_root / "04a_ga_internal" / "ga_target_results_with_wall_auc.csv")
    expected_seeds = set(config.root_seeds)
    observed_seeds = set(pd.to_numeric(ga.get("root_seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    per_seed = ga.groupby("root_seed").size().to_dict() if "root_seed" in ga else {}
    duplicate_targets = int(ga.duplicated(["root_seed", "target_id"]).sum()) if {"root_seed", "target_id"}.issubset(ga) else len(ga)
    completed_budget = (
        pd.to_numeric(ga.get("evaluations_completed", pd.Series(dtype=float)), errors="coerce")
        .eq(config.ga_evaluation_budget)
        .all()
        if not ga.empty
        else False
    )
    exact_finalists = ga.get("best_exact_candidate_id", pd.Series(index=ga.index, dtype=object)).notna().all() if not ga.empty else False
    check(
        "ga_internal_completed",
        observed_seeds == expected_seeds
        and all(int(per_seed.get(seed, 0)) == config.target_budget for seed in expected_seeds)
        and len(ga) == len(expected_seeds) * config.target_budget
        and duplicate_targets == 0
        and completed_budget
        and exact_finalists,
        {
            "expected_seeds": sorted(expected_seeds),
            "observed_seeds": sorted(observed_seeds),
            "targets_by_seed": per_seed,
            "duplicate_targets": duplicate_targets,
            "evaluation_budget_per_target": config.ga_evaluation_budget,
            "all_budgets_completed": bool(completed_budget),
            "all_finalists_recomputed_exactly": bool(exact_finalists),
        },
    )

    invalid_artifacts: list[dict[str, Any]] = []
    if not ga.empty and {"root_seed", "best_exact_candidate_id", "best_exact_instance_sha256"}.issubset(ga):
        for row in ga.itertuples(index=False):
            identifier = getattr(row, "best_exact_candidate_id", None)
            seed = getattr(row, "root_seed", None)
            expected_hash = getattr(row, "best_exact_instance_sha256", None)
            if pd.isna(identifier) or pd.isna(seed) or pd.isna(expected_hash):
                invalid_artifacts.append({"seed": seed, "candidate": identifier, "reason": "missing_reference"})
                continue
            instance_path = (
                result_root
                / "04a_ga_internal"
                / f"seed_{int(seed)}"
                / "generated_candidates"
                / f"{identifier}.instance.json"
            )
            try:
                payload = json.loads(instance_path.read_text(encoding="utf-8"))
                observed_hash = instance_content_hash(payload)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                invalid_artifacts.append(
                    {"seed": int(seed), "candidate": str(identifier), "reason": f"{type(exc).__name__}: {exc}"},
                )
                continue
            if observed_hash != str(expected_hash):
                invalid_artifacts.append(
                    {
                        "seed": int(seed),
                        "candidate": str(identifier),
                        "reason": "instance_hash_mismatch",
                        "expected": str(expected_hash),
                        "observed": observed_hash,
                    },
                )
    else:
        invalid_artifacts.append({"reason": "ga_result_columns_missing"})
    check(
        "exact_generated_instances_persisted",
        not ga.empty and not invalid_artifacts,
        {"instances_checked": len(ga) - len(invalid_artifacts), "invalid": invalid_artifacts},
    )

    accepted_rows = ga[ga.get("accepted", pd.Series(False, index=ga.index)).fillna(False).astype(bool)] if not ga.empty else ga
    invalid = accepted_rows[
        ~accepted_rows.get("domain_membership_confirmed_exact", pd.Series(False, index=accepted_rows.index)).fillna(False).astype(bool)
    ]
    check(
        "accepted_instances_recomputed_in_domain",
        not ga.empty and invalid.empty,
        {
            "accepted_rows": len(accepted_rows),
            "invalid_rows": len(invalid),
            "zero_acceptances_is_a_valid_negative_result": len(accepted_rows) == 0,
        },
    )
    status = "passed" if all(item["passed"] for item in checks) else "failed"
    result = {"status": status, "checks_total": len(checks), "checks_passed": sum(item["passed"] for item in checks), "checks": checks, "result_provenance": "measured"}
    write_json(result_root / "terminal_acceptance_validation.json", result)
    if status != "passed":
        raise RuntimeError("A validação terminal do TCC não foi aprovada")
    return result


def run_pipeline(root: Path, config: ExperimentConfig, *, resume: bool, command: list[str]) -> Path:
    started = time.perf_counter()
    result_root = root / "experiments" / "generation" / config.profile
    result_root.mkdir(parents=True, exist_ok=True)
    if not config.exact_topology:
        raise ValueError("O perfil do TCC exige topologia exata")
    if config.thresholds != (0.10,) or config.primary_threshold != 0.10:
        raise ValueError("O perfil do TCC exige exclusivamente o limiar MATILDA de 10%")

    resolved_config_path = result_root / "experiment_config_resolved.json"
    current_config = config.to_dict()
    current_environment = environment_manifest(root, command)
    current_environment["topology_hits_backend_id"] = HITS_BACKEND_ID
    if resume:
        previous_config = _json(resolved_config_path)
        previous_environment = _json(result_root / "environment_manifest.json")
        if not previous_config or not previous_environment:
            raise RuntimeError("Perfil sem configuração/ambiente inicial completo; retomada recusada")
        scientific_previous = {key: value for key, value in previous_config.items() if key != "workers"}
        scientific_current = {key: value for key, value in current_config.items() if key != "workers"}
        if scientific_previous != scientific_current:
            raise RuntimeError("A configuração científica do perfil difere da configuração da retomada")
        if previous_environment.get("root_git_commit") != current_environment.get("root_git_commit"):
            raise RuntimeError("A retomada exige exatamente o mesmo commit de código da execução inicial")
        resume_stamp = time.strftime("%Y%m%dT%H%M%S")
        write_json(result_root / f"environment_manifest_resume_{resume_stamp}.json", current_environment)
    else:
        write_json(resolved_config_path, current_config)
        write_json(result_root / "environment_manifest.json", current_environment)

    for name in RESULT_DIRECTORIES:
        (result_root / name).mkdir(exist_ok=True)
    _checkpoint(
        result_root / "pipeline_status.json",
        "running",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        resume=resume,
        result_provenance="measured",
    )

    records = load_wildfire_instances(root)
    report_hashes = _freeze_baseline(root, result_root, records)
    original = load_original_features(root)
    _validate_baseline_feature_recomputation(
        original,
        result_root / "00_baseline_55" / "features_recomputed_measured.csv",
        result_root / "00_baseline_55" / "baseline_feature_recomputation_validation.json",
    )
    performance_raw, performance = load_performance_matrix(root)
    performance_raw.to_csv(result_root / "00_baseline_55" / "performance_with_provenance.csv", index=False)

    topology_dir = result_root / "02_instancespace_topology"
    print("[topologia] cálculo exato das 55 instâncias", flush=True)
    precompute_exact_topology_cache(records, topology_dir, workers=config.workers)
    candidates, diagnostics = build_topology_table(
        records,
        topology_dir,
        exact_betweenness=True,
        max_exact_vertices=max(int(record["data"]["|V|"]) for record in records.values()),
        fallback_landmarks=0,
    )
    validate_exact_topology_recomputation(records, candidates, topology_dir / "exact_ga_topology_recomputation_validation.json")
    topology, _, _ = audit_and_prepare_topology(candidates, diagnostics, topology_dir)

    labels = label_audit(performance, config.thresholds)
    labels.to_csv(result_root / "05_final_isa" / "label_class_counts_and_folds.csv", index=False)
    original_prepared, _ = audit_feature_table(original, result_root / "01_instancespace_original", prefix="original")
    augmented = _augment_features(original_prepared, topology, topology_dir)
    original_result = _run_matilda("controle", original_prepared, result_root / "01_instancespace_original", performance, labels, config)
    augmented_result = _run_matilda("topologia", augmented, topology_dir, performance, labels, config)
    compare_embeddings(original_result["coordinates"], augmented_result["coordinates"], result_root / "05_final_isa", "controle_vs_topologia")
    compare_isa_artifacts(
        result_root / "01_instancespace_original" / "threshold_10",
        topology_dir / "threshold_10",
        result_root / "05_final_isa",
        "controle_vs_topologia",
    )

    coordinates, model, space = augmented_result["coordinates"], augmented_result["model"], augmented_result["space"]
    internal_dir = result_root / "04a_ga_internal"
    targets = make_initial_targets(
        coordinates,
        model,
        internal_dir,
        count=min(6, config.batch_size),
        sobol_seed=config.baseline_sobol_seed,
        sobol_points=config.sobol_points,
        config_hash=stable_json_hash(config.to_dict()),
    )

    evaluator = lambda directory: CandidateEvaluator(root, space, augmented, directory, search_landmarks=0)
    projection_evaluator = evaluator(result_root / "05_final_isa" / "projection_validation_cache")
    validate_candidate_projection(projection_evaluator, augmented, coordinates, result_root / "05_final_isa" / "ga_projection_validation.json", baseline_records=records)
    print("[GA] povoamento interno", flush=True)
    ga = run_ga_experiment(root, internal_dir, targets, coordinates, model, evaluator, config)
    ga = consolidate_wall_auc(internal_dir, ga)
    summarize_reach(internal_dir, coordinates, ga, reference_seed=config.root_seeds[0], sobol_seed=config.baseline_sobol_seed, sobol_points=config.sobol_points, cloister=cloister_polygon(model))

    final = result_root / "05_final_isa"
    primary = topology_dir / "threshold_10"
    shutil.copytree(primary, final / "matilda_primary", dirs_exist_ok=True)
    shutil.copy2(primary / "resolved_options.json", final / "resolved_options.json")
    _verify_baseline_reports(root, report_hashes, result_root / "00_baseline_55")
    generate_analysis_figures(result_root)
    provenance = _validate_output_provenance(result_root)
    _validate_terminal(result_root, config)
    report = write_report(root, result_root, config, provenance)
    write_json(result_root / "final_report_manifest.json", {"path": str(report.relative_to(root)), "size": report.stat().st_size, "sha256": sha256_file(report), "result_provenance": "measured"})
    _checkpoint(result_root / "pipeline_status.json", "completed", elapsed_seconds=time.perf_counter() - started, result_provenance="measured")
    manifest = result_root / "sha256_manifest.json"
    files = [path for path in result_root.rglob("*") if path.is_file() and path != manifest]
    write_json(manifest, file_manifest(files, result_root))
    return result_root
