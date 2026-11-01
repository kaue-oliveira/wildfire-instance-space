from __future__ import annotations

import json
import math
import pickle
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .data import numeric_feature_columns
from .provenance import sha256_bytes, write_json


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def _prelim_numeric_degenerate(values: np.ndarray) -> bool:
    """Detect columns that collapse under pyInstanceSpace's PRELIM shift.

    The fixed pyInstanceSpace commit applies ``x - min(x) + 1`` before its
    Box-Cox transform.  Values whose whole range is below floating-point
    resolution around one can therefore be non-constant in the source table
    yet become exactly constant inside PRELIM, yielding a zero standard
    deviation and NaNs.  This is a degeneracy check, not imputation.
    """
    if values.size < 2 or not np.isfinite(values).all():
        return False
    shifted = values - float(np.min(values)) + 1.0
    return np.unique(shifted).size <= 1 or float(np.std(shifted, ddof=1)) == 0.0


def instance_space_input_hash(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    epsilon: float,
    folds: int,
) -> str:
    """Hash the concrete matrix and label configuration behind an ISA run."""
    payload = bytearray()
    for frame in (features.reset_index(drop=True), performance.reset_index(drop=True)):
        payload.extend("\x1f".join(map(str, frame.columns)).encode("utf-8"))
        payload.extend(pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64).tobytes())
    payload.extend(np.asarray([float(epsilon), float(folds)], dtype=np.float64).tobytes())
    return sha256_bytes(bytes(payload))


def audit_feature_table(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only the pre-ISA exclusions authorised by the protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = numeric_feature_columns(frame)
    keep: list[str] = []
    vectors: dict[bytes, str] = {}
    rows: list[dict[str, Any]] = []
    for column in feature_cols:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        unique = int(pd.Series(values).nunique(dropna=False))
        q25, q75 = np.nanquantile(values, [0.25, 0.75]) if finite.any() else (math.nan, math.nan)
        iqr = float(q75 - q25)
        decision = "kept"
        duplicate_of: str | None = None
        if not finite.all():
            decision = "invalid_nonfinite"
        elif unique <= 1 or float(np.var(values)) == 0.0:
            decision = "constant_or_degenerate"
        elif _prelim_numeric_degenerate(values):
            decision = "prelim_numeric_degenerate"
        else:
            key = np.ascontiguousarray(values).tobytes()
            if key in vectors:
                decision = "exact_duplicate"
                duplicate_of = vectors[key]
            else:
                vectors[key] = column
                keep.append(column)
        rows.append(
            {
                "feature": column,
                "iqr": iqr,
                "iqr_zero": bool(iqr == 0),
                "unique_values": unique,
                "all_finite": bool(finite.all()),
                "decision": decision,
                "duplicate_of": duplicate_of,
                "result_provenance": "measured",
            },
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / f"{prefix}_feature_audit.csv", index=False)
    meta = [column for column in frame.columns if column not in feature_cols]
    prepared = frame[[*meta, *keep]].copy()
    prepared.to_csv(output_dir / f"{prefix}_features_for_isa.csv", index=False)
    return prepared, audit


def label_audit(performance: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    algorithms = [column for column in performance.columns if column != "instance"]
    for threshold in thresholds:
        for algorithm in algorithms:
            valid = pd.to_numeric(performance[algorithm], errors="coerce")
            missing = int(valid.isna().sum())
            observed = valid.dropna()
            positive = int((observed <= threshold).sum())
            negative = int((observed > threshold).sum())
            smaller = min(positive, negative)
            estimable = missing == 0 and smaller >= 2
            rows.append(
                {
                    "threshold": float(threshold),
                    "algorithm": algorithm,
                    "positive": positive,
                    "negative": negative,
                    "missing": missing,
                    "status": "estimable" if estimable else "not_estimable",
                    "cv_folds": min(5, smaller) if estimable else None,
                    "result_provenance": "measured" if estimable else "missing",
                },
            )
    return pd.DataFrame(rows)


def threshold_fold(label_counts: pd.DataFrame, threshold: float) -> int | None:
    subset = label_counts[np.isclose(label_counts["threshold"], threshold)]
    if subset.empty or (subset["status"] != "estimable").any():
        return None
    return int(subset["cv_folds"].min())


def resolved_instance_space_options(workers: int, epsilon: float, folds: int) -> tuple[Any, dict[str, Any]]:
    # pyInstanceSpace 0.2.1 accepts `uselibsvm` as the legacy input alias for the
    # dataclass field `use_grid_search`; the resolved object below is authoritative.
    from instancespace.data.options import InstanceSpaceOptions, PrelimOptions

    raw = {
        "parallel": {"flag": True, "ncores": int(workers)},
        "perf": {"MaxPerf": False, "AbsPerf": True, "epsilon": float(epsilon), "betaThreshold": 0.55},
        "auto": {"preproc": True},
        "bound": {"flag": False},
        "norm": {"flag": True},
        "selvars": {
            "smallScaleFlag": False,
            "smallScale": 0.5,
            "fileIdxFlag": False,
            "fileIdx": "",
            "feats": None,
            "algos": None,
            "type": "Ftr&Good",
            "mindistance": 0.1,
            "densityFlag": False,
        },
        "sifted": {"flag": True, "rho": 0.1, "K": 10, "NTREES": 50, "MaxIter": 1000, "Replicates": 100},
        "pilot": {"analytic": False, "ntries": 5},
        "cloister": {"pval": 0.05, "cthres": 0.7},
        "pythia": {"cvFolds": int(folds), "isPolyKrnl": False, "useWeights": True, "uselibsvm": True},
        "trace": {"usesim": True, "PI": 0.55},
        "outputs": {"csv": True, "web": True, "png": True},
    }
    options = InstanceSpaceOptions.from_dict(raw)
    resolved = asdict(options)
    resolved["prelim"] = asdict(PrelimOptions.from_options(options))
    resolved["requested_notation"] = {
        "selvars.type": "Ftr&&Good",
        "pythia.use_grid_search": True,
    }
    resolved["commit_compatibility_normalization"] = {
        "selvars.type": "Ftr&Good",
        "pythia.input_alias": "uselibsvm",
    }
    resolved["pythia_implementation"] = {
        "estimator": "sklearn.svm.SVC",
        "kernel": "rbf",
        "optimizer_when_use_grid_search_true": "sklearn.model_selection.RandomizedSearchCV",
        "n_iter": 30,
        "optimizer_random_state": 0,
        "parameter_space_generation": "LatinHypercube(d=2, n=30), C and gamma = 2**U[-10,4]",
        "parameter_space_seed": 0,
        "cross_validation": "StratifiedKFold(shuffle=True, random_state=0)",
        "class_weighting": "absolute deviation of algorithm performance from its mean, passed as sample_weight",
    }
    return options, resolved


def make_metadata(features: pd.DataFrame, performance: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    from instancespace.data.metadata import Metadata

    joined = features.merge(performance, on="instance", how="inner", validate="one_to_one")
    if len(joined) != len(features):
        raise ValueError("Features e desempenho não puderam ser alinhados um a um")
    feature_cols = numeric_feature_columns(features)
    algorithms = [column for column in performance if column != "instance"]
    metadata_columns: dict[str, Any] = {
        "instances": joined["instance"].astype(str),
        "source": joined.get("group", "baseline"),
    }
    for column in feature_cols:
        metadata_columns[f"feature_{_safe_name(column)}"] = pd.to_numeric(joined[column], errors="raise").astype(float)
    for algorithm in algorithms:
        metadata_columns[f"algo_{_safe_name(algorithm)}"] = pd.to_numeric(joined[algorithm], errors="raise").astype(float)
    metadata = pd.DataFrame(metadata_columns)
    if not np.isfinite(metadata.filter(regex="^(feature|algo)_").to_numpy(dtype=float)).all():
        raise ValueError("Metadata contém valor não finito; imputação é proibida")
    return metadata, Metadata.from_data_frame(metadata)


def _jsonable_parameters(model: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, svm, c, gamma in zip(model.data.algo_labels, model.pythia.svm, model.pythia.box_consnt, model.pythia.k_scale):
        rows.append(
            {
                "algorithm": name,
                "C": float(c),
                "gamma": float(gamma),
                "kernel": str(svm.kernel),
                "result_provenance": "predicted",
            },
        )
    return rows


def repeated_pythia_validation(model: Any, output_dir: Path, threshold: float, folds: int) -> pd.DataFrame:
    """Ten independently shuffled stratified evaluations using learned SVM parameters."""
    z = np.asarray(model.pilot.z, dtype=float)
    labels = np.asarray(model.data.y_bin, dtype=bool)
    rows: list[dict[str, Any]] = []
    for algorithm_index, algorithm in enumerate(model.data.algo_labels):
        y = labels[:, algorithm_index]
        if min(int(y.sum()), int((~y).sum())) < folds:
            rows.append(
                {
                    "algorithm": algorithm,
                    "threshold": threshold,
                    "seed": None,
                    "folds": None,
                    "status": "not_estimable",
                    "result_provenance": "missing",
                },
            )
            continue
        learned = model.pythia.svm[algorithm_index]
        for seed in range(10):
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            prediction = np.zeros(len(y), dtype=bool)
            score = np.zeros(len(y), dtype=float)
            for train, test in splitter.split(z, y):
                classifier = SVC(
                    kernel="rbf",
                    C=float(learned.C),
                    gamma=float(learned.gamma),
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                )
                classifier.fit(z[train], y[train])
                prediction[test] = classifier.predict(z[test])
                score[test] = classifier.predict_proba(z[test])[:, 1]
            rows.append(
                {
                    "algorithm": algorithm,
                    "threshold": threshold,
                    "seed": seed,
                    "folds": folds,
                    "positive": int(y.sum()),
                    "negative": int((~y).sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
                    "precision": float(precision_score(y, prediction, zero_division=0)),
                    "recall": float(recall_score(y, prediction, zero_division=0)),
                    "f1": float(f1_score(y, prediction, zero_division=0)),
                    "mcc": float(matthews_corrcoef(y, prediction)),
                    "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else math.nan,
                    "status": "estimated",
                    "result_provenance": "predicted",
                },
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "pythia_repeated_cv_raw.csv", index=False)
    numeric = ["balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]
    valid = raw[raw["status"] == "estimated"]
    summary_rows: list[dict[str, Any]] = []
    for algorithm, group in valid.groupby("algorithm"):
        row: dict[str, Any] = {"algorithm": algorithm, "threshold": threshold, "repetitions": len(group), "result_provenance": "predicted"}
        for metric in numeric:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "pythia_repeated_cv_summary.csv", index=False)
    return raw


def run_instance_space(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    output_dir: Path,
    *,
    epsilon: float,
    folds: int,
    workers: int,
    label_counts: pd.DataFrame,
    input_data_hash: str | None = None,
    performance_provenance: str = "original_source",
    feature_provenance: str | None = None,
    sifted_revision: str = "upstream",
) -> dict[str, Any]:
    from instancespace import InstanceSpace
    from contextlib import nullcontext

    correction_identity = None
    if sifted_revision != "upstream":
        from analysis_wsp.matilda74.sifted_correction import REVISION, identity, corrected_sifted
        if sifted_revision != REVISION:
            raise ValueError(f"Unknown SIFTED revision: {sifted_revision}")
        correction_identity = identity()

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_input_hash = input_data_hash or instance_space_input_hash(features, performance, epsilon, folds)
    metadata_frame, metadata = make_metadata(features, performance)
    metadata_frame.to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(
        {
            "instance": metadata_frame["instances"],
            "feature_result_provenance": feature_provenance or (
                "measured" if features["result_provenance"].eq("measured").all() else "original_source"
            ),
            "performance_result_provenance": performance_provenance,
        },
    ).to_csv(output_dir / "metadata_provenance.csv", index=False)
    feature_provenance = [
        {
            "metadata_field": f"feature_{_safe_name(column)}",
            "source_feature": column,
            "result_provenance": feature_provenance or ("measured" if column.startswith("topo_") else "original_source"),
        }
        for column in numeric_feature_columns(features)
    ]
    algorithm_provenance = [
        {
            "metadata_field": f"algo_{_safe_name(column)}",
            "source_feature": column,
            "result_provenance": performance_provenance,
        }
        for column in performance
        if column != "instance"
    ]
    pd.DataFrame([*feature_provenance, *algorithm_provenance]).to_csv(
        output_dir / "metadata_field_provenance.csv",
        index=False,
    )
    options, resolved = resolved_instance_space_options(workers, epsilon, folds)
    resolved["labels"] = label_counts[np.isclose(label_counts["threshold"], epsilon)].to_dict(orient="records")
    resolved["concrete_cv_folds"] = folds
    resolved["analysis_input_hash"] = analysis_input_hash
    resolved["sifted_correction"] = correction_identity
    write_json(output_dir / "resolved_options.json", resolved)
    started = time.perf_counter()
    pickle_path = output_dir / "instance_space_model.pkl"
    build_checkpoint = {}
    try:
        build_checkpoint = json.loads((output_dir / "build_completed.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if (
        pickle_path.exists()
        and build_checkpoint.get("status") == "completed"
        and build_checkpoint.get("analysis_input_hash") == analysis_input_hash
        and build_checkpoint.get("sifted_revision", "upstream") == sifted_revision
        and build_checkpoint.get("sifted_correction") == correction_identity
    ):
        with pickle_path.open("rb") as handle:
            model = pickle.load(handle)
        space = InstanceSpace(metadata, options)
        space._model = model  # compatibility restoration for the fixed 0.2.1 commit
        elapsed = 0.0
    else:
        space = InstanceSpace(metadata, options)
        context = (corrected_sifted(output_dir / "sifted_correction_trace.json")
                   if correction_identity else nullcontext())
        with context:
            model = space.build()
        elapsed = time.perf_counter() - started
        try:
            with pickle_path.open("wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            write_json(
                output_dir / "build_completed.json",
                {
                    "status": "completed",
                    "elapsed_seconds": elapsed,
                    "analysis_input_hash": analysis_input_hash,
                    "sifted_revision": sifted_revision,
                    "sifted_correction": correction_identity,
                    "result_provenance": "measured",
                },
            )
        except (OSError, pickle.PickleError, AttributeError, TypeError) as exc:
            pickle_path.unlink(missing_ok=True)
            write_json(
                output_dir / "build_completed.json",
                {
                    "status": "completed_without_pickle",
                    "elapsed_seconds": elapsed,
                    "analysis_input_hash": analysis_input_hash,
                    "sifted_revision": sifted_revision,
                    "sifted_correction": correction_identity,
                    "pickle_error": f"{type(exc).__name__}: {exc}",
                    "result_provenance": "measured",
                },
            )
    input_feature_names = numeric_feature_columns(features)
    if not (
        len(input_feature_names)
        == len(model.prelim.lambda_x)
        == len(model.prelim.mu_x)
        == len(model.prelim.sigma_x)
    ):
        raise RuntimeError("Parâmetros PRELIM não alinham com as features de entrada da MATILDA")
    pd.DataFrame(
        {
            "source_feature": input_feature_names,
            "metadata_field": [f"feature_{_safe_name(name)}" for name in input_feature_names],
            "boxcox_lambda": np.asarray(model.prelim.lambda_x, dtype=float),
            "mean_after_boxcox": np.asarray(model.prelim.mu_x, dtype=float),
            "std_after_boxcox": np.asarray(model.prelim.sigma_x, dtype=float),
            "positive_shift_before_boxcox": True,
            "zscore_after_boxcox": True,
            "result_provenance": "measured",
        },
    ).to_csv(output_dir / "prelim_feature_parameters.csv", index=False)
    csv_dir, web_dir, png_dir = output_dir / "csv", output_dir / "web", output_dir / "png"
    csv_dir.mkdir(exist_ok=True)
    web_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)
    selected_indices = np.asarray(model.sifted.selvars).astype(int).ravel()
    # pyInstanceSpace 0.2.1 keeps data.x before SIFTED but replaces feat_labels
    # with the selected labels. Its serializers assume both have the selected
    # width. Build a serialization-only immutable view; the trained model and
    # explore pipeline remain untouched.
    serial_idx = np.arange(len(selected_indices), dtype=int)
    serial_data = replace(model.data, x_raw=model.data.x_raw[:, selected_indices])
    serial_dense = replace(model.data_dense, x_raw=model.data_dense.x_raw[:, selected_indices])
    serial_model = replace(
        model,
        data=serial_data,
        data_dense=serial_dense,
        sifted=replace(model.sifted, idx=serial_idx),
        feat_sel=replace(model.feat_sel, idx=serial_idx),
    )
    serial_model.save_to_csv(csv_dir)
    serial_model.save_for_web(web_dir)
    serial_model.save_graphs(png_dir)
    coordinates = pd.DataFrame(
        {
            "instance": model.data.inst_labels.astype(str),
            "z1": model.pilot.z[:, 0],
            "z2": model.pilot.z[:, 1],
            "result_provenance": "measured",
        },
    )
    coordinates.to_csv(output_dir / "pilot_coordinates.csv", index=False)
    selected_names = list(model.data.feat_labels)
    write_json(output_dir / "sifted_selected_features.json", {"indices": selected_indices.tolist(), "features": selected_names, "result_provenance": "measured"})
    parameters = _jsonable_parameters(model)
    write_json(output_dir / "pythia_fitted_parameters.json", parameters)
    resolved["pythia_implementation"]["fitted_parameters"] = parameters
    resolved["sifted"]["selected_feature_indices"] = selected_indices.tolist()
    resolved["sifted"]["selected_feature_names"] = selected_names
    resolved["implementation_identity"] = {
        "pyinstancespace_version": "0.2.1",
        "pyinstancespace_commit": "289a940a79cee2001bd8c54621465ddc3e30a094",
        "known_internal_random_seeds": {"SIFTED": 0, "PYTHIA_parameter_space": 0, "PYTHIA_optimizer": 0, "PYTHIA_cv": 0},
        "serializer_compatibility_view": "x_raw restricted by sifted.selvars because commit serializer uses the unmodified idx field",
        "sifted_correction": correction_identity,
    }
    write_json(output_dir / "resolved_options.json", resolved)
    predictions: list[dict[str, Any]] = []
    for row_index, instance in enumerate(model.data.inst_labels.astype(str)):
        for algorithm_index, algorithm in enumerate(model.data.algo_labels):
            predictions.append(
                {
                    "instance": instance,
                    "algorithm": algorithm,
                    "predicted_good": bool(model.pythia.y_hat[row_index, algorithm_index]),
                    "predicted_probability": float(model.pythia.pr0_hat[row_index, algorithm_index]),
                    "result_provenance": "predicted",
                },
            )
    pd.DataFrame(predictions).to_csv(output_dir / "pythia_predictions.csv", index=False)
    repeated_pythia_validation(model, output_dir, epsilon, folds)
    status = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "n_instances": len(metadata_frame),
        "n_features_input": len(metadata.feature_names),
        "n_features_sifted": len(selected_names),
        "n_algorithms": len(metadata.algorithm_names),
        "epsilon": epsilon,
        "folds": folds,
        "analysis_input_hash": analysis_input_hash,
        "sifted_revision": sifted_revision,
        "sifted_correction": correction_identity,
        "result_provenance": "measured",
    }
    write_json(output_dir / "run_status.json", status)
    return {"model": model, "space": space, "status": status, "coordinates": coordinates}


def compare_embeddings(reference: pd.DataFrame, candidate: pd.DataFrame, output_dir: Path, label: str) -> dict[str, Any]:
    joined = reference.merge(candidate, on="instance", suffixes=("_ref", "_candidate"), validate="one_to_one")
    a = joined[["z1_ref", "z2_ref"]].to_numpy(dtype=float)
    b = joined[["z1_candidate", "z2_candidate"]].to_numpy(dtype=float)
    a_center = a - a.mean(axis=0)
    b_center = b - b.mean(axis=0)
    a_norm = a_center / max(np.linalg.norm(a_center), 1e-12)
    b_norm = b_center / max(np.linalg.norm(b_center), 1e-12)
    rotation, scale = orthogonal_procrustes(b_norm, a_norm)
    aligned = b_norm @ rotation * scale
    displacement = np.linalg.norm(a_norm - aligned, axis=1)
    k = min(5, len(joined) - 1)
    da, db = cdist(a_norm, a_norm), cdist(aligned, aligned)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    neigh_a = np.argsort(da, axis=1)[:, :k]
    neigh_b = np.argsort(db, axis=1)[:, :k]
    overlap = [len(set(x).intersection(y)) / k for x, y in zip(neigh_a, neigh_b)]
    result = {
        "comparison": label,
        "n_instances": len(joined),
        "procrustes_scale": float(scale),
        "mean_aligned_displacement": float(displacement.mean()),
        "median_aligned_displacement": float(np.median(displacement)),
        "neighbor_k": k,
        "mean_neighbor_overlap": float(np.mean(overlap)),
        "pairwise_distance_spearman": float(spearmanr(da[np.isfinite(da)], db[np.isfinite(db)]).statistic),
        "result_provenance": "measured",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"embedding_comparison_{_safe_name(label)}.json", result)
    return result


def compare_isa_artifacts(reference_dir: Path, candidate_dir: Path, output_dir: Path, label: str) -> dict[str, Any]:
    """Compare SIFTED, PYTHIA and TRACE products without changing their provenance."""
    selected_reference = json.loads((reference_dir / "sifted_selected_features.json").read_text(encoding="utf-8"))
    selected_candidate = json.loads((candidate_dir / "sifted_selected_features.json").read_text(encoding="utf-8"))
    reference_set = set(selected_reference.get("features", []))
    candidate_set = set(selected_candidate.get("features", []))
    union = reference_set | candidate_set
    result: dict[str, Any] = {
        "comparison": label,
        "sifted_reference_count": len(reference_set),
        "sifted_candidate_count": len(candidate_set),
        "sifted_intersection_count": len(reference_set & candidate_set),
        "sifted_jaccard": len(reference_set & candidate_set) / len(union) if union else 1.0,
        "sifted_reference_only": sorted(reference_set - candidate_set),
        "sifted_candidate_only": sorted(candidate_set - reference_set),
        "result_provenance": "measured",
    }
    reference_predictions = pd.read_csv(reference_dir / "pythia_predictions.csv")
    candidate_predictions = pd.read_csv(candidate_dir / "pythia_predictions.csv")
    predictions = reference_predictions.merge(
        candidate_predictions,
        on=["instance", "algorithm"],
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if not predictions.empty:
        result["pythia_binary_agreement"] = float(
            (predictions["predicted_good_reference"] == predictions["predicted_good_candidate"]).mean()
        )
        probabilities_reference = pd.to_numeric(predictions["predicted_probability_reference"], errors="coerce")
        probabilities_candidate = pd.to_numeric(predictions["predicted_probability_candidate"], errors="coerce")
        result["pythia_probability_mae"] = float(np.mean(np.abs(probabilities_reference - probabilities_candidate)))
        result["pythia_probability_spearman"] = float(
            spearmanr(probabilities_reference, probabilities_candidate).statistic
        )
        result["pythia_semantic_provenance"] = "predicted"
    reference_trace = pd.read_csv(reference_dir / "csv" / "footprint_performance.csv")
    candidate_trace = pd.read_csv(candidate_dir / "csv" / "footprint_performance.csv")
    trace = reference_trace.merge(candidate_trace, on="Row", suffixes=("_reference", "_candidate"), validate="one_to_one")
    numeric_reference = [column for column in trace if column.endswith("_reference")]
    trace_differences: list[float] = []
    for reference_column in numeric_reference:
        candidate_column = reference_column.removesuffix("_reference") + "_candidate"
        if candidate_column not in trace:
            continue
        left = pd.to_numeric(trace[reference_column], errors="coerce")
        right = pd.to_numeric(trace[candidate_column], errors="coerce")
        trace_differences.extend(np.abs(left - right).dropna().tolist())
    result["trace_footprint_metric_mae"] = float(np.mean(trace_differences)) if trace_differences else math.nan
    result["trace_algorithms_compared"] = int(len(trace))
    result["trace_semantic_provenance"] = "predicted"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"isa_artifact_comparison_{_safe_name(label)}.json", result)
    return result
