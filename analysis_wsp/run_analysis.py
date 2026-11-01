#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import heapq
import json
import math
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_wsp"
DATA_DIR = OUT / "data"
FIG_DIR = OUT / "figures"
STN_DIR = DATA_DIR / "stn"
STN_ORDERED_DIR = DATA_DIR / "stnOrdenado"
IBS_INSTANCE_DIR = DATA_DIR / "ibs_instances"

WF_DIR = ROOT / "third_party" / "Wildfire-Suppression"
IBS_DIR = ROOT / "third_party" / "Iterated-Beam-Search"
IBS_EXE = IBS_DIR / "build-plan-check" / "fire"

ARTICLE_STN_INSTANCES = [f"LA{i}" for i in range(8)] + [f"LB{i}" for i in range(8)]
STN_SELECTION = ARTICLE_STN_INSTANCES

STN_RUNS = 10
STN_MAXITER = 10
STN_TIMELIMIT = 30
STN_BETA = 25
STN_ETA = 35
STN_C = 10

OLD_IBS_NAMES = {
    **{f"LA{i}": f"L{i}_a.json" for i in range(8)},
    **{f"LB{i}": f"L{i}_b.json" for i in range(8)},
}


# Cria as pastas de saída usadas para dados, figuras e trajetórias STN.
def ensure_dirs() -> None:
    for path in [DATA_DIR, FIG_DIR, STN_DIR, STN_ORDERED_DIR, IBS_INSTANCE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# Monta o registro padrão de uma instância lida de um arquivo zip.
def canonical_zip_record(zip_path: Path, member: str, data: dict) -> dict:
    instance = Path(member).stem
    group = "literature" if zip_path.stem == "literature" else "generated"
    return {
        "instance": instance,
        "group": group,
        "experiment_group": zip_path.stem,
        "source_zip": zip_path.name,
        "data": data,
    }


# Carrega todas as instâncias WSP dos zips e remove duplicatas pelo nome.
def load_wildfire_instances() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for zip_path in sorted((WF_DIR / "instances").glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for member in sorted(zf.namelist()):
                if not member.endswith(".json"):
                    continue
                data = json.loads(zf.read(member).decode("utf-8"))
                rec = canonical_zip_record(zip_path, member, data)
                # The default generated instance appears in multiple experiment zips.
                # Keep the first occurrence to make the instance space unique.
                records.setdefault(rec["instance"], rec)
    return records


# Calcula os tempos mínimos de chegada do fogo a partir das ignições.
def dijkstra(n: int, arcs: list[tuple[int, int, float]], ignitions: list[int]) -> list[float]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in arcs:
        adj[u].append((v, float(w)))
    dist = [math.inf] * n
    pq: list[tuple[float, int]] = []
    for s in ignitions:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# Calcula estatísticas básicas de uma lista numérica com tratamento para lista vazia.
def safe_stats(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {f"{prefix}_{name}": math.nan for name in ["mean", "std", "min", "max", "p25", "p50", "p75", "cv"]}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p25": float(np.quantile(arr, 0.25)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p75": float(np.quantile(arr, 0.75)),
        f"{prefix}_cv": std / mean if mean else 0.0,
    }


# Mede quão diferentes são os tempos de propagação em direções opostas do grafo.
def directional_asymmetry(arcs: list[tuple[int, int, float]]) -> tuple[float, float]:
    weights = {(u, v): float(w) for u, v, w in arcs}
    vals = []
    seen = set()
    for (u, v), w in weights.items():
        if (v, u) in weights and (v, u) not in seen:
            wr = weights[(v, u)]
            denom = (w + wr) / 2.0
            vals.append(abs(w - wr) / denom if denom else 0.0)
            seen.add((u, v))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.max(vals))


# Calcula métricas do perímetro/frente do fogo nos tempos em que recursos são liberados.
def release_perimeter_features(n: int, arcs: list[tuple[int, int, float]], arrivals: list[float], release_times: list[float], h: float) -> dict:
    adj = defaultdict(list)
    for u, v, _ in arcs:
        adj[u].append(v)
    perimeters = []
    front_windows = []
    sorted_t = sorted(float(t) for t in release_times)
    for idx, t in enumerate(sorted_t):
        alpha = sorted_t[idx + 1] if idx + 1 < len(sorted_t) else h
        burned = [a < t for a in arrivals]
        perimeter = 0
        for u in range(n):
            if burned[u] and any(not burned[v] for v in adj[u]):
                perimeter += 1
        perimeters.append(perimeter)
        front_windows.append(sum(t <= a < alpha for a in arrivals))
    return {
        "release_perimeter_mean": float(np.mean(perimeters)) if perimeters else 0.0,
        "release_perimeter_max": float(np.max(perimeters)) if perimeters else 0.0,
        "release_front_window_mean": float(np.mean(front_windows)) if front_windows else 0.0,
        "release_front_window_max": float(np.max(front_windows)) if front_windows else 0.0,
    }


# Extrai todas as features estruturais, temporais e dinâmicas de uma instância.
def features_for_record(rec: dict) -> dict:
    data = rec["data"]
    n = int(data["|V|"])
    h = float(data["H"])
    arcs = [(int(u), int(v), float(w)) for u, v, w in data["arcs"]]
    weights = [w for _, _, w in arcs]
    release_times = [float(t) for t in data.get("t", [])]
    capacities = [float(c) for c in data.get("c", [])]
    deltas = [float(d) for d in data.get("delta", [])]
    ignitions = [int(i) for i in data.get("I", [])]
    arrivals = dijkstra(n, arcs, ignitions)
    finite_arrivals = [a for a in arrivals if math.isfinite(a)]
    asym_mean, asym_max = directional_asymmetry(arcs)
    coords = data.get("distance", {}).get("coordinates", [])
    x_vals = [float(c[0]) for c in coords] if coords else []
    y_vals = [float(c[1]) for c in coords] if coords else []
    z_vals = [float(c[2]) for c in coords if len(c) > 2] if coords else []
    burned_h = sum(a < h for a in finite_arrivals)
    row = {
        "instance": rec["instance"],
        "group": rec["group"],
        "experiment_group": rec["experiment_group"],
        "source_zip": rec["source_zip"],
        "n_vertices": n,
        "n_arcs": len(arcs),
        "arc_density": len(arcs) / n if n else 0.0,
        "horizon": h,
        "total_resources": float(sum(capacities)),
        "decision_points": len(release_times),
        "resources_per_decision": float(sum(capacities) / len(release_times)) if release_times else 0.0,
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_over_h": float(np.mean(deltas) / h) if deltas and h else 0.0,
        "first_release_over_h": min(release_times) / h if release_times and h else 0.0,
        "last_release_over_h": max(release_times) / h if release_times and h else 0.0,
        "release_window_over_h": (max(release_times) - min(release_times)) / h if len(release_times) > 1 and h else 0.0,
        "free_burning_time": max(finite_arrivals) if finite_arrivals else math.nan,
        "free_burning_over_h": max(finite_arrivals) / h if finite_arrivals and h else math.nan,
        "free_burned_fraction_h": burned_h / n if n else 0.0,
        "arrival_q05_over_h": float(np.quantile(finite_arrivals, 0.05) / h) if finite_arrivals and h else math.nan,
        "arrival_q50_over_h": float(np.quantile(finite_arrivals, 0.50) / h) if finite_arrivals and h else math.nan,
        "arrival_q95_over_h": float(np.quantile(finite_arrivals, 0.95) / h) if finite_arrivals and h else math.nan,
        "directional_asymmetry_mean": asym_mean,
        "directional_asymmetry_max": asym_max,
        "x_span": max(x_vals) - min(x_vals) if x_vals else 0.0,
        "y_span": max(y_vals) - min(y_vals) if y_vals else 0.0,
        "z_span": max(z_vals) - min(z_vals) if z_vals else 0.0,
        "z_std": float(np.std(z_vals)) if z_vals else 0.0,
    }
    row.update(safe_stats(weights, "arc_time"))
    row.update(release_perimeter_features(n, arcs, arrivals, release_times, h))
    return row


# Converte uma instância do formato Wildfire-Suppression para o formato aceito pelo executável do IBS.
def convert_to_ibs_instance(rec: dict, scale: int = 100) -> Path:
    data = rec["data"]
    coords = data.get("distance", {}).get("coordinates")
    if not coords:
        raise ValueError(f"Instance {rec['instance']} has no coordinates")
    xs = sorted({float(c[0]) for c in coords})
    ys = sorted({float(c[1]) for c in coords})
    x_rank = {x: idx for idx, x in enumerate(xs)}
    y_rank = {y: idx for idx, y in enumerate(ys)}
    id_to_coord = {
        idx: [x_rank[float(c[0])], y_rank[float(c[1])]]
        for idx, c in enumerate(coords)
    }
    res_at_time: dict[str, int] = defaultdict(int)
    for t, c in zip(data.get("t", []), data.get("c", [])):
        res_at_time[str(max(1, int(round(float(t) * scale))))] += int(c)
    arcs = {}
    for u, v, w in data["arcs"]:
        c1 = tuple(id_to_coord[int(u)])
        c2 = tuple(id_to_coord[int(v)])
        arcs[f"(({c1[0]}, {c1[1]}), ({c2[0]}, {c2[1]}))"] = max(1, int(round(float(w) * scale)))
    old = {
        "Delay": max(1, int(round(float(data.get("delta", [1])[0]) * scale))),
        "ArrivalTimeTarget": max(1, int(round(float(data["H"]) * scale))),
        "ResAtTime": dict(sorted(res_at_time.items(), key=lambda kv: int(kv[0]))),
        "Ignitions": [id_to_coord[int(data["I"][0])]],
        "Nodes": [id_to_coord[idx] for idx in sorted(id_to_coord)],
        "Arcs": arcs,
    }
    out = IBS_INSTANCE_DIR / f"{rec['instance']}.json"
    out.write_text(json.dumps(old, indent=2), encoding="utf-8")
    return out


# Gera a tabela de features de todas as instâncias e salva em CSV.
def build_feature_table(records: dict[str, dict]) -> pd.DataFrame:
    rows = [features_for_record(rec) for rec in records.values()]
    df = pd.DataFrame(rows).sort_values(["group", "instance"])
    df.to_csv(DATA_DIR / "features_instances.csv", index=False)
    return df


# Agrupa variantes do IBS sob o mesmo nome para comparação agregada.
def normalize_algorithm(name: str) -> str:
    if name.startswith("IBS"):
        return "IBS"
    return name


# Carrega os resultados experimentais e calcula o gap relativo ao ótimo ou BKV.
def load_performance() -> pd.DataFrame:
    rows = []
    lit_dir = WF_DIR / "dissertation" / "experimental_data" / "literature_instances"
    new_dir = WF_DIR / "dissertation" / "experimental_data" / "new_instances"
    opt = pd.read_csv(lit_dir / "opt.csv")
    opt_map = dict(zip(opt["instance"], opt["optimal_objective"]))
    benchmark = pd.read_csv(lit_dir / "benchmark.csv")
    for _, r in benchmark.iterrows():
        target = float(opt_map[r["instance"]])
        rows.append({
            "instance": r["instance"],
            "algorithm": r["algorithm"],
            "algorithm_norm": normalize_algorithm(str(r["algorithm"])),
            "seed": r["seed"],
            "objv": float(r["objv"]),
            "reference": target,
            "gap": (float(r["objv"]) - target) / target,
            "source": "literature_benchmark",
        })
    for csv_path in sorted(new_dir.glob("*.csv")):
        if csv_path.name in {"bkv.csv", "bklb.csv"}:
            continue
        df = pd.read_csv(csv_path)
        if "algorithm" not in df.columns:
            continue
        for _, r in df.iterrows():
            bkv = float(r["bkv"])
            rows.append({
                "instance": r["instance"],
                "algorithm": r["algorithm"],
                "algorithm_norm": normalize_algorithm(str(r["algorithm"])),
                "seed": r.get("seed", np.nan),
                "objv": float(r["objv"]),
                "reference": bkv,
                "gap": (float(r["objv"]) - bkv) / bkv if bkv else np.nan,
                "source": csv_path.stem,
            })
    perf = pd.DataFrame(rows)
    perf.to_csv(DATA_DIR / "performance_metadata.csv", index=False)
    return perf


# Carrega os ótimos conhecidos das instâncias de literatura LA/LB.
def load_literature_optima() -> dict[str, float]:
    opt_path = WF_DIR / "dissertation" / "experimental_data" / "literature_instances" / "opt.csv"
    opt = pd.read_csv(opt_path)
    return dict(zip(opt["instance"], opt["optimal_objective"]))


# Resume o desempenho por instância/algoritmo e identifica o melhor algoritmo por instância.
def performance_summary(perf: pd.DataFrame) -> pd.DataFrame:
    by_variant = (
        perf.groupby(["instance", "algorithm", "algorithm_norm"], as_index=False)
        .agg(median_gap=("gap", "median"), mean_gap=("gap", "mean"), mean_objv=("objv", "mean"), n=("gap", "size"))
    )
    # For IBS variants in instance_size, use the strongest observed IBS variant for the family-level ISA.
    family = (
        by_variant.sort_values("median_gap")
        .groupby(["instance", "algorithm_norm"], as_index=False)
        .first()
        .rename(columns={"algorithm": "representative_algorithm"})
    )
    best = family.sort_values("median_gap").groupby("instance", as_index=False).first()
    best = best[["instance", "algorithm_norm", "median_gap"]].rename(
        columns={"algorithm_norm": "best_algorithm", "median_gap": "best_gap"}
    )
    family.to_csv(DATA_DIR / "performance_by_algorithm.csv", index=False)
    best.to_csv(DATA_DIR / "instance_best_algorithm.csv", index=False)
    return best


# Constrói a projeção ISA com PCA e estima importância de features com Random Forest.
def make_isa(features: pd.DataFrame, best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    merged = features.merge(best, on="instance", how="left")
    numeric_cols = [
        c for c in features.columns
        if c not in {"instance", "group", "experiment_group", "source_zip"}
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    x_raw = merged[numeric_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(x_raw))
    pca = PCA(n_components=2, random_state=1)
    coords = pca.fit_transform(x)
    merged["pc1"] = coords[:, 0]
    merged["pc2"] = coords[:, 1]
    merged.to_csv(DATA_DIR / "isa_projection.csv", index=False)

    importance_rows = []
    y_alg = merged["best_algorithm"].fillna("unknown")
    if y_alg.nunique() > 1:
        clf = RandomForestClassifier(n_estimators=400, random_state=7, class_weight="balanced")
        clf.fit(x, y_alg)
        for col, imp in zip(numeric_cols, clf.feature_importances_):
            importance_rows.append({"target": "best_algorithm", "feature": col, "importance": imp})
    y_gap = merged["best_gap"].fillna(merged["best_gap"].median())
    if y_gap.notna().any():
        reg = RandomForestRegressor(n_estimators=400, random_state=8)
        reg.fit(x, y_gap)
        for col, imp in zip(numeric_cols, reg.feature_importances_):
            importance_rows.append({"target": "best_gap", "feature": col, "importance": imp})
    imp_df = pd.DataFrame(importance_rows).sort_values(["target", "importance"], ascending=[True, False])
    imp_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    meta = {
        "pca_var1": float(pca.explained_variance_ratio_[0]),
        "pca_var2": float(pca.explained_variance_ratio_[1]),
        "numeric_cols": numeric_cols,
    }
    return merged, imp_df, meta


# Desenha um mapa ISA em PC1/PC2 colorido por uma coluna categórica ou numérica.
def scatter_plot(df: pd.DataFrame, color_col: str, path: Path, title: str, cmap: str | None = None) -> None:
    plt.figure(figsize=(9, 6))
    if pd.api.types.is_numeric_dtype(df[color_col]):
        sc = plt.scatter(df["pc1"], df["pc2"], c=df[color_col], cmap=cmap or "viridis", s=70, edgecolor="black", linewidth=0.4)
        plt.colorbar(sc, label=color_col)
    else:
        categories = sorted(df[color_col].fillna("unknown").unique())
        palette = plt.get_cmap("tab10")
        for idx, cat in enumerate(categories):
            sub = df[df[color_col].fillna("unknown") == cat]
            plt.scatter(sub["pc1"], sub["pc2"], label=cat, s=70, edgecolor="black", linewidth=0.4, color=palette(idx % 10))
        plt.legend(loc="best", fontsize=8)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# Gera gráficos de barras com a importância das features do Random Forest.
def plot_feature_importance(imp: pd.DataFrame) -> None:
    for target in imp["target"].unique():
        sub = imp[imp["target"] == target].head(12).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(sub["feature"], sub["importance"], color="#4C78A8")
        plt.title(f"Importancia das features - {target}")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"feature_importance_{target}.png", dpi=180)
        plt.close()


# Gera as principais figuras da ISA e o boxplot de gaps por algoritmo.
def make_isa_figures(isa: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> None:
    scatter_plot(isa, "group", FIG_DIR / "isa_family.png", "ISA por familia de instancias")
    scatter_plot(isa, "best_algorithm", FIG_DIR / "isa_best_algorithm.png", "ISA por melhor algoritmo")
    scatter_plot(isa, "best_gap", FIG_DIR / "isa_best_gap.png", "ISA por gap do melhor algoritmo", cmap="magma_r")
    plot_feature_importance(imp)
    plt.figure(figsize=(9, 5))
    data = [
        perf.loc[perf["algorithm_norm"] == alg, "gap"].dropna().clip(lower=0, upper=2.0).values
        for alg in sorted(perf["algorithm_norm"].unique())
    ]
    labels = sorted(perf["algorithm_norm"].unique())
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Gap relativo")
    plt.title("Distribuicao de gaps por algoritmo")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "algorithm_gap_boxplot.png", dpi=180)
    plt.close()


# Localiza o arquivo antigo do IBS correspondente a uma instância LA/LB.
def old_ibs_instance_path(instance: str) -> Path | None:
    name = OLD_IBS_NAMES.get(instance)
    if not name:
        return None
    path = IBS_DIR / "instances" / name
    return path if path.exists() else None


# Executa o IBS nas instâncias selecionadas e salva as trajetórias brutas em CSV.
def run_ibs(records: dict[str, dict]) -> pd.DataFrame:
    if not IBS_EXE.exists():
        subprocess.run(["cmake", "-S", str(IBS_DIR), "-B", str(IBS_DIR / "build-plan-check"), "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(IBS_DIR / "build-plan-check"), "-j2"], check=True)

    summary_rows = []
    optima = load_literature_optima()
    for instance in STN_SELECTION:
        stn_csv = STN_DIR / f"{instance}_IBS.csv"
        if stn_csv.exists():
            stn_csv.unlink()
        inst_path = old_ibs_instance_path(instance)
        if inst_path is None:
            rec = records.get(instance)
            if rec is None:
                continue
            inst_path = convert_to_ibs_instance(rec)
        for seed in range(1, STN_RUNS + 1):
            cmd = [
                str(IBS_EXE),
                "--instance", str(inst_path),
                "--seed", str(seed),
                "--stn_output", str(stn_csv),
                "--stn_run", str(seed),
                "--timelimit", str(STN_TIMELIMIT),
                "--maxiter", str(STN_MAXITER),
                "--beta", str(STN_BETA),
                "--eta", str(STN_ETA),
                "--c", str(STN_C),
            ]
            if instance in optima:
                cmd += ["--target", str(int(optima[instance]))]
            proc = subprocess.run(cmd, cwd=IBS_DIR, text=True, capture_output=True, check=True)
            line = proc.stdout.strip().splitlines()[-1]
            parsed = next(csv.reader([line]))
            summary_rows.append({
                "instance": instance,
                "seed": seed,
                "reported_instance": parsed[0],
                "objv": float(parsed[8]),
                "time_to_best": float(parsed[9]),
                "iter_to_best": float(parsed[10]),
                "total_iter": float(parsed[11]),
                "elapsed_time": float(parsed[12]),
                "budget": float(parsed[13]),
                "trajectory": parsed[14] if len(parsed) > 14 else "[]",
                "stn_csv": str(stn_csv.relative_to(ROOT)),
            })
    df = pd.DataFrame(summary_rows)
    df.to_csv(DATA_DIR / "ibs_run_summary.csv", index=False)
    return df


# Cruza a posição ISA das instâncias com as métricas STN calculadas para o IBS.
def make_isa_stn_relation(isa: pd.DataFrame, stn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stn.empty:
        empty = pd.DataFrame()
        return empty, empty
    cols = ["instance", "group", "experiment_group", "pc1", "pc2", "best_algorithm", "best_gap"]
    rel = stn.merge(isa[cols], on="instance", how="left")
    rel["isa_region"] = np.where(
        rel["best_algorithm"].eq("IBS"),
        "IBS melhor",
        np.where(rel["best_algorithm"].isin(["MIP", "CUTH"]), "MIP/CUTH melhor", "Outro melhor"),
    )
    rel.to_csv(DATA_DIR / "isa_stn_relation.csv", index=False)
    grouped = (
        rel.groupby("isa_region", as_index=False)
        .agg(
            instances=("instance", "count"),
            mean_best_strength=("best_strength", "mean"),
            median_best_strength=("best_strength", "median"),
            mean_nend=("nend", "mean"),
            mean_overlap_ratio=("overlap_ratio", "mean"),
            mean_best_gap=("best_gap", "mean"),
        )
        .sort_values("isa_region")
    )
    grouped.to_csv(DATA_DIR / "isa_stn_cross_table.csv", index=False)
    plot_isa_stn_relation(rel, grouped)
    return rel, grouped


# Plota a relação ISA-STN usando cor para best_strength e tamanho para nend.
def plot_isa_stn_relation(rel: pd.DataFrame, grouped: pd.DataFrame) -> None:
    if rel.empty:
        return
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(
        rel["pc1"],
        rel["pc2"],
        c=rel["best_strength"],
        s=80 + 28 * rel["nend"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    for _, row in rel.iterrows():
        plt.text(row["pc1"], row["pc2"], row["instance"], fontsize=6, alpha=0.75)
    plt.colorbar(sc, label="best_strength")
    plt.title("Relacao ISA-STN: cor=best_strength, tamanho=nend")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "isa_stn_best_strength_nend.png", dpi=180)
    plt.close()

    if not grouped.empty:
        x = np.arange(len(grouped))
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(x - width / 2, grouped["mean_best_strength"], width, label="best_strength medio", color="#4C78A8")
        ax1.set_ylabel("best_strength medio")
        ax1.set_xticks(x)
        ax1.set_xticklabels(grouped["isa_region"], rotation=20, ha="right")
        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, grouped["mean_nend"], width, label="nend medio", color="#F58518", alpha=0.75)
        ax2.set_ylabel("nend medio")
        ax1.set_title("Metricas STN por regiao ISA")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "isa_stn_region_metrics.png", dpi=180)
        plt.close(fig)


# Carrega um CSV de trajetória STN no formato Run,Fitness,Solution.
def load_stn_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Run": int, "Fitness": float, "Solution": str})


# Cria trajetórias ordenadas pelo melhor valor encontrado até o momento.
def create_ordered_stn_csvs() -> None:
    """Create minimization-compatible best-so-far trajectories for STN Analytics.

    The IBS restart process may return a representative solution worse than the
    previous iteration. STN Analytics expects monotonic fitness for minimization,
    so this folder keeps only non-worsening steps per run.
    """
    for old in STN_ORDERED_DIR.glob("*_IBS.csv"):
        old.unlink()
    for instance in STN_SELECTION:
        src = STN_DIR / f"{instance}_IBS.csv"
        if not src.exists():
            continue
        df = load_stn_csv(src)
        rows = []
        for run, grp in df.groupby("Run", sort=True):
            best = math.inf
            for _, row in grp.iterrows():
                fitness = float(row["Fitness"])
                if fitness <= best:
                    best = fitness
                    rows.append({"Run": int(run), "Fitness": fitness, "Solution": str(row["Solution"])})
        out = STN_ORDERED_DIR / f"{instance}_IBS.csv"
        pd.DataFrame(rows, columns=["Run", "Fitness", "Solution"]).to_csv(out, index=False)


# Calcula métricas de rede para uma instância e gera a figura STN correspondente.
def stn_metrics_for(path: Path, instance: str, figure_prefix: str = "stnOrdenado") -> dict:
    df = load_stn_csv(path)
    graph = nx.DiGraph()
    visit_runs: dict[str, set[int]] = defaultdict(set)
    counts = Counter()
    fitness_by_node: dict[str, float] = {}
    starts = set()
    ends = set()
    for run, grp in df.groupby("Run"):
        grp = grp.reset_index(drop=True)
        if grp.empty:
            continue
        starts.add(grp.loc[0, "Solution"])
        ends.add(grp.loc[len(grp) - 1, "Solution"])
        prev = None
        for _, row in grp.iterrows():
            node = row["Solution"]
            fitness = float(row["Fitness"])
            graph.add_node(node)
            counts[node] += 1
            visit_runs[node].add(int(run))
            fitness_by_node[node] = min(fitness_by_node.get(node, fitness), fitness)
            if prev is not None and prev != node:
                graph.add_edge(prev, node, weight=graph.get_edge_data(prev, node, {"weight": 0})["weight"] + 1)
            prev = node
    best_fitness = min(fitness_by_node.values()) if fitness_by_node else math.nan
    best_nodes = {n for n, f in fitness_by_node.items() if f == best_fitness}
    incoming_strength = sum(
        data.get("weight", 1)
        for u, v, data in graph.edges(data=True)
        if v in best_nodes
    )
    nruns = df["Run"].nunique()
    nshared = sum(1 for runs in visit_runs.values() if len(runs) > 1)
    metrics = {
        "instance": instance,
        "nruns": nruns,
        "ntotal": graph.number_of_nodes(),
        "nedges": graph.number_of_edges(),
        "nbest": len(best_nodes),
        "nend": len({n for n in ends if n not in best_nodes}),
        "best_fitness": best_fitness,
        "best_strength": min(1.0, incoming_strength / nruns) if nruns else 0.0,
        "overlap_ratio": nshared / graph.number_of_nodes() if graph.number_of_nodes() else 0.0,
        "csv": str(path.relative_to(ROOT)),
    }
    draw_stn(graph, counts, fitness_by_node, starts, ends, best_nodes, FIG_DIR / f"{figure_prefix}_{instance}.png", instance)
    return metrics


# Desenha uma STN em Python usando nós como soluções e arestas como transições.
def draw_stn(graph: nx.DiGraph, counts: Counter, fitness: dict, starts: set, ends: set, best_nodes: set, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 7))
    if graph.number_of_nodes() == 0:
        plt.close()
        return
    pos = nx.spring_layout(graph, seed=4, weight="weight", k=0.8)
    y_by_fit = {n: -fitness.get(n, 0.0) for n in graph.nodes}
    # Blend force layout with fitness as vertical axis.
    if y_by_fit:
        ys = np.array(list(y_by_fit.values()), dtype=float)
        if np.ptp(ys) > 0:
            ys = (ys - ys.mean()) / np.ptp(ys)
            for n, y in zip(y_by_fit.keys(), ys):
                pos[n][1] = y
    colors = []
    for n in graph.nodes:
        if n in best_nodes:
            colors.append("#D62728")
        elif n in starts:
            colors.append("#F2C14E")
        elif n in ends:
            colors.append("#595959")
        else:
            colors.append("#72B7B2")
    sizes = [80 + 35 * math.sqrt(counts[n]) for n in graph.nodes]
    edge_widths = [0.5 + graph[u][v].get("weight", 1) * 0.25 for u, v in graph.edges]
    nx.draw_networkx_edges(graph, pos, alpha=0.35, arrows=True, arrowsize=8, width=edge_widths, edge_color="#777777")
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, edgecolors="black", linewidths=0.3)
    plt.title(f"STN IBS - {title}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# Calcula métricas STN para todos os CSVs de uma pasta e salva a tabela final.
def make_stn_outputs(input_dir: Path = STN_ORDERED_DIR, output_name: str = "stn_metrics_ibs_ordenado.csv", figure_prefix: str = "stnOrdenado") -> pd.DataFrame:
    rows = []
    for csv_path in sorted(input_dir.glob("*_IBS.csv")):
        instance = csv_path.name.replace("_IBS.csv", "")
        rows.append(stn_metrics_for(csv_path, instance, figure_prefix=figure_prefix))
    df = pd.DataFrame(rows).sort_values("instance")
    df.to_csv(DATA_DIR / output_name, index=False)
    return df


# Gera curvas de convergência das trajetórias do IBS por instância e seed.
def plot_convergence(input_dir: Path = STN_ORDERED_DIR, output_name: str = "ibs_convergence_selected_ordenado.png") -> None:
    paths = sorted(input_dir.glob("*_IBS.csv"))
    if not paths:
        return
    fig, axes = plt.subplots(len(paths), 1, figsize=(9, max(3, 2.4 * len(paths))), sharex=False)
    if len(paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, paths):
        instance = path.name.replace("_IBS.csv", "")
        df = load_stn_csv(path)
        for run, grp in df.groupby("Run"):
            grp = grp.reset_index(drop=True)
            ax.plot(range(1, len(grp) + 1), grp["Fitness"], alpha=0.35, linewidth=1)
        med = df.assign(iteration=df.groupby("Run").cumcount() + 1).groupby("iteration")["Fitness"].median()
        ax.plot(med.index, med.values, color="black", linewidth=2, label="mediana")
        ax.set_title(instance)
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Iteracao IBS")
    plt.tight_layout()
    plt.savefig(FIG_DIR / output_name, dpi=180)
    plt.close()


# Extrai do resumo do IBS os momentos em que houve melhoria de solução.
def parse_trajectory_improvements(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        text = str(r["trajectory"]).strip()
        if text in {"[]", "", "nan"}:
            continue
        for obj, elapsed, iteration, budget in ast.literal_eval(text):
            rows.append({
                "instance": r["instance"],
                "seed": r["seed"],
                "objv": obj,
                "elapsed_time": elapsed,
                "iteration": iteration,
                "budget": budget,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "ibs_improvement_trajectory.csv", index=False)
    if df.empty:
        return df
    plt.figure(figsize=(9, 5))
    for instance, grp in df.groupby("instance"):
        med = grp.groupby("elapsed_time")["objv"].median().sort_index()
        plt.step(med.index, med.values, where="post", label=instance)
    plt.xlabel("Tempo ate melhoria (s)")
    plt.ylabel("Melhor fitness registrada")
    plt.title("Trajetoria de melhorias do IBS")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ibs_improvement_time.png", dpi=180)
    plt.close()
    return df



# Executa o pipeline completo: carrega dados, gera ISA, roda IBS, cria STNs e salva artefatos.
def main() -> None:
    ensure_dirs()
    records = load_wildfire_instances()
    features = build_feature_table(records)
    perf = load_performance()
    best = performance_summary(perf)
    isa, imp, _ = make_isa(features, best)
    make_isa_figures(isa, imp, perf)
    summary = run_ibs(records)
    parse_trajectory_improvements(summary)
    create_ordered_stn_csvs()
    stn = make_stn_outputs()
    # Keep a copy under the old expected name pointing to the ordered final metrics.
    stn.to_csv(DATA_DIR / "stn_metrics_ibs.csv", index=False)
    plot_convergence()
    make_isa_stn_relation(isa, stn)
    print(f"features={len(features)} performance_rows={len(perf)} stn_instances={len(stn)}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
