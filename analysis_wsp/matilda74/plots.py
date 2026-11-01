"""Publication-oriented descriptive figures for the audited MATILDA74 fits."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_wsp.matilda74.prepare import OUTPUT


def main() -> None:
    output = OUTPUT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(OUTPUT / "inputs" / "native_source_manifest.csv")[["instance", "collection"]]
    specs = [
        ("55 v2.4: controle", OUTPUT / "comparison_55_v24" / "control"),
        ("74 v2.4: controle", OUTPUT / "control"),
        ("55 v2.4: topologia", OUTPUT / "comparison_55_v24" / "topology"),
        ("74 v2.4: topologia", OUTPUT / "topology"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    for ax, (title, folder) in zip(axes.flat, specs):
        coords = pd.read_csv(folder / "pilot_coordinates.csv", float_precision="round_trip")
        coords = coords.merge(source, on="instance", validate="one_to_one")
        old = coords[coords.collection == "original_55"]
        new = coords[coords.collection == "supplementary_19"]
        ax.scatter(old.z1, old.z2, s=30, c="#305d83", alpha=0.8, label=f"Originais ({len(old)})")
        if len(new):
            ax.scatter(new.z1, new.z2, s=46, c="#dc6f24", edgecolors="black", linewidths=0.25,
                       alpha=0.9, label=f"Novas ({len(new)})")
        ax.set_title(title)
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")
        ax.grid(alpha=0.16)
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("PILOT: coleção original e instâncias geradas", fontsize=14)
    fig.text(0.5, -0.012, "Cada painel é um ajuste distinto; coordenadas e distâncias não são diretamente comparáveis entre painéis.",
             ha="center", va="top", fontsize=9)
    fig.savefig(output / "pilot_55_74_controle_topologia.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    measured = pd.read_csv(OUTPUT / "inputs" / "performance_source_444.csv")
    measured["good"] = measured["gap_mean_pct"] <= 10.0
    counts = measured.groupby(["collection", "algorithm"])["good"].sum().unstack("algorithm").reindex(
        ["original_55", "supplementary_19"]
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    x = range(len(counts.columns))
    width = 0.36
    for offset, collection, label, color in [(-width / 2, "original_55", "Originais (55)", "#305d83"),
                                             (width / 2, "supplementary_19", "Novas (19)", "#dc6f24")]:
        values = counts.loc[collection].astype(int).tolist()
        bars = ax.bar([index + offset for index in x], values, width, label=label, color=color)
        ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xticks(list(x), [name.upper() for name in counts.columns])
    ax.set_ylabel("Instâncias com gap médio ≤ 10%")
    ax.set_title("Rótulos MATILDA por algoritmo e coleção")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "rotulos_bom_55_19.png", dpi=220)
    plt.close(fig)
    print("Created two MATILDA74 figures", flush=True)


if __name__ == "__main__":
    main()
