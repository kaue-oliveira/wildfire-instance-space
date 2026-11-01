#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STNS_V2 = ROOT / "third_party" / "STNs-v2"
SOURCE_DIR = ROOT / "analysis_wsp" / "data" / "stnOrdenado"
OUT_ROOT = ROOT / "experiments" / "stn_literature"
MANIFEST = ROOT / "analysis_wsp" / "data" / "stns_v2_manifest.csv"


# Cria o nome de pasta usado pelo STNs-v2 para uma instância WSP.
def stns_instance_name(instance: str) -> str:
    return f"wsp_{instance}"


# Copia uma trajetória IBS para o layout de entrada esperado pelo STNs-v2.
def prepare_input(csv_path: Path, instance: str) -> Path:
    stns_name = stns_instance_name(instance)
    stns_data_dir = STNS_V2 / "data" / stns_name
    archive_data_dir = OUT_ROOT / "data" / instance
    stns_data_dir.mkdir(parents=True, exist_ok=True)
    archive_data_dir.mkdir(parents=True, exist_ok=True)

    # STNs-v2 takes the algorithm name from the part before the first underscore.
    # Therefore IBS_out.csv is read as algorithm "IBS".
    target = stns_data_dir / "IBS_out.csv"
    archive = archive_data_dir / "IBS_out.csv"
    shutil.copy2(csv_path, target)
    shutil.copy2(csv_path, archive)
    return target


# Executa um script R do STNs-v2 com os argumentos informados.
def run_r(script: str, *args: str) -> None:
    subprocess.run(
        ["Rscript", script, *args],
        cwd=STNS_V2,
        check=True,
    )


# Copia os modelos RData e figuras PNG gerados pelo STNs-v2 para analysis_wsp/stns_v2.
def copy_outputs(instance: str) -> tuple[str, str]:
    stns_name = stns_instance_name(instance)
    src_stn_dir = STNS_V2 / "stns" / stns_name
    src_plot_dir = STNS_V2 / "plots" / stns_name
    dst_stn_dir = OUT_ROOT / "stns" / instance
    dst_plot_dir = OUT_ROOT / "plots" / instance
    dst_stn_dir.mkdir(parents=True, exist_ok=True)
    dst_plot_dir.mkdir(parents=True, exist_ok=True)

    rdata_out = ""
    plot_out = ""
    for path in sorted(src_stn_dir.glob("*.RData")):
        target = dst_stn_dir / path.name
        shutil.copy2(path, target)
        rdata_out = str(target.relative_to(ROOT))
    for path in sorted(src_plot_dir.glob("*.png")):
        target = dst_plot_dir / path.name
        shutil.copy2(path, target)
        plot_out = str(target.relative_to(ROOT))
    return rdata_out, plot_out


# Executa o fluxo completo do STNs-v2 para as 16 instâncias LA/LB do IBS.
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    csv_paths = sorted(SOURCE_DIR.glob("*_IBS.csv"))
    selected = [
        path for path in csv_paths
        if path.name.startswith("LA") or path.name.startswith("LB")
    ]
    if not selected:
        raise SystemExit(f"No LA/LB STN CSV files found in {SOURCE_DIR}")

    for csv_path in selected:
        instance = csv_path.name.removesuffix("_IBS.csv")
        stns_name = stns_instance_name(instance)
        input_csv = prepare_input(csv_path, instance)
        print(f"[STNs-v2] Creating STN model for {instance}")
        run_r("create.R", stns_name, "10", "NA", "TRUE")
        print(f"[STNs-v2] Plotting STN model for {instance}")
        run_r("gplot-single.R", stns_name)
        rdata_out, plot_out = copy_outputs(instance)
        rows.append({
            "instance": instance,
            "source_csv": str(csv_path.relative_to(ROOT)),
            "stns_v2_input": str(input_csv.relative_to(ROOT)),
            "archived_input": str((OUT_ROOT / "data" / instance / "IBS_out.csv").relative_to(ROOT)),
            "stns_v2_rdata": rdata_out,
            "stns_v2_plot": plot_out,
        })

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[STNs-v2] Manifest written to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
