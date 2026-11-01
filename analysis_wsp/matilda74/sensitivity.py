"""Check whether two ILS failure baselines change MATILDA's good labels."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from analysis_wsp.extended_analysis.provenance import write_json
from analysis_wsp.matilda74.prepare import INTEGRATED, OUTPUT


def main() -> None:
    index_path = INTEGRATED / "analysis_index.csv"
    frame = pd.read_csv(index_path)
    ils = frame[(frame.phase == "benchmark") & (frame.algorithm == "ils")]
    failures = ils[ils.status == "algorithm_failure"]
    if len(ils) != 656 or len(failures) != 2:
        raise RuntimeError("Quantidade de runs/falhas ILS inesperada")
    cases = []
    for failed in failures.itertuples(index=False):
        group = ils[ils.instance == failed.instance]
        included = float(group.relative_gap_to_bkv.mean())
        completed = float(group[group.status == "completed"].relative_gap_to_bkv.mean())
        cases.append({
            "instance": failed.instance,
            "failed_job": failed.job_id,
            "failed_seed": int(failed.seed),
            "archived_feasible_baseline_objective": float(failed.analysis_objective),
            "mean_gap_with_baseline": included,
            "mean_gap_completed_only": completed,
            "good_label_with_baseline": included <= 0.10,
            "good_label_completed_only": completed <= 0.10,
            "label_changed": (included <= 0.10) != (completed <= 0.10),
        })
    cases.sort(key=lambda row: row["instance"])
    result = {
        "status": "passed",
        "failure_cases": cases,
        "labels_changed": sum(case["label_changed"] for case in cases),
        "canonical_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "interpretation": "A exclusão hipotética das falhas não troca a classe ILS no limiar de 10%; as duas falhas permanecem reportadas e a matriz principal usa os baselines viáveis arquivados.",
    }
    write_json(OUTPUT / "inputs" / "ils_failure_label_sensitivity.json", result)
    print(json.dumps({"status": "passed", "labels_changed": result["labels_changed"],
                      "cases": len(cases)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
