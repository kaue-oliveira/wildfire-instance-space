"""Rebuild 418 published STNs without executing IBS or modifying measured data."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import networkx as nx
from analysis_wsp.ibs_decimal_pipeline.networks import network, ordered

def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def export(folder, graph):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "network.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
    network_graph = nx.DiGraph()
    for node in graph["nodes"]:
        network_graph.add_node(node["id"], **{k: json.dumps(v) if isinstance(v, list) else v for k, v in node.items() if k != "id"})
    for edge in graph["edges"]:
        network_graph.add_edge(edge["source"], edge["target"], Count=edge["Count"], Type=edge["Type"])
    nx.write_graphml(network_graph, folder / "network.graphml")
    for name, records, fields in (("nodes.csv", graph["nodes"], ["id", "Fitness", "Count", "Type", "seeds"]), ("edges.csv", graph["edges"], ["source", "target", "Count", "Type"])):
        with (folder / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

def compare(actual, expected):
    for key, value in actual.items():
        if isinstance(value, (int, float)):
            if not math.isclose(float(expected[key]), value, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Descriptor differs: {actual['instance']} {actual['representation']} {key}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/stn_rebuilt")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "results" or output.is_relative_to(ROOT / "results"):
        raise SystemExit("Choose an output outside the published results")
    if output.exists():
        raise SystemExit(f"Destination exists: {output}")
    expected = {(r['instance'], r['representation']): r for r in read(ROOT / "results/tables/stn_descriptors.csv")}
    individual = {(r['instance'], r['representation'], int(r['seed'])): r for r in read(ROOT / "results/tables/stn_per_seed_descriptors.csv")}
    count = 0
    for instance in sorted({key[0] for key in expected}):
        source = ROOT / "results/stn/trajectories" / instance
        runs = [(int(p.parent.name.split('_')[1]), read(p)) for p in sorted(source.glob("seed_*/trajectory_raw.csv"))]
        if [seed for seed, _ in runs] != list(range(1, 11)):
            raise ValueError(f"Missing seeds for {instance}")
        for representation in sorted({rep for name, rep in expected if name == instance}):
            selected = runs if representation == "RAW" else [(seed, ordered(rows)) for seed, rows in runs]
            descriptors, graph = network(instance, selected, representation)
            compare(descriptors, expected[instance, representation])
            export(output / "aggregate" / instance / representation, graph)
            count += 1
            for seed, rows in selected:
                descriptors, graph = network(instance, [(seed, rows)], representation)
                compare(descriptors, individual[instance, representation, seed])
                export(output / "per_seed" / instance / f"seed_{seed:02d}" / representation, graph)
                count += 1
    print(json.dumps({"status": "passed", "networks": count, "output": str(output)}, indent=2))

if __name__ == "__main__":
    main()
