"""STN semantics reused from the historical trajectory audit.

RAW and the historical nonworsening-best-so-far filter are separate outputs.
Edges connect consecutive distinct states only within a seed, never seeds.
"""
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean


def ordered(rows):
    best = float("inf")
    keep = []
    for row in rows:
        fitness = int(row["Fitness"])
        if fitness <= best:
            keep.append(row)
            best = fitness
    return keep


def network(instance, runs, representation):
    fit = defaultdict(set)
    visits = defaultdict(set)
    counts = Counter()
    edges = Counter()
    ends, starts, sets = [], [], []
    for seed, rows in runs:
        if not rows:
            raise ValueError("Empty run")
        prev = None
        ns = set()
        starts.append(rows[0]["Solution"])
        ends.append(rows[-1]["Solution"])
        for row in rows:
            state = row["Solution"]
            fit[state].add(int(row["Fitness"]))
            counts[state] += 1
            visits[state].add(seed)
            ns.add(state)
            if prev is not None and prev != state:
                edges[prev, state] += 1
            prev = state
        sets.append(ns)
    if any(len(values) != 1 for values in fit.values()):
        raise ValueError("Same signature has conflicting objectives")
    best = min(next(iter(f)) for f in fit.values())
    bestnodes = {s for s, fs in fit.items() if next(iter(fs)) == best}
    incoming = sum(v for (s, t), v in edges.items() if t in bestnodes)
    overlaps = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    nruns = len(runs)
    descriptors = dict(instance=instance, representation=representation, nruns=nruns,
        nrows=sum(len(rows) for _, rows in runs), ntotal=len(fit), nedges=len(edges),
        transition_count=sum(edges.values()), nbest=len(bestnodes), nend=len(set(ends)-bestnodes),
        best_fitness=best, best_strength_uncapped=incoming/nruns,
        best_strength_legacy_capped=min(1, incoming/nruns),
        endpoint_best_fraction=sum(s in bestnodes for s in ends)/nruns,
        shared_node_fraction_legacy=sum(len(v)>1 for v in visits.values())/len(fit),
        mean_pairwise_jaccard=mean(overlaps) if overlaps else 0,
        signature_fitness_conflicts=0)
    nodes = [{"id": s, "Fitness": next(iter(fs)), "Count": counts[s],
              "Type": "Best" if s in bestnodes else "End" if s in ends else "Start" if s in starts else "Medium",
              "seeds": sorted(visits[s])} for s, fs in sorted(fit.items())]
    links = [{"source": s, "target": t, "Count": count,
              "Type": "Improving" if next(iter(fit[t])) < next(iter(fit[s])) else
                      "Worsening" if next(iter(fit[t])) > next(iter(fit[s])) else "Neutral"}
             for (s, t), count in sorted(edges.items())]
    return descriptors, {"directed": True, "instance": instance,
        "representation": representation, "nodes": nodes, "edges": links,
        "starts": starts, "ends": ends, "best": best}
