from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT_SEEDS = (20260612, 104729, 1299709)


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    root_seeds: tuple[int, ...]
    population: int
    generations: int
    local_search_evaluations: int
    target_budget: int
    batch_size: int
    workers: int
    sobol_points: int
    thresholds: tuple[float, ...] = (0.10,)
    primary_threshold: float = 0.10
    baseline_sobol_seed: int = 20260612
    ga_priority: float = 0.8
    crossover_probability: float = 0.8
    mutation_probability: float = 1.0 / 9.0
    tournament_size: int = 3
    elitism: int = 2
    exact_topology: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ga_evaluation_budget(self) -> int:
        return self.population * self.generations + self.local_search_evaluations

    @classmethod
    def from_file(cls, path: Path, **overrides: Any) -> "ExperimentConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update({k: v for k, v in overrides.items() if v is not None})
        data["root_seeds"] = tuple(int(v) for v in data["root_seeds"])
        data["thresholds"] = tuple(float(v) for v in data.get("thresholds", (0.10,)))
        available_workers = max(1, (os.cpu_count() or 2) - 1)
        requested_workers = data.get("workers", "auto")
        data["workers"] = (
            available_workers
            if requested_workers == "auto"
            else max(1, min(int(requested_workers), available_workers))
        )
        known = set(cls.__dataclass_fields__)
        extra = {k: data.pop(k) for k in list(data) if k not in known}
        data["extra"] = extra
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for key, value in list(out.items()):
            if isinstance(value, tuple):
                out[key] = list(value)
        return out
