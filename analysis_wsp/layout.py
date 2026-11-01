"""Canonical project paths, independent of the checkout location."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BENCHMARK = RESULTS / "benchmark"
GENERATION = RESULTS / "generation"
ISA = RESULTS / "isa"
ISA_PREPARATION = RESULTS / "isa_preparation"
STN = RESULTS / "stn"
TABLES = RESULTS / "tables"
