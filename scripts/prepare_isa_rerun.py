"""Copy only the published input matrices to a new corrected ISA experiment."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "results/isa"
output = ROOT / "experiments/isa_corrected"
if output.exists():
    raise SystemExit(f"Destination already exists: {output}; choose or archive the existing experiment first")
for folder in ("inputs", "comparison_55_v24/inputs"):
    shutil.copytree(source / folder, output / folder)
print(f"Prepared inputs: {output}")
