"""Copy immutable prepared matrices and record provenance for corrected fits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys

from analysis_wsp.matilda74.prepare import OUTPUT as BASE, ROOT
from analysis_wsp.matilda74.sifted_correction import identity

OUTPUT=ROOT / "experiments" / "isa_corrected"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copied=[]
    for folder in ("inputs","comparison_55_v24/inputs"):
        for source in sorted((BASE/folder).iterdir()):
            if not source.is_file():continue
            target=OUTPUT/folder/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=source.read_bytes():
                raise RuntimeError(f"Refusing to overwrite divergent input: {target}")
            shutil.copy2(source,target)
            assert source.read_bytes()==target.read_bytes()
            copied.append({"source":str(source.relative_to(ROOT)),
                           "destination":str(target.relative_to(ROOT)),
                           "sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
    protected=["results/benchmark", "results/generation", "results/isa", "third_party"]
    gitroot=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],cwd=ROOT,text=True).strip())
    prefix=str(ROOT.relative_to(gitroot))
    trees={p:subprocess.check_output(["git","rev-parse",f"HEAD:{prefix}/{p}"],cwd=ROOT,text=True).strip()
           for p in protected}
    payload={
        "status":"prepared","created_utc":datetime.now(timezone.utc).isoformat(),
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        "implementation":identity(),"python_executable":sys.executable,
        "inputs_copied_byte_identically":copied,"protected_git_trees":trees,
        "wsp_solvers_executed":False,"stn_searches_executed":False,
        "topology_recomputed":False,"instance_generation_ga_executed":False,
        "planned_fits":["all74/control","all74/topology","original55_v24/control","original55_v24/topology"],
        "comparison_scope":"joint correction of sign, correlation-filter indexing and filtered-matrix mapping",
        "not_an_ablation_of_sign_alone":True,
    }
    (OUTPUT/"revision_manifest.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":"prepared","input_files":len(copied),"output":str(OUTPUT)},ensure_ascii=False))

if __name__=="__main__":main()
