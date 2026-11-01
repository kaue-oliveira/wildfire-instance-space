"""Run only the four corrected ISA fits, in pairs, with durable per-fit logs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

from analysis_wsp.matilda74.prepare_corrected import OUTPUT, ROOT


def run(scope, model):
    logdir=OUTPUT if scope=="all74" else OUTPUT/"comparison_55_v24"
    logfile=logdir/f"{model}_run.log"
    command=[sys.executable,"-u","-m","analysis_wsp.matilda74.run_model",model,
             "--scope",scope,"--workers","2","--output-root",str(OUTPUT),
             "--sifted-revision","sifted-corrected-v1"]
    print(f"START {scope}/{model} {datetime.now(timezone.utc).isoformat()}",flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    with logfile.open("w") as handle:
        result=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
    print(f"END {scope}/{model} code={result.returncode}",flush=True)
    return {"scope":scope,"model":model,"returncode":result.returncode,"command":command,"log":str(logfile.relative_to(OUTPUT))}


def main():
    outcomes=[]
    for scope in ("all74","original55_v24"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(run,scope,m) for m in ("control","topology")]
            for future in as_completed(pending):outcomes.append(future.result())
        if any(x["returncode"] for x in outcomes):break
    status={"status":"completed" if len(outcomes)==4 and all(x["returncode"]==0 for x in outcomes) else "failed",
            "ended_utc":datetime.now(timezone.utc).isoformat(),"fits":outcomes,
            "wsp_solvers_executed":False,"stn_searches_executed":False}
    (OUTPUT/"batch_status.json").write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n")
    if status["status"]!="completed":raise SystemExit(1)

if __name__=="__main__":main()
