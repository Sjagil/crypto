#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = "cannot instantiate 'WindowsPath' on your system"

def run_nodes(nodes, timeout=300):
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", *nodes],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return {"returncode": p.returncode, "windows_path_crash": SIGNATURE in out,
                "stdout_tail": (p.stdout or "")[-5000:], "stderr_tail": (p.stderr or "")[-5000:]}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": None, "timeout": True, "windows_path_crash": False,
                "stdout_tail": exc.stdout[-5000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": exc.stderr[-5000:] if isinstance(exc.stderr, str) else ""}

def collect_nodes():
    p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    nodes = [x.strip() for x in p.stdout.splitlines()
             if x.strip().startswith("tests/") and "::" in x]
    if not nodes:
        raise RuntimeError("no pytest node ids collected")
    return nodes

def suspicious(path):
    if not path.is_file():
        return []
    tokens = ("os.name","monkeypatch","patch.object","mock.patch","WindowsPath",'"nt"',"'nt'")
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if any(t in line for t in tokens):
            rows.append({"line":n,"text":line[:400]})
    return rows[:100]

def main():
    if sys.platform != "darwin":
        raise SystemExit(f"macOS-only diagnostic; got {sys.platform}")
    nodes=collect_nodes()
    probes=[]
    lower=0
    upper=None
    count=min(32,len(nodes))
    while True:
        result=run_nodes(nodes[:count])
        probes.append({"prefix_count":count,**result})
        if result.get("windows_path_crash"):
            upper=count
            break
        lower=count
        if count>=len(nodes):
            break
        count=min(len(nodes),count*2)
    culprit=None
    if upper is not None:
        lo,hi=lower+1,upper
        while lo<hi:
            mid=(lo+hi)//2
            result=run_nodes(nodes[:mid])
            probes.append({"prefix_count":mid,**result})
            if result.get("windows_path_crash"):
                hi=mid
            else:
                lo=mid+1
        culprit=nodes[lo-1]
    cfile=ROOT/culprit.split("::",1)[0] if culprit else None
    payload={"schema_version":"round32_macos_pytest_isolation_v1",
             "status":"CULPRIT_IDENTIFIED" if culprit else "WINDOWSPATH_NOT_REPRODUCED",
             "node_count":len(nodes),"culprit_node":culprit,
             "culprit_file":str(cfile.relative_to(ROOT)) if cfile else None,
             "culprit_isolated_run":run_nodes([culprit]) if culprit else None,
             "suspicious_lines":suspicious(cfile) if cfile else [],
             "probes":probes,"orders_generated":0,"orders_submitted":0,
             "private_exchange_requests":0}
    path=ROOT/"output"/"reports"/"round32_macos_pytest_isolation.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload["artifact"]=str(path)
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
