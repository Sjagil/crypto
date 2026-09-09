#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path

def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--crypto-root",default="/Users/ayoubalhari/Downloads/crypto"); p.add_argument("--swing-root",default="/Users/ayoubalhari/Downloads/crypto-ai-swing-layer"); p.add_argument("--markets",default="BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR"); p.add_argument("--timeframes",default="15m,1h,2h,4h,1d,1W"); p.add_argument("--as-of"); p.add_argument("--refresh-external",action="store_true"); p.add_argument("--output"); a=p.parse_args()
    crypto,swing=Path(a.crypto_root).expanduser().resolve(),Path(a.swing_root).expanduser().resolve(); sys.path.insert(0,str(crypto))
    from core.whole_market_context import build_whole_market_context
    markets=[x.strip().upper() for x in a.markets.split(",") if x.strip()]; tfs=[x.strip() for x in a.timeframes.split(",") if x.strip()]; base=crypto/"output/reports/whole_stack"; cp,ap=base/"canonical_context.json",base/"swing_advisory.json"; fp=Path(a.output).expanduser().resolve() if a.output else base/"latest.json"
    canonical=build_whole_market_context(crypto,markets=markets,timeframes=tfs,as_of=a.as_of); write(cp,canonical)
    py,worker=swing/".venv/bin/python",swing/"scripts/run_whole_context_advisor.py"
    if py.is_file() and worker.is_file():
        cmd=[str(py),str(worker),"--packet",str(cp),"--output",str(ap),"--crypto-root",str(crypto)]+(["--refresh-external"] if a.refresh_external else []); proc=subprocess.run(cmd,cwd=swing,capture_output=True,text=True,timeout=180,check=False)
        advisory=json.loads(ap.read_text(encoding="utf-8")) if proc.returncode==0 and ap.is_file() else {"status":"BLOCKED","reason":"SWING_ADVISORY_WORKER_FAILED","returncode":proc.returncode,"stdout_tail":proc.stdout[-4000:],"stderr_tail":proc.stderr[-4000:]}
    else: advisory={"status":"BLOCKED","reason":"SWING_WORKER_OR_VENV_MISSING"}
    evidence=dict(canonical.get("research_evidence") or {}); selector=dict(advisory.get("selector_policy") or {}); blockers=["ROUND33_ADVISORY_ONLY_NO_EXECUTION_PATH"]
    if canonical.get("status")!="READY_FOR_ADVISORY_ENRICHMENT": blockers.append(f"CANONICAL_CONTEXT_{canonical.get('status')}")
    if advisory.get("status")!="READY": blockers.append(f"SWING_CONTEXT_{advisory.get('status')}")
    if not selector.get("qualified",False): blockers.append("PROSPECTIVE_SELECTOR_NOT_QUALIFIED")
    if int(evidence.get("passed_gate_count") or 0)<1: blockers.append("NO_CANONICAL_RESEARCH_GATE_SURVIVOR")
    layers={"candles":canonical.get("status"),"technical":"READY" if all(any(dict(s).get("technical_score") is not None for s in dict(dict(canonical.get("market_context") or {}).get(m,{})).get("timeframes",{}).values()) for m in markets) else "DEGRADED","multi_timeframe":"READY" if all(dict(dict(canonical.get("market_context") or {}).get(m,{})).get("mtf",{}).get("status")=="READY" for m in markets) else "DEGRADED","news":"READY" if any(dict(dict(advisory.get("market_advisory") or {}).get(m,{})).get("news",{}).get("status")=="READY" for m in markets) else "MISSING","fundamentals_cmc":dict(advisory.get("cmc") or {}).get("status","MISSING"),"prospective_selector":selector.get("status","MISSING"),"research_evidence":evidence.get("status","MISSING")}
    final={"schema_version":"crypto_unified_whole_stack_snapshot_v1","generated_at":datetime.now(UTC).isoformat(),"status":"SHADOW_CONTEXT_READY" if canonical.get("status")=="READY_FOR_ADVISORY_ENRICHMENT" and advisory.get("status")=="READY" else "SHADOW_CONTEXT_DEGRADED","markets":markets,"timeframes":tfs,"layer_status":layers,"canonical_context":canonical,"swing_advisory":advisory,"live":{"eligible":False,"blockers":list(dict.fromkeys(blockers)),"execution_authority":"SJAGIL_CRYPTO_ONLY","automatic_live_promotion":False,"autoscale":False},"safety":{"spot_only":True,"long_only":True,"private_exchange_requests":0,"orders_generated":0,"orders_submitted":0},"artifacts":{"canonical_context":str(cp),"swing_advisory":str(ap),"latest":str(fp)}}; write(fp,final); print(json.dumps(final,indent=2,sort_keys=True,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
