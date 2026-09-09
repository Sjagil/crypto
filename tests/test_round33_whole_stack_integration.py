from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import numpy as np,pandas as pd
from core.whole_market_context import _closed_frame,_mtf

def test_round33_closed_candle_and_real_provenance(tmp_path:Path):
    p=tmp_path/"BTC-EUR_1h.parquet"; idx=pd.date_range("2026-01-01",periods=260,freq="1h",tz="UTC"); f=pd.DataFrame({"open":np.arange(260)+100,"high":np.arange(260)+101,"low":np.arange(260)+99,"close":np.arange(260)+100.5,"volume":10.0},index=idx); f.index.name="timestamp"; f.to_parquet(p); Path(str(p)+".provenance.json").write_text(json.dumps({"source_type":"REAL_PROVIDER_DATA","provider":"bitvavo","rows":260}),encoding="utf-8"); closed,q=_closed_frame(p,"BTC-EUR","1h",idx[-1].to_pydatetime() + timedelta(hours=1)); assert len(closed)==260; assert q["provenance"]["status"]=="READY"

def test_round33_mtf_includes_all_available_timeframes():
    x=_mtf({"15m":{"technical_score":.2},"1h":{"technical_score":.6},"2h":{"technical_score":.5},"4h":{"technical_score":.7},"1d":{"technical_score":.4},"1W":{"technical_score":.1}}); assert x["status"]=="READY"; assert "2h" in x["available_timeframes"]
