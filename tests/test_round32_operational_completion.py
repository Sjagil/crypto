from __future__ import annotations
import inspect
from pathlib import Path
from core.contracts import DataValidationError
from data.downloader import _is_historical_prefix_absence
from reporting import top_existing_strategies
ROOT = Path(__file__).resolve().parents[1]

def test_prefix_absence_is_narrow():
    assert _is_historical_prefix_absence(DataValidationError("Bitvavo returned no candles for SOL-EUR 1d"))
    assert _is_historical_prefix_absence(DataValidationError("OHLCV data must be a non-empty DataFrame"))
    assert not _is_historical_prefix_absence(DataValidationError("provider candles conflict with stored history"))
    assert not _is_historical_prefix_absence(DataValidationError("Bitvavo candle pagination did not advance"))

def test_rotation_lead_optional():
    source=inspect.getsource(top_existing_strategies.collect_longlist)
    assert "rotation_research_lead_v1.json" in source
    assert "lead_path.is_file()" in source

def test_runner_contract():
    p=Path("/Users/ayoubalhari/Downloads/crypto-ai-swing-layer/scripts/run_operational_stack.py")
    source=p.read_text(encoding="utf-8")
    assert '"history_sync"' in source
    assert "ADVISORY_PHASES" in source
    assert "required_blocked" in source
