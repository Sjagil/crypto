from __future__ import annotations

from pathlib import Path
from core import swing_layer_live
from research.seven_year import _is_canonical_ohlcv_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_feature_artifact_is_not_canonical_ohlcv() -> None:
    assert not _is_canonical_ohlcv_dataset(Path("BTC-EUR_features.parquet"))
    assert _is_canonical_ohlcv_dataset(Path("BTC-EUR_1h.parquet"))


def test_canonical_swing_defaults_are_level2() -> None:
    defaults = swing_layer_live._defaults()
    assert defaults["capital_level"] == 2
    assert defaults["maximum_order_eur"] == "25"
    assert defaults["maximum_total_exposure_eur"] == "75"
    assert defaults["maximum_open_positions"] == 3
    assert defaults["maximum_new_orders_per_day"] == 3
    assert defaults["maximum_risk_per_trade_eur"] == "2"
    assert defaults["autoscale"] is False


def test_level2_approval_contract_is_explicit() -> None:
    source = (ROOT / "core" / "swing_layer_live.py").read_text(encoding="utf-8")
    assert "LEVEL_2_APPROVAL_PHRASE" in source
    assert "body = _defaults()" in source
    assert '"capital_level": CAPITAL_LEVEL' in source


def test_public_provider_contracts() -> None:
    source = (ROOT / "data" / "downloader.py").read_text(encoding="utf-8")
    assert '"pair": f"{pair_base}{quote_currency}"' in source
    bitvavo = source[source.index("class BitvavoProvider:"):source.index("class KrakenProvider:")]
    assert '"end": cursor_end' in bitvavo
    assert '"start": start_ms' not in bitvavo
    assert "next_end = oldest" in bitvavo
