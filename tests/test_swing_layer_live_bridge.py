from datetime import datetime,timedelta,timezone
from decimal import Decimal
from types import SimpleNamespace
import core.swing_layer_live as bridge

class _Eligibility: status=SimpleNamespace(value="ALLOWED")
class _Settings: shariah=SimpleNamespace(eligibility=lambda market:_Eligibility())

def _payload(notional="10"):
    now=datetime.now(timezone.utc)
    return {"intent_id":"test","created_at":now.isoformat(),"market":"BTC-EUR","side":"BUY",
        "notional_eur":notional,"expected_edge_bps":100,"estimated_round_trip_cost_bps":60,
        "net_edge_bps":40,"stop_pct":.01,"take_profit_pct":.03,"trailing_stop_pct":.01,
        "strategy":"TREND_PULLBACK","authority":"LIVE","expires_at":(now+timedelta(minutes=2)).isoformat(),
        "metadata":{"signal_score":.8,"crypto_repo_context":{"data_source":"Sjagil/crypto",
        "entry_blocked":False,"nlp_severe_negative":False}}}

def test_level2_rejects_above_twenty_five(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "_authority",
        lambda: {**bridge._defaults(), "active": True},
    )
    authority, blockers = bridge._validate_intent(
        {
            "authority": "LIVE",
            "market": "BTC-EUR",
            "notional_eur": "25.01",
            "net_edge_bps": "50",
            "stop_pct": "0.02",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
    )
    assert authority["capital_level"] == 2
    assert "ORDER_CAP_EXCEEDED" in blockers


def test_level2_canary_constants():
    assert bridge.CAPITAL_LEVEL == 2
    assert bridge.MAXIMUM_ORDER_EUR == Decimal("25")
    assert bridge.MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR == Decimal("75")
    assert bridge.MAXIMUM_MANAGED_POSITIONS == 3
    assert bridge.MAXIMUM_NEW_ORDERS_PER_DAY == 3
    assert bridge.MAXIMUM_RISK_PER_TRADE_EUR == Decimal("2")
    assert bridge.LEVEL_2_AUTOSCALE is False
