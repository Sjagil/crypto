from reporting.bitvavo_l2_performance import benchmark_bitvavo_l2_v2


def test_reconstruction_performance_probe_is_bounded_and_fail_closed() -> None:
    result = benchmark_bitvavo_l2_v2(2_000)
    assert result["events_per_second"] > 100
    assert result["peak_traced_memory_bytes"] > 0
    assert result["bounded_recent_event_capacity"] == 20_000
    assert result["final_sequence"] == 2_001
    assert result["final_state"] == "VALID"
    assert result["features_available"]
    assert result["orders_generated"] == 0
