from __future__ import annotations

from reporting.multi_source_data_platform import REPORT_SECTIONS


def test_required_multi_source_report_topology_is_exact_a_through_w() -> None:
    assert REPORT_SECTIONS == tuple("ABCDEFGHIJKLMNOPQRSTUVW")
    assert len(REPORT_SECTIONS) == 23
