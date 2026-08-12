from __future__ import annotations

from reporting.multi_source_maturation_platform import REPORT_SECTIONS


def test_required_p1_2_2_report_topology_is_exact_a_through_v() -> None:
    assert REPORT_SECTIONS == tuple("ABCDEFGHIJKLMNOPQRSTUV")
    assert len(REPORT_SECTIONS) == 22
