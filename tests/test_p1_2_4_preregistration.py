from reporting.p1_2_4_preregistration import (
    DEFINITION_OF_DONE,
    REPORT_SECTIONS,
    REQUIREMENT_TITLES,
)


def test_p1_2_4_final_report_contract_is_complete() -> None:
    assert REPORT_SECTIONS == tuple("ABCDEFGHIJKLMNOPQRSTUV")
    assert len(REQUIREMENT_TITLES) == 96
    assert REQUIREMENT_TITLES[0] == "DO NOT STOP THE COLLECTOR"
    assert REQUIREMENT_TITLES[-1] == "FINAL REPORT"
    assert len(DEFINITION_OF_DONE) == 22
    assert DEFINITION_OF_DONE[0].startswith("P1.3 experiment design")
    assert DEFINITION_OF_DONE[-1] == "zero exchange mutations occur"
