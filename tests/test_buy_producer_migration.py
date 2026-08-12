from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _submit_keywords(relative: str, function_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(functions) == 1, (relative, function_name)
    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit_order"
    ]
    assert len(calls) == 1, (relative, function_name)
    return {keyword.arg for keyword in calls[0].keywords if keyword.arg}


def test_every_live_buy_submission_passes_canonical_chain() -> None:
    producers = {
        "core/autonomous_trading.py": ("submit_reserved_rr_entry",),
        "core/event_driven_live.py": ("submit_with_fresh_portfolio",),
        "core/generated_strategy_live.py": (
            "submit_reserved_replacement",
            "submit_reserved_generated_entry",
        ),
        "core/cli.py": ("submit_reserved_manual_entry",),
    }
    for relative, functions in producers.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "canonicalize_approved_buy_order" in source
        for function_name in functions:
            assert "canonical_chain" in _submit_keywords(
                relative, function_name
            )


def test_inventory_reallocation_is_sell_only_safety_path() -> None:
    tree = ast.parse(
        (ROOT / "core" / "inventory_reallocation.py").read_text(
            encoding="utf-8"
        )
    )
    order_intents = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OrderIntent"
    ]
    assert order_intents
    for call in order_intents:
        side = next(keyword.value for keyword in call.keywords if keyword.arg == "side")
        assert isinstance(side, ast.Attribute)
        assert side.attr == "SELL"
