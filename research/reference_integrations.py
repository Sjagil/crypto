"""Pinned, research-only adapters for the repositories in ``crypto-references``.

The adapters deliberately load or execute the local upstream source tree.  They
do not copy upstream implementations into this project and they never receive
credentials, an exchange client, or execution authority.  Every invocation is
hashed so reports can prove that a named upstream callable actually ran.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from utils.common import sha256_file, stable_hash, utc_iso

REFERENCE_INTEGRATION_SCHEMA = "crypto_reference_integration_v2"


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    name: str
    commit: str
    license: str
    callable: str
    role: str
    isolation: str


REFERENCE_SPECS: tuple[ReferenceSpec, ...] = (
    ReferenceSpec(
        "finrl-trading",
        "e65d6f0483ead7d2ef4a5fc940cdf960392a25c1",
        "Apache-2.0",
        "src.strategies.adaptive_rotation.utils.robust_stats.robust_zscore",
        "robust feature diagnostic supporting the bounded RL state contract",
        "direct standalone source-module load; no FinRL agent or execution engine",
    ),
    ReferenceSpec(
        "freqtrade",
        "89d469fe638eaf116d45a8f92598aeed4d9f6dde",
        "GPL-3.0",
        "freqtrade.exchange.exchange_utils_timeframe.timeframe_to_seconds",
        "feature-cadence validation",
        "source import; no exchange object or strategy engine",
    ),
    ReferenceSpec(
        "lean",
        "c6cc3b743ed7b65d5e0b9fa2bfc18b7d3ac2aea0",
        "Apache-2.0",
        "QuantConnect.Statistics.Statistics.SharpeRatio(double,double,double)",
        "independent research-metric cross-check",
        "separate local process compiled from the pinned upstream source file",
    ),
    ReferenceSpec(
        "nautilus_trader",
        "e8be4522cd12a4a65a4d1350f791d414ad246439",
        "LGPL-3.0",
        "nautilus_model::orderbook::OrderBook::{spread,midpoint}",
        "independent L2 top-of-book cross-check",
        "separate Rust process linked to the pinned local path dependency",
    ),
    ReferenceSpec(
        "pybroker",
        "e0e7b08886343274efb05b96f7399ca3de280aa5",
        "Apache-2.0 with Commons Clause",
        "pybroker.vect.returnv",
        "feature-return cross-check",
        "direct source-module load with a no-JIT decorator shim",
    ),
    ReferenceSpec(
        "qlib",
        "79633dd9506ea689e5400dea0197717b5b3d74b7",
        "MIT",
        "qlib.utils.index_data.sum_by_index",
        "indexed feature aggregation cross-check",
        "direct standalone source-module load",
    ),
    ReferenceSpec(
        "rd-agent",
        "6762f84f9bc0f5c6486c50a00e128a57ac6c3683",
        "MIT",
        "rdagent.core.evaluation.Feedback.is_acceptable",
        "research feedback lifecycle cross-check",
        "direct standalone source-module load; no RD loop, LLM or workspace mutation",
    ),
    ReferenceSpec(
        "tradingagents",
        "a33fd4c0f134485a43553a2c23a63cb14adbd88f",
        "Apache-2.0",
        "tradingagents.agents.schemas.ResearchPlan.model_validate",
        "structured research-plan schema cross-check",
        "direct standalone source-module load; no graph, provider, agent or trading action",
    ),
    ReferenceSpec(
        "vectorbt",
        "34b6d5935e3ea3eccd549e2592bc0f455b8045f5",
        "Apache-2.0 with Commons Clause",
        "vectorbt.utils.math_.{is_less_nb,is_close_nb}",
        "numeric book-invariant cross-check",
        "direct source-module load with a no-JIT decorator shim",
    ),
)


def _repository_root(workspace: Path, name: str) -> Path:
    root = workspace / "crypto-references" / name
    if not root.is_dir():
        raise FileNotFoundError(f"reference repository is missing: {root}")
    return root


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


@contextmanager
def _temporary_modules(replacements: dict[str, types.ModuleType]) -> Iterator[None]:
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _no_jit_module() -> types.ModuleType:
    module = types.ModuleType("numba")

    def njit(*args: Any, **_kwargs: Any) -> Any:
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    module.njit = njit  # type: ignore[attr-defined]
    return module


def _load_standalone(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freqtrade_timeframe_seconds(workspace: Path, timeframe: str) -> int:
    root = _repository_root(workspace, "freqtrade")
    exchange_pkg = types.ModuleType("freqtrade.exchange")
    exchange_pkg.__path__ = [str(root / "freqtrade" / "exchange")]  # type: ignore[attr-defined]
    util_pkg = types.ModuleType("freqtrade.util")
    util_pkg.__path__ = [str(root / "freqtrade" / "util")]  # type: ignore[attr-defined]
    datetime_helpers = types.ModuleType("freqtrade.util.datetime_helpers")
    # The selected callable does not use these functions; placeholders prevent
    # importing Freqtrade's unrelated CLI dependency surface.
    datetime_helpers.dt_from_ts = lambda value: value  # type: ignore[attr-defined]
    datetime_helpers.dt_ts = lambda value: value  # type: ignore[attr-defined]
    sys.path.insert(0, str(root))
    try:
        with _temporary_modules(
            {
                "freqtrade.exchange": exchange_pkg,
                "freqtrade.util": util_pkg,
                "freqtrade.util.datetime_helpers": datetime_helpers,
            }
        ):
            sys.modules.pop("freqtrade.exchange.exchange_utils_timeframe", None)
            module = importlib.import_module("freqtrade.exchange.exchange_utils_timeframe")
            return int(module.timeframe_to_seconds(timeframe))
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("freqtrade.exchange.exchange_utils_timeframe", None)


def pybroker_returns(workspace: Path, values: list[float], period: int = 1) -> list[float | None]:
    path = _repository_root(workspace, "pybroker") / "src" / "pybroker" / "vect.py"
    with _temporary_modules({"numba": _no_jit_module()}):
        module = _load_standalone(path, "_crypto_reference_pybroker_vect")
        result = module.returnv(np.asarray(values, dtype=np.float64), int(period))
    return [None if np.isnan(value) else float(value) for value in result]


def qlib_sum_by_index(
    workspace: Path,
    rows: list[dict[int, float]],
    index: list[int],
) -> dict[str, float]:
    path = _repository_root(workspace, "qlib") / "qlib" / "utils" / "index_data.py"
    module = _load_standalone(path, "_crypto_reference_qlib_index_data")
    values = [module.SingleData(row) for row in rows]
    result = module.sum_by_index(values, index).to_dict()
    return {str(key): float(value) for key, value in result.items()}


def vectorbt_book_invariants(
    workspace: Path,
    best_bid: float,
    best_ask: float,
) -> dict[str, bool]:
    path = _repository_root(workspace, "vectorbt") / "vectorbt" / "utils" / "math_.py"
    with _temporary_modules({"numba": _no_jit_module()}):
        module = _load_standalone(path, "_crypto_reference_vectorbt_math")
        return {
            "strictly_uncrossed": bool(module.is_less_nb(best_bid, best_ask)),
            "locked_within_tolerance": bool(module.is_close_nb(best_bid, best_ask)),
        }


def finrl_robust_zscores(
    workspace: Path,
    values: list[float],
    *,
    window: int,
) -> list[float | None]:
    path = (
        _repository_root(workspace, "finrl-trading")
        / "src"
        / "strategies"
        / "adaptive_rotation"
        / "utils"
        / "robust_stats.py"
    )
    module = _load_standalone(path, "_crypto_reference_finrl_robust_stats")
    result = module.robust_zscore(module.pd.Series(values, dtype=float), window=int(window))
    return [None if module.pd.isna(value) else float(value) for value in result]


def rdagent_feedback_acceptability(workspace: Path) -> dict[str, bool]:
    path = _repository_root(workspace, "rd-agent") / "rdagent" / "core" / "evaluation.py"
    module = _load_standalone(path, "_crypto_reference_rdagent_evaluation")
    feedback = module.Feedback()
    return {"acceptable": bool(feedback.is_acceptable()), "finished": bool(feedback.finished())}


def tradingagents_research_plan(workspace: Path) -> dict[str, Any]:
    path = (
        _repository_root(workspace, "tradingagents")
        / "tradingagents"
        / "agents"
        / "schemas.py"
    )
    module = _load_standalone(path, "_crypto_reference_tradingagents_schemas")
    module.ResearchPlan.model_rebuild(_types_namespace=vars(module))
    plan = module.ResearchPlan.model_validate(
        {
            "recommendation": "Hold",
            "rationale": "Bull and bear evidence are balanced.",
            "strategic_actions": "Collect more point-in-time evidence.",
        }
    )
    return dict(plan.model_dump(mode="json"))


def _run_json_probe(
    command: list[str],
    *,
    timeout: float = 20.0,
    extra_path: Path | None = None,
) -> dict[str, Any]:
    selected_path = os.environ.get("PATH", "")
    if extra_path is not None:
        selected_path = f"{extra_path}{os.pathsep}{selected_path}"
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "NO_COLOR": "1", "PATH": selected_path},
    )
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


def lean_sharpe_ratio(
    workspace: Path,
    average: float,
    standard_deviation: float,
    risk_free_rate: float = 0.0,
) -> float:
    executable = workspace / "tools" / "reference_probes" / "lean" / "lean_reference_probe.exe"
    if not executable.is_file():
        raise FileNotFoundError("Lean probe is not built; run scripts/build_reference_probes.py")
    payload = _run_json_probe(
        [str(executable), str(average), str(standard_deviation), str(risk_free_rate)]
    )
    return float(payload["sharpe_ratio"])


def nautilus_top_of_book(
    workspace: Path,
    best_bid: float,
    best_ask: float,
    bid_size: float,
    ask_size: float,
) -> dict[str, float]:
    base = workspace / "tools" / "reference_probes" / "nautilus" / "target" / "release"
    executable = base / (
        "nautilus_reference_probe.exe" if os.name == "nt" else "nautilus_reference_probe"
    )
    if not executable.is_file():
        raise FileNotFoundError(
            "Nautilus probe is not built; run scripts/build_reference_probes.py"
        )
    payload = _run_json_probe(
        [str(executable), str(best_bid), str(best_ask), str(bid_size), str(ask_size)],
        extra_path=(
            workspace
            / ".tools"
            / "rustup"
            / "toolchains"
            / "1.97.1-x86_64-pc-windows-gnullvm"
            / "lib"
            / "rustlib"
            / "x86_64-pc-windows-gnullvm"
            / "bin"
        ),
    )
    return {"spread": float(payload["spread"]), "midpoint": float(payload["midpoint"])}


def verify_reference_repositories(workspace: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in REFERENCE_SPECS:
        root = _repository_root(workspace, spec.name)
        actual = _git_commit(root)
        rows.append(
            {
                **asdict(spec),
                "path": str(root),
                "actual_commit": actual,
                "commit_verified": actual == spec.commit,
                "license_file_sha256": sha256_file(
                    root / ("LICENSE.md" if spec.name == "vectorbt" else "LICENSE")
                ),
            }
        )
    return rows


def run_reference_integration_probes(workspace: Path) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "finrl-trading": {"values": [1.0, 1.0, 2.0, 3.0], "window": 3},
        "freqtrade": {"timeframe": "5m"},
        "lean": {"average": 0.01, "standard_deviation": 0.02, "risk_free_rate": 0.0},
        "nautilus_trader": {
            "best_bid": 100.0,
            "best_ask": 101.0,
            "bid_size": 2.0,
            "ask_size": 3.0,
        },
        "pybroker": {"values": [100.0, 101.0, 99.0], "period": 1},
        "qlib": {"rows": [{1: 2.0, 3: 4.0}, {1: 1.0, 2: 5.0}], "index": [1, 2, 3]},
        "rd-agent": {},
        "tradingagents": {
            "recommendation": "Hold",
            "rationale": "Bull and bear evidence are balanced.",
            "strategic_actions": "Collect more point-in-time evidence.",
        },
        "vectorbt": {"best_bid": 100.0, "best_ask": 101.0},
    }
    outputs = {
        "finrl-trading": {
            "robust_zscores": finrl_robust_zscores(
                workspace, [1.0, 1.0, 2.0, 3.0], window=3
            )
        },
        "freqtrade": {"seconds": freqtrade_timeframe_seconds(workspace, "5m")},
        "lean": {"sharpe_ratio": lean_sharpe_ratio(workspace, 0.01, 0.02)},
        "nautilus_trader": nautilus_top_of_book(workspace, 100.0, 101.0, 2.0, 3.0),
        "pybroker": {"returns": pybroker_returns(workspace, [100.0, 101.0, 99.0])},
        "qlib": {"sum_by_index": qlib_sum_by_index(workspace, inputs["qlib"]["rows"], [1, 2, 3])},
        "rd-agent": rdagent_feedback_acceptability(workspace),
        "tradingagents": tradingagents_research_plan(workspace),
        "vectorbt": vectorbt_book_invariants(workspace, 100.0, 101.0),
    }
    repositories = verify_reference_repositories(workspace)
    executions = []
    for spec in REFERENCE_SPECS:
        executions.append(
            {
                "repository": spec.name,
                "callable": spec.callable,
                "input_hash": stable_hash(inputs[spec.name]),
                "output_hash": stable_hash(outputs[spec.name]),
                "output": outputs[spec.name],
                "executed": True,
                "execution_authority": False,
                "orders_generated": 0,
            }
        )
    body = {
        "schema_version": REFERENCE_INTEGRATION_SCHEMA,
        "generated_at": utc_iso(),
        "repositories": repositories,
        "executions": executions,
        "all_commits_verified": all(row["commit_verified"] for row in repositories),
        "all_functions_executed": all(row["executed"] for row in executions),
        "private_exchange_requests": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return {**body, "evidence_hash": stable_hash(body)}


__all__ = [
    "REFERENCE_INTEGRATION_SCHEMA",
    "REFERENCE_SPECS",
    "finrl_robust_zscores",
    "freqtrade_timeframe_seconds",
    "lean_sharpe_ratio",
    "nautilus_top_of_book",
    "pybroker_returns",
    "qlib_sum_by_index",
    "rdagent_feedback_acceptability",
    "run_reference_integration_probes",
    "tradingagents_research_plan",
    "vectorbt_book_invariants",
    "verify_reference_repositories",
]
