"""E1.4: Monte Carlo estimator intervals in the JSON report (offline, seeded).

``results.risk_intervals`` is an optional block inside schema v1: new reports
carry it, older v1 reports without it must still replay, and tampering with
it must be detected. The intervals are Monte Carlo estimator intervals
(mu/sigma fixed), not price percentiles.
"""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import main as app
from risk.intervals import cvar_interval, prob_loss_interval, var_interval
from simulation.report import (SimulationConfig, historical_snapshot, read_replay,
                               read_report, run_simulation, write_report)


@pytest.fixture
def snapshot():
    # Same deterministic fixture pattern as tests/test_report.py (copied).
    prices = pd.Series(100 * np.exp([0, 0.01, -0.01, 0.02]),
                       index=pd.date_range("2026-01-05", periods=4, freq="B"))
    return historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                               retrieved_at="2026-01-09T12:00:00+00:00",
                               source="deterministic_test_fixture")


CONFIG = SimulationConfig(horizon=0.5, steps=4, n_sims=400, seed=321, confidence=0.9)


def test_report_block_matches_direct_interval_calls(snapshot):
    report, paths = run_simulation(snapshot, CONFIG)
    block = report["results"]["risk_intervals"]
    terminal, s0 = paths[-1], report["last_price"]
    assert s0 == paths[0, 0]
    assert set(block) == {"interval_kind", "level", "prob_loss", "var", "cvar"}
    assert block["interval_kind"] == "monte_carlo_estimator"
    assert block["level"] == 0.95
    assert block["prob_loss"] == prob_loss_interval(terminal, s0, level=0.95,
                                                    method="wilson")
    assert block["var"] == var_interval(terminal, s0, confidence=CONFIG.confidence,
                                        level=0.95)
    assert block["cvar"] == cvar_interval(terminal, s0, confidence=CONFIG.confidence,
                                          level=0.95, method="asymptotic")
    assert block["var"]["confidence"] == block["cvar"]["confidence"] == 0.9
    risk = report["results"]["risk"]
    assert block["prob_loss"]["estimate"] == risk["prob_loss"]
    assert block["var"]["estimate"] == risk["var"]
    assert block["cvar"]["estimate"] == risk["cvar"]


def test_report_with_block_round_trips_and_replays(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    json.dumps(report, allow_nan=False)
    path = tmp_path / "run.json"
    write_report(report, path)
    loaded = read_report(path)
    assert loaded["results"]["risk_intervals"] == report["results"]["risk_intervals"]
    replay_snapshot, replay_config = read_replay(path)
    replay, _ = run_simulation(replay_snapshot, replay_config)
    assert replay == report


def test_single_path_report_is_json_safe_with_undefined_cvar(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, SimulationConfig(steps=1, n_sims=1))
    cvar = report["results"]["risk_intervals"]["cvar"]
    assert cvar["n_tail"] == 1
    assert cvar["lower"] is None and cvar["upper"] is None
    assert cvar["standard_error"] is None and isinstance(cvar["note"], str)
    json.dumps(report, allow_nan=False)
    path = tmp_path / "single.json"
    write_report(report, path)
    assert "NaN" not in path.read_text()
    assert read_report(path)["results"]["risk_intervals"] == report["results"][
        "risk_intervals"]


def test_v1_report_without_block_still_replays(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    old = deepcopy(report)
    del old["results"]["risk_intervals"]
    path = tmp_path / "old_v1.json"
    write_report(old, path)
    loaded = read_report(path)
    assert "risk_intervals" not in loaded["results"]
    assert loaded["schema_version"] == 1
    replay_snapshot, replay_config = read_replay(path)
    assert replay_snapshot == report["historical_data"]
    assert replay_config == CONFIG


@pytest.mark.parametrize("mutate", [
    lambda b: b["var"].__setitem__("upper", b["var"]["upper"] + 1),
    lambda b: b["prob_loss"].__setitem__("lower", b["prob_loss"]["lower"] / 2),
    lambda b: b["cvar"].__setitem__("standard_error", b["cvar"]["standard_error"] * 2),
    lambda b: b.__setitem__("level", 0.9),
    lambda b: b.__setitem__("interval_kind", "price_percentile"),
    lambda b: b.pop("cvar"),
])
def test_tampered_block_is_rejected(snapshot, tmp_path, mutate):
    report, _ = run_simulation(snapshot, CONFIG)
    mutate(report["results"]["risk_intervals"])
    path = tmp_path / "tampered.json"
    write_report(report, path)
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)


@pytest.mark.parametrize("damage", ["delete", "empty"])
def test_removed_results_still_rejected(snapshot, tmp_path, damage):
    report, _ = run_simulation(snapshot, CONFIG)
    if damage == "delete":
        del report["results"]
    else:
        report["results"] = {}
    path = tmp_path / "no_results.json"
    write_report(report, path)
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)


def test_cli_lines_present_with_block(snapshot):
    report, _ = run_simulation(snapshot, CONFIG)
    block = report["results"]["risk_intervals"]
    lines = app._risk_interval_lines(block)
    assert len(lines) == 4
    assert "95%" in lines[0] and "Monte Carlo" in lines[0]
    for line, key, label in zip(lines[1:], ("prob_loss", "var", "cvar"),
                                ("P(perda)", "VaR", "CVaR")):
        item = block[key]
        assert line.strip().startswith(label)
        assert f"[{item['lower']:.6g}, {item['upper']:.6g}]" in line
        assert "indefinido" not in line
    summary = app.format_simulation_summary(report, CONFIG, 1.0, 1.0)
    assert all(line in summary for line in lines)


def test_cli_lines_absent_for_old_report(snapshot):
    report, _ = run_simulation(snapshot, CONFIG)
    del report["results"]["risk_intervals"]
    assert app._risk_interval_lines(report["results"].get("risk_intervals")) == []
    assert app._risk_interval_lines(None) == []
    summary = app.format_simulation_summary(report, CONFIG, 1.0, 1.0)
    assert "Monte Carlo" not in summary and "indefinido" not in summary


def test_cli_lines_show_undefined_when_lower_is_none(snapshot):
    report, _ = run_simulation(snapshot, SimulationConfig(steps=1, n_sims=1))
    lines = app._risk_interval_lines(report["results"]["risk_intervals"])
    cvar_line = next(line for line in lines if line.strip().startswith("CVaR"))
    assert "indefinido" in cvar_line
    assert sum("indefinido" in line for line in lines) == 1


def test_block_is_deterministic(snapshot):
    first, _ = run_simulation(snapshot, CONFIG)
    second, _ = run_simulation(snapshot, CONFIG)
    assert first["results"]["risk_intervals"] == second["results"]["risk_intervals"]
    assert (json.dumps(first["results"]["risk_intervals"], sort_keys=True)
            == json.dumps(second["results"]["risk_intervals"], sort_keys=True))


def test_old_report_without_block_still_detects_other_tampering(snapshot, tmp_path):
    # The compatibility pop must not weaken the check of the remaining results.
    report, _ = run_simulation(snapshot, CONFIG)
    del report["results"]["risk_intervals"]
    report["results"]["risk"]["var"] += 1
    path = tmp_path / "old_tampered.json"
    write_report(report, path)
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)
