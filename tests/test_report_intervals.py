"""E1.4: ``results.risk_intervals`` in the JSON report (offline, seeded).

The block holds Monte Carlo *estimator* intervals (sampling error with mu and
sigma fixed), labelled ``interval_kind == "monte_carlo_estimator"`` so they
are not confused with ``calibration_sensitivity`` (parameter uncertainty).
It is an optional field of schema v1: reports saved before it existed must
still replay, while a tampered block must be rejected.
"""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import main as app
from risk.intervals import cvar_interval, prob_loss_interval, var_interval
from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                               run_simulation, write_report)


@pytest.fixture
def snapshot():
    prices = pd.Series(100 * np.exp([0, 0.01, -0.01, 0.02]),
                       index=pd.date_range("2026-01-05", periods=4, freq="B"))
    return historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                               retrieved_at="2026-01-09T12:00:00+00:00",
                               source="deterministic_test_fixture")


@pytest.fixture
def constant_snapshot():
    # Zero log-returns -> sigma = 0 and mu = 0: every terminal price is s0.
    return historical_snapshot(
        pd.Series([100.0] * 4, index=pd.date_range("2026-01-01", periods=4)),
        ticker="CONSTANT", period="fixture", retrieved_at="2026-01-09T00:00:00Z",
        source="deterministic_test_fixture")


CONFIG = SimulationConfig(horizon=0.5, steps=4, n_sims=300, seed=7, confidence=0.9)


def _write(report, path):
    write_report(report, path)
    return path


def test_report_contains_labelled_intervals_consistent_with_risk(snapshot):
    report, paths = run_simulation(snapshot, CONFIG)
    intervals = report["results"]["risk_intervals"]
    risk = report["results"]["risk"]
    assert intervals["interval_kind"] == "monte_carlo_estimator"
    assert intervals["level"] == 0.95
    assert set(intervals) == {"interval_kind", "level", "prob_loss", "var", "cvar"}
    assert intervals["prob_loss"]["estimate"] == risk["prob_loss"]
    assert intervals["var"]["estimate"] == risk["var"]
    assert intervals["cvar"]["estimate"] == risk["cvar"]
    assert intervals["cvar"]["var"] == risk["var"]
    for key in ("prob_loss", "var", "cvar"):
        item = intervals[key]
        assert item["level"] == 0.95
        assert item["n"] == CONFIG.n_sims
        assert item["lower"] <= item["upper"]
    assert intervals["var"]["confidence"] == CONFIG.confidence
    assert intervals["cvar"]["confidence"] == CONFIG.confidence
    assert intervals["prob_loss"]["method"] == "wilson"
    assert intervals["var"]["method"] == "order_statistics"
    assert intervals["cvar"]["method"] == "asymptotic"
    # Built from the simulated terminal row with the public functions.
    s0 = report["last_price"]
    assert paths[0, 0] == s0
    assert intervals["prob_loss"] == prob_loss_interval(paths[-1], s0)
    assert intervals["var"] == var_interval(paths[-1], s0, CONFIG.confidence)
    assert intervals["cvar"] == cvar_interval(paths[-1], s0, CONFIG.confidence)


def test_report_with_intervals_is_deterministic_and_json_safe(snapshot, tmp_path):
    first, _ = run_simulation(snapshot, CONFIG)
    second, _ = run_simulation(snapshot, CONFIG)
    assert first["results"]["risk_intervals"] == second["results"]["risk_intervals"]
    path = _write(first, tmp_path / "run.json")
    assert json.loads(path.read_text())["results"]["risk_intervals"] == \
        first["results"]["risk_intervals"]


def test_new_report_replays(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    path = _write(report, tmp_path / "run.json")
    assert read_report(path) == report


@pytest.mark.parametrize("damage", [
    "cvar_upper", "var_lower", "prob_loss_estimate", "interval_kind", "level",
    "drop_cvar",
])
def test_tampered_risk_intervals_are_rejected(snapshot, tmp_path, damage):
    report, _ = run_simulation(snapshot, CONFIG)
    damaged = deepcopy(report)
    block = damaged["results"]["risk_intervals"]
    if damage == "cvar_upper":
        block["cvar"]["upper"] += 1e-9
    elif damage == "var_lower":
        block["var"]["lower"] -= 1
    elif damage == "prob_loss_estimate":
        block["prob_loss"]["estimate"] = 0.5
    elif damage == "interval_kind":
        block["interval_kind"] = "calibration_sensitivity"
    elif damage == "level":
        block["level"] = 0.9
    else:
        del block["cvar"]
    path = _write(damaged, tmp_path / "damaged.json")
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)


def test_legacy_v1_report_without_block_still_replays(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    legacy = deepcopy(report)
    del legacy["results"]["risk_intervals"]
    path = _write(legacy, tmp_path / "legacy.json")
    assert "risk_intervals" not in json.loads(path.read_text())["results"]
    replayed = read_report(path)
    assert replayed == legacy
    assert "risk_intervals" not in replayed["results"]


def test_legacy_report_still_checks_remaining_results(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    legacy = deepcopy(report)
    del legacy["results"]["risk_intervals"]
    legacy["results"]["risk"]["cvar"] += 1
    path = _write(legacy, tmp_path / "legacy-damaged.json")
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)


def test_empty_results_still_fail_replay(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    report["results"] = {}
    path = _write(report, tmp_path / "empty.json")
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)


def test_results_with_only_the_block_fail_replay(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    report["results"] = {"risk_intervals": report["results"]["risk_intervals"]}
    path = _write(report, tmp_path / "only-block.json")
    with pytest.raises(ValueError, match="Replay result mismatch"):
        read_report(path)


@pytest.mark.parametrize("n_sims", [1, 50])
def test_degenerate_runs_serialize_without_nan(constant_snapshot, tmp_path, n_sims):
    config = SimulationConfig(steps=1, n_sims=n_sims)
    report, _ = run_simulation(constant_snapshot, config)
    assert report["parameters"]["sigma"] == 0.0
    intervals = report["results"]["risk_intervals"]
    path = _write(report, tmp_path / f"constant-{n_sims}.json")
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert read_report(path) == report
    # Every terminal price equals s0: no loss, VaR = CVaR = 0.
    assert intervals["prob_loss"]["estimate"] == 0.0
    assert intervals["var"]["estimate"] == 0.0
    assert intervals["cvar"]["estimate"] == 0.0
    if n_sims == 1:
        assert intervals["cvar"]["lower"] is None
        assert intervals["cvar"]["upper"] is None
        assert intervals["cvar"]["standard_error"] is None
        assert intervals["cvar"]["note"]
    else:
        assert intervals["cvar"]["lower"] == intervals["cvar"]["upper"] == 0.0
        assert intervals["cvar"]["standard_error"] == 0.0


def test_single_path_gbm_run_serializes(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=1, seed=3))
    intervals = report["results"]["risk_intervals"]
    assert intervals["cvar"]["lower"] is None and intervals["cvar"]["upper"] is None
    assert intervals["var"]["saturated"] is True
    assert intervals["var"]["lower"] == intervals["var"]["upper"] == \
        intervals["var"]["estimate"]
    path = _write(report, tmp_path / "single.json")
    assert "NaN" not in path.read_text()
    assert read_report(path) == report


def _summary(report):
    config = SimulationConfig(**report["simulation"])
    return app.format_simulation_summary(report, config, 1.0, 0.0)


def test_cli_summary_shows_intervals(snapshot):
    report, _ = run_simulation(snapshot, CONFIG)
    text = _summary(report)
    assert "IC 95% do estimador Monte Carlo" in text
    for label in ("P(perda)", "VaR", "CVaR"):
        assert f"  {label}: " in text
    assert "indefinido" not in text


def test_cli_summary_handles_undefined_bounds(constant_snapshot):
    report, _ = run_simulation(constant_snapshot, SimulationConfig(steps=1, n_sims=1))
    lines = app._risk_interval_lines(report["results"]["risk_intervals"])
    cvar_line = next(line for line in lines if line.startswith("  CVaR"))
    assert "indefinido" in cvar_line
    assert "IC 95%" in _summary(report)


def test_cli_summary_without_block_for_legacy_reports(snapshot):
    report, _ = run_simulation(snapshot, CONFIG)
    del report["results"]["risk_intervals"]
    assert app._risk_interval_lines(None) == []
    assert "estimador Monte Carlo" not in _summary(report)


def test_cli_replay_prints_intervals_offline(snapshot, tmp_path, monkeypatch, capsys):
    report, _ = run_simulation(snapshot, CONFIG)
    source = _write(report, tmp_path / "source.json")

    def forbidden(*args, **kwargs):
        pytest.fail("Replay must not access the data provider")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "download_prices", forbidden)
    monkeypatch.setattr(app, "download_company", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    app.main(["--replay", str(source), "--report", str(tmp_path / "replay.json"),
              "--no-plot"])
    output = capsys.readouterr().out
    assert "IC 95% do estimador Monte Carlo" in output
    assert (tmp_path / "replay" / "report.json").read_bytes() == source.read_bytes()
