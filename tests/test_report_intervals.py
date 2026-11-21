"""Risk intervals in the JSON report and replay compatibility with old reports."""

import numpy as np
import pandas as pd
import pytest

from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                               run_simulation, write_report)


@pytest.fixture
def snapshot():
    prices = pd.Series(100 * np.exp([0, 0.01, -0.01, 0.02]),
                       index=pd.date_range("2026-01-05", periods=4, freq="B"))
    return historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                               retrieved_at="2026-01-09T12:00:00+00:00",
                               source="deterministic_test_fixture")


CONFIG = SimulationConfig(horizon=0.5, steps=6, n_sims=500, seed=7, confidence=0.9)


def test_report_contains_deterministic_risk_intervals(snapshot):
    report, _ = run_simulation(snapshot, CONFIG)
    again, _ = run_simulation(snapshot, CONFIG)
    assert "risk_intervals" in report["results"]
    assert report["results"]["risk_intervals"] == again["results"]["risk_intervals"]


def test_replay_accepts_reports_without_risk_intervals(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    del report["results"]["risk_intervals"]
    path = tmp_path / "old.json"
    write_report(report, path)
    assert "risk_intervals" not in read_report(path)["results"]


def test_replay_accepts_current_report(snapshot, tmp_path):
    report, _ = run_simulation(snapshot, CONFIG)
    path = tmp_path / "new.json"
    write_report(report, path)
    assert read_report(path)["results"]["risk_intervals"] == report["results"]["risk_intervals"]
