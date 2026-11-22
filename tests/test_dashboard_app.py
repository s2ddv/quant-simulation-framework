"""Streamlit dashboard smoke tests with a saved offline report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from simulation.report import SimulationConfig, historical_snapshot, run_simulation, write_report

APP = str(Path(__file__).resolve().parents[1] / "dashboard.py")


@pytest.fixture
def report_folder(tmp_path):
    log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
    prices = pd.Series(100 * np.exp(log_prices),
                       index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B"))
    snapshot = historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                                   retrieved_at="2026-03-01T12:00:00+00:00",
                                   source="deterministic_test_fixture")
    report, _ = run_simulation(
        snapshot, SimulationConfig(horizon=0.5, steps=20, n_sims=300, seed=5),
        {"calibration_observations": 10, "horizon_days": 1, "alpha": 0.95},
        {"mu_delta": 0.02, "sigma_relative_delta": 0.2, "scenario_levels": 3})
    write_report(report, tmp_path / "run" / "report.json")
    return tmp_path, report


def run_app(folder):
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["folder"] = str(folder)
    return app.run()


def test_dashboard_renders_a_saved_report(report_folder):
    folder, report = report_folder
    app = run_app(folder)
    assert not app.exception and not app.error
    assert "SYNTHETIC" in app.markdown[0].value
    values = {metric.label: metric.value for metric in app.metric}
    assert values["VaR 95%"] == f"{report['results']['risk']['var']:,.2f}"
    assert values["Violações"].endswith(
        f"/ {report['out_of_sample']['evaluation']['metrics']['forecast_count']}")
    app.radio(key="metric").set_value("q0.05").run()
    assert not app.exception


def test_dashboard_reports_empty_folder_and_unreproducible_report(report_folder, tmp_path):
    empty = run_app(tmp_path / "missing")
    assert not empty.exception and "Nenhum relatório" in empty.info[0].value
    folder, _ = report_folder
    target = folder / "run" / "report.json"
    target.write_text(target.read_text().replace('"model": "single_asset_gbm"',
                                                 '"model": "other"'))
    broken = run_app(folder)
    assert not broken.exception and "Unsupported report" in broken.error[0].value


def test_new_simulation_runs_offline_with_a_fake_provider_and_saves_on_request(monkeypatch, tmp_path):
    import calibration.data

    def fake_download(ticker, period="3y"):
        log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
        return pd.Series(100 * np.exp(log_prices),
                         index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B"))

    monkeypatch.setattr(calibration.data, "download_prices", fake_download)
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["source"] = "Nova simulação"
    app.run()
    assert not app.exception and "Simular" in app.info[0].value
    app.text_input(key="ticker").set_value("petr4")
    app.number_input(key="n_sims").set_value(400)
    app.number_input(key="steps").set_value(20)
    app.number_input(key="calibration").set_value(10)
    app.selectbox(key="levels").set_value(3)
    next(b for b in app.sidebar.button if b.label == "Simular").click()
    app.run()
    assert not app.exception and not app.error
    assert app.markdown[0].value.startswith("### PETR4.SA")
    assert not (tmp_path / "reports").exists()
    app.button(key="save").click().run()
    assert not app.exception and not app.error
    saved = list((tmp_path / "reports").glob("PETR4.SA-*/report.json"))
    assert len(saved) == 1 and app.success


def test_new_simulation_reports_memory_limit_without_downloading(monkeypatch):
    import calibration.data

    def fail(*args, **kwargs):
        raise AssertionError("must not download")

    monkeypatch.setattr(calibration.data, "download_prices", fail)
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["source"] = "Nova simulação"
    app.run()
    app.number_input(key="n_sims").set_value(1_000_000)
    next(b for b in app.sidebar.button if b.label == "Simular").click()
    app.run()
    assert not app.exception and "limit" in app.error[0].value


def test_dashboard_renders_a_single_path_report(tmp_path):
    log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
    snapshot = historical_snapshot(
        pd.Series(100 * np.exp(log_prices),
                  index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B")),
        ticker="ONE", period="fixture", retrieved_at="2026-03-01T12:00:00+00:00")
    report, _ = run_simulation(snapshot, SimulationConfig(horizon=0.5, steps=5, n_sims=1, seed=1),
                               None, {"mu_delta": 0.0, "sigma_relative_delta": 0.2,
                                      "scenario_levels": 3})
    write_report(report, tmp_path / "one" / "report.json")
    app = run_app(tmp_path)
    assert not app.exception and not app.error
    assert "indefinido" in app.metric[0].help
