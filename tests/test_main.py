"""Command-line parsing and an offline entry-point smoke test."""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import main as app
from calibration.fundamentals import company_snapshot


@pytest.fixture(autouse=True)
def offline_company(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "download_company", lambda ticker: company_snapshot(
        ticker, {"longName": "Test Company", "currency": "USD", "financialCurrency": "USD",
                 "totalRevenue": 123456789, "trailingPE": 15, "profitMargins": 0.125},
        retrieved_at="2026-01-09T12:00:00+00:00"))


@pytest.mark.parametrize("arguments", [
    ["--steps", "0"], ["--n-sims", "-1"],
    ["--horizon", "nan"], ["--horizon", "inf"],
    ["--confidence", "0"], ["--confidence", "1"], ["--seed", "-1"],
])
def test_argparse_rejects_invalid_values(arguments):
    with pytest.raises(SystemExit) as error:
        app.build_parser().parse_args(arguments)
    assert error.value.code == 2


def test_main_runs_with_arguments_without_plot_or_network(monkeypatch, capsys, tmp_path):
    prices = pd.Series([100.0, 101.0, 102.0, 103.0],
                       index=pd.date_range("2026-01-01", periods=4))

    def fake_download(ticker, period):
        assert (ticker, period) == ("TEST", "1y")
        return prices

    monkeypatch.setattr(app, "download_prices", fake_download)
    app.main(["--ticker", "TEST", "--period", "1y", "--horizon", "0.5",
              "--steps", "10", "--n-sims", "1000", "--seed", "42",
              "--confidence", "0.9", "--no-plot", "--report", str(tmp_path / "run.json")])

    output = capsys.readouterr().out
    assert "'expected_price':" in output
    assert "'prob_loss':" in output
    assert "'var':" in output
    assert "'cvar':" in output
    assert "Preço esperado teórico (0.5 ano(s))" in output
    assert (tmp_path / "run" / "report.json").is_file()
    assert (tmp_path / "run" / "convergence.png").read_bytes().startswith(b"\x89PNG")
    for name in ("paths.png", "final-distribution.png"):
        assert (tmp_path / "run" / name).read_bytes().startswith(b"\x89PNG")
    assert "Test Company (TEST)" in output
    assert "P/L (TTM): 15.00x" in output
    assert "Receita (revenue, TTM): USD 123,456,789.00" in output


def test_prompt_retries_and_normalizes_brazilian_ticker(monkeypatch, capsys, tmp_path):
    answers = iter(["", "two tickers", " petr4 "])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    received = []

    def prices(ticker, period):
        received.append(ticker)
        return pd.Series([100., 101., 102.], index=pd.date_range("2026-01-01", periods=3))

    monkeypatch.setattr(app, "download_prices", prices)
    app.main(["--no-plot", "--steps", "2", "--n-sims", "10",
              "--report", str(tmp_path / "prompt.json")])
    assert received == ["PETR4.SA"]
    assert capsys.readouterr().out.count("Informe apenas um ticker válido") == 2


def test_explicit_ticker_does_not_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("Unexpected prompt"))
    monkeypatch.setattr(app, "download_prices", lambda *a, **kw: pd.Series(
        [100., 101., 102.], index=pd.date_range("2026-01-01", periods=3)))
    app.main(["--ticker", "aapl", "--no-plot", "--steps", "2", "--n-sims", "10",
              "--report", str(tmp_path / "explicit.json")])


def test_prompt_eof_explains_cli_option(monkeypatch, capsys):
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(SystemExit) as exc:
        app.main([])
    assert exc.value.code == 2
    assert "--ticker" in capsys.readouterr().err


def test_unavailable_price_history_has_actionable_error(monkeypatch, capsys, tmp_path):
    def unavailable(*args, **kwargs):
        raise ValueError("No data")
    monkeypatch.setattr(app, "download_prices", unavailable)
    destination = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as exc:
        app.main(["--ticker", "INVALID", "--report", str(destination)])
    assert exc.value.code == 1
    assert "Confira o ticker, o período e a conexão" in capsys.readouterr().err
    assert not destination.exists()


def test_repeated_implicit_runs_preserve_existing_output(monkeypatch, tmp_path, capsys):
    """Reproduce the user's occupied default path and repeated PyCharm runs."""
    original = tmp_path / "reports" / "simulation"
    original.mkdir(parents=True)
    saved = original / "report.json"
    saved.write_bytes(b"previous report must remain unchanged\n")

    class FrozenClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(app, "datetime", FrozenClock)
    monkeypatch.setattr("builtins.input", lambda prompt: "AAPL")
    monkeypatch.setattr(app, "download_prices", lambda *a, **kw: pd.Series(
        [100., 101., 100.5, 102.], index=pd.date_range("2026-01-01", periods=4)))
    for _ in range(2):
        app.main(["--no-plot", "--steps", "2", "--n-sims", "10"])

    assert saved.read_bytes() == b"previous report must remain unchanged\n"
    runs = sorted((tmp_path / "reports").glob("simulation-*"))
    assert len(runs) == 2
    for run in runs:
        report = json.loads((run / "report.json").read_text())
        assert len(report["calibration_sensitivity"]["scenarios"]) == 10_000
        assert report["calibration_sensitivity"]["configuration"]["scenario_levels"] == 100
        for name in ("convergence.png", "paths.png", "final-distribution.png"):
            assert (run / name).read_bytes().startswith(b"\x89PNG")
    assert "10000 cenários" in capsys.readouterr().out


def test_implicit_name_avoids_existing_json(tmp_path):
    occupied = tmp_path / "reports" / "simulation.json"
    occupied.parent.mkdir()
    occupied.write_text("previous JSON")
    result = app.default_report_path()
    assert result != app.Path("reports/simulation.json")
    assert not result.exists()
    assert not result.with_suffix("").exists()
    assert occupied.read_text() == "previous JSON"


def test_cli_operational_error_is_actionable_without_traceback(tmp_path):
    occupied = tmp_path / "existing.json"
    occupied.write_text("existing data")
    with pytest.raises(SystemExit, match="Erro:.*Choose another --report"):
        app.cli(["--ticker", "AAPL", "--report", str(occupied)])
    assert occupied.read_text() == "existing data"


def test_sensitivity_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "download_prices", lambda *a, **kw: pd.Series(
        [100., 101., 102.], index=pd.date_range("2026-01-01", periods=3)))
    app.main(["--ticker", "AAPL", "--no-plot", "--steps", "2", "--n-sims", "10",
              "--no-calibration-sensitivity", "--report", str(tmp_path / "base.json")])
    report = json.loads((tmp_path / "base" / "report.json").read_text())
    assert "calibration_sensitivity" not in report


@pytest.mark.parametrize("levels", ["0", "2", "4", "10", "1000", "nan"])
def test_cli_rejects_invalid_scenario_levels(levels):
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--scenario-levels", levels])
