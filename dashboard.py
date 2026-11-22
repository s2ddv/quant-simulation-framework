"""Offline Streamlit dashboard for saved single-asset GBM reports.

Run from the repository root:

    pip install -e ".[dashboard]"
    streamlit run dashboard.py

Saved ``report.json`` files (from a folder or an upload) open offline: paths
are regenerated from the report's seed and checked against the saved terminal
mean. "Nova simulação" downloads prices from yfinance (network required) and
runs the same pipeline as ``main.py``; it writes files only when you press
save, always to a new run directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:  # `main` (save helpers) lives at the repository root
    sys.path.append(str(ROOT))

import streamlit as st  # noqa: E402

from datetime import datetime, timezone  # noqa: E402

from simulation.dashboard_data import build_dashboard_data  # noqa: E402
from simulation.dashboard_run import (DEFAULT_MAX_PATH_BYTES, path_memory_bytes,  # noqa: E402
                                      run_new_simulation)
from simulation.report import SimulationConfig  # noqa: E402
from simulation.dashboard_figures import (SENSITIVITY_LABELS, backtest_figure,  # noqa: E402
                                          convergence_figure, fan_figure,
                                          sensitivity_figure, terminal_figure)

DEFAULT_FOLDER = "reports"
MAX_LISTED_REPORTS = 200
MAX_FAN_POINTS = 400
COMPARISON_LABELS = {"mean": "Média", "variance": "Variância", "quantile_0.05": "Quantil 5%",
                     "quantile_0.5": "Mediana", "quantile_0.95": "Quantil 95%"}


def find_reports(folder: str) -> list[Path]:
    """``report.json`` files under ``folder``, newest first (at most 200).

    The model is checked when a file is opened, so other report types appear
    in the list and show a clear error instead of charts.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    found = sorted(root.rglob("report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:MAX_LISTED_REPORTS]


@st.cache_data(show_spinner="Reconstruindo trajetórias a partir da semente…", max_entries=8)
def load_dashboard(content: bytes, digest: str) -> dict:
    """Parse a report and build chart data; ``digest`` keys the cache."""
    report = json.loads(content.decode("utf-8"))
    return build_dashboard_data(report, max_points=MAX_FAN_POINTS)


def money(value: float) -> str:
    return f"{value:,.2f}"


PERIODS = ("1y", "2y", "3y", "5y", "10y", "max")


def new_simulation_form() -> None:
    """Sidebar form; a successful run is kept in ``st.session_state['new_run']``."""
    with st.sidebar.form("new_run_form"):
        ticker = st.text_input("Ticker", "AAPL", key="ticker",
                               help="Brazilian shorthand such as PETR4 becomes PETR4.SA.")
        period = st.selectbox("Histórico para calibrar", PERIODS, index=2, key="period")
        horizon = st.number_input("Horizonte (anos)", 0.01, 10.0, 1.0, 0.25, key="horizon")
        steps = st.number_input("Passos", 1, 2520, 252, key="steps")
        n_sims = st.number_input("Trajetórias", 1, 1_000_000, 1000, 1000, key="n_sims")
        seed = st.number_input("Semente", 0, 2**32 - 1, 42, key="seed")
        confidence = st.slider("Confiança do VaR", 0.80, 0.99, 0.95, 0.01, key="confidence")
        levels = st.selectbox("Grade de sensibilidade (níveis por eixo)",
                              (100, 9, 7, 5, 3, 0), key="levels",
                              format_func=lambda n: "desligada" if n == 0 else f"{n}×{n}")
        backtest = st.checkbox("Backtest histórico do VaR", value=True, key="backtest")
        calibration = st.number_input("Janela de calibração (pregões)", 3, 5000, 252,
                                      key="calibration")
        mode = st.selectbox("Calibração", ("rolling", "expanding", "fixed"), key="mode")
        submitted = st.form_submit_button("Simular", type="primary")
    mib = path_memory_bytes(int(steps), int(n_sims)) / 2**20
    st.sidebar.caption(f"Trajetórias em memória: {mib:,.1f} MiB "
                       f"(limite {DEFAULT_MAX_PATH_BYTES / 2**20:.0f} MiB). Requer internet.")
    if not submitted:
        return
    try:
        config = SimulationConfig(float(horizon), int(steps), int(n_sims), int(seed),
                                  round(float(confidence), 4))
        out_of_sample = ({"calibration_observations": int(calibration), "horizon_days": 1,
                          "alpha": config.confidence, "calibration_mode": mode,
                          "recalibration_frequency": 1} if backtest else None)
        sensitivity = ({"mu_delta": 0.02, "sigma_relative_delta": 0.2,
                        "scenario_levels": int(levels)} if levels else None)
        with st.spinner(f"Baixando preços de {ticker} e simulando…"):
            report, paths = run_new_simulation(ticker, period or "3y", config,
                                               out_of_sample=out_of_sample,
                                               sensitivity=sensitivity)
            data = build_dashboard_data(report, paths, max_points=MAX_FAN_POINTS)
    except Exception as exc:  # network, provider and validation errors alike
        st.session_state.pop("new_run", None)
        st.session_state["new_run_error"] = f"Não foi possível simular {ticker}: {exc}"
        return
    st.session_state.pop("new_run_error", None)
    st.session_state["new_run"] = {"report": report, "paths": paths, "config": config,
                                   "data": data, "saved": None}


def save_controls(run: dict) -> None:
    """Explicit save to a new run directory, plus an in-browser JSON download."""
    report = run["report"]
    ticker = report["historical_data"]["ticker"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    left, right, info = st.columns([1, 1, 3])
    left.download_button("Baixar report.json", payload, file_name=f"{ticker}-{stamp}.json",
                         mime="application/json", key="download")
    if right.button(f"Salvar em {DEFAULT_FOLDER}/", key="save", disabled=run["saved"] is not None):
        from main import check_destination, publish_run
        try:
            target = Path(DEFAULT_FOLDER) / f"{ticker}-{stamp}.json"
            check_destination(target)
            run["saved"] = str(publish_run(report, run["paths"], run["config"], target,
                                           no_plot=True))
        except Exception as exc:
            info.error(f"Não foi possível salvar: {exc}")
    if run["saved"]:
        info.success(f"Salvo em `{run['saved']}` (report.json e figuras PNG).")


def main() -> None:
    st.set_page_config(page_title="GBM Terminal", layout="wide")
    st.sidebar.header("Relatório")
    source = st.sidebar.radio("Origem", ("Pasta", "Upload", "Nova simulação"),
                              horizontal=True, key="source")
    run = None
    if source == "Nova simulação":
        new_simulation_form()
        if "new_run_error" in st.session_state:
            st.error(st.session_state["new_run_error"])
        run = st.session_state.get("new_run")
        if run is None:
            st.info("Escolha o ticker e os parâmetros na barra lateral e clique em Simular. "
                    "Os preços são baixados do yfinance; nada é gravado até você salvar.")
            return
        data, label = run["data"], "nova simulação (não salva)"
    else:
        content: bytes | None = None
        label = ""
        if source == "Pasta":
            folder = st.sidebar.text_input("Pasta de relatórios", DEFAULT_FOLDER, key="folder")
            reports = find_reports(folder)
            if not reports:
                st.info(f"Nenhum relatório GBM encontrado em `{folder}`. Gere um com "
                        "`python main.py --ticker AAPL --no-plot`, envie um arquivo ou "
                        "use Nova simulação.")
                return
            choice = st.sidebar.selectbox("Arquivo", reports, format_func=str, key="report")
            if choice is None:
                return
            content, label = choice.read_bytes(), str(choice)
        else:
            upload = st.sidebar.file_uploader("report.json", type="json", key="upload")
            if upload is None:
                st.info("Envie um `report.json` gerado por `main.py`.")
                return
            content, label = upload.getvalue(), upload.name
        try:
            data = load_dashboard(content, hashlib.sha256(content).hexdigest())
        except (ValueError, KeyError, TypeError, MemoryError) as exc:
            st.error(f"Não foi possível abrir `{label}`: {exc}")
            return
    show_paths = st.sidebar.toggle("Trajetórias de exemplo", value=True, key="paths")
    if run is not None:
        save_controls(run)

    s = data["summary"]
    level = round(100 * s["confidence"])
    st.markdown(f"### {s['ticker']} · {money(s['last_price'])} · {s['last_date']}")
    st.caption(f"GBM com incrementos lognormais exatos · μ = {s['mu']:.4f} a⁻¹ · "
               f"σ = {s['sigma']:.4f} a⁻¹ᐟ² · horizonte {s['horizon']:g} ano(s), "
               f"{s['steps']} passos · {s['n_sims']:,} trajetórias · semente {s['seed']} · "
               "cenários sob parâmetros fixos, não previsão.")

    ci = s["terminal_mean_confidence_interval"]
    columns = st.columns(5)
    ci_text = (f"IC {round(100 * ci['level'])}% da média: "
               f"{money(ci['lower'])} – {money(ci['upper'])}"
               if ci["lower"] is not None and ci["upper"] is not None
               else "IC da média indefinido: o erro-padrão exige ao menos 2 trajetórias.")
    columns[0].metric("E[S(T)] simulado", money(s["expected_price"]), help=ci_text)
    columns[1].metric(f"VaR {level}%", money(s["var"]),
                      help="Perda S0 − S(T) em unidades de preço; pode ser negativa.")
    columns[2].metric(f"CVaR {level}%", money(s["cvar"]),
                      help="Média das perdas no limiar do VaR ou acima dele (inclui empates).")
    columns[3].metric("P(S(T) < S0)", f"{100 * s['prob_loss']:.1f}%")
    columns[4].metric("VaR / S0", f"{100 * s['var'] / s['last_price']:.1f}%")

    st.subheader("Histórico e leque de cenários")
    st.caption("Faixas: percentis 5–95 e 25–75 da amostra em cada passo. Tracejado: média e "
               "quantis 5%/95% analíticos. Datas futuras são dias úteis aproximados.")
    st.plotly_chart(fan_figure(data, show_sample_paths=show_paths), width="stretch")

    left, right = st.columns([7, 5])
    with left:
        st.subheader("Distribuição de S(T)")
        st.plotly_chart(terminal_figure(data), width="stretch")
    with right:
        st.subheader("Simulado vs analítico")
        st.dataframe(
            [{"Estatística": COMPARISON_LABELS.get(row["statistic"], row["statistic"]),
              "Simulado": row["estimate"], "Analítico": row["theoretical"],
              "Erro relativo": row["relative_error"]} for row in data["comparison"]],
            hide_index=True, width="stretch",
            column_config={"Erro relativo": st.column_config.NumberColumn(format="percent")})
        st.caption("Erro Monte Carlo da amostra; não mede incerteza de calibração.")
        st.plotly_chart(convergence_figure(data), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Sensibilidade a μ e σ")
        if data["sensitivity"] is None:
            st.info("Relatório sem grade de sensibilidade.")
        else:
            options = [m for m in SENSITIVITY_LABELS if m in data["sensitivity"]["metrics"]]
            metric = st.radio("Métrica", options, format_func=SENSITIVITY_LABELS.get,
                              horizontal=True, key="metric")
            st.plotly_chart(sensitivity_figure(data, metric or "var"), width="stretch")
    with right:
        st.subheader("Backtest histórico do VaR")
        bt = data["backtest"]
        if bt is None:
            st.info("Relatório sem validação histórica (`--out-of-sample`).")
        else:
            m, w = bt["metrics"], bt["violation_rate_interval"]
            st.plotly_chart(backtest_figure(data), width="stretch")
            cols = st.columns(3)
            cols[0].metric("Violações", f"{m['violation_count']} / {m['forecast_count']}")
            cols[1].metric("Taxa observada", f"{100 * m['observed_violation_rate']:.1f}%",
                           help=f"Nominal: {100 * m['nominal_violation_rate']:.1f}%")
            if w is not None:
                cols[2].metric(f"Wilson {round(100 * w['confidence_level'])}%",
                               f"{100 * w['lower']:.1f}–{100 * w['upper']:.1f}%")

    with st.expander("Modelo, unidades e proveniência"):
        snap = data["provenance"]["snapshot"]
        st.markdown(
            "dS = μS dt + σS dW. μ é o drift histórico do preço "
            "(252·média(log-ret) + σ²/2), não o drift risco-neutro; σ = √252·desvio(log-ret). "
            "As faixas não incluem incerteza de estimação de μ e σ nem erro de modelo.\n\n"
            f"Arquivo: `{label}` · dados {snap['source']} {snap['price_field']}, "
            f"{snap['count']} fechamentos de {snap['first_date'][:10]} a {snap['last_date'][:10]}, "
            f"obtidos em {snap['retrieved_at'][:10]} · sha256 `{snap['sha256'][:16]}…`")
        st.json({"conventions": data["provenance"]["conventions"],
                 "environment": data["provenance"]["environment"]}, expanded=False)


if __name__ == "__main__":
    main()
