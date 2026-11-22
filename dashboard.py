"""Offline Streamlit dashboard for saved single-asset GBM reports.

Run from the repository root:

    pip install -e ".[dashboard]"
    streamlit run dashboard.py

The app only reads existing ``report.json`` files (from a folder or an upload);
it never downloads prices or writes files. Paths are regenerated from the
report's seed and checked against the saved terminal mean.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from simulation.dashboard_data import build_dashboard_data  # noqa: E402
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


def main() -> None:
    st.set_page_config(page_title="GBM Terminal", layout="wide")
    st.sidebar.header("Relatório")
    source = st.sidebar.radio("Origem", ("Pasta", "Upload"), horizontal=True, key="source")
    content: bytes | None = None
    label = ""
    if source == "Pasta":
        folder = st.sidebar.text_input("Pasta de relatórios", DEFAULT_FOLDER, key="folder")
        reports = find_reports(folder)
        if not reports:
            st.info(f"Nenhum relatório GBM encontrado em `{folder}`. Gere um com "
                    "`python main.py --ticker AAPL --no-plot` ou envie um arquivo.")
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
    show_paths = st.sidebar.toggle("Trajetórias de exemplo", value=True, key="paths")

    try:
        data = load_dashboard(content, hashlib.sha256(content).hexdigest())
    except (ValueError, KeyError, TypeError, MemoryError) as exc:
        st.error(f"Não foi possível abrir `{label}`: {exc}")
        return

    s = data["summary"]
    level = round(100 * s["confidence"])
    st.markdown(f"### {s['ticker']} · {money(s['last_price'])} · {s['last_date']}")
    st.caption(f"GBM com incrementos lognormais exatos · μ = {s['mu']:.4f} a⁻¹ · "
               f"σ = {s['sigma']:.4f} a⁻¹ᐟ² · horizonte {s['horizon']:g} ano(s), "
               f"{s['steps']} passos · {s['n_sims']:,} trajetórias · semente {s['seed']} · "
               "cenários sob parâmetros fixos, não previsão.")

    ci = s["terminal_mean_confidence_interval"]
    columns = st.columns(5)
    columns[0].metric("E[S(T)] simulado", money(s["expected_price"]),
                      help=f"IC {round(100 * ci['level'])}% da média: "
                           f"{money(ci['lower'])} – {money(ci['upper'])}")
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
