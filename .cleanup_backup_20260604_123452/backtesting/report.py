"""
Generador de Informes HTML de Backtesting
==========================================
Produce un informe HTML interactivo con Plotly que incluye:
  - Equity curve con anotaciones de trades
  - Drawdown chart
  - Distribución de retornos (histogram)
  - Distribución de R multiples
  - Scatter plot Entry Price vs R obtenido
  - Tabla completa de todos los trades
  - Métricas resumen en cards visuales
  - Breakdown por calidad de setup y por mes

El informe es auto-contenido (un solo .html) — se puede abrir en cualquier browser.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template

from config.settings import BACKTEST, REPORTS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)

# ── HTML Template ──────────────────────────────────────────────────────────────
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {{ symbol }} {{ strategy }}</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --accent: #4f7cff;
    --green: #00c896; --red: #ff4d4f; --yellow: #ffd700;
    --text: #e0e0e0; --muted: #6b7280;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
  .header { background: linear-gradient(135deg, var(--surface), #242837); padding: 2rem 3rem; border-bottom: 1px solid #2d3148; }
  .header h1 { font-size: 1.8rem; color: var(--accent); margin-bottom: 0.5rem; }
  .header p { color: var(--muted); font-size: 0.9rem; }
  .cards { display: flex; flex-wrap: wrap; gap: 1rem; padding: 2rem 3rem 1rem; }
  .card { background: var(--surface); border-radius: 12px; padding: 1.2rem 1.5rem;
          flex: 1; min-width: 160px; border: 1px solid #2d3148; }
  .card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
  .card .value { font-size: 1.5rem; font-weight: 700; }
  .card .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; margin-top: 0.3rem; display: inline-block; }
  .green { color: var(--green); } .red { color: var(--red); } .yellow { color: var(--yellow); }
  .badge-pass { background: rgba(0,200,150,0.15); color: var(--green); border: 1px solid var(--green); }
  .badge-fail { background: rgba(255,77,79,0.15); color: var(--red); border: 1px solid var(--red); }
  .section { padding: 1rem 3rem; }
  .section h2 { font-size: 1.1rem; color: var(--muted); margin-bottom: 1rem; font-weight: 500; }
  .chart-container { background: var(--surface); border-radius: 12px; padding: 1rem; border: 1px solid #2d3148; margin-bottom: 1.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  thead { background: #242837; }
  th { padding: 0.7rem 1rem; text-align: left; color: var(--muted); font-weight: 500; }
  td { padding: 0.5rem 1rem; border-bottom: 1px solid #1e2130; }
  tr:hover { background: rgba(79,124,255,0.05); }
  .phase1 { padding: 1rem 3rem 2rem; }
  .phase1 .result { font-size: 1.3rem; font-weight: 700; padding: 1rem 2rem; border-radius: 10px; display: inline-block; }
  .pass { background: rgba(0,200,150,0.15); color: var(--green); border: 1px solid var(--green); }
  .fail { background: rgba(255,77,79,0.15); color: var(--red); border: 1px solid var(--red); }
  .footer { padding: 1.5rem 3rem; color: var(--muted); font-size: 0.8rem; border-top: 1px solid #2d3148; }
</style>
</head>
<body>
<div class="header">
  <h1>📊 Backtest Report — {{ symbol }} | {{ strategy }} | {{ timeframe }}</h1>
  <p>Período: {{ start_date }} → {{ end_date }} &nbsp;|&nbsp; Capital inicial: ${{ initial_capital }} &nbsp;|&nbsp; Generado: {{ generated_at }}</p>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Total Trades</div>
    <div class="value">{{ metrics.total_trades }}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value {% if metrics.win_rate >= 0.45 %}green{% else %}red{% endif %}">
      {{ "%.1f"|format(metrics.win_rate * 100) }}%
    </div>
    <div class="badge {% if metrics.win_rate >= 0.45 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: ≥45%
    </div>
  </div>
  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value {% if metrics.profit_factor >= 1.5 %}green{% else %}red{% endif %}">
      {{ "%.2f"|format(metrics.profit_factor) }}
    </div>
    <div class="badge {% if metrics.profit_factor >= 1.5 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: ≥1.5
    </div>
  </div>
  <div class="card">
    <div class="label">Sharpe (Anual)</div>
    <div class="value {% if metrics.sharpe_ratio_annual >= 1.0 %}green{% else %}red{% endif %}">
      {{ "%.2f"|format(metrics.sharpe_ratio_annual) }}
    </div>
    <div class="badge {% if metrics.sharpe_ratio_annual >= 1.0 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: ≥1.0
    </div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value {% if metrics.max_drawdown_pct <= 0.15 %}green{% else %}red{% endif %}">
      {{ "%.1f"|format(metrics.max_drawdown_pct * 100) }}%
    </div>
    <div class="badge {% if metrics.max_drawdown_pct <= 0.15 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: ≤15%
    </div>
  </div>
  <div class="card">
    <div class="label">Expectancy</div>
    <div class="value {% if metrics.expectancy > 0 %}green{% else %}red{% endif %}">
      ${{ "%.2f"|format(metrics.expectancy) }}
    </div>
  </div>
  <div class="card">
    <div class="label">Return Total</div>
    <div class="value {% if metrics.total_return_pct > 0 %}green{% else %}red{% endif %}">
      {{ "%+.1f"|format(metrics.total_return_pct) }}%
    </div>
  </div>
  <div class="card">
    <div class="label">Avg R Multiple</div>
    <div class="value {% if metrics.avg_r_multiple > 0 %}green{% else %}red{% endif %}">
      {{ "%.2f"|format(metrics.avg_r_multiple) }}R
    </div>
  </div>
</div>

<div class="section">
  <h2>Equity Curve</h2>
  <div class="chart-container">{{ equity_chart }}</div>
  <h2>Drawdown</h2>
  <div class="chart-container">{{ drawdown_chart }}</div>
</div>

<div class="section">
  <h2>Distribución de Retornos</h2>
  <div class="chart-container">{{ distribution_chart }}</div>
</div>

<div class="section">
  <h2>Journal de Trades</h2>
  <div class="chart-container" style="overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Fecha</th><th>Par</th><th>Dir</th><th>Calidad</th>
          <th>Entrada</th><th>SL</th><th>TP1</th><th>Salida</th>
          <th>Tamaño</th><th>PnL ($)</th><th>PnL (%)</th><th>R</th>
          <th>Duración</th><th>Razón Salida</th>
        </tr>
      </thead>
      <tbody>
        {% for t in trades %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ t.entry_time }}</td>
          <td>{{ t.symbol }}</td>
          <td class="{% if t.direction == 'LONG' %}green{% else %}red{% endif %}">{{ t.direction }}</td>
          <td>{{ t.setup_quality }}</td>
          <td>{{ "%.4f"|format(t.entry_price) }}</td>
          <td>{{ "%.4f"|format(t.stop_loss) }}</td>
          <td>{{ "%.4f"|format(t.take_profit_1) }}</td>
          <td>{{ "%.4f"|format(t.exit_price) }}</td>
          <td>{{ "%.6f"|format(t.position_size) }}</td>
          <td class="{% if t.pnl_usd > 0 %}green{% else %}red{% endif %}">
            {{ "%+.2f"|format(t.pnl_usd) }}
          </td>
          <td class="{% if t.pnl_pct > 0 %}green{% else %}red{% endif %}">
            {{ "%+.2f"|format(t.pnl_pct) }}%
          </td>
          <td class="{% if t.r_multiple > 0 %}green{% else %}red{% endif %}">
            {{ "%.2f"|format(t.r_multiple) }}R
          </td>
          <td>{{ "%.1f"|format(t.duration_hours) }}h</td>
          <td>{{ t.exit_reason }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<div class="phase1">
  <h2 style="margin-bottom:1rem;">Veredicto — Objetivos Fase 1</h2>
  <div class="result {% if passes %}pass{% else %}fail{% endif %}">
    {% if passes %}✅ PASA FASE 1 — Listo para Paper Trading{% else %}❌ NO PASA FASE 1 — Optimizar estrategia{% endif %}
  </div>
</div>

<div class="footer">
  Sistema de Trading Algorítmico | Fase 1: Backtesting | Capital: ${{ initial_capital }} | {{ generated_at }}
</div>
</body>
</html>"""


class BacktestReporter:
    """Genera informes HTML interactivos de backtesting."""

    def generate(
        self,
        result,  # BacktestResult
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Genera el informe HTML completo.
        Retorna la ruta al archivo generado.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.symbol.replace('/', '_')}_{result.strategy}_{ts}.html"
            output_path = REPORTS_DIR / filename

        trades_df = result.portfolio.get_trades_dataframe()
        equity_df = result.portfolio.equity_curve

        # Generar gráficos Plotly
        equity_chart = self._equity_chart(equity_df, result.initial_capital)
        drawdown_chart = self._drawdown_chart(equity_df)
        distribution_chart = self._distribution_chart(trades_df)

        # Preparar datos para el template
        trades_list = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                t = row.to_dict()
                # Formatear timestamps
                for ts_col in ["entry_time", "exit_time"]:
                    if ts_col in t and hasattr(t[ts_col], "strftime"):
                        t[ts_col] = t[ts_col].strftime("%Y-%m-%d %H:%M")
                trades_list.append(type("Trade", (), t)())

        template = Template(_HTML_TEMPLATE)
        html = template.render(
            symbol=result.symbol,
            strategy=result.strategy,
            timeframe=result.timeframe,
            start_date=result.start_date.strftime("%Y-%m-%d") if result.start_date else "N/A",
            end_date=result.end_date.strftime("%Y-%m-%d") if result.end_date else "N/A",
            initial_capital=f"{result.initial_capital:.0f}",
            final_capital=f"{result.final_capital:.2f}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            metrics=result.metrics,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            distribution_chart=distribution_chart,
            trades=trades_list,
            passes=result.passes_phase1_criteria(),
        )

        output_path.write_text(html, encoding="utf-8")
        logger.info("report_generated", path=str(output_path))
        print(f"\n📊 Informe generado: {output_path}")
        return output_path

    def _equity_chart(self, equity_df: pd.DataFrame, initial_capital: float) -> str:
        """Equity curve interactiva con Plotly."""
        if equity_df is None or equity_df.empty:
            return "<p>Sin datos de equity</p>"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["capital"],
            mode="lines",
            name="Capital",
            line=dict(color="#4f7cff", width=2),
            fill="tozeroy",
            fillcolor="rgba(79,124,255,0.1)",
        ))
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="#6b7280",
            annotation_text=f"Capital inicial ${initial_capital:.0f}",
        )
        fig.update_layout(
            **self._dark_layout("Equity Curve"),
            yaxis_title="Capital ($)",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _drawdown_chart(self, equity_df: pd.DataFrame) -> str:
        """Drawdown chart."""
        if equity_df is None or equity_df.empty:
            return "<p>Sin datos</p>"

        capital = equity_df["capital"].values
        peak = np.maximum.accumulate(capital)
        drawdown = (capital - peak) / np.where(peak > 0, peak, 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["timestamp"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color="#ff4d4f", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,77,79,0.15)",
        ))
        fig.add_hline(y=-15, line_dash="dash", line_color="#ffd700",
                      annotation_text="Límite Fase 1 (-15%)")
        fig.update_layout(
            **self._dark_layout("Drawdown (%)"),
            yaxis_title="Drawdown (%)",
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _distribution_chart(self, trades_df: pd.DataFrame) -> str:
        """Histograma de distribución de PnL + scatter de R multiples."""
        if trades_df is None or trades_df.empty:
            return "<p>Sin trades</p>"

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Distribución PnL ($)", "R Multiples por Trade"]
        )

        pnl = trades_df["pnl_usd"].values
        colors = ["#00c896" if p > 0 else "#ff4d4f" for p in pnl]

        fig.add_trace(go.Histogram(
            x=pnl,
            nbinsx=30,
            name="PnL",
            marker_color=colors,
        ), row=1, col=1)

        if "r_multiple" in trades_df.columns:
            r = trades_df["r_multiple"].values
            r_colors = ["#00c896" if v > 0 else "#ff4d4f" for v in r]
            fig.add_trace(go.Bar(
                x=list(range(len(r))),
                y=r,
                name="R",
                marker_color=r_colors,
            ), row=1, col=2)

        fig.update_layout(**self._dark_layout("Distribución de Retornos"))
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _dark_layout(title: str) -> dict:
        return {
            "title": title,
            "paper_bgcolor": "#1a1d27",
            "plot_bgcolor": "#1a1d27",
            "font": {"color": "#e0e0e0", "family": "Segoe UI"},
            "xaxis": {"gridcolor": "#2d3148", "showgrid": True},
            "yaxis": {"gridcolor": "#2d3148", "showgrid": True},
            "showlegend": False,
            "height": 350,
            "margin": {"t": 40, "b": 30, "l": 50, "r": 20},
        }
