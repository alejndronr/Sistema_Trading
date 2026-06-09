"""Tests para el motor de backtesting, métricas y position sizer."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.metrics import BacktestMetrics
from backtesting.engine import Portfolio, Position
from risk.position_sizer import PositionSizer, DrawdownTracker
from config.settings import SetupQuality, SignalDirection


@pytest.fixture
def sample_trades_df():
    """DataFrame de trades de ejemplo con mezcla de wins y losses."""
    n = 50
    np.random.seed(123)
    pnl = np.random.choice([-8, -10, 15, 20, 25, -5], size=n,
                            p=[0.15, 0.15, 0.2, 0.2, 0.15, 0.15])

    entry_times = pd.date_range("2023-01-01", periods=n, freq="2D", tz="UTC")
    exit_times = entry_times + pd.Timedelta(hours=8)

    return pd.DataFrame({
        "trade_id": [f"t{i:03d}" for i in range(n)],
        "symbol": ["BTC/USDT"] * n,
        "strategy": ["trend_following"] * n,
        "direction": np.where(pnl > 0, "LONG", "LONG"),
        "setup_quality": np.random.choice(["A+", "A", "B", "C"], size=n,
                                           p=[0.1, 0.5, 0.3, 0.1]),
        "entry_price": np.random.uniform(30000, 60000, n),
        "exit_price": np.random.uniform(30000, 60000, n),
        "stop_loss": np.random.uniform(28000, 55000, n),
        "take_profit_1": np.random.uniform(35000, 65000, n),
        "take_profit_2": np.random.uniform(38000, 70000, n),
        "position_size": np.random.uniform(0.001, 0.01, n),
        "risk_amount": [10.0] * n,
        "pnl_usd": pnl.astype(float),
        "pnl_pct": pnl / 300 * 100,
        "r_multiple": pnl / 10.0,
        "entry_time": entry_times,
        "exit_time": exit_times,
        "duration_hours": [8.0] * n,
        "exit_reason": np.random.choice(["stop_loss", "take_profit_1", "take_profit_2"], n),
        "entry_reason": ["EMA alignment"] * n,
        "market_regime": ["bullish_trend"] * n,
    })


@pytest.fixture
def sample_equity_curve():
    """Equity curve de ejemplo."""
    n = 200
    capital = [300.0]
    for i in range(1, n):
        change = np.random.randn() * 2
        capital.append(max(200, capital[-1] + change))

    timestamps = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "capital": capital})


class TestBacktestMetrics:
    def test_win_rate_calculation(self, sample_trades_df, sample_equity_curve):
        m = BacktestMetrics.from_trades(sample_trades_df, sample_equity_curve, 300.0)
        manual_wr = (sample_trades_df["pnl_usd"] > 0).mean()
        assert abs(m.win_rate - manual_wr) < 0.001

    def test_profit_factor_positive_when_profitable(self, sample_equity_curve):
        trades = pd.DataFrame({
            "pnl_usd": [10.0, 15.0, 20.0, -5.0, -8.0],
            "r_multiple": [1.0, 1.5, 2.0, -0.5, -0.8],
            "setup_quality": ["A"] * 5,
            "exit_reason": ["take_profit_1"] * 5,
            "duration_hours": [8.0] * 5,
        })
        m = BacktestMetrics.from_trades(trades, sample_equity_curve, 300.0)
        assert m.profit_factor > 1.0

    def test_expectancy_sign_matches_total_pnl(self, sample_trades_df, sample_equity_curve):
        m = BacktestMetrics.from_trades(sample_trades_df, sample_equity_curve, 300.0)
        assert (m.expectancy > 0) == (m.total_pnl_usd > 0)

    def test_max_drawdown_between_0_and_1(self, sample_trades_df, sample_equity_curve):
        m = BacktestMetrics.from_trades(sample_trades_df, sample_equity_curve, 300.0)
        assert 0 <= m.max_drawdown_pct <= 1.0

    def test_streak_calculation(self):
        pnl = np.array([10, 10, 10, -5, -5, 10, -5, -5, -5, -5])
        max_wins, max_losses = BacktestMetrics._calculate_streaks(pnl)
        assert max_wins == 3
        assert max_losses == 4

    def test_empty_trades_returns_zero_metrics(self):
        m = BacktestMetrics.from_trades(pd.DataFrame(), pd.DataFrame(), 300.0)
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0

    def test_phase1_criteria_all_pass(self, sample_equity_curve):
        trades = pd.DataFrame({
            "pnl_usd": [12.0] * 60 + [-5.0] * 40,
            "r_multiple": [1.2] * 60 + [-0.5] * 40,
            "setup_quality": ["A"] * 100,
            "exit_reason": ["take_profit_1"] * 100,
            "duration_hours": [8.0] * 100,
        })
        m = BacktestMetrics.from_trades(trades, sample_equity_curve, 300.0)
        # Con 60% win rate y avg win >> avg loss, debe pasar
        assert m.win_rate == 0.60
        assert m.profit_factor > 1.5


class TestPortfolio:
    def setup_method(self):
        self.portfolio = Portfolio(initial_capital=300.0)
        self.now = datetime.now(timezone.utc)

    def test_open_position_reduces_capital_by_commission(self):
        initial = self.portfolio.capital
        self.portfolio.open_position(
            symbol="BTC/USDT", strategy="trend_following",
            direction=SignalDirection.LONG,
            entry_price=50000, stop_loss=49000, take_profit_1=52000,
            position_size=0.001, risk_amount=10.0, entry_time=self.now,
        )
        # Capital debe bajar solo por la comisión (0.1%)
        commission = 50000 * 0.001 * 0.001
        assert abs(self.portfolio.capital - (initial - commission)) < 0.01

    def test_close_position_profitable(self):
        trade_id = self.portfolio.open_position(
            symbol="BTC/USDT", strategy="trend_following",
            direction=SignalDirection.LONG,
            entry_price=50000, stop_loss=49000, take_profit_1=52000,
            position_size=0.001, risk_amount=10.0, entry_time=self.now,
        )
        assert trade_id is not None
        trade = self.portfolio.close_position(
            trade_id=trade_id, exit_price=52000,
            exit_time=self.now, exit_reason="take_profit_1",
        )
        assert trade is not None
        assert trade.pnl_usd > 0  # Ganancia

    def test_close_position_stop_loss(self):
        trade_id = self.portfolio.open_position(
            symbol="BTC/USDT", strategy="trend_following",
            direction=SignalDirection.LONG,
            entry_price=50000, stop_loss=49000, take_profit_1=52000,
            position_size=0.001, risk_amount=10.0, entry_time=self.now,
        )
        trade = self.portfolio.close_position(
            trade_id=trade_id, exit_price=49000,
            exit_time=self.now, exit_reason="stop_loss",
        )
        assert trade.pnl_usd < 0  # Pérdida

    def test_max_capital_loss_near_risk_amount(self):
        """La pérdida máxima al SL debe ser aprox. el risk_amount."""
        # Usar capital suficiente para que la posición pueda abrirse
        portfolio = Portfolio(initial_capital=10_000.0)
        risk = 10.0
        entry = 50000
        sl = 49000
        size = risk / (entry - sl)  # = 0.01

        trade_id = portfolio.open_position(
            symbol="BTC/USDT", strategy="trend_following",
            direction=SignalDirection.LONG,
            entry_price=entry, stop_loss=sl, take_profit_1=52000,
            position_size=size, risk_amount=risk, entry_time=self.now,
        )
        assert trade_id is not None, "La posición debe abrirse con capital suficiente"
        trade = portfolio.close_position(
            trade_id=trade_id, exit_price=sl,
            exit_time=self.now, exit_reason="stop_loss",
        )
        assert trade is not None
        # La pérdida neta debe estar cerca del risk_amount (± comisiones y slippage)
        assert abs(trade.pnl_usd) < risk * 1.5

    def test_stop_loss_only_moves_in_favor(self):
        trade_id = self.portfolio.open_position(
            symbol="BTC/USDT", strategy="trend_following",
            direction=SignalDirection.LONG,
            entry_price=50000, stop_loss=49000, take_profit_1=52000,
            position_size=0.001, risk_amount=10.0, entry_time=self.now,
        )
        pos = self.portfolio.open_positions[trade_id]
        original_sl = pos.current_sl

        # Intentar bajar el SL (no permitido para long)
        self.portfolio.update_stop_loss(trade_id, 48000)
        assert pos.current_sl == original_sl  # No debe cambiar

        # Subir el SL sí está permitido
        self.portfolio.update_stop_loss(trade_id, 49500)
        assert pos.current_sl == 49500


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(initial_capital=300.0)

    def test_risk_amount_fixed_below_1000(self):
        """Capital < $1000 → riesgo fijo de $10"""
        risk = self.sizer.get_risk_amount()
        assert risk == 10.0

    def test_position_size_formula(self):
        """Verifica la fórmula: size = risk / (entry - sl)"""
        # Con capital de $1000 (tier 1%), no hay cap de capital que interfiera
        sizer = PositionSizer(initial_capital=1_000.0)
        entry, sl = 50_000.0, 49_000.0  # diff = $1000
        result = sizer.calculate_position_size(
            entry_price=entry,
            stop_loss_price=sl,
            quality=SetupQuality.A,
        )
        # Tier $1000-$5000 → 1% = $10
        expected_risk = 1_000.0 * 0.01  # $10
        expected_size = expected_risk / (entry - sl)  # 10/1000 = 0.01
        assert abs(result["risk_amount"] - expected_risk) < 0.01
        assert abs(result["position_size"] - expected_size) < 0.0001

    def test_position_size_a_plus_uses_double_risk(self):
        sizer_rich = PositionSizer(initial_capital=5000.0)  # 1% = $50
        a = sizer_rich.calculate_position_size(50000, 49000, quality=SetupQuality.A)
        a_plus = sizer_rich.calculate_position_size(50000, 49000, quality=SetupQuality.A_PLUS)
        assert a_plus["risk_amount"] > a["risk_amount"]

    def test_max_open_positions(self):
        self.sizer.register_position_opened()
        self.sizer.register_position_opened()
        self.sizer.register_position_opened()
        ok, reason = self.sizer.can_open_position()
        assert not ok
        assert "3" in reason

    def test_drawdown_tracker_daily(self):
        tracker = DrawdownTracker(initial_capital=300.0)
        tracker.record_capital(300.0)
        tracker.record_capital(285.0)  # -5% → dispara 3%
        triggered = tracker.check_circuit_breakers()
        assert triggered["daily"]

    def test_drawdown_tracker_can_trade_false(self):
        tracker = DrawdownTracker(initial_capital=300.0)
        tracker.record_capital(300.0)
        tracker.record_capital(280.0)  # -6.7% diario
        # Simular mismo día
        sizer = PositionSizer(initial_capital=300.0)
        sizer.drawdown_tracker = tracker
        can, reason = sizer.drawdown_tracker.can_trade()
        assert not can
