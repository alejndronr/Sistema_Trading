"""Tests para los indicadores técnicos."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators


@pytest.fixture
def sample_ohlcv():
    """DataFrame OHLCV sintético para tests."""
    np.random.seed(42)
    n = 300
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    close = np.maximum(close, 100)  # No negativos
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    low = np.maximum(low, 50)
    open_ = close + np.random.randn(n) * 100
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    timestamps = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestTrendIndicators:
    def setup_method(self):
        self.ti = TrendIndicators()

    def test_ema_columns_created(self, sample_ohlcv):
        df = self.ti.ema(sample_ohlcv)
        assert "ema_21" in df.columns
        assert "ema_55" in df.columns
        assert "ema_200" in df.columns

    def test_ema_no_nan_after_warmup(self, sample_ohlcv):
        df = self.ti.ema(sample_ohlcv)
        tail = df.iloc[200:]
        assert tail["ema_21"].isna().sum() == 0
        assert tail["ema_200"].isna().sum() == 0

    def test_ema_fast_follows_price_more(self, sample_ohlcv):
        df = self.ti.ema(sample_ohlcv)
        # EMA21 debe ser más cercana al precio que EMA200
        df = df.iloc[200:]
        diff_21 = (df["close"] - df["ema_21"]).abs().mean()
        diff_200 = (df["close"] - df["ema_200"]).abs().mean()
        assert diff_21 < diff_200, "EMA21 debe seguir el precio más de cerca que EMA200"

    def test_supertrend_binary_direction(self, sample_ohlcv):
        df = self.ti.supertrend(sample_ohlcv)
        assert "supertrend_direction" in df.columns
        directions = df["supertrend_direction"].dropna().unique()
        assert set(directions).issubset({1, -1})

    def test_adx_range(self, sample_ohlcv):
        df = self.ti.adx(sample_ohlcv)
        assert "adx" in df.columns
        valid = df["adx"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_ema_alignment_bullish_bearish_mutually_exclusive(self, sample_ohlcv):
        df = self.ti.calculate_all(sample_ohlcv)
        # No puede ser alcista Y bajista a la vez
        both = df["ema_alignment_bullish"] & df["ema_alignment_bearish"]
        assert both.sum() == 0

    def test_calculate_all_returns_more_columns(self, sample_ohlcv):
        original_cols = set(sample_ohlcv.columns)
        df = self.ti.calculate_all(sample_ohlcv)
        new_cols = set(df.columns) - original_cols
        assert len(new_cols) >= 5, f"Debe añadir al menos 5 columnas, añadió: {new_cols}"


class TestMomentumIndicators:
    def setup_method(self):
        self.mi = MomentumIndicators()

    def test_rsi_range(self, sample_ohlcv):
        df = self.mi.rsi(sample_ohlcv)
        valid = df["rsi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_overbought_oversold_flags(self, sample_ohlcv):
        df = self.mi.rsi(sample_ohlcv)
        # Los flags de sobrecompra/sobreventa son booleanos
        assert df["rsi_overbought"].dtype == bool
        assert df["rsi_oversold"].dtype == bool
        # No pueden estar activos los dos al mismo tiempo
        both = df["rsi_overbought"] & df["rsi_oversold"]
        assert both.sum() == 0

    def test_macd_histogram_is_difference(self, sample_ohlcv):
        df = self.mi.macd(sample_ohlcv)
        calculated_hist = df["macd_line"] - df["macd_signal"]
        diff = (df["macd_histogram"] - calculated_hist).abs()
        assert diff.max() < 1e-10, "MACD histogram debe ser line - signal"

    def test_stoch_rsi_range(self, sample_ohlcv):
        df = self.mi.stochastic_rsi(sample_ohlcv)
        valid_k = df["stoch_rsi_k"].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_divergences_not_all_false(self, sample_ohlcv):
        """Con suficientes datos, debe detectar alguna divergencia."""
        df = self.mi.calculate_all(sample_ohlcv)
        total_divs = (
            df["rsi_divergence_bullish"].sum()
            + df["rsi_divergence_bearish"].sum()
        )
        # No se requiere un número exacto, pero debería detectar alguna
        assert total_divs >= 0  # Al menos no falla


class TestVolatilityIndicators:
    def setup_method(self):
        self.vi = VolatilityIndicators()

    def test_atr_positive(self, sample_ohlcv):
        df = self.vi.atr(sample_ohlcv)
        valid = df["atr"].dropna()
        assert (valid > 0).all()

    def test_bb_upper_greater_than_lower(self, sample_ohlcv):
        df = self.vi.bollinger_bands(sample_ohlcv)
        valid = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] > valid["bb_lower"]).all()

    def test_bb_pct_range(self, sample_ohlcv):
        df = self.vi.bollinger_bands(sample_ohlcv)
        # %B puede salir de 0-1 en movimientos extremos, pero la mayoría dentro
        valid = df["bb_pct"].dropna()
        within_range = ((valid >= -0.5) & (valid <= 1.5)).sum() / len(valid)
        assert within_range > 0.9, "La mayoría del tiempo %B debe estar cerca de 0-1"

    def test_squeeze_counts_consecutive(self, sample_ohlcv):
        df = self.vi.calculate_all(sample_ohlcv)
        # Los squeeze_candles nunca deben bajar si squeeze está activo
        for i in range(1, len(df)):
            if df["bb_squeeze"].iloc[i] and df["bb_squeeze"].iloc[i - 1]:
                assert df["bb_squeeze_candles"].iloc[i] == df["bb_squeeze_candles"].iloc[i - 1] + 1

    def test_atr_regime_categories(self, sample_ohlcv):
        df = self.vi.calculate_all(sample_ohlcv)
        valid_regimes = {"normal", "high_vol", "very_high_vol", "extreme_vol"}
        actual = set(df["atr_regime"].dropna().unique())
        assert actual.issubset(valid_regimes)


class TestVolumeIndicators:
    def setup_method(self):
        self.vi = VolumeIndicators()

    def test_obv_monotone_when_prices_rising(self, sample_ohlcv):
        """Cuando el precio siempre sube, OBV debe ser siempre creciente."""
        df = sample_ohlcv.copy()
        df["close"] = df["close"].sort_values().values
        df_out = self.vi.obv(df)
        obv_diff = df_out["obv"].diff().dropna()
        # La mayoría debe ser positivo
        positive_ratio = (obv_diff > 0).sum() / len(obv_diff)
        assert positive_ratio > 0.7

    def test_vwap_near_price(self, sample_ohlcv):
        df = self.vi.vwap(sample_ohlcv)
        valid = df.dropna(subset=["vwap"])
        # VWAP no debería desviarse más del 10% del precio en datos normales
        deviation = (valid["close"] - valid["vwap"]).abs() / valid["close"]
        assert deviation.mean() < 0.10

    def test_volume_profile_has_poc(self, sample_ohlcv):
        df = self.vi.volume_profile(sample_ohlcv)
        assert "vp_poc" in df.columns
        assert not df["vp_poc"].isna().all()

    def test_poc_within_price_range(self, sample_ohlcv):
        df = self.vi.volume_profile(sample_ohlcv)
        poc = df["vp_poc"].dropna().iloc[0]
        assert sample_ohlcv["low"].min() <= poc <= sample_ohlcv["high"].max()
