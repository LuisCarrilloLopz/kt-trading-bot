"""Feature engineering and asset loading."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.core.config import ASSETS, TRAIN_YEAR_MAX, TRAIN_SPLIT, VOL_LOOKBACK

FEATURE_COLS = [
    "C", "dist_ma200", "bb_pos", "rsi_norm", "vol_rel", "log_ret", "market_vol",
    # V1.3 nuevas features
    "macd_norm", "macd_hist_norm", "atr_norm",
    "hour_sin", "hour_cos",
    "log_ret_lag1", "log_ret_lag5", "log_ret_lag10",
]


def _engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["T"] = pd.to_datetime(df_raw["T"])
    df["C"] = df_raw["C"].astype(float)
    df["dist_ma200"] = df_raw["C"] / df_raw["MA200"]
    df["bb_pos"] = (df_raw["C"] - df_raw["BOLLINGER_lower"]) / (
        df_raw["BOLLINGER_upper"] - df_raw["BOLLINGER_lower"]
    )
    df["rsi_norm"] = df_raw["RSI"] / 100.0
    df["vol_rel"] = df_raw["V"] / (df_raw["V"].rolling(window=20).mean() + 1e-5)
    df["log_ret"] = np.log(df_raw["C"] / df_raw["C"].shift(1)).fillna(0.0).clip(-0.15, 0.15)
    # Pre-computed so the env step is O(1) instead of O(VOL_LOOKBACK)
    df["market_vol"] = df["log_ret"].rolling(window=VOL_LOOKBACK).std().fillna(0.0)

    # --- V1.3: features de momentum / volatilidad / temporales / lags ---
    # MACD normalizado por precio, escalado x100 (típicamente 0.1 - 1.2)
    df["macd_norm"] = (df_raw["MACD_outmacd"] / df_raw["C"]).clip(-0.05, 0.05) * 100.0
    df["macd_hist_norm"] = (df_raw["MACD_outmacdhist"] / df_raw["C"]).clip(-0.05, 0.05) * 100.0

    # ATR(14) calculado desde H/L/C, normalizado por close, escalado x100 (~0.5-3)
    high, low, close = df_raw["H"], df_raw["L"], df_raw["C"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df["atr_norm"] = (atr / close).clip(0.0, 0.1).fillna(0.0) * 100.0

    # Hora del día → sin/cos para que la red vea la ciclicidad correctamente
    hours = df_raw["T"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Lagged log_returns para que LSTM (y critic) tengan referencia explícita
    df["log_ret_lag1"] = (df["log_ret"].shift(1).fillna(0.0) * 100.0).clip(-15.0, 15.0)
    df["log_ret_lag5"] = (df["log_ret"].shift(5).fillna(0.0) * 100.0).clip(-15.0, 15.0)
    df["log_ret_lag10"] = (df["log_ret"].shift(10).fillna(0.0) * 100.0).clip(-15.0, 15.0)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def load_assets(split: str = "train") -> dict[str, pd.DataFrame]:
    """Return {asset_name: dataframe} ready for the env, sliced by split."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    out: dict[str, pd.DataFrame] = {}
    for name, path in ASSETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {name}: {path}")
        df_raw = pd.read_csv(path)
        df_raw["T"] = pd.to_datetime(df_raw["T"])
        df_raw = df_raw.sort_values("T").reset_index(drop=True)
        df_raw = df_raw[df_raw["T"].dt.year < TRAIN_YEAR_MAX].reset_index(drop=True)
        df = _engineer(df_raw)
        split_idx = int(len(df) * TRAIN_SPLIT)
        df = (df.iloc[:split_idx] if split == "train" else df.iloc[split_idx:]).reset_index(drop=True)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {name}: {missing}")
        out[name] = df
    return out
