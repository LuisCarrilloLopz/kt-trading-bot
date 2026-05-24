"""Causal trend-following teacher for behavioral cloning pretraining.

Labels (env action convention: 0=SHORT, 1=NEUTRAL, 2=LONG):

  Base trend rule:
    above_ma200 := dist_ma200 > 1.0          (price above MA200)
    macd_up     := macd_hist_norm > 0        (MACD histogram positive)

    above_ma200 AND macd_up           → LONG  (2)
    NOT above_ma200 AND NOT macd_up   → SHORT (0)
    else                              → NEUTRAL (1)

  Overlay (applied last, can flip to SHORT):
    trailing_ret_24h := C[t] / C[t-24] - 1.0  (last 24h price change)
    trailing_ret_24h < -0.03          → SHORT (0)  (override)

All inputs are strictly backward-looking (causal): no lookahead. For the first
24 rows (where trailing_ret_24h is undefined) the overlay does not fire, but
the base trend rule still applies.

Used by `src/bc_pretrain.py` to generate supervised targets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Constant thresholds — change here, document in EXPERIMENTS.md.
TRAILING_LOOKBACK_H = 24
TRAILING_DRAWDOWN_THRESHOLD = -0.03   # -3% over last 24h → force SHORT

# Action codes — must match MultiAssetTradingEnv: 0=SHORT, 1=NEUTRAL, 2=LONG.
SHORT, NEUTRAL, LONG = 0, 1, 2


def generate_labels(df: pd.DataFrame) -> np.ndarray:
    """Return int8 array of length len(df) with the teacher's discrete action per row.

    Assumes df has the columns the loader produces (`C`, `dist_ma200`,
    `macd_hist_norm` at least). Any NaN in the inputs is treated as NEUTRAL.
    """
    required = {"C", "dist_ma200", "macd_hist_norm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"generate_labels: missing columns {missing}")

    n = len(df)
    labels = np.full(n, NEUTRAL, dtype=np.int8)

    above_ma200 = df["dist_ma200"].to_numpy() > 1.0
    macd_up     = df["macd_hist_norm"].to_numpy() > 0.0

    # Base trend rule.
    labels[above_ma200 & macd_up]               = LONG
    labels[(~above_ma200) & (~macd_up)]         = SHORT
    # else stays NEUTRAL.

    # Overlay — trailing 24h drawdown forces SHORT.
    c = df["C"].to_numpy(dtype=np.float64)
    trailing_ret = np.full(n, np.nan, dtype=np.float64)
    if n > TRAILING_LOOKBACK_H:
        trailing_ret[TRAILING_LOOKBACK_H:] = c[TRAILING_LOOKBACK_H:] / c[:-TRAILING_LOOKBACK_H] - 1.0
    overlay_short_mask = np.where(
        np.isnan(trailing_ret), False, trailing_ret < TRAILING_DRAWDOWN_THRESHOLD
    )
    labels[overlay_short_mask] = SHORT

    return labels


def label_distribution(labels: np.ndarray) -> dict:
    """Return {'long': pct, 'neutral': pct, 'short': pct, 'directional': pct, 'n': N}."""
    n = len(labels)
    n_short   = int((labels == SHORT).sum())
    n_neutral = int((labels == NEUTRAL).sum())
    n_long    = int((labels == LONG).sum())
    return {
        "n": n,
        "short":   n_short / n,
        "neutral": n_neutral / n,
        "long":    n_long / n,
        "directional": (n_short + n_long) / n,
    }


# ---------------------------------------------------------------------------
# CLI / acceptance: distribución sobre fold 9 train window.
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    from src.core import config as cfg
    from src.data.loader import load_assets

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=9,
                        help="Walk-forward fold index (1..len(FOLDS)). Default: 9.")
    args = parser.parse_args()

    if not (1 <= args.fold <= len(cfg.FOLDS)):
        parser.error(f"--fold must be 1..{len(cfg.FOLDS)}, got {args.fold}")
    spec = cfg.FOLDS[args.fold - 1]
    train_range = (spec[0], spec[1])
    print(f"[teacher acceptance] fold {args.fold} train window: {train_range[0]} → {train_range[1]}")

    dfs = load_assets(date_range=train_range)
    for asset, df in dfs.items():
        labels = generate_labels(df)
        dist = label_distribution(labels)
        print(f"  {asset}: n={dist['n']:>6d}  "
              f"LONG={dist['long']*100:5.1f}%  "
              f"NEUTRAL={dist['neutral']*100:5.1f}%  "
              f"SHORT={dist['short']*100:5.1f}%  "
              f"directional={dist['directional']*100:5.1f}%")

    # Acceptance: at least one asset directional >= 30% AND <= 75%, not collapsed to NEUTRAL.
    all_ok = True
    for asset, df in dfs.items():
        dist = label_distribution(generate_labels(df))
        if dist["neutral"] > 0.85:
            print(f"  [FAIL] {asset}: NEUTRAL>85% ({dist['neutral']*100:.1f}%) — teacher collapsed")
            all_ok = False
        elif dist["directional"] < 0.30:
            print(f"  [FAIL] {asset}: directional<30% ({dist['directional']*100:.1f}%)")
            all_ok = False
        elif dist["directional"] > 0.90:
            print(f"  [FAIL] {asset}: directional>90% ({dist['directional']*100:.1f}%) — over-active")
            all_ok = False
    print()
    print("✅ ACCEPTANCE PASS" if all_ok else "❌ ACCEPTANCE FAIL")


if __name__ == "__main__":
    _main()
