"""
Backtest a saved model on **unseen** 2025 data and compare to buy-and-hold.

Usage:
    # Single model, all assets
    python -m src.backtest --model models/checkpoints/v1.1/best_model.zip \
                           --vec-norm models/checkpoints/v1.1/vec_normalize.pkl

    # Specific asset
    python -m src.backtest --model ... --vec-norm ... --asset BTC

    # Compare two models side-by-side
    python -m src.backtest --compare \
        --model-a models/checkpoints/v1.1/best_model.zip \
        --vec-norm-a models/checkpoints/v1.1/vec_normalize.pkl \
        --model-b models/checkpoints/v1.1/ppo_lstm_1310000_steps.zip \
        --vec-norm-b models/checkpoints/v1.1/vec_normalize_1310000_steps.pkl
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.core.config import ASSETS
from src.data.loader import _engineer
from src.envs.trading_env import MultiAssetTradingEnv
from src.utils.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, total_return,
    CANDLES_PER_YEAR,
)


def load_2025(asset: str | None = None) -> dict[str, pd.DataFrame]:
    """Years ≥ 2025: unseen by training."""
    out: dict[str, pd.DataFrame] = {}
    items = [(asset, ASSETS[asset])] if asset else list(ASSETS.items())
    for name, path in items:
        df_raw = pd.read_csv(path)
        df_raw["T"] = pd.to_datetime(df_raw["T"])
        df_raw = df_raw.sort_values("T").reset_index(drop=True)
        df_raw = df_raw[df_raw["T"].dt.year >= 2025].reset_index(drop=True)
        if len(df_raw) == 0:
            print(f"[warn] no 2025 data for {name}")
            continue
        out[name] = _engineer(df_raw)
    return out


def buy_and_hold(df: pd.DataFrame, initial: float = 10_000.0) -> np.ndarray:
    prices = df["C"].values
    units = initial / prices[0]
    return units * prices


def _build_env(df: pd.DataFrame, asset: str, vec_norm_path: str) -> VecNormalize:
    env = DummyVecEnv([lambda: MultiAssetTradingEnv(
        {asset: df}, randomize_start=False, max_episode_steps=None
    )])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    return env


def run_backtest(model_path: str, vec_norm_path: str, df: pd.DataFrame, asset: str):
    env = _build_env(df, asset, vec_norm_path)
    model = RecurrentPPO.load(model_path)

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    equity: list[float] = []
    positions: list[int] = []
    actions_count = {0: 0, 1: 0, 2: 0}
    position_changes = 0
    last_pos = None

    max_steps = len(df) + 100
    for _ in range(max_steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True,
        )
        actions_count[int(action[0])] += 1
        obs, reward, done, info = env.step(action)
        episode_starts = done
        equity.append(float(info[0]["net_worth"]))
        pos = int(info[0]["position"])
        positions.append(pos)
        if last_pos is not None and pos != last_pos:
            position_changes += 1
        last_pos = pos
        if done[0]:
            break

    return {
        "equity": np.array(equity, dtype=np.float64),
        "positions": np.array(positions, dtype=np.int8),
        "actions": actions_count,
        "changes": position_changes,
    }


def report_one(label: str, asset: str, result: dict, bh: np.ndarray) -> None:
    eq = result["equity"]
    n = len(eq)
    pct = total_return(eq) * 100
    bh_pct = (bh[-1] / bh[0] - 1) * 100
    s = sharpe_ratio(eq)
    bh_s = sharpe_ratio(bh)
    so = sortino_ratio(eq)
    mdd = max_drawdown(eq) * 100
    bh_mdd = max_drawdown(bh) * 100
    n_year_frac = n / CANDLES_PER_YEAR
    cagr = (eq[-1] / eq[0]) ** (1 / max(n_year_frac, 1e-6)) - 1 if eq[-1] > 0 else -1
    bh_cagr = (bh[-1] / bh[0]) ** (1 / max(n_year_frac, 1e-6)) - 1
    pct_long = 100 * (result["positions"] == 1).mean()
    pct_short = 100 * (result["positions"] == -1).mean()
    pct_neutral = 100 * (result["positions"] == 0).mean()
    print(f"\n=== {label} | {asset} | n={n} ({n_year_frac*12:.2f} months) ===")
    print(f"  Total return:     {pct:+7.2f}%    (B&H {bh_pct:+7.2f}%)   Δ = {pct - bh_pct:+.2f}%")
    print(f"  CAGR (annual):    {cagr*100:+7.2f}%    (B&H {bh_cagr*100:+7.2f}%)")
    print(f"  Sharpe (annual):  {s:+7.3f}     (B&H {bh_s:+.3f})")
    print(f"  Sortino:          {so:+7.3f}")
    print(f"  Max drawdown:     {mdd:7.2f}%    (B&H {bh_mdd:.2f}%)")
    print(f"  Position mix:     LONG {pct_long:.1f}%  NEUTRAL {pct_neutral:.1f}%  SHORT {pct_short:.1f}%")
    print(f"  Action counts:    SHORT={result['actions'][0]}  NEUTRAL={result['actions'][1]}  LONG={result['actions'][2]}")
    print(f"  Position changes: {result['changes']}  (~{result['changes']/max(n,1)*100:.2f}% of steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--vec-norm")
    parser.add_argument("--asset", default=None, help="BTC or ETH (omit = both)")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--model-a")
    parser.add_argument("--vec-norm-a")
    parser.add_argument("--model-b")
    parser.add_argument("--vec-norm-b")
    parser.add_argument("--label-a", default="model-A")
    parser.add_argument("--label-b", default="model-B")
    args = parser.parse_args()

    dfs = load_2025(args.asset)
    print(f"2025 data: { {k: len(v) for k, v in dfs.items()} }")

    if args.compare:
        for label, mp, vp in [
            (args.label_a, args.model_a, args.vec_norm_a),
            (args.label_b, args.model_b, args.vec_norm_b),
        ]:
            print(f"\n##### {label}: {Path(mp).name} #####")
            for name, df in dfs.items():
                result = run_backtest(mp, vp, df, name)
                report_one(label, name, result, buy_and_hold(df))
    else:
        if not args.model or not args.vec_norm:
            parser.error("--model and --vec-norm required (or use --compare)")
        print(f"Model:    {args.model}")
        print(f"VecNorm:  {args.vec_norm}")
        for name, df in dfs.items():
            result = run_backtest(args.model, args.vec_norm, df, name)
            report_one("model", name, result, buy_and_hold(df))


if __name__ == "__main__":
    main()
