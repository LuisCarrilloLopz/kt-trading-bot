"""Deterministic post-training evaluation of a single walk-forward fold.

Loads the best_model (preferred) or the final ppo_lstm (fallback) of fold N and
runs ONE deterministic pass over each asset's eval window. Saves metrics to
`models/final/walk_forward_v1/fold_N/backtest_metrics.json`.

Reuses run_backtest() from src/backtest.py for the rollout (already correctly
wires VecNormalize.training=False to prevent stat contamination).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.core import config as cfg
from src.data.loader import load_assets
from src.backtest import run_backtest, buy_and_hold
from src.utils.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, total_return,
)


def _resolve_model_paths(fold: int) -> tuple[Path, Path, str]:
    """Return (model_path, vec_norm_path, source_label).

    Prefer the best_model from the checkpoint dir. Fall back to the final model
    if best_model doesn't exist (happens when training was too short to reach
    smoothing warmup of 5 evals — useful for acceptance tests).
    """
    ckpt_dir = cfg.CHECKPOINT_DIR / cfg.WALK_FORWARD_RUN_NAME / f"fold_{fold}"
    final_dir = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME / f"fold_{fold}"

    best_model = ckpt_dir / "best_model.zip"
    best_norm = ckpt_dir / "vec_normalize.pkl"
    final_model = final_dir / "ppo_lstm.zip"
    final_norm = final_dir / "vec_normalize.pkl"

    if best_model.exists() and best_norm.exists():
        return best_model, best_norm, "best_model"
    if final_model.exists() and final_norm.exists():
        return final_model, final_norm, "final_model (best_model unavailable)"
    raise FileNotFoundError(
        f"No usable model for fold {fold}. Tried:\n"
        f"  {best_model}\n  {final_model}"
    )


def evaluate_fold(fold: int) -> dict:
    if not (1 <= fold <= len(cfg.FOLDS)):
        raise ValueError(f"fold must be 1..{len(cfg.FOLDS)}, got {fold}")
    spec = cfg.FOLDS[fold - 1]
    eval_range = (spec[2], spec[3])

    model_path, vec_norm_path, source = _resolve_model_paths(fold)
    print(f"[fold {fold}] loading: {model_path.name}  ({source})")
    print(f"[fold {fold}] eval window: {eval_range[0]} → {eval_range[1]}")

    eval_dfs = load_assets(date_range=eval_range)

    out: dict = {
        "fold": fold,
        "train_start": spec[0], "train_end": spec[1],
        "eval_start": eval_range[0], "eval_end": eval_range[1],
        "model_source": source,
    }

    for asset, df in eval_dfs.items():
        if len(df) < 50:
            print(f"  [warn] {asset}: only {len(df)} rows, skipping")
            out[asset] = {"error": f"insufficient data ({len(df)} rows)"}
            continue
        print(f"  [fold {fold}] backtest {asset}  ({len(df)} rows)...")
        res = run_backtest(str(model_path), str(vec_norm_path), df, asset)
        eq = res["equity"]
        positions = res["positions"]
        bh = buy_and_hold(df)
        n = len(positions)
        ret = total_return(eq)
        bh_ret = total_return(bh)
        pct_long = float((positions == 1).sum() / n) if n else 0.0
        pct_neutral = float((positions == 0).sum() / n) if n else 0.0
        pct_short = float((positions == -1).sum() / n) if n else 0.0

        out[asset] = {
            "n_steps": int(n),
            "return": float(ret),
            "BH_return": float(bh_ret),
            "vs_BH": float(ret - bh_ret),
            "sharpe": float(sharpe_ratio(eq)),
            "sortino": float(sortino_ratio(eq)),
            "mdd": float(max_drawdown(eq)),
            "BH_sharpe": float(sharpe_ratio(bh)),
            "BH_mdd": float(max_drawdown(bh)),
            "n_trades": int(res["changes"]),
            "pct_long": pct_long,
            "pct_neutral": pct_neutral,
            "pct_short": pct_short,
            "actions_short": int(res["actions"][0]),
            "actions_neutral": int(res["actions"][1]),
            "actions_long": int(res["actions"][2]),
        }
        # Acceptance sanity: position mix should sum to 1.0 ± 1e-6
        s = pct_long + pct_neutral + pct_short
        if abs(s - 1.0) > 1e-6:
            print(f"    [warn] {asset} position mix sums to {s}, expected 1.0")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True,
                        help=f"Fold index (1..{len(cfg.FOLDS)})")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    metrics = evaluate_fold(args.fold)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME / f"fold_{args.fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "backtest_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[fold {args.fold}] backtest_metrics.json saved → {out_path}")

    # Console summary
    for k, v in metrics.items():
        if isinstance(v, dict) and "return" in v:
            print(f"  {k}: return={v['return']*100:+.2f}% (BH {v['BH_return']*100:+.2f}%) "
                  f"sharpe={v['sharpe']:+.3f} mdd={v['mdd']*100:.1f}% "
                  f"L/N/S={v['pct_long']*100:.0f}/{v['pct_neutral']*100:.0f}/{v['pct_short']*100:.0f}")


if __name__ == "__main__":
    main()
