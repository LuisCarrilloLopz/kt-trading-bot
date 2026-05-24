"""Out-of-sample backtest of a fold's best_model on an arbitrary date range.

Reuses `run_backtest()` from src/backtest.py for the rollout (so VecNormalize is
wired identically: training=False, norm_reward=False, deterministic=True). Adds
a monthly breakdown on top of the aggregate metrics that `evaluate_fold.py`
already produces.

Genuinely OOS = the requested range falls **after** the fold's training window.
For fold 9 (trained 2023-01-01 → 2024-12-31), any data ≥ 2025-01-01 is OOS;
the original eval window 2025-01-01 → 2025-06-30 is technically in-sample for
checkpoint selection (best_model was picked using metrics computed there), so
we log which portion of the requested range is the *strictly new* OOS subset.

Usage:
    python -m src.evaluate_oos --fold 9 --start 2025-01-01 --end 2026-01-31
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as cfg
from src.data.loader import load_assets
from src.backtest import run_backtest, buy_and_hold
from src.utils.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, total_return,
)


def _resolve_model_paths(fold: int) -> tuple[Path, Path, str]:
    """Return (model_path, vec_norm_path, source_label). Mirrors evaluate_fold."""
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
        f"No usable model for fold {fold}. Tried:\n  {best_model}\n  {final_model}"
    )


def _changes_per_step(positions: np.ndarray) -> np.ndarray:
    """Boolean array len=n: True when positions[i] != positions[i-1] (i>=1).

    Matches run_backtest()'s counting semantics: the first step is never a
    trade (last_pos=None in run_backtest), so `.sum()` == `res['changes']`.
    """
    out = np.zeros(len(positions), dtype=bool)
    if len(positions) > 1:
        out[1:] = positions[1:] != positions[:-1]
    return out


def _monthly_breakdown(
    df: pd.DataFrame,
    positions: np.ndarray,
    equity: np.ndarray,
    changes: np.ndarray,
) -> list[dict]:
    """Per-calendar-month slice of the rollout.

    The env starts at df.iloc[VOL_LOOKBACK], so rollout step i maps to
    df.iloc[VOL_LOOKBACK + i]. We use that timestamp to assign each step to
    a YYYY-MM bucket.
    """
    n = len(positions)
    if n == 0:
        return []

    ts = df["T"].iloc[cfg.VOL_LOOKBACK : cfg.VOL_LOOKBACK + n].reset_index(drop=True)
    prices = df["C"].iloc[cfg.VOL_LOOKBACK : cfg.VOL_LOOKBACK + n].reset_index(drop=True).to_numpy()
    months = ts.dt.strftime("%Y-%m").to_numpy()

    out: list[dict] = []
    for month_str in sorted(set(months.tolist())):
        mask = months == month_str
        idx = np.where(mask)[0]
        start_i, end_i = int(idx[0]), int(idx[-1])

        # Equity curve INCLUDING the carry-in from the previous step so that
        # the first month's return reflects the price/equity move during that
        # month, not against INITIAL_BALANCE (which would be the same).
        if start_i == 0:
            eq_slice = equity[0 : end_i + 1]
        else:
            eq_slice = np.concatenate([[equity[start_i - 1]], equity[start_i : end_i + 1]])

        # B&H month: same convention — price at month start vs end.
        if start_i == 0:
            bh_start = prices[0]
        else:
            bh_start = prices[start_i - 1]
        bh_end = prices[end_i]
        bh_return_month = float(bh_end / bh_start - 1.0) if bh_start > 0 else 0.0

        n_month = int(mask.sum())
        pos_slice = positions[mask]
        pct_long = float((pos_slice == 1).sum() / n_month)
        pct_neutral = float((pos_slice == 0).sum() / n_month)
        pct_short = float((pos_slice == -1).sum() / n_month)
        n_trades = int(changes[mask].sum())

        ret = float(total_return(eq_slice))

        out.append({
            "month": month_str,
            "n_steps": n_month,
            "return": ret,
            "BH_return": bh_return_month,
            "vs_BH": float(ret - bh_return_month),
            "sharpe": float(sharpe_ratio(eq_slice)),
            "mdd": float(max_drawdown(eq_slice)),
            "n_trades": n_trades,
            "pct_long": pct_long,
            "pct_neutral": pct_neutral,
            "pct_short": pct_short,
        })

        if abs(pct_long + pct_neutral + pct_short - 1.0) > 1e-6:
            print(f"  [warn] {month_str} position mix sums to "
                  f"{pct_long + pct_neutral + pct_short}")
    return out


def _read_fold_eval_window(fold: int) -> tuple[str, str] | None:
    """Read original eval window from the fold's backtest_metrics.json (if any)."""
    p = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME / f"fold_{fold}" / "backtest_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("eval_start"), data.get("eval_end")
    except Exception:
        return None


def _genuinely_oos_range(
    requested: tuple[str, str], eval_window: tuple[str, str] | None
) -> tuple[str, str] | None:
    """Return the portion of `requested` that is strictly after `eval_window` end."""
    if eval_window is None or eval_window[1] is None:
        return requested
    eval_end = pd.Timestamp(eval_window[1])
    req_start = pd.Timestamp(requested[0])
    req_end = pd.Timestamp(requested[1])
    new_start = max(eval_end + pd.Timedelta(days=1), req_start)
    if new_start > req_end:
        return None
    return new_start.strftime("%Y-%m-%d"), req_end.strftime("%Y-%m-%d")


def evaluate_oos(
    fold: int, start: str, end: str
) -> dict:
    if not (1 <= fold <= len(cfg.FOLDS)):
        raise ValueError(f"fold must be 1..{len(cfg.FOLDS)}, got {fold}")

    model_path, vec_norm_path, source = _resolve_model_paths(fold)
    print(f"[oos fold {fold}] loading: {model_path.name}  ({source})")

    eval_window = _read_fold_eval_window(fold)
    if eval_window and eval_window[0] and eval_window[1]:
        print(f"[oos fold {fold}] original eval window: {eval_window[0]} → {eval_window[1]}")
        new_oos = _genuinely_oos_range((start, end), eval_window)
        if new_oos:
            print(f"[oos fold {fold}] strictly OOS sub-range: {new_oos[0]} → {new_oos[1]}")
        else:
            print(f"[oos fold {fold}] requested range is fully within original eval window")

    dfs = load_assets(date_range=(start, end))

    out: dict = {
        "fold": fold,
        "model_path": str(model_path),
        "model_source": source,
        "range_requested": {"start": start, "end": end},
        "original_eval_window": {
            "start": eval_window[0] if eval_window else None,
            "end": eval_window[1] if eval_window else None,
        },
    }

    actual_starts: list[pd.Timestamp] = []
    actual_ends: list[pd.Timestamp] = []

    for asset, df in dfs.items():
        if len(df) < cfg.VOL_LOOKBACK + 50:
            print(f"  [warn] {asset}: only {len(df)} rows, skipping")
            out[asset] = {"error": f"insufficient data ({len(df)} rows)"}
            continue

        ts_first = df["T"].iloc[0]
        ts_last = df["T"].iloc[-1]
        actual_starts.append(ts_first)
        actual_ends.append(ts_last)
        if ts_last < pd.Timestamp(end):
            print(f"  [warn] {asset}: data ends at {ts_last.date()}, "
                  f"requested end was {end} — using available data")

        print(f"  [oos] backtest {asset}  ({len(df)} rows, "
              f"{ts_first.date()} → {ts_last.date()})...")
        res = run_backtest(str(model_path), str(vec_norm_path), df, asset)

        eq = res["equity"]
        positions = res["positions"]
        bh = buy_and_hold(df)
        changes = _changes_per_step(positions)

        n = len(positions)
        ret = total_return(eq)
        bh_ret = total_return(bh)

        aggregate = {
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
            "pct_long": float((positions == 1).sum() / n) if n else 0.0,
            "pct_neutral": float((positions == 0).sum() / n) if n else 0.0,
            "pct_short": float((positions == -1).sum() / n) if n else 0.0,
            "actions_short": int(res["actions"][0]),
            "actions_neutral": int(res["actions"][1]),
            "actions_long": int(res["actions"][2]),
        }

        s = aggregate["pct_long"] + aggregate["pct_neutral"] + aggregate["pct_short"]
        if abs(s - 1.0) > 1e-6:
            print(f"    [warn] {asset} aggregate position mix sums to {s}")

        # Cross-check the changes-per-step counting matches run_backtest's tally.
        if int(changes.sum()) != int(res["changes"]):
            print(f"    [warn] {asset} changes mismatch: "
                  f"per-step={int(changes.sum())} vs run_backtest={int(res['changes'])}")

        monthly = _monthly_breakdown(df, positions, eq, changes)

        out[asset] = {"aggregate": aggregate, "monthly": monthly}

    if actual_starts and actual_ends:
        out["range_actual"] = {
            "start": min(actual_starts).strftime("%Y-%m-%d"),
            "end":   max(actual_ends).strftime("%Y-%m-%d"),
        }
    else:
        out["range_actual"] = {"start": start, "end": end}

    return out


def _print_report(metrics: dict) -> None:
    print("\n" + "=" * 78)
    print(f"OOS REPORT — fold {metrics['fold']}  ({metrics['model_source']})")
    print("=" * 78)
    print(f"Original eval window: {metrics['original_eval_window']['start']} → "
          f"{metrics['original_eval_window']['end']}")
    print(f"Requested range:      {metrics['range_requested']['start']} → "
          f"{metrics['range_requested']['end']}")
    print(f"Actual range used:    {metrics['range_actual']['start']} → "
          f"{metrics['range_actual']['end']}")

    for asset in ("BTC", "ETH"):
        if asset not in metrics or "error" in metrics[asset]:
            continue
        agg = metrics[asset]["aggregate"]
        print(f"\n--- {asset} aggregate ---")
        print(f"  return:   {agg['return']*100:+7.2f}%  (B&H {agg['BH_return']*100:+7.2f}%)  "
              f"Δ = {agg['vs_BH']*100:+.2f}%")
        print(f"  sharpe:   {agg['sharpe']:+7.3f}     (B&H {agg['BH_sharpe']:+.3f})")
        print(f"  sortino:  {agg['sortino']:+7.3f}")
        print(f"  mdd:      {agg['mdd']*100:7.2f}%  (B&H {agg['BH_mdd']*100:.2f}%)")
        print(f"  n_trades: {agg['n_trades']}  ({agg['n_trades']/max(agg['n_steps'],1)*100:.2f}% of steps)")
        print(f"  L/N/S:    {agg['pct_long']*100:.1f}% / "
              f"{agg['pct_neutral']*100:.1f}% / {agg['pct_short']*100:.1f}%")

        print(f"\n  Monthly breakdown ({asset}):")
        print(f"  {'month':<8} {'n':>5} {'ret%':>8} {'B&H%':>8} {'Δ%':>7} "
              f"{'sharpe':>7} {'mdd%':>6} {'trd':>4} {'L%':>5} {'N%':>5} {'S%':>5}")
        for m in metrics[asset]["monthly"]:
            print(f"  {m['month']:<8} {m['n_steps']:>5} "
                  f"{m['return']*100:>+8.2f} {m['BH_return']*100:>+8.2f} "
                  f"{m['vs_BH']*100:>+7.2f} {m['sharpe']:>+7.2f} {m['mdd']*100:>6.2f} "
                  f"{m['n_trades']:>4} "
                  f"{m['pct_long']*100:>5.1f} {m['pct_neutral']*100:>5.1f} {m['pct_short']*100:>5.1f}")

        # Sanity: compounding the monthly returns should match aggregate (within float tolerance)
        compounded = np.prod([1.0 + m["return"] for m in metrics[asset]["monthly"]]) - 1.0
        delta = compounded - agg["return"]
        print(f"  [check] sum(months) compounded = {compounded*100:+.4f}%, "
              f"aggregate = {agg['return']*100:+.4f}%, diff = {delta*100:+.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=9,
                        help=f"Fold index (1..{len(cfg.FOLDS)}). Default: 9")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--output-name", type=str, default="oos_extended_metrics.json",
                        help="Output JSON filename inside the fold's final/ dir")
    args = parser.parse_args()

    metrics = evaluate_oos(args.fold, args.start, args.end)

    out_dir = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[oos fold {args.fold}] JSON saved → {out_path}")

    _print_report(metrics)


if __name__ == "__main__":
    main()
