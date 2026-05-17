"""Aggregate walk-forward reporter.

Reads `eval_metrics.json` (from training) and `backtest_metrics.json` (from
evaluate_fold) for each fold and produces:
  - `models/final/walk_forward_v1/aggregate_report.md` — markdown with per-fold
    tables, aggregate stats, and regime-conditional analysis (the key
    diagnostic).
  - `models/final/walk_forward_v1/aggregate_report.csv` — flat tabular form
    (one row per (fold, asset)).

Missing folds appear as "N/A" instead of crashing.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

from src.core import config as cfg

ASSETS = ("BTC", "ETH")
BULL_THRESHOLD = 0.20   # BH_return BTC > +20% → Bull
BEAR_THRESHOLD = -0.20  # BH_return BTC < -20% → Bear


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _regime_label(btc_bh_return: float | None) -> str:
    if btc_bh_return is None:
        return "?"
    if btc_bh_return > BULL_THRESHOLD:
        return "Bull"
    if btc_bh_return < BEAR_THRESHOLD:
        return "Bear"
    return "Mixed"


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:+.{digits}f}%"


def _fmt_num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    return f"{x:+.{digits}f}"


def _collect() -> list[dict]:
    """Return one record per fold, merging eval_metrics + backtest_metrics."""
    records = []
    for fold_idx in range(1, len(cfg.FOLDS) + 1):
        spec = cfg.FOLDS[fold_idx - 1]
        fold_dir = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME / f"fold_{fold_idx}"
        eval_metrics = _safe_load_json(fold_dir / "eval_metrics.json")
        backtest_metrics = _safe_load_json(fold_dir / "backtest_metrics.json")
        records.append({
            "fold": fold_idx,
            "spec": spec,
            "eval_metrics": eval_metrics,
            "backtest_metrics": backtest_metrics,
        })
    return records


def _markdown(records: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# Walk-Forward Aggregate Report — `{cfg.WALK_FORWARD_RUN_NAME}`\n")
    lines.append(f"**Folds:** {len(records)}  ·  "
                 f"**Bull threshold (BH BTC):** > +{BULL_THRESHOLD*100:.0f}%  ·  "
                 f"**Bear threshold:** < {BEAR_THRESHOLD*100:.0f}%\n")

    # ------------------------------------------------------------------ Per fold
    lines.append("\n## Per-fold summary\n")
    lines.append("| Fold | Train | Eval | Regime | BH ret BTC | Model ret BTC | vs BH BTC | Sharpe BTC | MDD BTC | L/N/S BTC | "
                 "BH ret ETH | Model ret ETH | vs BH ETH | Sharpe ETH | MDD ETH | L/N/S ETH |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        fold = r["fold"]
        spec = r["spec"]
        em = r["eval_metrics"]
        bm = r["backtest_metrics"]
        if bm is None:
            lines.append(
                f"| {fold} | {spec[0]} → {spec[1]} | {spec[2]} → {spec[3]} | "
                f"N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        btc = bm.get("BTC", {})
        eth = bm.get("ETH", {})
        if "error" in btc: btc = {}
        if "error" in eth: eth = {}
        regime = _regime_label(btc.get("BH_return"))
        lns_btc = (f"{btc.get('pct_long', 0)*100:.0f}/{btc.get('pct_neutral', 0)*100:.0f}/"
                   f"{btc.get('pct_short', 0)*100:.0f}" if btc else "N/A")
        lns_eth = (f"{eth.get('pct_long', 0)*100:.0f}/{eth.get('pct_neutral', 0)*100:.0f}/"
                   f"{eth.get('pct_short', 0)*100:.0f}" if eth else "N/A")
        lines.append(
            f"| {fold} | {spec[0]} → {spec[1]} | {spec[2]} → {spec[3]} | "
            f"**{regime}** | "
            f"{_fmt_pct(btc.get('BH_return'))} | {_fmt_pct(btc.get('return'))} | "
            f"{_fmt_pct(btc.get('vs_BH'))} | {_fmt_num(btc.get('sharpe'))} | "
            f"{_fmt_pct(btc.get('mdd'))} | {lns_btc} | "
            f"{_fmt_pct(eth.get('BH_return'))} | {_fmt_pct(eth.get('return'))} | "
            f"{_fmt_pct(eth.get('vs_BH'))} | {_fmt_num(eth.get('sharpe'))} | "
            f"{_fmt_pct(eth.get('mdd'))} | {lns_eth} |"
        )

    # ------------------------------------------- Training-eval metrics summary
    lines.append("\n## Training summary (per-fold best_smoothed_sharpe)\n")
    lines.append("| Fold | best_smoothed_sharpe | peak_unsmoothed_sharpe | timesteps | elapsed (s) | best_model saved |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        em = r["eval_metrics"]
        if em is None:
            lines.append(f"| {r['fold']} | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {r['fold']} | {_fmt_num(em.get('best_smoothed_sharpe'))} | "
            f"{_fmt_num(em.get('peak_unsmoothed_sharpe'))} | "
            f"{em.get('total_timesteps', 'N/A')} | "
            f"{em.get('elapsed_seconds', 0):.0f} | "
            f"{'✅' if em.get('best_model_saved') else '❌'} |"
        )

    # ----------------------------------------------- Aggregate stats per asset
    def _aggregate(metric: str, asset: str) -> dict[str, float]:
        vals = []
        for r in records:
            bm = r["backtest_metrics"]
            if bm is None: continue
            a = bm.get(asset, {})
            if "error" in a: continue
            v = a.get(metric)
            if v is None or (isinstance(v, float) and np.isnan(v)): continue
            vals.append(float(v))
        if not vals:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"),
                    "pct_positive": float("nan"), "pct_beats_bh": float("nan")}
        arr = np.array(vals)
        # pct_beats_bh requires comparison with BH_return per fold
        pct_beats = float("nan")
        if metric == "return":
            beat = []
            for r in records:
                bm = r["backtest_metrics"]
                if bm is None: continue
                a = bm.get(asset, {})
                if "error" in a or "return" not in a or "BH_return" not in a: continue
                beat.append(1.0 if a["return"] > a["BH_return"] else 0.0)
            pct_beats = float(np.mean(beat)) if beat else float("nan")
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "pct_positive": float((arr > 0).mean()),
            "pct_beats_bh": pct_beats,
        }

    lines.append("\n## Aggregate across all folds (per asset)\n")
    lines.append("| Metric | Asset | n | mean | std | min | max | % > 0 | % beats B&H |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for metric in ("return", "sharpe", "mdd", "vs_BH"):
        for asset in ASSETS:
            agg = _aggregate(metric, asset)
            fmt = _fmt_pct if metric in ("return", "mdd", "vs_BH") else _fmt_num
            beats_str = f"{agg['pct_beats_bh']*100:.0f}%" if not np.isnan(agg['pct_beats_bh']) else "N/A"
            lines.append(
                f"| {metric} | {asset} | {agg['n']} | "
                f"{fmt(agg['mean']) if agg['n'] else 'N/A'} | "
                f"{fmt(agg['std']) if agg['n'] else 'N/A'} | "
                f"{fmt(agg['min']) if agg['n'] else 'N/A'} | "
                f"{fmt(agg['max']) if agg['n'] else 'N/A'} | "
                f"{agg['pct_positive']*100:.0f}% | "
                f"{beats_str} |"
            )

    # --------------------------------------- Regime-conditional analysis (KEY)
    lines.append("\n## Regime-conditional analysis (THE diagnostic)\n")
    lines.append(
        "Splits folds by regime (Bull/Bear/Mixed via BH BTC return on the eval window). "
        "If the model is **regime-robust**, Bull and Bear means should be comparable. "
        "If it's a **bull specialist**, Bear will have terrible numbers vs Bull.\n"
    )

    regime_buckets: dict[str, list[dict]] = {"Bull": [], "Bear": [], "Mixed": [], "?": []}
    for r in records:
        bm = r["backtest_metrics"]
        if bm is None:
            regime_buckets["?"].append(r)
            continue
        btc = bm.get("BTC", {})
        regime = _regime_label(btc.get("BH_return") if "error" not in btc else None)
        regime_buckets[regime].append(r)

    lines.append("| Regime | n folds | mean Sharpe BTC | mean Return BTC | mean vs B&H BTC | "
                 "mean Sharpe ETH | mean Return ETH | mean vs B&H ETH |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for regime in ("Bull", "Mixed", "Bear", "?"):
        bucket = regime_buckets[regime]
        if not bucket:
            continue
        def _mean(field: str, asset: str) -> float | None:
            vs = []
            for r in bucket:
                bm = r["backtest_metrics"]
                if bm is None: continue
                a = bm.get(asset, {})
                if "error" in a: continue
                v = a.get(field)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vs.append(v)
            return float(np.mean(vs)) if vs else None
        lines.append(
            f"| **{regime}** | {len(bucket)} | "
            f"{_fmt_num(_mean('sharpe', 'BTC'))} | {_fmt_pct(_mean('return', 'BTC'))} | "
            f"{_fmt_pct(_mean('vs_BH', 'BTC'))} | "
            f"{_fmt_num(_mean('sharpe', 'ETH'))} | {_fmt_pct(_mean('return', 'ETH'))} | "
            f"{_fmt_pct(_mean('vs_BH', 'ETH'))} |"
        )

    # ------------------------------------------------------------ Folds listed
    lines.append("\n### Folds by regime\n")
    for regime in ("Bull", "Mixed", "Bear", "?"):
        bucket = regime_buckets[regime]
        if not bucket: continue
        ids = ", ".join(f"#{r['fold']}" for r in bucket)
        lines.append(f"- **{regime}** ({len(bucket)}): {ids}")

    lines.append("")
    return "\n".join(lines)


def _write_csv(records: list[dict], out_path: Path) -> None:
    """Flat per-(fold, asset) rows."""
    fieldnames = [
        "fold", "asset", "train_start", "train_end", "eval_start", "eval_end",
        "regime", "n_steps", "return", "BH_return", "vs_BH",
        "sharpe", "sortino", "mdd", "BH_sharpe", "BH_mdd",
        "n_trades", "pct_long", "pct_neutral", "pct_short",
        "best_smoothed_sharpe", "peak_unsmoothed_sharpe",
        "training_timesteps", "training_elapsed_seconds",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            spec = r["spec"]
            em = r["eval_metrics"] or {}
            bm = r["backtest_metrics"] or {}
            btc = bm.get("BTC", {})
            regime = _regime_label(btc.get("BH_return") if "error" not in btc else None)
            for asset in ASSETS:
                a = bm.get(asset, {}) if bm else {}
                if "error" in a: a = {}
                w.writerow({
                    "fold": r["fold"], "asset": asset,
                    "train_start": spec[0], "train_end": spec[1],
                    "eval_start": spec[2], "eval_end": spec[3],
                    "regime": regime,
                    "n_steps": a.get("n_steps"),
                    "return": a.get("return"),
                    "BH_return": a.get("BH_return"),
                    "vs_BH": a.get("vs_BH"),
                    "sharpe": a.get("sharpe"),
                    "sortino": a.get("sortino"),
                    "mdd": a.get("mdd"),
                    "BH_sharpe": a.get("BH_sharpe"),
                    "BH_mdd": a.get("BH_mdd"),
                    "n_trades": a.get("n_trades"),
                    "pct_long": a.get("pct_long"),
                    "pct_neutral": a.get("pct_neutral"),
                    "pct_short": a.get("pct_short"),
                    "best_smoothed_sharpe": em.get("best_smoothed_sharpe"),
                    "peak_unsmoothed_sharpe": em.get("peak_unsmoothed_sharpe"),
                    "training_timesteps": em.get("total_timesteps"),
                    "training_elapsed_seconds": em.get("elapsed_seconds"),
                })


def main():
    out_dir = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _collect()
    n_with_eval = sum(1 for r in records if r["eval_metrics"] is not None)
    n_with_bt = sum(1 for r in records if r["backtest_metrics"] is not None)
    print(f"[report] {n_with_eval}/{len(records)} folds with eval_metrics.json")
    print(f"[report] {n_with_bt}/{len(records)} folds with backtest_metrics.json")

    md_path = out_dir / "aggregate_report.md"
    csv_path = out_dir / "aggregate_report.csv"
    with open(md_path, "w") as f:
        f.write(_markdown(records))
    _write_csv(records, csv_path)
    print(f"[report] wrote {md_path}")
    print(f"[report] wrote {csv_path}")


if __name__ == "__main__":
    main()
