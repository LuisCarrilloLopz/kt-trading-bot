"""
Parse TensorBoard event files and produce a structured training summary.

Usage:
    python -m src.utils.analyze_logs                       # latest run under logs/tensorboard
    python -m src.utils.analyze_logs --run-name v1
    python -m src.utils.analyze_logs --logdir /path/to/specific/run
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.core.config import LOG_DIR


KEY_TAGS = [
    # Eval (custom callback)
    "eval/sharpe_ratio", "eval/sharpe_std", "eval/total_return",
    "eval/max_drawdown", "eval/sortino_ratio", "eval/mean_reward",
    # Rollout
    "rollout/ep_rew_mean", "rollout/ep_len_mean",
    # PPO training health
    "train/explained_variance", "train/approx_kl", "train/clip_fraction",
    "train/entropy_loss", "train/policy_gradient_loss", "train/value_loss",
    "train/learning_rate",
]


def _latest_run_dir() -> Path:
    candidates = [p for p in LOG_DIR.rglob("events.out.tfevents.*") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No tfevents found under {LOG_DIR}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def load_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {tag: (steps, values)}."""
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        values = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, values)
    return out


def _fmt(x: float) -> str:
    if abs(x) >= 1000 or (0 < abs(x) < 0.01):
        return f"{x:+.3e}"
    return f"{x:+.4f}"


def _trend(values: np.ndarray) -> str:
    if len(values) < 4:
        return "n/a"
    first = values[: max(1, len(values) // 4)].mean()
    last = values[-max(1, len(values) // 4):].mean()
    delta = last - first
    if abs(delta) < 1e-6:
        return "flat"
    return f"{first:+.3f} → {last:+.3f}  ({delta:+.3f})"


def summarize(scalars: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    print(f"\n{'tag':<35} {'n':>5} {'last_step':>10}   {'last':>12}   {'min':>10}   {'max':>10}   trend (first 25% → last 25%)")
    print("-" * 130)
    for tag in KEY_TAGS:
        if tag not in scalars:
            continue
        steps, values = scalars[tag]
        if len(values) == 0:
            continue
        print(
            f"{tag:<35} {len(values):>5} {int(steps[-1]):>10}   "
            f"{_fmt(values[-1]):>12}   {_fmt(values.min()):>10}   {_fmt(values.max()):>10}   "
            f"{_trend(values)}"
        )

    # Extras
    other = [t for t in scalars if t not in KEY_TAGS]
    if other:
        print(f"\n(other tags present, not shown: {', '.join(sorted(other))})")


def head_tail_sample(scalars: dict[str, tuple[np.ndarray, np.ndarray]], tag: str, k: int = 8) -> None:
    if tag not in scalars:
        print(f"[no data for {tag!r}]")
        return
    steps, values = scalars[tag]
    n = len(values)
    if n <= 2 * k:
        idx = np.arange(n)
    else:
        idx = np.unique(np.concatenate([
            np.arange(k),
            np.linspace(k, n - k - 1, k, dtype=int),
            np.arange(n - k, n),
        ]))
    print(f"\n--- {tag} (n={n}) ---")
    for i in idx:
        print(f"  step={int(steps[i]):>10}   value={_fmt(float(values[i]))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None,
                        help="Pick a specific run dir under logs/tensorboard/<run-name>")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Absolute path to a run dir (overrides --run-name)")
    parser.add_argument("--detail", type=str, action="append", default=[],
                        help="Print head/tail samples for this tag (can repeat)")
    args = parser.parse_args()

    if args.logdir:
        run_dir = Path(args.logdir)
    elif args.run_name:
        candidates = list((LOG_DIR / args.run_name).rglob("events.out.tfevents.*"))
        if not candidates:
            print(f"No events under {LOG_DIR / args.run_name}", file=sys.stderr)
            sys.exit(1)
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime).parent
    else:
        run_dir = _latest_run_dir()

    print(f"Reading: {run_dir}")
    scalars = load_scalars(run_dir)
    if not scalars:
        print("(no scalar tags found)")
        return
    summarize(scalars)
    for tag in args.detail:
        head_tail_sample(scalars, tag)


if __name__ == "__main__":
    main()
