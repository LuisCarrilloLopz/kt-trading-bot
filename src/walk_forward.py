"""Walk-forward sequential orchestrator: trains + evaluates folds in sequence.

For each fold N in [start_fold, end_fold]:
  1. subprocess: python -m src.main_train --fold N --total-timesteps T
  2. subprocess: python -m src.evaluate_fold --fold N

Subprocesses are used to fully isolate SB3/torch state between folds (avoids any
cross-fold memory or RNG leakage). stdout/stderr stream live to the caller's
terminal — no silencing.

Meta-events (start/end/returncode/errors per fold) are logged with timestamps to
`<FINAL_DIR>/walk_forward_v1/walk_forward.log` AND echoed to console.

Flags:
  --start-fold N      (default 1)
  --end-fold M        (default = len(FOLDS))
  --total-timesteps T (overrides WALK_FORWARD_TIMESTEPS for each fold)
  --skip-evaluate     (only run training, skip evaluate_fold per fold)
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.core import config as cfg


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TeeLogger:
    """Writes a line to both stdout and a file (with timestamp prefix)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", buffering=1)  # line-buffered

    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


def _run_subprocess(cmd: list[str], logger: TeeLogger, label: str) -> int:
    """Run `cmd`, stream output live, return returncode."""
    logger.log(f"=== {label} START === cmd: {' '.join(cmd)}")
    t0 = time.time()
    # Inherit stdout/stderr → subprocess output streams to terminal directly.
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    logger.log(f"=== {label} END   === returncode={proc.returncode}  elapsed={elapsed:.1f}s")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=len(cfg.FOLDS))
    parser.add_argument("--total-timesteps", type=int, default=cfg.WALK_FORWARD_TIMESTEPS,
                        help="Per-fold training budget (default: WALK_FORWARD_TIMESTEPS)")
    parser.add_argument("--skip-evaluate", action="store_true",
                        help="Only run training, skip evaluate_fold per fold.")
    args = parser.parse_args()

    if not (1 <= args.start_fold <= args.end_fold <= len(cfg.FOLDS)):
        parser.error(f"Invalid fold range [{args.start_fold}..{args.end_fold}]; "
                     f"must be within 1..{len(cfg.FOLDS)}")

    log_path = cfg.FINAL_DIR / cfg.WALK_FORWARD_RUN_NAME / "walk_forward.log"
    logger = TeeLogger(log_path)

    logger.log(f"WALK-FORWARD orchestrator starting")
    logger.log(f"  folds:       {args.start_fold}..{args.end_fold} (of {len(cfg.FOLDS)})")
    logger.log(f"  timesteps:   {args.total_timesteps} per fold")
    logger.log(f"  skip_eval:   {args.skip_evaluate}")
    logger.log(f"  log file:    {log_path}")

    summary: list[tuple[int, str, int]] = []  # (fold, phase, returncode)
    t_global = time.time()

    try:
        for fold in range(args.start_fold, args.end_fold + 1):
            spec = cfg.FOLDS[fold - 1]
            logger.log(
                f"\n##### FOLD {fold}/{len(cfg.FOLDS)} #####  "
                f"train={spec[0]}→{spec[1]}  eval={spec[2]}→{spec[3]}"
            )

            # Phase 1: training
            train_cmd = [
                sys.executable, "-m", "src.main_train",
                "--fold", str(fold),
                "--total-timesteps", str(args.total_timesteps),
            ]
            rc = _run_subprocess(train_cmd, logger, f"fold-{fold} TRAIN")
            summary.append((fold, "train", rc))
            if rc != 0:
                logger.log(f"❌ TRAIN failed (rc={rc}) — aborting walk-forward")
                break

            if args.skip_evaluate:
                logger.log(f"--skip-evaluate set; skipping fold-{fold} evaluate")
                continue

            # Phase 2: deterministic post-training evaluation
            eval_cmd = [sys.executable, "-m", "src.evaluate_fold", "--fold", str(fold)]
            rc = _run_subprocess(eval_cmd, logger, f"fold-{fold} EVALUATE")
            summary.append((fold, "evaluate", rc))
            if rc != 0:
                logger.log(f"❌ EVALUATE failed (rc={rc}) — aborting walk-forward")
                break
    finally:
        total_elapsed = time.time() - t_global
        logger.log(f"\n===== walk-forward DONE =====  total elapsed: {total_elapsed:.1f}s")
        logger.log("Summary:")
        for fold, phase, rc in summary:
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            logger.log(f"  fold {fold:>2} {phase:>10}: {status}")
        logger.close()


if __name__ == "__main__":
    main()
