"""PPO fine-tune of a BC-initialized policy.

Loads `models/final/bc_v1/fold_N/bc_model.zip` and continues training with
RecurrentPPO, using REWARD_MODE='absolute' (V1 baseline), lowered LR and
ent_coef so the BC init survives.

Hyperparams that differ from walk-forward:
  - learning_rate: 1e-4 * progress     (vs 3e-4 * progress)
  - ent_coef initial: 0.01             (vs 0.05)
  - ent_coef final:   0.001            (proportionally lower)
  - total_timesteps: 400_000           (vs 1_500_000)

Output: models/{checkpoints,final}/bc_v1/fold_N/ ; TB logs/tensorboard/bc_v1/fold_N/.

Action-distribution monitor: at every rollout end, computes pct L/N/S over the
PPO rollout buffer. If pct_neutral > 0.85 for 3 consecutive rollouts, stops
training — that's the explicit "BC init didn't hold under v1 reward" signal.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.core import config as cfg
from src.data.loader import load_assets
from src.envs.trading_env import MultiAssetTradingEnv
from src.callbacks.sharpe_eval_callback import SharpeEvalCallback
from src.callbacks.ent_coef_schedule import EntCoefScheduleCallback
from src.callbacks.checkpoint_callback import CheckpointWithVecNormalizeCallback


BC_FINETUNE_RUN_NAME = "bc_v1"
FINETUNE_TIMESTEPS = 400_000
FINETUNE_LR = 1e-4
FINETUNE_ENT_INITIAL = 0.01
FINETUNE_ENT_FINAL = 0.001
NEUTRAL_COLLAPSE_THRESHOLD = 0.85
NEUTRAL_COLLAPSE_PATIENCE = 3   # consecutive rollouts >= threshold to stop


class ActionDistributionMonitor(BaseCallback):
    """Logs rollout-buffer L/N/S distribution per rollout, stops on NEUTRAL collapse.

    PPO actions are stored in `rollout_buffer.actions` with shape
    (n_steps, n_envs). For Discrete actions they are float-encoded ints in
    [0, n_actions). We aggregate the full buffer at every rollout boundary.
    """
    def __init__(self, threshold: float = NEUTRAL_COLLAPSE_THRESHOLD,
                 patience: int = NEUTRAL_COLLAPSE_PATIENCE, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.consecutive_collapses = 0
        self._should_stop = False

    def _on_step(self) -> bool:
        # Honor the stop flag here — _on_rollout_end can't directly abort.
        return not self._should_stop

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        actions = np.asarray(buf.actions).flatten().astype(int)
        n = len(actions)
        if n == 0:
            return
        pct_short   = (actions == 0).sum() / n
        pct_neutral = (actions == 1).sum() / n
        pct_long    = (actions == 2).sum() / n
        self.logger.record("rollout/pct_short", float(pct_short))
        self.logger.record("rollout/pct_neutral", float(pct_neutral))
        self.logger.record("rollout/pct_long", float(pct_long))
        if pct_neutral > self.threshold:
            self.consecutive_collapses += 1
            if self.verbose:
                print(f"  [WARN] rollout pct_neutral={pct_neutral*100:.1f}% "
                      f"> {self.threshold*100:.0f}%  "
                      f"(consecutive: {self.consecutive_collapses}/{self.patience})")
            if self.consecutive_collapses >= self.patience:
                print(f"  [STOP] NEUTRAL collapse — {self.patience} consecutive rollouts "
                      f"with pct_neutral > {self.threshold*100:.0f}%. BC init lost under v1 reward.")
                self._should_stop = True
        else:
            if self.consecutive_collapses > 0 and self.verbose:
                print(f"  [recover] rollout pct_neutral={pct_neutral*100:.1f}% — reset counter")
            self.consecutive_collapses = 0


def _make_env(dfs, randomize_start: bool):
    return MultiAssetTradingEnv(dfs, randomize_start=randomize_start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=9,
                        help="Walk-forward fold index (1..len(FOLDS)).")
    parser.add_argument("--total-timesteps", type=int, default=FINETUNE_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args()

    if not (1 <= args.fold <= len(cfg.FOLDS)):
        parser.error(f"--fold must be 1..{len(cfg.FOLDS)}")
    spec = cfg.FOLDS[args.fold - 1]
    train_range = (spec[0], spec[1])
    eval_range = (spec[2], spec[3])
    run_name = f"{BC_FINETUNE_RUN_NAME}/fold_{args.fold}"

    # Safety: refuse to overwrite walk_forward_v1/v2 artifacts (defensive).
    forbidden = {"walk_forward_v1", "walk_forward_v2"}
    if BC_FINETUNE_RUN_NAME in forbidden:
        parser.error(f"BC_FINETUNE_RUN_NAME='{BC_FINETUNE_RUN_NAME}' collides with walk-forward — refusing to overwrite.")

    bc_dir = cfg.FINAL_DIR / BC_FINETUNE_RUN_NAME / f"fold_{args.fold}"
    bc_model_path = bc_dir / "bc_model.zip"
    bc_vecnorm_path = bc_dir / "vec_normalize.pkl"
    if not bc_model_path.exists():
        raise FileNotFoundError(f"BC checkpoint not found at {bc_model_path}. Run bc_pretrain first.")

    print(f"--- BC fine-tune — fold {args.fold}  run='{run_name}' ---")
    print(f"  train: {train_range[0]} → {train_range[1]}")
    print(f"  eval:  {eval_range[0]} → {eval_range[1]}")
    print(f"  REWARD_MODE={cfg.REWARD_MODE}  (must be 'absolute' for v1 baseline test)")
    print(f"  load BC from: {bc_model_path}")
    print(f"  timesteps={args.total_timesteps}  LR={FINETUNE_LR}  ent_coef={FINETUNE_ENT_INITIAL}→{FINETUNE_ENT_FINAL}")
    set_random_seed(args.seed)

    train_dfs = load_assets(date_range=train_range)
    eval_dfs = load_assets(date_range=eval_range)
    print(f"  train sizes: { {k: len(v) for k, v in train_dfs.items()} }")
    print(f"  eval  sizes: { {k: len(v) for k, v in eval_dfs.items()} }")

    # Build train env with BC's VecNormalize stats so the obs scale matches what
    # the BC actor saw (matters: VecNormalize wraps reward, not obs in our config,
    # but obs scale would matter if we'd had norm_obs=True).
    env_train = DummyVecEnv([lambda: _make_env(train_dfs, randomize_start=True)])
    env_train = VecNormalize.load(str(bc_vecnorm_path), env_train)
    env_train.training = True
    env_train.norm_reward = True

    env_eval = DummyVecEnv([lambda: _make_env(eval_dfs, randomize_start=True)])
    env_eval = VecNormalize(
        env_eval, norm_obs=False, norm_reward=False,
        training=False, clip_obs=10.0, gamma=cfg.GAMMA,
    )

    # Load BC model and override LR + ent_coef.
    model = RecurrentPPO.load(
        str(bc_model_path),
        env=env_train,
        custom_objects={
            "learning_rate": lambda p: FINETUNE_LR * p,
            "lr_schedule":   lambda p: FINETUNE_LR * p,
            "ent_coef":      FINETUNE_ENT_INITIAL,
            "clip_range":    0.2,
        },
    )

    train_size = min(len(df) for df in train_dfs.values())
    eval_freq = max(10_000, train_size // 4)

    run_log_dir = cfg.LOG_DIR / run_name
    run_ckpt_dir = cfg.CHECKPOINT_DIR / run_name
    run_final_dir = cfg.FINAL_DIR / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_final_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = SharpeEvalCallback(
        eval_env=env_eval,
        train_env_ref=env_train,
        n_eval_episodes=cfg.N_EVAL_EPISODES,
        eval_freq=eval_freq,
        best_model_save_path=str(run_ckpt_dir),
        deterministic=True,
        verbose=1,
    )
    ent_cb = EntCoefScheduleCallback(
        initial_coef=FINETUNE_ENT_INITIAL,
        final_coef=FINETUNE_ENT_FINAL,
        total_timesteps=args.total_timesteps,
    )
    ckpt_cb = CheckpointWithVecNormalizeCallback(
        save_freq=eval_freq,
        save_path=str(run_ckpt_dir),
        name_prefix="ppo_lstm",
        verbose=1,
    )
    monitor_cb = ActionDistributionMonitor(verbose=1)

    # Sb3 model.learn() resets num_timesteps unless we tell it not to.
    # We want a clean 0→400k schedule, so default is fine.
    model.tensorboard_log = str(run_log_dir)

    print()
    print(f"--- starting fine-tune  eval_freq={eval_freq} ---")
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, ent_cb, ckpt_cb, monitor_cb],
        tb_log_name="run",
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t0

    model.save(str(run_final_dir / "ppo_lstm"))
    env_train.save(str(run_final_dir / "vec_normalize.pkl"))

    eval_metrics = {
        "fold": args.fold,
        "train_start": spec[0], "train_end": spec[1],
        "eval_start":  spec[2], "eval_end":  spec[3],
        "bc_source": str(bc_model_path),
        "best_smoothed_sharpe": (
            float(eval_cb.best_smoothed_sharpe)
            if eval_cb.best_smoothed_sharpe != float("-inf") else None
        ),
        "peak_unsmoothed_sharpe": (
            float(eval_cb.peak_unsmoothed_sharpe)
            if eval_cb.peak_unsmoothed_sharpe != float("-inf") else None
        ),
        "total_timesteps": int(args.total_timesteps),
        "elapsed_seconds": float(elapsed),
        "seed": int(args.seed),
        "smoothing_window": int(eval_cb.smoothing_window),
        "best_model_saved": (run_ckpt_dir / "best_model.zip").exists(),
        "stopped_on_collapse": bool(monitor_cb._should_stop),
        "finetune_lr": FINETUNE_LR,
        "finetune_ent_initial": FINETUNE_ENT_INITIAL,
        "finetune_ent_final": FINETUNE_ENT_FINAL,
    }
    with open(run_final_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"--- fine-tune complete ({elapsed:.1f}s) ---")
    if monitor_cb._should_stop:
        print(f"[NOTICE] training stopped early due to NEUTRAL collapse signal.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
