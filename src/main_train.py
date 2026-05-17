"""Train RecurrentPPO on the multi-asset (BTC + ETH) trading env.

Two modes:
- Default (no --fold): legacy single train/eval 80/20 chronological split.
- Walk-forward (--fold N, N in 1..9): trains on the N-th fold's date_range from
  config.FOLDS, using config.WALK_FORWARD_TIMESTEPS as the default budget,
  output dirs under <CHECKPOINT_DIR>/walk_forward_v1/fold_N/.
"""
from __future__ import annotations
import argparse
import json
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.core import config as cfg
from src.data.loader import load_assets
from src.envs.trading_env import MultiAssetTradingEnv
from src.callbacks.sharpe_eval_callback import SharpeEvalCallback
from src.callbacks.ent_coef_schedule import EntCoefScheduleCallback
from src.callbacks.checkpoint_callback import CheckpointWithVecNormalizeCallback


def _make_env(dfs, randomize_start: bool):
    def _init():
        return MultiAssetTradingEnv(dfs, randomize_start=randomize_start)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override timesteps. Default: TOTAL_TIMESTEPS or "
                             "WALK_FORWARD_TIMESTEPS when --fold is set.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override run name. Default: 'v1' or 'walk_forward_v1/fold_N' "
                             "when --fold is set.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Walk-forward fold index (1..len(FOLDS)). If set, "
                             "loads train/eval from FOLDS[fold-1] date ranges.")
    args = parser.parse_args()

    # --- Resolve walk-forward parameters ---
    if args.fold is not None:
        if not (1 <= args.fold <= len(cfg.FOLDS)):
            parser.error(f"--fold must be in 1..{len(cfg.FOLDS)}, got {args.fold}")
        fold_spec = cfg.FOLDS[args.fold - 1]
        train_range = (fold_spec[0], fold_spec[1])
        eval_range = (fold_spec[2], fold_spec[3])
        if args.total_timesteps is None:
            args.total_timesteps = cfg.WALK_FORWARD_TIMESTEPS
        if args.run_name is None:
            args.run_name = f"{cfg.WALK_FORWARD_RUN_NAME}/fold_{args.fold}"
    else:
        if args.total_timesteps is None:
            args.total_timesteps = cfg.TOTAL_TIMESTEPS
        if args.run_name is None:
            args.run_name = "v1"

    print(f"--- kt-trading-bot — run='{args.run_name}' seed={args.seed} "
          f"timesteps={args.total_timesteps} fold={args.fold} ---")
    set_random_seed(args.seed)

    if args.fold is not None:
        print(f"WALK-FORWARD fold {args.fold}:")
        print(f"  train: {train_range[0]} → {train_range[1]}")
        print(f"  eval:  {eval_range[0]} → {eval_range[1]}")
        train_dfs = load_assets(date_range=train_range)
        eval_dfs = load_assets(date_range=eval_range)
    else:
        train_dfs = load_assets(split="train")
        eval_dfs = load_assets(split="eval")
    print(f"train sizes: { {k: len(v) for k, v in train_dfs.items()} }")
    print(f"eval  sizes: { {k: len(v) for k, v in eval_dfs.items()} }")

    # VecNormalize: norm_obs=False because features are pre-scaled in the env;
    # norm_reward=True still helps PPO stability.
    env_train = DummyVecEnv([_make_env(train_dfs, randomize_start=True)])
    env_train = VecNormalize(
        env_train, norm_obs=False, norm_reward=True,
        clip_reward=cfg.REWARD_CLIP, gamma=cfg.GAMMA,
    )

    env_eval = DummyVecEnv([_make_env(eval_dfs, randomize_start=True)])
    env_eval = VecNormalize(
        env_eval, norm_obs=False, norm_reward=False,
        training=False, clip_obs=10.0, gamma=cfg.GAMMA,
    )

    train_size = min(len(df) for df in train_dfs.values())
    eval_freq = max(10_000, train_size // 4)

    run_log_dir = cfg.LOG_DIR / args.run_name
    run_ckpt_dir = cfg.CHECKPOINT_DIR / args.run_name
    run_final_dir = cfg.FINAL_DIR / args.run_name
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
        initial_coef=cfg.ENT_COEF_INITIAL,
        final_coef=cfg.ENT_COEF_FINAL,
        total_timesteps=args.total_timesteps,
    )
    ckpt_cb = CheckpointWithVecNormalizeCallback(
        save_freq=eval_freq,
        save_path=str(run_ckpt_dir),
        name_prefix="ppo_lstm",
        verbose=1,
    )

    print(f"RecurrentPPO  critic_lstm={cfg.ENABLE_CRITIC_LSTM}  n_steps={cfg.N_STEPS}  "
          f"batch={cfg.BATCH_SIZE}  eval_freq={eval_freq}")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env_train,
        verbose=1,
        learning_rate=cfg.LEARNING_RATE,
        n_steps=cfg.N_STEPS,
        batch_size=cfg.BATCH_SIZE,
        n_epochs=cfg.N_EPOCHS,
        gamma=cfg.GAMMA,
        gae_lambda=cfg.GAE_LAMBDA,
        ent_coef=cfg.ENT_COEF_INITIAL,
        vf_coef=cfg.VF_COEF,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        policy_kwargs={
            "enable_critic_lstm": cfg.ENABLE_CRITIC_LSTM,
            "lstm_hidden_size": cfg.LSTM_HIDDEN_SIZE,
            "net_arch": cfg.NET_ARCH,
        },
        seed=args.seed,
        tensorboard_log=str(run_log_dir),
    )

    t_start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, ent_cb, ckpt_cb],
        tb_log_name="run",
    )
    elapsed = time.time() - t_start

    model.save(str(run_final_dir / "ppo_lstm"))
    env_train.save(str(run_final_dir / "vec_normalize.pkl"))

    if args.fold is not None:
        # Persist final eval summary for walk-forward orchestrator/report.
        eval_metrics = {
            "fold": args.fold,
            "train_start": train_range[0],
            "train_end": train_range[1],
            "eval_start": eval_range[0],
            "eval_end": eval_range[1],
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
        }
        out_path = run_final_dir / "eval_metrics.json"
        with open(out_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"[fold {args.fold}] eval_metrics.json saved → {out_path}")

    print(f"--- training complete ({elapsed:.1f}s) ---")


if __name__ == "__main__":
    main()
