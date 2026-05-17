"""Train RecurrentPPO on the multi-asset (BTC + ETH) trading env."""
from __future__ import annotations
import argparse

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
    parser.add_argument("--total-timesteps", type=int, default=cfg.TOTAL_TIMESTEPS)
    parser.add_argument("--run-name", type=str, default="v1")
    args = parser.parse_args()

    print(f"--- kt-trading-bot — run='{args.run_name}' seed={args.seed} ---")
    set_random_seed(args.seed)

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

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, ent_cb, ckpt_cb],
        tb_log_name="run",
    )

    model.save(str(run_final_dir / "ppo_lstm"))
    env_train.save(str(run_final_dir / "vec_normalize.pkl"))
    print("--- training complete ---")


if __name__ == "__main__":
    main()
