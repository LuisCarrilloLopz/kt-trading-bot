"""
Evaluation callback that:
- Syncs VecNormalize stats from train env to eval env (no leakage).
- Rolls out N eval episodes, reconstructs the equity curve from `info['net_worth']`.
- Logs Sharpe / Sortino / MDD / total return to TensorBoard.
- Saves `best_model` selected by **smoothed** mean Sharpe across the last K evaluations
  (V1.5+: prevents a single lucky eval from becoming best_model).
"""
from __future__ import annotations
import collections
from copy import deepcopy
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.utils.metrics import sharpe_ratio, sortino_ratio, max_drawdown, total_return


class SharpeEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecNormalize,
        train_env_ref: VecNormalize,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_env_ref = train_env_ref
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = max(1, int(eval_freq))
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.deterministic = deterministic
        # Smoothing — best_model is selected by mean Sharpe of the last K evaluations,
        # not by a single eval. Avoids one lucky outlier becoming best_model.
        self.smoothing_window = 5
        self.sharpe_history: collections.deque[float] = collections.deque(maxlen=self.smoothing_window)
        self.best_smoothed_sharpe = -np.inf
        self.peak_unsmoothed_sharpe = -np.inf   # for TB comparison only, not used to save
        if self.best_model_save_path is not None:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Sync normalization stats — eval env uses training=False, so this is the
        # only way it sees up-to-date stats without contaminating them.
        if hasattr(self.train_env_ref, "obs_rms") and self.train_env_ref.obs_rms is not None:
            self.eval_env.obs_rms = deepcopy(self.train_env_ref.obs_rms)
        if hasattr(self.train_env_ref, "ret_rms") and self.train_env_ref.ret_rms is not None:
            self.eval_env.ret_rms = deepcopy(self.train_env_ref.ret_rms)

        rewards, sharpes, sortinos, mdds, returns = [], [], [], [], []
        for _ in range(self.n_eval_episodes):
            equity, total_r = self._run_episode()
            rewards.append(total_r)
            sharpes.append(sharpe_ratio(equity))
            sortinos.append(sortino_ratio(equity))
            mdds.append(max_drawdown(equity))
            returns.append(total_return(equity))

        mean_sharpe = float(np.mean(sharpes))

        # Update history (deque keeps only the last `smoothing_window` values).
        self.sharpe_history.append(mean_sharpe)
        self.peak_unsmoothed_sharpe = max(self.peak_unsmoothed_sharpe, mean_sharpe)

        # Smoothed metric: only valid once the window is full (warmup).
        is_warm = len(self.sharpe_history) >= self.smoothing_window
        smoothed_sharpe = float(np.mean(self.sharpe_history)) if is_warm else float("nan")

        self.logger.record("eval/mean_reward", float(np.mean(rewards)))
        self.logger.record("eval/sharpe_ratio", mean_sharpe)
        self.logger.record("eval/sharpe_std", float(np.std(sharpes)))
        self.logger.record("eval/sortino_ratio", float(np.mean(sortinos)))
        self.logger.record("eval/max_drawdown", float(np.mean(mdds)))
        self.logger.record("eval/total_return", float(np.mean(returns)))
        self.logger.record("eval/sharpe_smoothed_5", smoothed_sharpe)
        self.logger.record("eval/best_smoothed_sharpe", self.best_smoothed_sharpe)
        self.logger.record("eval/peak_unsmoothed_sharpe", self.peak_unsmoothed_sharpe)
        if self.verbose:
            smoothed_str = f"smoothed5={smoothed_sharpe:+.3f}" if is_warm else "smoothed5=warmup"
            print(
                f"[eval @ {self.num_timesteps}] "
                f"Sharpe={mean_sharpe:+.3f} (std={np.std(sharpes):.3f})  "
                f"{smoothed_str}  "
                f"Return={np.mean(returns) * 100:+.2f}%  MDD={np.mean(mdds) * 100:.2f}%"
            )

        # Save best_model only when window is full AND smoothed mean strictly improves.
        if is_warm and smoothed_sharpe > self.best_smoothed_sharpe:
            self.best_smoothed_sharpe = smoothed_sharpe
            if self.best_model_save_path is not None:
                self.model.save(str(self.best_model_save_path / "best_model"))
                self.eval_env.save(str(self.best_model_save_path / "vec_normalize.pkl"))
                if self.verbose:
                    print(f"[eval] new best smoothed_sharpe={smoothed_sharpe:+.3f}")
        return True

    def _run_episode(self) -> tuple[np.ndarray, float]:
        obs = self.eval_env.reset()
        lstm_states = None
        episode_starts = np.ones((self.eval_env.num_envs,), dtype=bool)
        equity: list[float] = []
        total_r = 0.0
        for _ in range(20_000):  # safety cap; env truncates at MAX_EPISODE_STEPS
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=self.deterministic,
            )
            obs, reward, done, info = self.eval_env.step(action)
            episode_starts = done
            total_r += float(reward[0])
            equity.append(float(info[0]["net_worth"]))
            if done[0]:
                break
        return np.array(equity, dtype=np.float64), total_r
