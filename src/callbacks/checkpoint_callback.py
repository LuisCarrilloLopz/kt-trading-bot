"""Periodic checkpoint of (model, VecNormalize stats)."""
from __future__ import annotations
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointWithVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ppo_lstm",
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        model_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
        vec_path = self.save_path / f"vec_normalize_{self.num_timesteps}_steps.pkl"
        self.model.save(str(model_path))
        if self.training_env is not None:
            self.training_env.save(str(vec_path))
        if self.verbose > 0:
            print(f"[checkpoint] {model_path.name}")
        return True
