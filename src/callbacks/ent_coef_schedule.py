"""Linear decay of `ent_coef` from initial to final across the training horizon."""
from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback


class EntCoefScheduleCallback(BaseCallback):
    def __init__(self, initial_coef: float, final_coef: float, total_timesteps: int):
        super().__init__()
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.total_timesteps = max(1, int(total_timesteps))

    def _on_step(self) -> bool:
        progress_remaining = max(0.0, 1.0 - self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = (
            self.final_coef + progress_remaining * (self.initial_coef - self.final_coef)
        )
        return True
