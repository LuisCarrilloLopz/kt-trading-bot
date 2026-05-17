"""Multi-asset trading environment with corrected mechanics and a clean reward."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.core.config import (
    TRADING_FEE, INITIAL_BALANCE, VOL_LOOKBACK, TARGET_VOL_PER_STEP,
    LEVERAGE_MIN, LEVERAGE_MAX, MAX_ALLOCATION, MIN_TRADE_USD,
    RUIN_THRESHOLD, MAX_EPISODE_STEPS,
    REWARD_LOG_RET_SCALE, REWARD_DRAWDOWN_COEF, REWARD_CHURN_COEF, REWARD_CLIP,
)

N_FEATURES = 13   # V1.4: 18 → 13 (ablation — quitadas hour sin/cos y log_ret lags 1/5/10 por sospecha de ruido; MACD/MACD_hist/ATR retenidas)


class MultiAssetTradingEnv(gym.Env):
    """
    Actions: 0=SHORT, 1=NEUTRAL, 2=LONG.

    Reward (per step):
        REWARD_LOG_RET_SCALE * log(nw_t / nw_{t-1})
      - REWARD_DRAWDOWN_COEF * max(0, drawdown - prev_drawdown)
      - REWARD_CHURN_COEF    * 1[position changed]

    The DD term is **derivative**: it fires only on new drawdown, not on
    existing drawdown. Total DD penalty per episode is bounded by
    REWARD_DRAWDOWN_COEF * max_drawdown_ever_reached. (V1.1 fix.)

    Each reset() samples one asset from `dfs`. The episode runs at most
    MAX_EPISODE_STEPS steps; ends with `truncated=True` on data/length cap and
    `terminated=True` only on ruin (net_worth <= RUIN_THRESHOLD * initial).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame],
        randomize_start: bool = True,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        super().__init__()
        if not dfs:
            raise ValueError("dfs must be a non-empty dict[str, DataFrame]")
        self.dfs = dfs
        self.asset_names = list(dfs.keys())
        self.randomize_start = randomize_start
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self.current_asset: str = self.asset_names[0]
        self.df: pd.DataFrame = self.dfs[self.current_asset]
        self.df_len: int = len(self.df)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = int(self.np_random.integers(0, len(self.asset_names)))
        self.current_asset = self.asset_names[idx]
        self.df = self.dfs[self.current_asset]
        self.df_len = len(self.df)

        if self.randomize_start:
            max_start = max(VOL_LOOKBACK + 1, self.df_len - self.max_episode_steps - 1)
            self.start_step = int(self.np_random.integers(VOL_LOOKBACK, max_start))
        else:
            self.start_step = VOL_LOOKBACK
        self.current_step = self.start_step
        self.steps_in_episode = 0

        self.cash = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.peak_net_worth = INITIAL_BALANCE
        self.prev_drawdown = 0.0
        self.current_position = 0
        self.entry_price = 0.0
        self.position_size = 0.0

        # Reward decomposition accumulators (V1.5+): emitted in info on episode end.
        self._r_log_ret_cum = 0.0
        self._r_dd_cum = 0.0
        self._r_churn_cum = 0.0
        self._r_clipped_cum = 0.0

        return self._get_observation(), {"asset": self.current_asset}

    # -------------------------------------------------------------- accessors
    def _position_value(self, price: float) -> float:
        if self.current_position == 1:
            return self.position_size * price
        if self.current_position == -1:
            # Mark-to-market a short: colateral + unrealized PnL
            return self.position_size * (2.0 * self.entry_price - price)
        return 0.0

    def _unrealized_pnl(self, price: float) -> float:
        if self.current_position == 1:
            return self.position_size * (price - self.entry_price)
        if self.current_position == -1:
            return self.position_size * (self.entry_price - price)
        return 0.0

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        price = float(row["C"])
        market_vol = float(row["market_vol"])

        position_value = self._position_value(price)
        unrealized_pnl = self._unrealized_pnl(price)
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth \
            if self.peak_net_worth > 0 else 0.0

        obs = np.array([
            # Market features (originales)
            float(row["dist_ma200"]) - 1.0,
            float(row["bb_pos"]) - 0.5,
            float(row["rsi_norm"]) - 0.5,
            float(np.log1p(max(float(row["vol_rel"]), 0.0))) - np.log1p(1.0),
            float(row["log_ret"]) * 100.0,
            market_vol * 100.0,
            # Portfolio features
            float(self.current_position),
            position_value / INITIAL_BALANCE,
            unrealized_pnl / INITIAL_BALANCE,
            drawdown,
            # V1.3+ retenidas — momentum y volatilidad realizada
            float(row["macd_norm"]),
            float(row["macd_hist_norm"]),
            float(row["atr_norm"]),
            # V1.4 — quitadas: hour_sin/cos (cripto 24/7) y log_ret_lag1/5/10 (LSTM ya mantiene memoria)
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    # ----------------------------------------------------------------- step
    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = float(row["C"])
        market_vol = float(row["market_vol"])

        target_dir = int(action) - 1
        prev_position = self.current_position

        safe_vol = max(market_vol, 1e-4)
        leverage = float(np.clip(TARGET_VOL_PER_STEP / safe_vol, LEVERAGE_MIN, LEVERAGE_MAX))

        if target_dir != self.current_position:
            self._execute_transition(target_dir, price, leverage)

        prev_net_worth = self.net_worth
        self.net_worth = float(np.clip(self.cash + self._position_value(price), 1.0, 1e8))
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)

        log_ret = float(np.clip(np.log(self.net_worth / max(prev_net_worth, 1e-6)), -0.2, 0.2))
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth \
            if self.peak_net_worth > 0 else 0.0
        new_drawdown = max(0.0, drawdown - self.prev_drawdown)
        self.prev_drawdown = drawdown
        position_changed = float(target_dir != prev_position)

        # Reward components (pre-clip) — exposed via info on episode end for diagnostics.
        r_log_ret = REWARD_LOG_RET_SCALE * log_ret
        r_dd      = -REWARD_DRAWDOWN_COEF * new_drawdown
        r_churn   = -REWARD_CHURN_COEF * position_changed
        reward_unclipped = r_log_ret + r_dd + r_churn
        reward = float(np.clip(reward_unclipped, -REWARD_CLIP, REWARD_CLIP))

        self._r_log_ret_cum += r_log_ret
        self._r_dd_cum      += r_dd
        self._r_churn_cum   += r_churn
        self._r_clipped_cum += reward

        terminated = False
        truncated = False
        if self.net_worth <= INITIAL_BALANCE * RUIN_THRESHOLD:
            terminated = True
            reward = -REWARD_CLIP

        self.current_step += 1
        self.steps_in_episode += 1
        if self.current_step >= self.df_len - 1:
            truncated = truncated or (not terminated)
        elif self.max_episode_steps is not None and self.steps_in_episode >= self.max_episode_steps:
            truncated = truncated or (not terminated)

        info = {
            "net_worth": self.net_worth,
            "price": price,
            "position": self.current_position,
            "asset": self.current_asset,
            "drawdown": drawdown,
            "leverage": leverage,
        }
        if terminated or truncated:
            info["ep_r_log_ret"] = self._r_log_ret_cum
            info["ep_r_dd"]      = self._r_dd_cum
            info["ep_r_churn"]   = self._r_churn_cum
            info["ep_r_clipped"] = self._r_clipped_cum
        return self._get_observation(), reward, terminated, truncated, info

    # ----------------------------------------------------------- transitions
    def _execute_transition(self, target: int, price: float, leverage: float) -> None:
        # CLOSE current position; fees on notional traded
        if self.current_position == 1:
            notional = self.position_size * price
            self.cash += notional - abs(notional) * TRADING_FEE
            self.position_size = 0.0
            self.current_position = 0
        elif self.current_position == -1:
            buyback_notional = self.position_size * price
            colateral = self.position_size * self.entry_price
            pnl = colateral - buyback_notional  # (entry - current) * size
            self.cash += colateral + pnl - abs(buyback_notional) * TRADING_FEE
            self.position_size = 0.0
            self.current_position = 0

        if target == 0:
            self.entry_price = 0.0
            return

        allocation = float(np.clip(leverage, LEVERAGE_MIN, MAX_ALLOCATION))
        investable = max(self.cash, 0.0) * allocation
        if investable < MIN_TRADE_USD:
            return

        size = (investable - investable * TRADING_FEE) / price
        self.cash -= investable
        self.position_size = size
        self.entry_price = price
        self.current_position = target
