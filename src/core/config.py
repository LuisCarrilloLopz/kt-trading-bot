"""
Centralized configuration for kt-trading-bot.

Rationale for each value is documented in EXPERIMENTS.md (V1.0).
"""
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs" / "tensorboard"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FINAL_DIR = MODEL_DIR / "final"

ASSETS: dict[str, Path] = {
    "BTC": DATA_DIR / "df_eval_btc.csv",
    "ETH": DATA_DIR / "df_eval_eth.csv",
}
TRAIN_YEAR_MAX = 2025
TRAIN_SPLIT = 0.80

# --- Environment ---
TRADING_FEE = 0.0005
INITIAL_BALANCE = 10_000.0
VOL_LOOKBACK = 30
TARGET_VOL_PER_STEP = 0.005
LEVERAGE_MIN = 0.1
LEVERAGE_MAX = 2.0
MAX_ALLOCATION = 0.99
MIN_TRADE_USD = 10.0
RUIN_THRESHOLD = 0.5
MAX_EPISODE_STEPS = 2000   # V1.2: 4000 → 2000 (velas horarias → ~3 meses por episodio)

# --- Reward ---
REWARD_LOG_RET_SCALE = 100.0
REWARD_DRAWDOWN_COEF = 20.0   # V1.1: applied to derivative DD (one-shot per new DD), not chronic. Coef raised to compensate.
REWARD_CHURN_COEF = 0.05
REWARD_CLIP = 10.0

# --- PPO ---
LEARNING_RATE = lambda p: 3e-4 * p
N_STEPS = 4096
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF_INITIAL = 0.05
ENT_COEF_FINAL = 0.005
VF_COEF = 1.0   # V1.2: 0.5 → 1.0, dar más peso al value loss para que el crítico siga el ritmo
MAX_GRAD_NORM = 0.5

# --- Recurrent policy ---
ENABLE_CRITIC_LSTM = True
LSTM_HIDDEN_SIZE = 256              # V1.2: 128 → 256
NET_ARCH = dict(pi=[64, 64], vf=[128, 128])   # V1.2: vf [64,64] → [128,128]

# --- Training ---
TOTAL_TIMESTEPS = 3_000_000
RANDOM_SEED = 42
N_EVAL_EPISODES = 5
