# kt-trading-bot

PPO + LSTM trading bot (BTC + ETH multi-asset). Fresh iteration of
`trading-bot-project`, with the bugs and design issues found in the V2.3 review fixed.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full list of changes vs. V2.3 and the rationale behind each one.

## Structure

```
src/
├── main_train.py                       # Training entrypoint
├── core/config.py                      # All hyperparameters and paths
├── data/loader.py                      # Feature engineering, train/eval split
├── envs/trading_env.py                 # MultiAssetTradingEnv
├── callbacks/
│   ├── sharpe_eval_callback.py         # best_model by Sharpe (not reward)
│   ├── ent_coef_schedule.py            # 0.05 → 0.005 linear decay
│   └── checkpoint_callback.py          # Model + VecNormalize checkpoints
└── utils/metrics.py                    # Sharpe / Sortino / MDD / total return
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.main_train --run-name v1
tensorboard --logdir=logs/tensorboard
```

## Multi-seed sweep

```bash
make train-seeds          # seeds 13, 42, 77
# or
python -m src.main_train --seed 77 --run-name seed-77
```

## Key TensorBoard metrics

- `eval/sharpe_ratio` — primary metric, **used to pick `best_model`**
- `eval/sharpe_std` — across the N=5 eval episodes; if high → unstable strategy
- `eval/total_return` — % portfolio return per episode
- `eval/max_drawdown` — worst peak-to-trough on the equity curve
- `eval/mean_reward` — kept for compatibility; not the selection metric

## Data

CSVs live in `data/processed/`. The loader builds:
`dist_ma200, bb_pos, rsi_norm, vol_rel, log_ret, market_vol` from
the raw OHLCV+indicators columns. `market_vol` is pre-computed (rolling
30-period std of `log_ret`) to keep `step()` O(1).
