"""Behavioral Cloning pretraining for the actor (+actor LSTM) of RecurrentPPO.

Pipeline:
  1. Load fold-N train data (BTC + ETH).
  2. For each asset, roll the env following the teacher's discrete actions and
     record (obs, teacher_action) pairs. Critically the env runs realistically,
     so portfolio features (position, unrealized_pnl, drawdown) match the
     trajectory the BC-initialized policy will see later in PPO.
  3. Create a RecurrentPPO model with the same architecture as walk-forward
     (LSTM_HIDDEN_SIZE, NET_ARCH, ENABLE_CRITIC_LSTM). The env wrapping is the
     same DummyVecEnv+VecNormalize setup so that the saved checkpoint loads
     unchanged in T4 fine-tune.
  4. Supervised loop over the actor parameters only:
        - chunks of L=SEQ_LEN steps, hidden state carried across chunks
          (detached → truncated BPTT)
        - cross-entropy(action_logits, teacher_action)
        - Adam over: lstm_actor + mlp_extractor.policy_net + action_net
        - critic is left untouched; PPO trains it fresh in T4.

Acceptance:
  - Cross-entropy drops clearly across epochs (random ≈ ln(3) ≈ 1.10 → < 0.5).
  - Policy action distribution (deterministic inference on the train trajectory)
    matches the teacher labels within ±10pp per class.
  - If the policy collapses to >85% NEUTRAL → STOP (BC failed).

Output: bc_v1/fold_N/{bc_model.zip, vec_normalize.pkl, bc_acceptance.json}.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.core import config as cfg
from src.data.loader import load_assets
from src.envs.trading_env import MultiAssetTradingEnv
from src.bc_teacher import generate_labels, label_distribution


# ---------------------------------------------------------------------------
# Constants — keep here, not in config.py (these are BC-specific).
# ---------------------------------------------------------------------------
BC_RUN_NAME = "bc_v1"
SEQ_LEN = 256                  # truncated BPTT window
BC_EPOCHS = 8                  # spec: 5-10
BC_LR = 1e-3                   # higher than RL fine-tune; pure supervised
ACCEPTANCE_CE_MAX = 0.5
ACCEPTANCE_DIST_TOLERANCE = 0.10   # ±10pp per class vs teacher


# ---------------------------------------------------------------------------
# Trajectory collection — one env per asset to keep clean per-asset trajectories.
# ---------------------------------------------------------------------------
def collect_teacher_trajectory(df, asset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Roll the env following teacher actions, return (obs_array, action_array).

    `obs_array` shape (T, N_FEATURES); `action_array` shape (T,) int64.
    The trajectory ends at episode boundary (truncated at df_len-1 or terminated
    on ruin). Realistic portfolio features are baked into each obs.
    """
    labels = generate_labels(df)
    # max_episode_steps = full df length so the trajectory runs end-to-end,
    # not truncated at MAX_EPISODE_STEPS=2000 (env's PPO-rollout default).
    env = MultiAssetTradingEnv(
        {asset_name: df},
        randomize_start=False,
        max_episode_steps=len(df),
    )
    obs, _ = env.reset(seed=cfg.RANDOM_SEED)

    obs_list, act_list = [], []
    while True:
        # Decision at this step uses the teacher label for env.current_step.
        a = int(labels[env.current_step])
        obs_list.append(obs.copy())
        act_list.append(a)
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Build a RecurrentPPO model wrapped around the same env shape walk-forward uses.
# ---------------------------------------------------------------------------
def _make_env(dfs):
    return MultiAssetTradingEnv(dfs, randomize_start=True)


def build_model_and_env(train_dfs):
    env_train = DummyVecEnv([lambda: _make_env(train_dfs)])
    env_train = VecNormalize(
        env_train, norm_obs=False, norm_reward=True,
        clip_reward=cfg.REWARD_CLIP, gamma=cfg.GAMMA,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env_train,
        verbose=0,
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
        seed=cfg.RANDOM_SEED,
    )
    return model, env_train


# ---------------------------------------------------------------------------
# Supervised loop over actor params only.
# ---------------------------------------------------------------------------
def actor_param_list(policy):
    """Return the list of params we train (actor side only)."""
    return (
        list(policy.lstm_actor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )


def forward_chunk(policy, obs_chunk, h, c):
    """Forward a sequence chunk through actor LSTM + actor MLP + action head.
    Returns logits of shape (L, n_actions) and new (h, c).

    Reproduces what RecurrentActorCriticPolicy.get_distribution does, but
    keeping the returned LSTM states so we can pass them across chunks.
    """
    L = obs_chunk.shape[0]
    # Features extractor (FlattenExtractor — pass-through for vector obs).
    features = policy.extract_features(obs_chunk, policy.pi_features_extractor)
    episode_starts = th.zeros(L, dtype=th.float32)   # state already initialized; no per-step resets
    latent_pi, (h_new, c_new) = policy._process_sequence(
        features, (h, c), episode_starts, policy.lstm_actor
    )
    latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
    # _get_action_dist_from_latent → distribution; for Discrete it wraps a Categorical.
    dist = policy._get_action_dist_from_latent(latent_pi)
    logits = dist.distribution.logits   # (L, n_actions)
    return logits, h_new, c_new


def bc_train_loop(policy, obs_all, act_all, epochs: int = BC_EPOCHS,
                  seq_len: int = SEQ_LEN, lr: float = BC_LR) -> dict:
    """Truncated-BPTT supervised loop. Returns dict with per-epoch metrics."""
    optimizer = th.optim.Adam(actor_param_list(policy), lr=lr)

    obs_t = th.as_tensor(obs_all, dtype=th.float32)
    act_t = th.as_tensor(act_all, dtype=th.int64)
    T = obs_t.shape[0]
    n_chunks = T // seq_len
    print(f"  trajectory: T={T} steps  → {n_chunks} chunks of {seq_len}")

    metrics = {"epoch_ce": [], "epoch_acc": []}
    for epoch in range(epochs):
        h = th.zeros(1, 1, cfg.LSTM_HIDDEN_SIZE)
        c = th.zeros(1, 1, cfg.LSTM_HIDDEN_SIZE)
        epoch_losses, epoch_corrects, epoch_total = [], 0, 0
        for k in range(n_chunks):
            start = k * seq_len
            stop = start + seq_len
            obs_chunk = obs_t[start:stop]
            act_chunk = act_t[start:stop]
            logits, h_new, c_new = forward_chunk(policy, obs_chunk, h, c)
            loss = F.cross_entropy(logits, act_chunk)
            optimizer.zero_grad()
            loss.backward()
            # Match PPO's grad-clip convention so the policy stays well-conditioned.
            th.nn.utils.clip_grad_norm_(actor_param_list(policy), cfg.MAX_GRAD_NORM)
            optimizer.step()
            # Truncate BPTT — detach hidden state from graph for next chunk.
            h = h_new.detach()
            c = c_new.detach()
            epoch_losses.append(loss.item())
            epoch_corrects += int((logits.argmax(dim=-1) == act_chunk).sum().item())
            epoch_total    += int(act_chunk.shape[0])
        mean_ce = float(np.mean(epoch_losses))
        acc = epoch_corrects / max(1, epoch_total)
        metrics["epoch_ce"].append(mean_ce)
        metrics["epoch_acc"].append(acc)
        print(f"  epoch {epoch+1}/{epochs}  CE={mean_ce:.4f}  acc={acc*100:.1f}%")
    return metrics


# ---------------------------------------------------------------------------
# Acceptance: deterministic inference and class distribution check.
# ---------------------------------------------------------------------------
def predict_distribution(policy, obs_all) -> dict:
    """Run the policy deterministically over the full obs sequence and return
    a class-distribution dict {'long', 'neutral', 'short'} of predicted actions."""
    policy.eval()
    with th.no_grad():
        obs_t = th.as_tensor(obs_all, dtype=th.float32)
        T = obs_t.shape[0]
        # Single sequence through the LSTM, no chunking, no grad.
        h = th.zeros(1, 1, cfg.LSTM_HIDDEN_SIZE)
        c = th.zeros(1, 1, cfg.LSTM_HIDDEN_SIZE)
        logits, _, _ = forward_chunk(policy, obs_t, h, c)
        actions = logits.argmax(dim=-1).numpy()
    policy.train()
    n = len(actions)
    return {
        "n": int(n),
        "short":   float((actions == 0).sum() / n),
        "neutral": float((actions == 1).sum() / n),
        "long":    float((actions == 2).sum() / n),
    }


def acceptance(policy, obs_all, act_all, teacher_dist: dict, metrics: dict) -> tuple[bool, dict]:
    final_ce = metrics["epoch_ce"][-1]
    pred_dist = predict_distribution(policy, obs_all)
    diffs = {
        cls: abs(pred_dist[cls] - teacher_dist[cls])
        for cls in ("short", "neutral", "long")
    }
    max_diff = max(diffs.values())

    ce_ok      = final_ce < ACCEPTANCE_CE_MAX
    dist_ok    = max_diff < ACCEPTANCE_DIST_TOLERANCE
    no_collapse = pred_dist["neutral"] < 0.85
    overall = ce_ok and dist_ok and no_collapse

    out = {
        "final_ce": final_ce,
        "ce_threshold": ACCEPTANCE_CE_MAX,
        "ce_ok": ce_ok,
        "teacher_dist": teacher_dist,
        "policy_dist": pred_dist,
        "max_class_diff": max_diff,
        "dist_tolerance": ACCEPTANCE_DIST_TOLERANCE,
        "dist_ok": dist_ok,
        "no_collapse_ok": no_collapse,
        "overall_pass": overall,
    }
    return overall, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=9,
                        help="Walk-forward fold index (1..len(FOLDS)).")
    parser.add_argument("--epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--lr", type=float, default=BC_LR)
    args = parser.parse_args()

    if not (1 <= args.fold <= len(cfg.FOLDS)):
        parser.error(f"--fold must be 1..{len(cfg.FOLDS)}")
    spec = cfg.FOLDS[args.fold - 1]
    train_range = (spec[0], spec[1])
    run_name = f"{BC_RUN_NAME}/fold_{args.fold}"

    print(f"--- BC pretrain — fold {args.fold}  train: {train_range[0]} → {train_range[1]} ---")
    set_random_seed(cfg.RANDOM_SEED)

    train_dfs = load_assets(date_range=train_range)
    print(f"train sizes: { {k: len(v) for k, v in train_dfs.items()} }")

    # Collect trajectories per asset, then concat (LSTM treats each segment in order
    # with state continuity; episode boundary between assets is one "reset" — we
    # handle this by re-initializing the hidden state for each new asset segment
    # via the chunk loop carrying h/c across chunks but starting from zero each epoch).
    print()
    print("[1] Collecting teacher trajectories...")
    obs_segments, act_segments = [], []
    teacher_aggregate = {"n": 0, "short": 0, "neutral": 0, "long": 0}
    for asset, df in train_dfs.items():
        obs_seg, act_seg = collect_teacher_trajectory(df, asset)
        obs_segments.append(obs_seg)
        act_segments.append(act_seg)
        n = len(act_seg)
        teacher_aggregate["n"] += n
        teacher_aggregate["short"]   += int((act_seg == 0).sum())
        teacher_aggregate["neutral"] += int((act_seg == 1).sum())
        teacher_aggregate["long"]    += int((act_seg == 2).sum())
        print(f"  {asset}: {n} steps  L/N/S = "
              f"{(act_seg==2).mean()*100:5.1f}% / "
              f"{(act_seg==1).mean()*100:5.1f}% / "
              f"{(act_seg==0).mean()*100:5.1f}%")

    obs_all = np.concatenate(obs_segments, axis=0)
    act_all = np.concatenate(act_segments, axis=0)
    teacher_dist = {
        "short":   teacher_aggregate["short"]   / teacher_aggregate["n"],
        "neutral": teacher_aggregate["neutral"] / teacher_aggregate["n"],
        "long":    teacher_aggregate["long"]    / teacher_aggregate["n"],
    }
    print(f"  combined: {len(act_all)} steps")

    # Initial CE baseline before any training.
    print()
    print("[2] Building RecurrentPPO model (same arch as walk-forward)...")
    model, env_train = build_model_and_env(train_dfs)
    policy = model.policy
    n_actor_params = sum(p.numel() for p in actor_param_list(policy))
    print(f"  actor params (trained): {n_actor_params:,}")

    initial_pred = predict_distribution(policy, obs_all)
    print(f"  initial policy dist:  L/N/S = "
          f"{initial_pred['long']*100:5.1f}% / "
          f"{initial_pred['neutral']*100:5.1f}% / "
          f"{initial_pred['short']*100:5.1f}%")

    # Train.
    print()
    print(f"[3] BC training — {args.epochs} epochs, seq_len={args.seq_len}, lr={args.lr}")
    t0 = time.time()
    metrics = bc_train_loop(policy, obs_all, act_all,
                            epochs=args.epochs, seq_len=args.seq_len, lr=args.lr)
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s")

    # Acceptance.
    print()
    print("[4] Acceptance check")
    passed, report = acceptance(policy, obs_all, act_all, teacher_dist, metrics)
    print(f"  CE final  = {report['final_ce']:.4f}   (threshold < {ACCEPTANCE_CE_MAX}) → {'OK' if report['ce_ok'] else 'FAIL'}")
    print(f"  teacher  L/N/S = {teacher_dist['long']*100:5.1f}% / {teacher_dist['neutral']*100:5.1f}% / {teacher_dist['short']*100:5.1f}%")
    print(f"  policy   L/N/S = {report['policy_dist']['long']*100:5.1f}% / {report['policy_dist']['neutral']*100:5.1f}% / {report['policy_dist']['short']*100:5.1f}%")
    print(f"  max class diff = {report['max_class_diff']*100:.1f}pp   (tol ±{ACCEPTANCE_DIST_TOLERANCE*100:.0f}pp) → {'OK' if report['dist_ok'] else 'FAIL'}")
    print(f"  no NEUTRAL collapse → {'OK' if report['no_collapse_ok'] else 'FAIL'}")
    print()
    print("  ✅ ACCEPTANCE PASS" if passed else "  ❌ ACCEPTANCE FAIL")

    # Persist.
    print()
    print("[5] Saving artifacts")
    out_dir = cfg.FINAL_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "bc_model.zip"
    vecnorm_path = out_dir / "vec_normalize.pkl"
    model.save(str(model_path))
    env_train.save(str(vecnorm_path))

    acceptance_path = out_dir / "bc_acceptance.json"
    payload = {
        "fold": args.fold,
        "train_start": spec[0], "train_end": spec[1],
        "bc_run_name": BC_RUN_NAME,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "elapsed_seconds": elapsed,
        "trajectory_steps": int(len(act_all)),
        "actor_params_trained": n_actor_params,
        "epoch_ce":  metrics["epoch_ce"],
        "epoch_acc": metrics["epoch_acc"],
        "initial_policy_dist": initial_pred,
        "acceptance": report,
        "model_path": str(model_path),
        "vec_normalize_path": str(vecnorm_path),
    }
    with open(acceptance_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  model → {model_path}")
    print(f"  vec_normalize → {vecnorm_path}")
    print(f"  bc_acceptance.json → {acceptance_path}")
    print()
    print("--- BC pretrain complete ---")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
