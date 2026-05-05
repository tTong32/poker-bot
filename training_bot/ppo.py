"""
ppo.py — PPO trainer with Generalized Advantage Estimation (GAE).

Hyperparameter summary
----------------------
  GAMMA            0.999  discount factor
  LAMBDA           0.95   GAE smoothing (0 = pure TD, 1 = pure MC)
  CLIP_EPSILON     0.2    PPO surrogate clip range
  VALUE_COEFF      0.5    value loss weight in joint loss
  ENTROPY_COEFF    0.05   starting entropy bonus (decayed by train.py)
  POLICY_PASSES    4      joint policy+value passes per rollout
  VALUE_EXTRA_PASSES 4    additional value-only passes per rollout
  MINI_BATCH_SIZE  512    experiences per gradient step
  POLICY_LR        3e-4   backbone + policy head LR
  VALUE_LR         1e-3   value head LR (also extra-pass backbone LR)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from collections import deque

from .training_bot import Experience
from .network import PokerNetwork

# Hyperparameters

CLIP_EPSILON      = 0.1
VALUE_COEFF       = 0.5
ENTROPY_COEFF     = 0.05
GAMMA             = 0.999
LAMBDA            = 0.7        # GAE smoothing factor
POLICY_PASSES     = 4
VALUE_EXTRA_PASSES= 4           # value-only passes on top of joint passes
MINI_BATCH_SIZE   = 512
POLICY_LR         = 5e-5
VALUE_LR          = 1e-3
NORM_WINDOW       = 50_000
ENTROPY_FLOOR     = 0.01        # absolute minimum entropy coefficient


# Running return normalizer (unchanged)

class RunningNormalizer:
    """Maintains a running mean/std over a sliding window of return values."""

    def __init__(self, window: int = NORM_WINDOW):
        self._buf  = deque(maxlen=window)
        self._mean = 0.0
        self._std  = 1.0

    def update(self, values: np.ndarray):
        self._buf.extend(values.tolist())
        if len(self._buf) > 10:
            arr        = np.array(self._buf, dtype=np.float32)
            self._mean = float(arr.mean())
            self._std  = float(arr.std()) + 1e-8

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self._mean) / self._std

    @property
    def mean(self): return self._mean

    @property
    def std(self):  return self._std

    def state_dict(self) -> dict:
        return {"buf": list(self._buf), "mean": self._mean, "std": self._std}

    def load_state_dict(self, d: dict):
        self._buf  = deque(d.get("buf", []), maxlen=NORM_WINDOW)
        self._mean = d.get("mean", 0.0)
        self._std  = d.get("std", 1.0)


# PPO Trainer

class PPOTrainer:

    def __init__(self, network: PokerNetwork):
        self.network        = network
        self._entropy_coeff = ENTROPY_COEFF
        self._reward_clip: Optional[float] = None

        # Backbone parameter list (shared between heads)
        backbone_params = (
            list(network.fc1.parameters()) + list(network.ln1.parameters()) +
            list(network.fc2.parameters()) + list(network.ln2.parameters()) +
            list(network.fc3.parameters()) + list(network.ln3.parameters())
        )

        # Single optimizer with explicit per-group learning rates.

        self.optimizer = torch.optim.Adam([
            {"params": backbone_params,
             "lr": POLICY_LR, "weight_decay": 1e-4},
            {"params": list(network.policy_head.parameters()),
             "lr": POLICY_LR, "weight_decay": 1e-4},
            {"params": list(network.value_head.parameters()),
             "lr": VALUE_LR,  "weight_decay": 1e-4},
        ])

        # Separate optimizer for the extra value-only passes.
        # Trains backbone + value head at VALUE_LR so the backbone learns
        # value-useful representations on top of the joint passes.
        self.value_optimizer = torch.optim.Adam(
            list(network.value_head.parameters()),
            lr=VALUE_LR, weight_decay=1e-4,
        )

        self.normalizer = RunningNormalizer()

    # Public setters (called by train.py)

    def set_entropy_coeff(self, value: float):
        self._entropy_coeff = max(value, ENTROPY_FLOOR)

    def set_reward_clip(self, value: Optional[float]):
        """Clip terminal rewards to ±value BB before GAE computation."""
        self._reward_clip = value

    # GAE advantage estimation

    def _compute_gae(
        self, experiences: List[Experience]
    ) -> tuple:
        """
        Compute GAE advantages and normalised returns from stored behavior-
        policy values.

        Using behavior-policy values for bootstrapping (rather than
        recomputing with the current network) is standard PPO practice
        (cf. stable-baselines3).  The slight off-policy bias is acceptable
        and saves one full forward pass over the entire buffer.

        Returns:
            advantages_t    FloatTensor (N,)  — normalised to μ=0, σ=1
            norm_returns_t  FloatTensor (N,)  — normalised returns (value targets)
            explained_var   float             — EV of behavior-policy value estimates
        """
        n = len(experiences)

        values      = np.array([e.value      for e in experiences], dtype=np.float32)
        next_values = np.array([e.next_value for e in experiences], dtype=np.float32)
        rewards     = np.array([e.reward     for e in experiences], dtype=np.float32)
        dones       = np.array([float(e.done) for e in experiences], dtype=np.float32)

        # Clip terminal rewards if requested (intermediate rewards are all 0)
        if self._reward_clip is not None:
            rewards = np.clip(rewards, -self._reward_clip, self._reward_clip)

        # GAE — computed backwards through the flattened experience list.
        # The (1 - dones[t]) factor zeroes out the carry-over gae at every
        # hand boundary, so interleaved trajectories are handled correctly
        # without needing to split the buffer into per-hand chunks first.
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            # δt = rt + γ·V(st+1)·(1−done) − V(st)
            delta = rewards[t] + GAMMA * next_values[t] * (1.0 - dones[t]) - values[t]
            # GAE carry: reset at episode boundaries via (1−done)
            gae = delta + GAMMA * LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae

        # Returns = advantages + values  →  targets for the value head
        returns = advantages + values

        # Normalise returns for consistent value loss scale across stages
        self.normalizer.update(returns)
        norm_returns  = self.normalizer.normalize(returns)
        norm_values   = self.normalizer.normalize(values)

        # Explained variance — computed BEFORE advantage normalisation so it
        # reflects how well the BEHAVIOR policy value estimates track returns
        var_ret = float(np.var(norm_returns))
        if var_ret > 1e-8:
            explained_var = 1.0 - float(np.var(norm_returns - norm_values)) / var_ret
        else:
            explained_var = 0.0

        # Normalise advantages to μ=0, σ=1 for stable policy gradient updates
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return (
            torch.FloatTensor(advantages),
            torch.FloatTensor(norm_returns),
            explained_var,
        )

    # Loss computation (single forward pass, shared graph)

    def _compute_loss(
        self,
        obs_t:          torch.Tensor,
        actions_t:      torch.Tensor,
        old_log_probs_t:torch.Tensor,
        advantages_t:   torch.Tensor,
        norm_returns_t: torch.Tensor,
    ):
        action_probs, values, _ = self.network(obs_t)
        values = values.squeeze(-1)

        dist          = torch.distributions.Categorical(probs=action_probs)
        new_log_probs = dist.log_prob(actions_t)
        entropy       = dist.entropy().mean()

        ratio   = torch.exp(new_log_probs - old_log_probs_t)
        clipped = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
        policy_loss = -torch.min(ratio * advantages_t, clipped * advantages_t).mean()
        value_loss  = nn.MSELoss()(values, norm_returns_t)

        return policy_loss, value_loss, entropy

    # Main update

    def update(self, experiences: List[Experience]) -> dict:
        """
        Run POLICY_PASSES joint policy+value passes, then VALUE_EXTRA_PASSES
        value-only passes.  Returns a stats dict for logging.
        """
        advantages, norm_returns, explained_variance = self._compute_gae(experiences)

        # Pre-stack tensors — indexing into these is O(1) per mini-batch
        obs_t           = torch.FloatTensor(np.array([e.obs      for e in experiences]))
        actions_t       = torch.LongTensor ([e.action             for e in experiences])
        old_log_probs_t = torch.FloatTensor([e.log_prob           for e in experiences])
        n = len(experiences)

        stats = {
            "policy_loss":        0.0,
            "value_loss":         0.0,
            "entropy":            0.0,
            "explained_variance": explained_variance,
        }
        total_passes = 0

        # Joint policy + value passes
        for _ in range(POLICY_PASSES):
            perm = torch.randperm(n)
            pass_pl = pass_vl = pass_ent = 0.0
            num_batches = 0

            for start in range(0, n, MINI_BATCH_SIZE):
                idx = perm[start : start + MINI_BATCH_SIZE]
                if len(idx) < 4:
                    continue

                pl, vl, ent = self._compute_loss(
                    obs_t[idx], actions_t[idx], old_log_probs_t[idx],
                    advantages[idx], norm_returns[idx],
                )

                loss = pl + VALUE_COEFF * vl - self._entropy_coeff * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                pass_pl  += pl.item()
                pass_vl  += vl.item()
                pass_ent += ent.item()
                num_batches += 1

            if num_batches > 0:
                stats["policy_loss"] += pass_pl  / num_batches
                stats["value_loss"]  += pass_vl  / num_batches
                stats["entropy"]     += pass_ent / num_batches
                total_passes += 1

        if total_passes > 0:
            stats["policy_loss"] /= total_passes
            stats["value_loss"]  /= total_passes
            stats["entropy"]     /= total_passes

        for _ in range(VALUE_EXTRA_PASSES):
            perm = torch.randperm(n)
            for start in range(0, n, MINI_BATCH_SIZE):
                idx = perm[start : start + MINI_BATCH_SIZE]
                if len(idx) < 4:
                    continue

                _, values, _ = self.network(obs_t[idx])
                values   = values.squeeze(-1)
                vl_extra = nn.MSELoss()(values, norm_returns[idx])

                self.value_optimizer.zero_grad()
                vl_extra.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.value_optimizer.step()

        return stats