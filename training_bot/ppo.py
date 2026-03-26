import torch
import torch.nn as nn
import numpy as np
from typing import List
from collections import deque

from .training_bot import Experience
from .network import PokerNetwork

CLIP_EPSILON    = 0.2
VALUE_COEFF     = 0.5
ENTROPY_COEFF   = 0.05
GAMMA           = 0.999
POLICY_PASSES   = 4
MINI_BATCH_SIZE = 512       # experiences per gradient step (was: entire buffer at once)
POLICY_LR       = 3e-4
VALUE_LR        = 1e-3
NORM_WINDOW     = 50_000


class RunningNormalizer:
    """Maintains running mean and std of returns for normalization."""
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


class PPOTrainer:
    def __init__(self, network: PokerNetwork):
        self.network        = network
        self._entropy_coeff = ENTROPY_COEFF
        self._reward_clip   = None   # None = no clipping; set via set_reward_clip()

        backbone_params = (
            list(network.fc1.parameters()) +
            list(network.fc2.parameters()) +
            list(network.fc3.parameters()) +
            list(network.ln1.parameters()) +
            list(network.ln2.parameters()) +
            list(network.ln3.parameters())
        )
        self.policy_optimizer = torch.optim.Adam(
            backbone_params + list(network.policy_head.parameters()),
            lr=POLICY_LR,
            weight_decay=1e-4,
        )
        self.value_optimizer = torch.optim.Adam(
            list(network.value_head.parameters()),
            lr=VALUE_LR,
            weight_decay=1e-4,
        )
        self.normalizer = RunningNormalizer()

    def set_entropy_coeff(self, value: float):
        self._entropy_coeff = value

    def set_reward_clip(self, value: float):
        self._reward_clip = value

    def _compute_mc_returns(self, experiences: List[Experience]) -> tuple:
        """
        Pure Monte Carlo returns for each trajectory.

        """
        trajectories, current = [], []
        for exp in experiences:
            current.append(exp)
            if exp.done:
                trajectories.append(current)
                current = []
        if current:
            trajectories.append(current)

        obs_all = torch.FloatTensor(np.array([e.obs for e in experiences]))
        with torch.no_grad():
            _, values_all, _ = self.network(obs_all)
        values_all = values_all.squeeze(-1).numpy()

        all_raw_returns = []
        for traj in trajectories:
            n               = len(traj)
            terminal_reward = traj[-1].reward

            # Apply reward clipping on the raw reward before normalization.
            # This was previously a no-op because set_reward_clip() did nothing.
            if self._reward_clip is not None:
                terminal_reward = float(np.clip(terminal_reward,
                                                -self._reward_clip,
                                                 self._reward_clip))

            mc_returns = np.array(
                [GAMMA ** (n - 1 - t) * terminal_reward for t in range(n)],
                dtype=np.float32,
            )
            all_raw_returns.append(mc_returns)

        raw_returns = np.concatenate(all_raw_returns)   # shape (N,)

        self.normalizer.update(raw_returns)
        norm_returns = self.normalizer.normalize(raw_returns)
        norm_values  = self.normalizer.normalize(values_all)

        advantages = norm_returns - norm_values

        var_ret = np.var(norm_returns)
        ev = (float(1.0 - np.var(norm_returns - norm_values) / var_ret)
              if var_ret > 1e-8 else 0.0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            torch.FloatTensor(advantages),
            torch.FloatTensor(norm_returns),
            ev,
        )

    def _compute_loss(self, obs_t, actions_t, old_log_probs_t, advantages_t, norm_returns_t):
        """
        Compute PPO loss for one mini-batch.
        Accepts pre-stacked tensors (indexed slices of the full rollout)
        so mini-batch selection is O(1) rather than rebuilding numpy arrays.
        """
        action_probs, values, _ = self.network(obs_t)
        values = values.squeeze(-1)

        dist          = torch.distributions.Categorical(probs=action_probs)
        new_log_probs = dist.log_prob(actions_t)
        entropy       = dist.entropy().mean()

        ratio       = torch.exp(new_log_probs - old_log_probs_t)
        clipped     = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
        policy_loss = -torch.min(ratio * advantages_t, clipped * advantages_t).mean()

        value_loss  = nn.MSELoss()(values, norm_returns_t)
        loss        = policy_loss + VALUE_COEFF * value_loss - self._entropy_coeff * entropy

        return loss, policy_loss.item(), value_loss.item(), entropy.item()

    def update(self, experiences: List[Experience]) -> dict:
        advantages, norm_returns, explained_variance = self._compute_mc_returns(experiences)

        # Pre-stack the full buffer once; mini-batch loops just index into tensors.
        obs_t           = torch.FloatTensor(np.array([e.obs      for e in experiences]))
        actions_t       = torch.LongTensor ([e.action             for e in experiences])
        old_log_probs_t = torch.FloatTensor([e.log_prob           for e in experiences])

        n = len(experiences)
        stats = {
            'policy_loss':        0.0,
            'value_loss':         0.0,
            'entropy':            0.0,
            'explained_variance': explained_variance,
        }

        total_passes = 0

        for _ in range(POLICY_PASSES):
            perm = torch.randperm(n)

            pass_pl = pass_vl = pass_ent = 0.0
            num_batches = 0

            for start in range(0, n, MINI_BATCH_SIZE):
                idx = perm[start : start + MINI_BATCH_SIZE]
                if len(idx) < 4:
                    continue  # skip degenerate tail batches

                loss, pl, vl, ent = self._compute_loss(
                    obs_t[idx],
                    actions_t[idx],
                    old_log_probs_t[idx],
                    advantages[idx],
                    norm_returns[idx],
                )

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                pass_pl  += pl
                pass_vl  += vl
                pass_ent += ent
                num_batches += 1

            if num_batches > 0:
                stats['policy_loss'] += pass_pl  / num_batches
                stats['value_loss']  += pass_vl  / num_batches
                stats['entropy']     += pass_ent / num_batches
                total_passes += 1

        if total_passes > 0:
            stats['policy_loss'] /= total_passes
            stats['value_loss']  /= total_passes
            stats['entropy']     /= total_passes

        return stats