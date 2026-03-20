import torch
import torch.nn as nn
import numpy as np
from typing import List

from .training_bot import Experience
from .network import PokerNetwork

CLIP_EPSILON    = 0.2      # max policy change per update step
VALUE_COEFF     = 1.0      # how much value loss contributes
ENTROPY_COEFF   = 0.01     # how much unpredictability is rewarded
GAMMA           = 0.999    # future reward discount
LAMBDA          = 0.95     # GAE smoothing factor
VALUE_PASSES    = 2        # how many times to reuse each rollout for values
POLICY_PASSES   = 8        # how many times the value head gets passed
LEARNING_RATE   = 3e-4     # optimizer step size
REWARD_CLIP     = 3.0      # to maintain value_loss values
BUST_PENALTY_CLIP = 50.0   # lower bound for bust_penalty (already negative)

class PPOTrainer:
    def __init__(self, network: PokerNetwork):
        self.network = network
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )
        self._reward_clip = REWARD_CLIP

    def set_reward_clip(self, value: float):
        self._reward_clip = value

    def _compute_advantages(self, experiences: List[Experience]):
        # Split buffer into per-episode trajectories at every done boundary
        # Each segment is one player's hand
        trajectories = []
        current = []
        for exp in experiences:
            current.append(exp)
            if exp.done:
                trajectories.append(current)
                current = []
        if current:          # any trailing incomplete episode
            trajectories.append(current)

        all_advantages = []
        all_returns    = []

        for traj in trajectories:
            clipped = np.clip(
                [e.reward for e in traj],
                -self._reward_clip, self._reward_clip,
            ).astype(np.float32)
            bust    = np.clip(
                [e.bust_penalty for e in traj],
                -BUST_PENALTY_CLIP, 0.0,
            ).astype(np.float32)
            rewards = clipped + bust
            obs     = np.array([e.obs for e in traj])

            with torch.no_grad():
                _, values = self.network(torch.FloatTensor(obs))
            values = values.squeeze(-1).numpy()

            advantages = np.zeros(len(traj), dtype=np.float32)
            last_gae   = 0.0

            for t in reversed(range(len(traj))):
                # Always bootstrap from the next step WITHIN this trajectory only
                # The last step is always terminal (done=True), so next_value = 0
                next_value = values[t + 1] if t + 1 < len(traj) else 0.0
                delta      = rewards[t] + GAMMA * next_value - values[t]
                last_gae   = delta + GAMMA * LAMBDA * last_gae
                advantages[t] = last_gae

            returns = advantages + values
            all_advantages.append(advantages)
            all_returns.append(returns)

        advantages = np.concatenate(all_advantages)
        returns    = np.concatenate(all_returns)

        # Recompute values for all experiences in one batch for EV calculation
        obs_all = np.array([e.obs for e in experiences])
        with torch.no_grad():
            _, v_all = self.network(torch.FloatTensor(obs_all))
        v_all = v_all.squeeze(-1).numpy()

        var_returns = np.var(returns)
        explained_variance = float(
            1.0 - np.var(returns - v_all) / var_returns
        ) if var_returns > 1e-8 else 0.0

        # Normalise advantages across all trajectories together
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (    
            torch.FloatTensor(advantages),
            torch.FloatTensor(returns),
            explained_variance,
        )

    def _compute_loss(
        self,
        experiences: List[Experience],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple:
        obs       = torch.FloatTensor(np.array([e.obs      for e in experiences]))
        actions   = torch.LongTensor ([e.action             for e in experiences])
        old_probs = torch.FloatTensor([e.log_prob           for e in experiences])

        action_probs, values = self.network(obs)
        values = values.squeeze(-1)

        dist          = torch.distributions.Categorical(probs=action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy       = dist.entropy().mean()

        ratio          = torch.exp(new_log_probs - old_probs)
        clipped_ratio  = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
        policy_loss    = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        value_loss = nn.MSELoss()(values, returns)
        loss       = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

        return loss, policy_loss.item(), value_loss.item(), entropy.item()

    def update(self, experiences: List[Experience]) -> dict:
        advantages, returns, explained_variance = self._compute_advantages(experiences)

        stats = {
            'policy_loss':       0.0,
            'value_loss':        0.0,
            'entropy':           0.0,
            'explained_variance': explained_variance,   # computed once, before weight updates
        }

        obs_t = torch.FloatTensor(np.array([e.obs for e in experiences]))

        # Value-only passes first — get the value head working before policy updates
        for param in list(self.network.fc1.parameters()) + \
                    list(self.network.fc2.parameters()) + \
                    list(self.network.fc3.parameters()) + \
                    list(self.network.ln1.parameters()) + \
                    list(self.network.ln2.parameters()) + \
                    list(self.network.ln3.parameters()) + \
                    list(self.network.policy_head.parameters()):
            param.requires_grad = False

        for _ in range(VALUE_PASSES):
            _, values = self.network(obs_t)
            values     = values.squeeze(-1)
            value_loss = nn.MSELoss()(values, returns)

            self.optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.value_head.parameters(), max_norm=2.0
            )
            self.optimizer.step()
            stats['value_loss'] += value_loss.item()

        # Unfreeze backbone for policy passes
        for param in self.network.parameters():
            param.requires_grad = True

        stats['value_loss'] /= VALUE_PASSES

        # Policy passes after value head has warmed up
        for _ in range(POLICY_PASSES):
            loss, policy_loss, value_loss, entropy = self._compute_loss(
                experiences, advantages, returns
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            stats['policy_loss'] += policy_loss
            stats['entropy']     += entropy

        stats['policy_loss'] /= POLICY_PASSES
        stats['entropy']     /= POLICY_PASSES

        return stats