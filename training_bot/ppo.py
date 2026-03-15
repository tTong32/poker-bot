import torch
import torch.nn as nn
import numpy as np
from typing import List

from .training_bot import Experience
from .network import PokerNetwork

CLIP_EPSILON    = 0.2      # max policy change per update step
VALUE_COEFF     = 0.5      # how much value loss contributes
ENTROPY_COEFF   = 0.05     # how much unpredictability is rewarded
GAMMA           = 0.999    # future reward discount
LAMBDA          = 0.95     # GAE smoothing factor
UPDATE_PASSES   = 4        # how many times to reuse each rollout
LEARNING_RATE   = 3e-4     # optimizer step size
REWARD_CLIP     = 10       # to maintain value_loss values

class PPOTrainer:
    def __init__(self, network: PokerNetwork):
        self.network = network
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )

    def _compute_advantages(self, experiences: List[Experience]):
        rewards = np.clip([e.reward for e in experiences], -REWARD_CLIP, REWARD_CLIP)
        dones   = np.array([e.done   for e in experiences])
        obs     = np.array([e.obs    for e in experiences])

        # Get value estimates for all observations
        with torch.no_grad():
            _, values = self.network(torch.FloatTensor(obs))
        values = values.squeeze(-1).numpy()

        # GAE calculation
        advantages = np.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                last_gae   = 0.0
            else:
                next_value = values[t + 1] if t + 1 < len(values) else 0.0

            delta      = rewards[t] + GAMMA * next_value - values[t]
            last_gae   = delta + GAMMA * LAMBDA * last_gae
            advantages[t] = last_gae

        returns = advantages + values   # raw, before normalising

        # Variance Explained:
        # Measures how well the value function predicts actual returns.
        # 1.0 = perfect prediction  |  0.0 = no better than baseline
        # Negative values indicate the value head is actively misleading.
        var_returns = np.var(returns)
        if var_returns > 1e-8:
            explained_variance = float(
                1.0 - np.var(returns - values) / var_returns
            )
        else:
            explained_variance = 0.0

        # Normalise advantages for stable gradient magnitude
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

        for _ in range(UPDATE_PASSES):
            loss, policy_loss, value_loss, entropy = self._compute_loss(
                experiences, advantages, returns
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            stats['policy_loss'] += policy_loss
            stats['value_loss']  += value_loss
            stats['entropy']     += entropy

        # Average the per-pass losses
        for key in ('policy_loss', 'value_loss', 'entropy'):
            stats[key] /= UPDATE_PASSES

        return stats