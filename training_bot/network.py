"""Shared actor–critic MLP: ``OBS_SIZE``-dim poker observations → policy + value."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_engine.observation import OBS_SIZE


class PokerNetwork(nn.Module):
    """LayerNorm MLP with seven-way policy logits and scalar ``V(s)``."""

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(OBS_SIZE, 256)
        self.ln1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)

        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        self.policy_head = nn.Linear(128, 7)

        self.value_head = nn.Linear(128, 1)

        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        # return policy logits too so callers can apply
        # action masking before softmax, not after
        policy_logits = self.policy_head(x)
        action_probs = torch.softmax(policy_logits, dim=-1)

        # Value head
        value = self.value_head(x)

        return action_probs, value, policy_logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.value_head or module is self.policy_head:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.policy_head.weight, -0.03, 0.03)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.uniform_(self.value_head.weight, -0.03, 0.03)
        nn.init.zeros_(self.value_head.bias)