import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_engine.observation import OBS_SIZE

class PokerNetwork(nn.Module):
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

       # Policy head
        policy_logits = self.policy_head(x)
        action_probs = torch.softmax(policy_logits, dim=-1)

        # Value head
        value = self.value_head(x)

        return action_probs, value

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias) 