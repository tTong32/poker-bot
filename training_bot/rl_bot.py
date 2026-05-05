"""
Inference-only ``Bot`` that runs a ``PokerNetwork`` without training side-effects.

Used from the CLI/simulation wizards, ladder evaluation, and frozen opponents
loaded from ``SelfPlayPool`` checkpoints.

Does not append ``Experience`` rows or run backward passes — only masked softmax
(or greedy argmax) over legal actions.

Usage::

    bot = RLBot.from_checkpoint(
        player_id=0,
        checkpoint_path="runs/checkpoints/my_run/latest.pt",
    )
    bot = RLBot(player_id=0, network=net, starting_stack=1000.0)
    bot = RLBot(player_id=0, network=net, greedy=True)   # argmax vs sample
"""

import os
import torch
from typing import List

from poker_engine.bots import Bot
from poker_engine.action import Action, ActionType
from poker_engine.state import GameState
from poker_engine.observation import encode_observation
from .network import PokerNetwork


class RLBot(Bot):
    """
    Plays using a frozen PokerNetwork.

    Args:
        player_id:     Seat index (0–5).
        network:       A PokerNetwork instance (will be set to eval mode).
        starting_stack: Used for observation normalisation; updated each hand
                        by the caller when the stack changes between hands.
        name:          Display name shown in CLI and logs.
        greedy:        If True, picks argmax action instead of sampling.
                       Sampling is better during self-play; greedy is better
                       for final evaluation / human-facing play.
    """

    def __init__(
        self,
        player_id: int,
        network: PokerNetwork,
        starting_stack: float = 1000.0,
        name: str = "RLBot",
        greedy: bool = False,
    ):
        super().__init__(player_id, name=name)
        self.network = network
        self.network.eval()
        self.starting_stack = starting_stack
        self.greedy = greedy

    # Bot interface

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        obs = encode_observation(
            state=state,
            player_id=self.player_id,
            legal_actions=legal_actions,
            starting_stack=self.starting_stack,
        )
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)   # (1, OBS_SIZE)

        with torch.no_grad():
            _, _, policy_logits = self.network(obs_tensor)  # (1, 7) raw logits

        # Mask illegal actions in logit space before softmax
        mask = torch.zeros(len(ActionType))
        for a in legal_actions:
            mask[a.type.value] = 1.0
        logit_mask = (1.0 - mask) * -1e9
        masked_logits = policy_logits.squeeze(0) + logit_mask
        masked_probs = torch.softmax(masked_logits, dim=-1)

        if self.greedy:
            action_idx = int(masked_probs.argmax().item())
        else:
            dist       = torch.distributions.Categorical(probs=masked_probs)
            action_idx = int(dist.sample().item())

        chosen_type = ActionType(action_idx)
        for a in legal_actions:
            if a.type == chosen_type:
                return a
        return legal_actions[0]   # safety fallback

    def notify_hand_end(self, hand_result) -> None:
        new_stack = hand_result.stacks_after.get(self.player_id, self.starting_stack)
        if new_stack > 0:
            self.starting_stack = new_stack

    # Factory helpers 

    @classmethod
    def from_checkpoint(
        cls,
        player_id: int,
        checkpoint_path: str,
        starting_stack: float = 1000.0,
        name: str | None = None,
        greedy: bool = False,
    ) -> "RLBot":
        """
        Load ``network_state`` from a ``train.py`` checkpoint (``runs/checkpoints/.../*.pt``).

        Raises ``FileNotFoundError`` if the path does not exist.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ck = torch.load(checkpoint_path, weights_only=False)
        network = PokerNetwork()

        # Support both bare state-dicts (snapshots) and full checkpoints
        if isinstance(ck, dict) and "network_state" in ck:
            network.load_state_dict(ck["network_state"])
            display_name = name or f"RLBot(u{ck.get('update_num', '?')})"
        else:
            network.load_state_dict(ck)
            display_name = name or "RLBot"

        return cls(player_id, network, starting_stack, name=display_name, greedy=greedy)

    @classmethod
    def make_factory(
        cls,
        checkpoint_path: str,
        starting_stack: float = 1000.0,
        greedy: bool = False,
    ):
        """
        Return a factory function ``(seat: int) -> RLBot`` that shares one
        loaded network across all seats.  Pass this to
        ``session.fill_empty_seats_with()`` or the simulate/CLI setup wizards.

        The network is loaded once at factory-creation time, so multiple
        seats share the same weights without redundant disk reads.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ck = torch.load(checkpoint_path, weights_only=False)
        network = PokerNetwork()
        if isinstance(ck, dict) and "network_state" in ck:
            network.load_state_dict(ck["network_state"])
            update_label = ck.get("update_num", "?")
        else:
            network.load_state_dict(ck)
            update_label = "?"
        network.eval()

        def factory(seat: int) -> "RLBot":
            return cls(
                player_id=seat,
                network=network,
                starting_stack=starting_stack,
                name=f"RLBot(u{update_label})",
                greedy=greedy,
            )

        return factory