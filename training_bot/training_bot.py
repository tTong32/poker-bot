"""
training_bot.py — A bot with a pluggable neural network policy for RL training.

"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Set
from poker_engine.action import Action, ActionType
from poker_engine.bots import Bot
from poker_engine.observation import encode_observation
from poker_engine.state import GameState
import torch


@dataclass
class Experience:
    obs: np.ndarray                # OBS_SIZE-element observation vector
    action: int                    # ActionType value (0–6)
    log_prob: float                # log π(a|s) at decision time
    value: float = 0.0             # V(st) from behavior policy
    next_value: float = 0.0        # V(st+1); set on next decision, or 0 at done
    reward: float = 0.0            # terminal chip delta / BB (all intermediate = 0)
    done: bool = False             # True only on the last experience of a hand


class TrainingBot(Bot):
    """
    A bot whose decisions are driven by a shared neural network.

    Six instances share one network.  Each sees only its own perspective
    and feeds experiences into a single buffer that PPO trains on.

    Caller must invoke finish_hand() after each hand ends — wired via
    session.on_hand_end in train.py.
    """

    def __init__(
        self,
        player_id: int,
        network,
        shared_buffer: List[Experience],
        starting_stack: float = 1000.0,
        big_blind: float = 10.0,
    ):
        super().__init__(player_id, name=f"TrainingBot_{player_id}")
        self.network = network
        self.shared_buffer = shared_buffer
        self.big_blind = big_blind

        self.starting_stack = starting_stack
        self.hand_experiences: List[Experience] = []
        self.restricted_actions: Set[ActionType] = set()

        self._last_betting_round = None 
        self._street_start_pot   = 0.0
        self._street_start_invested = 0.0

    # Bot interface

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:

        if (self.hand_experiences
                and self._last_betting_round is not None
                and state.betting_round != self._last_betting_round):
            
            me = state.players[self.player_id]
            pot_now = state.pot
            my_invested_now = me.total_invested

            # How much did opponents put in this street vs how much I put in
            total_in_street = pot_now - self._street_start_pot
            i_put_in        = my_invested_now - self._street_start_invested
            opponents_put_in = total_in_street - i_put_in

            # Positive if I got others to put in more than me (built pot profitably)
            # Negative if I called/called and opponents didn't match
            street_reward = (opponents_put_in - i_put_in) / self.big_blind
            street_reward = max(-1.0, min(1.0, street_reward * 0.1))  # small scale + clip

            self.hand_experiences[-1].reward += street_reward

            # Update street tracking
            self._street_start_pot      = pot_now
            self._street_start_invested = my_invested_now

        self._last_betting_round = state.betting_round
        # Apply action-restriction curriculum
        if self.restricted_actions:
            filtered = [a for a in legal_actions if a.type not in self.restricted_actions]
            if filtered:
                legal_actions = filtered

        obs = encode_observation(
            state=state,
            player_id=self.player_id,
            legal_actions=legal_actions,
            starting_stack=self.starting_stack,
        )

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)   # (1, OBS_SIZE)
        with torch.no_grad():
            _, value_t, policy_logits = self.network(obs_tensor)

        value_scalar = value_t.squeeze().item()
        
        if self.hand_experiences:
            self.hand_experiences[-1].next_value = value_scalar

        # Build legal-action mask and sample
        mask = torch.zeros(len(ActionType))
        for a in legal_actions:
            mask[a.type.value] = 1.0
        logit_mask = (1.0 - mask) * -1e9
        masked_logits = policy_logits.squeeze(0) + logit_mask
        masked_probs  = torch.softmax(masked_logits, dim=-1)

        dist        = torch.distributions.Categorical(probs=masked_probs)
        action_idx  = dist.sample()
        log_prob    = dist.log_prob(action_idx).item()
        chosen_type = ActionType(action_idx.item())

        exp = Experience(
            obs=obs,
            action=action_idx.item(),
            log_prob=log_prob,
            value=value_scalar,
            # next_value will be filled on the next call, or 0 at hand end
        )
        self.hand_experiences.append(exp)

        for a in legal_actions:
            if a.type == chosen_type:
                return a
        return legal_actions[0]   # safety fallback

    # End-of-hand reward assignment

    def finish_hand(self, hand_result) -> None:
        """
        Called at the end of each hand by the training loop.

        Assigns the terminal chip-delta reward (normalised by BB) to the
        last experience, marks it done, zeroes next_value (episode boundary),
        and flushes everything to the shared buffer.
        """
        if not self.hand_experiences:
            return

        current_stack = hand_result.stacks_after.get(self.player_id, self.starting_stack)
        chip_delta    = current_stack - self.starting_stack
        reward        = chip_delta / self.big_blind

        last = self.hand_experiences[-1]
        last.reward     += reward
        last.done       = True
        last.next_value = 0.0   # V(terminal) = 0 by definition

        self.shared_buffer.extend(self.hand_experiences)
        self.hand_experiences = []

        self._last_betting_round    = None
        self._street_start_pot      = 0.0
        self._street_start_invested = 0.0

    def reset_starting_stack(self, stack: float) -> None:
        """Call at the start of a new session to reset the stack reference point."""
        self.starting_stack  = stack
        self.hand_experiences = []

        self._last_betting_round    = None
        self._street_start_pot      = 0.0
        self._street_start_invested = 0.0 