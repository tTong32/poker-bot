"""
training_bot.py — A bot with a pluggable neural network policy for RL training.

The bot is AI-agnostic: it doesn't care whether the policy comes from a
neural network, CFR lookup table, or anything else. The network is passed
in at construction time

Usage:
    network = PokerNetwork()
    shared_buffer = []

    bots = [TrainingBot(seat, network, shared_buffer) for seat in range(6)]

    session = GameSession(starting_stack=1000)
    for bot in bots:
        session.assign_bot(bot.player_id, bot)

    # After each hand, call finish_hand() on all bots
    session.on_hand_end = lambda result: [bot.finish_hand(result) for bot in bots]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from poker_engine.action import Action, ActionType
from poker_engine.bots import Bot
from poker_engine.observation import encode_observation
from poker_engine.state import GameState
import torch    

@dataclass
class Experience:
    obs: np.ndarray                    # 414-element observation vector
    action: int                        # ActionType value (0-6)
    log_prob: float                    # log probability of action at time of decision
    reward: float = 0.0                # filled in retroactively at hand end
    done: bool = False                 # set to True on the last experience of a hand
    bust_penalty: float = 0.0          # Penalty for completely busting



class TrainingBot(Bot):
    """
    A bot whose decisions are made by a shared neural network.

    Six instances share one network — each sees only its own perspective
    (own hole cards, own stack) but feeds experience into a single buffer
    that PPO trains on.

    The training loop is responsible for calling finish_hand() after each
    hand ends — this is wired up via session.on_hand_end in train.py.
    """

    def __init__(
        self,
        player_id: int,
        network: "PokerNetwork",
        shared_buffer: List[Experience],
        starting_stack: float = 1000.0,
        big_blind: float = 10.0,
    ):
        super().__init__(player_id, name=f"TrainingBot_{player_id}")
        self.network = network
        self.shared_buffer = shared_buffer
        self.big_blind = big_blind

        # Per-hand state
        self.starting_stack = starting_stack
        self.hand_experiences: List[Experience] = []


    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """
        Encode the current state, pass through the network, sample an action,
        record the experience, and return the chosen action.
        """
        # Encode observation from this bot's perspective
        obs = encode_observation(
            state=state,
            player_id=self.player_id,
            legal_actions=legal_actions,
            starting_stack=self.starting_stack,
        )

        # Forward pass through network — get action probabilities
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)   # (1, 414)
        with torch.no_grad():
            action_probs, _ = self.network(obs_tensor)    # (1, 7), (1, 1)

        # Build legal action mask and apply it
        mask = torch.zeros(len(ActionType))
        for a in legal_actions:
            mask[a.type.value] = 1.0
        masked_probs = action_probs.squeeze(0) * mask
        
        # Renormalize after masking
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)

        # Sample action from distribution
        distribution = torch.distributions.Categorical(probs=masked_probs)
        action_idx = distribution.sample()
        log_prob = distribution.log_prob(action_idx).item()
        chosen_type = ActionType(action_idx.item())

        # Record experience (reward and done filled in at hand end)

        # Reduce the values of all_in (but winning will still reward the bot)
        # immediate_reward = -0.3 if chosen_type == ActionType.ALL_IN else 0.0
        immediate_reward = 0

        exp = Experience(
            obs=obs,
            action=action_idx.item(),
            log_prob=log_prob,
            reward=immediate_reward,
        )
        self.hand_experiences.append(exp)

        # Return the matching legal action
        for a in legal_actions:
            if a.type == chosen_type:
                return a

        # Fallback — should never happen if mask is correct
        return legal_actions[0]


    def finish_hand(self, hand_result) -> None:
        """
        Called at the end of each hand by the training loop.

        Calculates the reward as chip delta normalised by big blind,
        assigns it to all experiences from this hand, marks the last
        experience as done, and flushes to the shared buffer.
        """
        if not self.hand_experiences:
            return

        # Chip delta normalised by big blind    
        current_stack = hand_result.stacks_after.get(self.player_id, self.starting_stack)
        chip_delta = current_stack - self.starting_stack
        reward = chip_delta / self.big_blind

        # went_bust = (current_stack == 0) #
        # bust_penalty = -50.0 if went_bust else 0.0 #
        bust_penalty = 0

        # Assign reward to all experiences this hand, mark last as done
        self.hand_experiences[-1].reward      += reward
        self.hand_experiences[-1].done         = True
        self.hand_experiences[-1].bust_penalty = bust_penalty

        # Flush to shared buffer
        self.shared_buffer.extend(self.hand_experiences)

        # Reset for next hand
        self.hand_experiences = []

    def reset_starting_stack(self, stack: float) -> None:
        """Call at the start of a new session to reset the stack reference point."""
        self.starting_stack = stack
        self.hand_experiences = []
