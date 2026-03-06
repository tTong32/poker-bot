"""
Bot interface for Texas Hold'em AI players.

To create a new bot, subclass Bot and implement choose_action().
The method receives the full GameState and the list of legal actions,
and must return one of those actions.

Included bots:
  - RandomBot       — picks uniformly at random (baseline / fuzz tester)
  - CallBot         — always calls/checks, never folds or raises (calling station)
  - TightBot        — folds weak hands preflop, bets strong hands aggressively
  - PositionBot     — like TightBot but loosens up in late position
"""

import random
from abc import ABC, abstractmethod
from typing import List

from .action import Action, ActionType
from .state import GameState, BettingRound
from .evaluator import evaluate_hand

# Abstract base
class Bot(ABC):
    """
    All bots implement this interface. The engine calls choose_action() on
    each bot's turn; bots must return one of the provided legal_actions.
    """

    def __init__(self, player_id: int, name: str = "Bot"):
        self.player_id = player_id
        self.name = name

    @abstractmethod
    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """
        Select and return one action from legal_actions.

        Args:
            state:         Current GameState (read-only — do not mutate).
            legal_actions: Non-empty list of valid Action objects for this turn.

        Returns:
            One of the Action objects from legal_actions.
        """
        ...

    # Helpers
    def _my_player(self, state: GameState):
        return state.players[self.player_id]

    def _get_action(self, legal_actions: List[Action], *preferred: ActionType) -> Action:
        """Return the first preferred action found, else the last legal action."""
        for atype in preferred:
            for a in legal_actions:
                if a.type == atype:
                    return a
        return legal_actions[-1]

    def _has_action(self, legal_actions: List[Action], atype: ActionType) -> bool:
        return any(a.type == atype for a in legal_actions)

    def __repr__(self):
        return f"{self.name}(id={self.player_id})"


class RandomBot(Bot):
    """Picks a random legal action every turn. Good for stress-testing."""

    def __init__(self, player_id: int, seed: int | None = None):
        super().__init__(player_id, name="RandomBot")
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        return self._rng.choice(legal_actions)


class CallBot(Bot):
    """Always calls or checks — never raises, rarely folds. Calling station."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="CallBot")

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        return self._get_action(legal_actions, ActionType.CHECK, ActionType.CALL)


# Hand-strength helpers (used by smarter bots)

# Preflop hand strength buckets based on hole cards.
# Returns a score 0.0–1.0 (higher = stronger).
def _preflop_strength(hole_cards, player_id: int) -> float:
    ranks = sorted([c.rank for c in hole_cards], reverse=True)
    suits = [c.suit for c in hole_cards]
    suited = suits[0] == suits[1]
    paired = ranks[0] == ranks[1]
    hi, lo = ranks[0], ranks[1]
    gap = hi - lo

    if paired:
        # Pair value: AA=1.0, 22~0.5
        return 0.5 + (hi - 2) / 24

    score = (hi - 2) / 24 * 0.6 + (lo - 2) / 24 * 0.3
    if suited:
        score += 0.05
    if gap <= 1:
        score += 0.05   # connected
    return min(score, 1.0)


def _postflop_strength(hole_cards, board) -> float:
    """Rough postflop strength: hand rank / 8, normalised to 0–1."""
    if len(board) < 3:
        return 0.5
    score = evaluate_hand(hole_cards, board)
    return score[0] / 8.0


class TightBot(Bot):
    """
    A simple rule-based bot with reasonable default strategy:
      - Preflop: plays only top ~40% of hands, raises strong ones
      - Postflop: bets made hands (pair+), folds weak holdings to aggression
    """

    PREFLOP_CALL_THRESHOLD = 0.38    # minimum hand strength to call
    PREFLOP_RAISE_THRESHOLD = 0.65   # minimum strength to raise

    def __init__(self, player_id: int, seed: int | None = None):
        super().__init__(player_id, name="TightBot")
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        me = self._my_player(state)
        board = state.board

        if state.betting_round == BettingRound.PREFLOP:
            return self._preflop(state, legal_actions, me)
        else:
            return self._postflop(state, legal_actions, me, board)

    def _preflop(self, state, legal_actions, me):
        strength = _preflop_strength(me.hand, me.id)

        if strength >= self.PREFLOP_RAISE_THRESHOLD:
            # Strong hand — raise if possible, else call
            return self._get_action(
                legal_actions,
                ActionType.BET_POT,
                ActionType.BET_HALF_POT,
                ActionType.CALL,
                ActionType.CHECK,
            )

        if strength >= self.PREFLOP_CALL_THRESHOLD:
            # Playable hand — call or check
            return self._get_action(
                legal_actions,
                ActionType.CHECK,
                ActionType.CALL,
            )

        # Weak hand — check if free, otherwise fold
        if self._has_action(legal_actions, ActionType.CHECK):
            return self._get_action(legal_actions, ActionType.CHECK)
        return self._get_action(legal_actions, ActionType.FOLD)

    def _postflop(self, state, legal_actions, me, board):
        strength = _postflop_strength(me.hand, board)
        # strength is score[0]/8 where score[0] is hand rank 0-8:
        #   0.000 = high card,  0.125 = pair,   0.250 = two pair,
        #   0.375 = trips,      0.500 = straight, 0.625 = flush,
        #   0.750 = full house, 0.875 = quads,   1.000 = str. flush

        if strength >= 0.375:     # trips or better — pot-size bet/raise
            return self._get_action(
                legal_actions,
                ActionType.BET_POT,
                ActionType.BET_HALF_POT,
                ActionType.CALL,
                ActionType.CHECK,
            )

        if strength >= 0.250:     # two pair — half-pot bet, call raises
            return self._get_action(
                legal_actions,
                ActionType.BET_HALF_POT,
                ActionType.BET_POT,
                ActionType.CALL,
                ActionType.CHECK,
            )

        if strength >= 0.125:     # one pair — lead half pot, call one raise
            return self._get_action(
                legal_actions,
                ActionType.BET_HALF_POT,
                ActionType.CALL,
                ActionType.CHECK,
            )

        # High card — check if free, fold to any bet
        if self._has_action(legal_actions, ActionType.CHECK):
            return self._get_action(legal_actions, ActionType.CHECK)
        return self._get_action(legal_actions, ActionType.FOLD)


class PositionBot(TightBot):
    """
    Extends TightBot with positional awareness:
      - Loosens preflop thresholds in late position (closer to dealer button)
      - Bluffs occasionally on the river when last to act
    """

    BLUFF_RATE = 0.15   # probability of bluffing on river in position

    def __init__(self, player_id: int, seed: int | None = None):
        super().__init__(player_id, seed=seed)
        self.name = "PositionBot"

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        me = self._my_player(state)
        position_bonus = self._position_bonus(state)

        # Temporarily loosen thresholds based on position
        orig_call = self.PREFLOP_CALL_THRESHOLD
        orig_raise = self.PREFLOP_RAISE_THRESHOLD
        self.PREFLOP_CALL_THRESHOLD -= position_bonus * 0.12
        self.PREFLOP_RAISE_THRESHOLD -= position_bonus * 0.08

        action = super().choose_action(state, legal_actions)

        # Restore thresholds
        self.PREFLOP_CALL_THRESHOLD = orig_call
        self.PREFLOP_RAISE_THRESHOLD = orig_raise

        # River bluff when in late position and we'd normally check/fold
        if (state.betting_round == BettingRound.RIVER
                and position_bonus > 0.5
                and action.type in (ActionType.CHECK, ActionType.FOLD)
                and self._rng.random() < self.BLUFF_RATE):
            bet = self._get_action(legal_actions, ActionType.BET_HALF_POT)
            if bet.type == ActionType.BET_HALF_POT:
                return bet

        return action

    def _position_bonus(self, state: GameState) -> float:
        """
        Returns 0.0 (early position) to 1.0 (dealer/button).
        Computed as how far clockwise we are from UTG relative to total players.
        """
        n = len(state.players)
        dealer = state.dealer_index
        me = self.player_id
        # Distance from dealer (0 = dealer, n-1 = UTG+1)
        distance_from_dealer = (dealer - me) % n
        # Late position = small distance from dealer
        return 1.0 - (distance_from_dealer / n)