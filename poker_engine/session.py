"""
GameSession: orchestrates multi-hand poker sessions.

Responsibilities:
  - Assigns bots to player seats
  - Runs hands end-to-end, calling bot.choose_action() for bot seats
    and pausing for human input on human seats
  - Rotates the dealer button between hands
  - Eliminates busted players (stack == 0)
  - Records per-hand results for later analysis
  - Detects the end of the session (one player left, or hand limit reached)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from .game import PokerGame, NUM_PLAYERS
from .bots import Bot
from .player import Player
from .action import Action, ActionType
from .state import GameState
from .evaluator import hand_rank_name, evaluate_hand


# --------------------------------------------------------------------------- #
# Result records                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class HandResult:
    hand_number: int
    winner_ids: List[int]
    pot: float
    board: list
    stacks_after: Dict[int, float]   # player_id -> stack


@dataclass
class SessionResult:
    total_hands: int
    final_stacks: Dict[int, float]
    hand_results: List[HandResult]
    session_winner_id: Optional[int]   # None if session ended by hand limit


# --------------------------------------------------------------------------- #
# GameSession                                                                  #
# --------------------------------------------------------------------------- #

class GameSession:
    """
    Runs a full session of Texas Hold'em.

    Usage:
        session = GameSession(starting_stack=1000)
        session.assign_bot(0, TightBot(0))
        session.assign_bot(1, PositionBot(1))
        # seat 2 is a human player — GameSession will call human_input_fn
        session.set_human_seat(2)
        result = session.run(max_hands=100)
    """

    def __init__(
        self,
        starting_stack: float = 1000.0,
        seed: int | None = None,
        on_state_change: Optional[Callable[[GameState], None]] = None,
        on_hand_end: Optional[Callable[[HandResult], None]] = None,
    ):
        self.game = PokerGame(starting_stack=starting_stack, seed=seed)
        self.bots: Dict[int, Bot] = {}          # player_id -> Bot
        self.human_seats: set = set()           # player ids that are human
        self.hand_results: List[HandResult] = []
        self.hand_number = 0

        # Optional hooks (used by CLI / UI layers)
        self.on_state_change = on_state_change  # called after every action
        self.on_hand_end = on_hand_end          # called after each hand

        # Human input function: (state, legal_actions) -> Action
        # Override this to plug in a UI. Default raises (must be set for human seats).
        self.human_input_fn: Optional[Callable[[GameState, List[Action]], Action]] = None

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def assign_bot(self, seat: int, bot: Bot):
        """Assign a Bot to a seat (0–5)."""
        if not (0 <= seat < NUM_PLAYERS):
            raise ValueError(f"Seat must be 0–{NUM_PLAYERS - 1}.")
        self.bots[seat] = bot

    def set_human_seat(self, seat: int):
        """Mark a seat as human-controlled."""
        self.human_seats.add(seat)

    def fill_empty_seats_with(self, bot_factory: Callable[[int], Bot]):
        """Fill any unassigned, non-human seats with bots from a factory."""
        for i in range(NUM_PLAYERS):
            if i not in self.bots and i not in self.human_seats:
                self.bots[i] = bot_factory(i)

    # ------------------------------------------------------------------ #
    # Running                                                              #
    # ------------------------------------------------------------------ #

    def run(self, max_hands: int = 200) -> SessionResult:
        """
        Play up to max_hands hands. Returns a SessionResult.
        Stops early if only one player has chips left.
        """
        while self.hand_number < max_hands:
            active = [p for p in self.game.players if p.stack > 0]
            if len(active) <= 1:
                break

            result = self._play_one_hand()
            self.hand_results.append(result)

            if self.on_hand_end:
                self.on_hand_end(result)

            self.game.rotate_dealer()

        # Determine session winner (last player standing, or chip leader)
        final_stacks = {p.id: p.stack for p in self.game.players}
        alive = [p for p in self.game.players if p.stack > 0]
        session_winner = alive[0].id if len(alive) == 1 else None

        return SessionResult(
            total_hands=self.hand_number,
            final_stacks=final_stacks,
            hand_results=self.hand_results,
            session_winner_id=session_winner,
        )

    def run_one_hand(self) -> HandResult:
        """Play exactly one hand and return its result. Useful for step-by-step UIs."""
        result = self._play_one_hand()
        self.hand_results.append(result)
        self.game.rotate_dealer()
        return result
        
    def _play_one_hand(self) -> HandResult:
        self.hand_number += 1
        state = self.game.start_new_hand()

        if self.on_state_change:
            self.on_state_change(state)

        while not state.terminal:
            pid = state.current_player_index
            legal = self.game.get_legal_actions()

            if pid in self.human_seats:
                action = self._get_human_action(state, legal)
            elif pid in self.bots:
                action = self.bots[pid].choose_action(state, legal)
            else:
                # Unassigned seat defaults to check/call
                action = _default_action(legal)

            state = self.game.apply_action(action)

            if self.on_state_change:
                self.on_state_change(state)

        return HandResult(
            hand_number=self.hand_number,
            winner_ids=state.winners,
            pot=state.pot,
            board=list(state.board),
            stacks_after={p.id: p.stack for p in self.game.players},
        )

    def _get_human_action(self, state: GameState, legal: List[Action]) -> Action:
        if self.human_input_fn is None:
            raise RuntimeError(
                "Human seat requires human_input_fn to be set on the session."
            )
        return self.human_input_fn(state, legal)


def _default_action(legal: List[Action]) -> Action:
    """Fallback for unassigned seats: check if possible, else call."""
    for a in legal:
        if a.type == ActionType.CHECK:
            return a
    for a in legal:
        if a.type == ActionType.CALL:
            return a
    return legal[0]