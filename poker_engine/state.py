from dataclasses import dataclass, field
from typing import List, Optional
from .player import Player
from .card import Card


class BettingRound:
    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"
    SHOWDOWN = "SHOWDOWN"


@dataclass
class GameState:
    players: List[Player]
    dealer_index: int          # index of the dealer button
    current_player_index: int  # index of the player whose turn it is
    board: List[Card] = field(default_factory=list)
    pot: float = 0.0
    betting_round: str = BettingRound.PREFLOP
    current_bet: float = 0.0   # highest bet on the table this street
    last_raise_size: float = 0.0
    raise_count: int = 0       # number of raises this street
    players_acted: set = field(default_factory=set)  # ids who have acted this street
    terminal: bool = False
    winners: List[int] = field(default_factory=list)  # player ids

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    @property
    def active_players(self) -> List[Player]:
        """Players still in the hand (not folded, has chips or is all-in from this hand)."""
        return [p for p in self.players if not p.has_folded and (p.stack > 0 or p.is_all_in)]

    @property
    def players_who_can_act(self) -> List[Player]:
        """Active players who are not all-in."""
        return [p for p in self.active_players if not p.is_all_in]

    def __repr__(self):
        board_str = " ".join(str(c) for c in self.board) or "(none)"
        return (
            f"[{self.betting_round}] Board: {board_str} | "
            f"Pot: {self.pot:.1f} | "
            f"To act: Player {self.current_player_index}"
        )