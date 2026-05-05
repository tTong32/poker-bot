"""Player stacks, bets, and eligibility flags used by ``PokerGame``."""

from dataclasses import dataclass, field
from typing import List
from .card import Card


@dataclass
class Player:
    id: int
    stack: float
    hand: List[Card] = field(default_factory=list)
    has_folded: bool = False
    is_all_in: bool = False
    current_bet: float = 0.0   # amount committed THIS street
    total_invested: float = 0.0  # amount committed THIS hand (for side-pot logic)
    human_player: bool = False

    def reset_for_new_hand(self):
        self.hand.clear()
        self.has_folded = False
        self.is_all_in = False
        self.current_bet = 0.0
        self.total_invested = 0.0

    def reset_for_new_street(self):
        self.current_bet = 0.0

    @property
    def is_active(self) -> bool:
        """Player can still act (not folded, not all-in, has chips)."""
        return not self.has_folded and not self.is_all_in and self.stack > 0

    def __repr__(self):
        return (f"Player(id={self.id}, stack={self.stack:.1f}, "
                f"bet={self.current_bet:.1f}, folded={self.has_folded})")