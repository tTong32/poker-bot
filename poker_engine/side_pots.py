"""
Side pot calculator for Texas Hold'em.

When players go all-in for different amounts, the pot must be split into
separate pots, each with a defined set of eligible winners.

Example:
  Player A invested 100 (all-in)
  Player B invested 300 (all-in)
  Player C invested 300

  Main pot:  3 x 100 = 300  -> A, B, C eligible
  Side pot:  2 x 200 = 400  -> B, C eligible only

Key rule: if a pot layer has no eligible winners (everyone who contributed
at that level folded), those chips are rolled up into the next layer rather
than disappearing. This preserves chip conservation.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SidePot:
    amount: float
    eligible_player_ids: List[int]

    def __repr__(self):
        return f"SidePot(amount={self.amount:.1f}, eligible={self.eligible_player_ids})"


def calculate_side_pots(players) -> List[SidePot]:
    contributors = [p for p in players if p.total_invested > 0]
    non_folded_ids = {p.id for p in players if not p.has_folded}
    original_invested = {p.id: p.total_invested for p in contributors}

    if not contributors:
        return []

    remaining = {p.id: p.total_invested for p in contributors}
    active_ids = [p.id for p in contributors]

    result = []
    carry = 0.0

    while any(v > 0 for v in remaining.values()):
        cap = min(v for v in remaining.values() if v > 0)

        pot_amount = carry
        carry = 0.0
        for pid in active_ids:
            contribution = min(remaining[pid], cap)
            pot_amount += contribution
            remaining[pid] -= contribution

        eligible = [
            pid for pid in active_ids
            if pid in non_folded_ids and original_invested[pid] >= cap
        ]

        if eligible:
            result.append(SidePot(amount=pot_amount, eligible_player_ids=eligible))
        else:
            carry += pot_amount

        active_ids = [pid for pid in active_ids if remaining[pid] > 0]

    if carry > 0:
        if result:
            last = result[-1]
            result[-1] = SidePot(
                amount=last.amount + carry,
                eligible_player_ids=last.eligible_player_ids,
            )
        elif non_folded_ids:
            result.append(SidePot(amount=carry, eligible_player_ids=list(non_folded_ids)))

    return result