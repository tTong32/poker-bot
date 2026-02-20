"""
Hand evaluator for Texas Hold'em.

Returns a comparable tuple: (hand_rank, tiebreakers...)
Higher is better. Hand ranks:
  8 = Straight Flush
  7 = Four of a Kind
  6 = Full House
  5 = Flush
  4 = Straight
  3 = Three of a Kind
  2 = Two Pair
  1 = One Pair
  0 = High Card
"""

from itertools import combinations
from typing import List, Tuple
from .card import Card


HandScore = Tuple


def evaluate_hand(hole_cards: List[Card], board: List[Card]) -> HandScore:
    """Return the best 5-card HandScore from hole cards + board."""
    all_cards = hole_cards + board
    if len(all_cards) < 5:
        raise ValueError("Need at least 5 cards to evaluate.")
    best = max(_score_five(combo) for combo in combinations(all_cards, 5))
    return best


def _score_five(cards: Tuple[Card, ...]) -> HandScore:
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _check_straight(ranks)

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    # Sort groups: first by count desc, then by rank desc (for tiebreaking)
    groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    group_counts = [g[1] for g in groups]
    group_ranks = [g[0] for g in groups]

    if is_straight and is_flush:
        return (8, straight_high)

    if group_counts[0] == 4:
        return (7, group_ranks[0], group_ranks[1])

    if group_counts[:2] == [3, 2]:
        return (6, group_ranks[0], group_ranks[1])

    if is_flush:
        return (5, *ranks)

    if is_straight:
        return (4, straight_high)

    if group_counts[0] == 3:
        return (3, group_ranks[0], *group_ranks[1:])

    if group_counts[:2] == [2, 2]:
        top_pair, bot_pair = max(group_ranks[:2]), min(group_ranks[:2])
        kicker = group_ranks[2]
        return (2, top_pair, bot_pair, kicker)

    if group_counts[0] == 2:
        return (1, group_ranks[0], *group_ranks[1:])

    return (0, *ranks)


def _check_straight(sorted_ranks: List[int]) -> Tuple[bool, int]:
    unique = sorted(set(sorted_ranks), reverse=True)
    # Normal straight
    for i in range(len(unique) - 4):
        window = unique[i:i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return True, window[0]
    # Wheel: A-2-3-4-5
    if set([14, 2, 3, 4, 5]).issubset(set(sorted_ranks)):
        return True, 5
    return False, 0


def hand_rank_name(score: HandScore) -> str:
    names = {
        8: "Straight Flush",
        7: "Four of a Kind",
        6: "Full House",
        5: "Flush",
        4: "Straight",
        3: "Three of a Kind",
        2: "Two Pair",
        1: "One Pair",
        0: "High Card",
    }
    return names[score[0]]