"""
observation.py — Numeric observation encoder for RL/CFR agents.

Converts a GameState into a fixed-size float32 numpy array from the
perspective of a single player (the observing player only sees their
own hole cards, not opponents').

Design decisions:
  Cards:         One-hot, 52 bits per card.  Sparse but lets the network
                 learn rank×suit interactions without inductive bias.
  Hole cards:    2 × 52 = 104 bits, sorted for order-invariance.
  Board cards:   5 × 52 = 260 bits, zero-padded for unseen streets.
  Pot / scalars: Normalised by starting_stack for stable scale.
  Position:      One-hot over 6 seats relative to dealer.
  Street:        One-hot over 4 streets.
  Action mask:   7-bit mask, one per ActionType.
  Hand strength: Preflop heuristic + postflop rank/8, injected as domain
                 knowledge to speed up early hand-selection learning.
"""

import numpy as np
from typing import List

from .state import GameState, BettingRound
from .action import Action, ActionType
from .card import Card


# Constants

NUM_PLAYERS   = 6
NUM_RANKS     = 13    # 2–A
NUM_SUITS     = 4
CARD_SIZE     = NUM_RANKS * NUM_SUITS   # 52 — one-hot per card
NUM_STREETS   = 4
NUM_ACTIONS   = 7

# ---------------------------------------------------------------------------
# Vector layout (all offsets are compile-time constants)
#
#   [0   : 104)   hole cards              (2 × 52)
#   [104 : 364)   board cards             (5 × 52)
#   [364 : 365)   pot                     (1 float, normalised)
#   [365 : 371)   stacks per seat         (6 floats, RELATIVE order)
#   [371 : 377)   current street bets     (6 floats, RELATIVE order)
#   [377 : 383)   total invested          (6 floats, RELATIVE order)
#   [383 : 389)   folded flags            (6 bits,   RELATIVE order)
#   [389 : 395)   all-in flags            (6 bits,   RELATIVE order)
#   [395 : 401)   position one-hot        (6 bits — seat relative to dealer)
#   [401 : 405)   street one-hot          (4 bits)
#   [405 : 406)   current bet             (1 float, normalised)
#   [406 : 407)   amount to call          (1 float, normalised)
#   [407 : 414)   legal action mask       (7 bits)
#   [414 : 456)   last action per player  (6 × 7, RELATIVE order)
#   [456 : 458)   hand strength           (preflop heuristic, postflop rank/8)
# ---------------------------------------------------------------------------

_HOLE_START     = 0
_HOLE_END       = 104
_BOARD_START    = 104
_BOARD_END      = 364
_POT            = 364
_STACKS_START   = 365
_STACKS_END     = 371
_BETS_START     = 371
_BETS_END       = 377
_INVESTED_START = 377
_INVESTED_END   = 383
_FOLDED_START   = 383
_FOLDED_END     = 389
_ALLIN_START    = 389
_ALLIN_END      = 395
_POS_START      = 395
_POS_END        = 401
_STREET_START   = 401
_STREET_END     = 405
_CURRENT_BET    = 405
_TO_CALL        = 406
_MASK_START     = 407
_MASK_END       = 414
_LAST_ACT_START = 414
_LAST_ACT_END   = 456   # 6 × 7
_STRENGTH_START = 456
_STRENGTH_END   = 458

OBS_SIZE = _STRENGTH_END   # 458

_STREET_INDEX = {
    BettingRound.PREFLOP: 0,
    BettingRound.FLOP:    1,
    BettingRound.TURN:    2,
    BettingRound.RIVER:   3,
}


# Card encoding helpers

def _card_index(card: Card) -> int:
    """Map a Card to 0–51.  rank 2=0 … A=12, suit 0–3."""
    return (card.rank - 2) * NUM_SUITS + card.suit


def _encode_cards(cards: List[Card], vec: np.ndarray, offset: int, max_cards: int):
    """
    Write up to max_cards cards as one-hot blocks starting at offset.
    Cards are sorted for order-invariance; unused slots stay zero.
    """
    for i, card in enumerate(sorted(cards, key=lambda c: (c.rank, c.suit))[:max_cards]):
        vec[offset + i * CARD_SIZE + _card_index(card)] = 1.0

# Public API

def encode_observation(
    state: GameState,
    player_id: int,
    legal_actions: List[Action],
    starting_stack: float,
) -> np.ndarray:
    """
    Return a float32 numpy array of shape (OBS_SIZE,) representing the game
    from player_id's perspective.

    Args:
        state:          Current GameState.
        player_id:      The observing player's seat index (0–5).
        legal_actions:  Output of game.get_legal_actions() for this player.
        starting_stack: Used for normalisation (kept stable across a session).

    Returns:
        obs: np.ndarray, dtype=float32, shape=(OBS_SIZE,)
    """
    obs  = np.zeros(OBS_SIZE, dtype=np.float32)
    norm = starting_stack

    me = state.players[player_id]

    # Hole cards (observer's cards only
    _encode_cards(me.hand, obs, _HOLE_START, 2)

    # Board cards
    _encode_cards(state.board, obs, _BOARD_START, 5)

    # Pot
    obs[_POT] = state.pot / norm

    # Per-player features in OBSERVER-RELATIVE seat order
    # offset 0 = observer (player_id)
    # offset 1 = next player clockwise
    # offset 2 = two seats clockwise
    # ...
    # This means every training seat presents its own features at offset 0
    # regardless of which physical seat it occupies, allowing the shared
    # network to learn one universal "my stack / my bet / ..." pattern.
    for offset in range(NUM_PLAYERS):
        seat = (player_id + offset) % NUM_PLAYERS
        p    = state.players[seat]
        obs[_STACKS_START   + offset] = p.stack          / norm
        obs[_BETS_START     + offset] = p.current_bet    / norm
        obs[_INVESTED_START + offset] = p.total_invested / norm
        obs[_FOLDED_START   + offset] = float(p.has_folded)
        obs[_ALLIN_START    + offset] = float(p.is_all_in)

    # Position one-hot (relative to dealer)
    # 0 = dealer button, 1 = SB, 2 = BB, 3 = UTG, 4 = UTG+1, 5 = UTG+2
    relative_pos = (player_id - state.dealer_index) % NUM_PLAYERS
    obs[_POS_START + relative_pos] = 1.0

    # Street one-hot
    obs[_STREET_START + _STREET_INDEX.get(state.betting_round, 0)] = 1.0

    # Current bet and amount to call
    obs[_CURRENT_BET] = state.current_bet / norm
    obs[_TO_CALL]     = max(0.0, state.current_bet - me.current_bet) / norm

    # Legal action mask
    for action in legal_actions:
        obs[_MASK_START + action.type.value] = 1.0

    # Last action per player
    # Provides a minimal betting-history signal without recurrence.
    # Using the same relative ordering as per-player features keeps the
    # network's spatial structure consistent.
    for pid, action_val in state.last_actions.items():
        if 0 <= action_val < NUM_ACTIONS:
            relative_seat = (pid - player_id) % NUM_PLAYERS
            obs[_LAST_ACT_START + relative_seat * NUM_ACTIONS + action_val] = 1.0

    # ---- Hand strength (domain knowledge injection) ----
    if me.hand:
        ranks  = sorted([c.rank for c in me.hand], reverse=True)
        suits  = [c.suit for c in me.hand]
        hi, lo = ranks[0], ranks[1]
        suited = suits[0] == suits[1]
        paired = hi == lo

        if paired:
            preflop_str = 0.5 + (hi - 2) / 24.0
        else:
            preflop_str = (hi - 2) / 24.0 * 0.6 + (lo - 2) / 24.0 * 0.3
            if suited:        preflop_str += 0.05
            if hi - lo <= 1:  preflop_str += 0.05
        obs[_STRENGTH_START] = float(min(preflop_str, 1.0))

        if len(state.board) >= 3:
            from .evaluator import evaluate_hand
            score = evaluate_hand(me.hand, state.board)
            obs[_STRENGTH_START + 1] = score[0] / 8.0

    return obs


def decode_action_mask(obs: np.ndarray) -> List[ActionType]:
    """Extract legal ActionTypes from an observation vector."""
    return [ActionType(i) for i, v in enumerate(obs[_MASK_START:_MASK_END]) if v > 0.5]


def obs_size() -> int:
    """Return the observation vector length.  Use this for your input-layer size."""
    return OBS_SIZE