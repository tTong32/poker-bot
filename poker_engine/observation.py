"""
observation.py — Numeric observation encoder for RL/CFR agents.

Converts a GameState into a fixed-size float32 numpy array from the
perspective of a single player (the observing player only sees their
own hole cards, not opponents').

Design decisions (each is a deliberate choice you may want to revisit):

  Cards:          One-hot encoded. Each card = 52 bits (13 ranks × 4 suits).
                  Alternative: [rank/13, suit/3] = 2 floats. One-hot is
                  sparser but lets the network learn suit/rank interactions
                  without inductive bias.

  Hole cards:     2 × 52 = 104 bits. Order-invariant (sorted by rank then
                  suit before encoding) so 3♠ 7♥ and 7♥ 3♠ produce the
                  same vector.

  Board cards:    5 × 52 = 260 bits (padded with zeros for unseen streets).
                  Again order-invariant (sorted).

  Pot / stacks:   Normalised by starting_stack so values stay in [0, ~2].
                  Using starting_stack rather than current max-stack keeps
                  the scale stable across a session.

  Betting history: Per street, per player: how much they've committed this
                  street, normalised. This lets the agent infer aggression
                  patterns without needing sequence modelling.

  Position:       One-hot over 6 seats relative to dealer (0 = dealer,
                  1 = SB, 2 = BB, 3 = UTG, ...). Positional encoding is
                  critical for any serious strategy.

  Street:         One-hot over 4 streets (preflop/flop/turn/river).

  Action validity: 7-bit mask, one per ActionType. Lets the agent zero out
                  invalid actions in its policy head without a separate call.

Total vector length: see OBS_SIZE below.
"""

import numpy as np
from typing import List

from .state import GameState, BettingRound
from .action import Action, ActionType
from .card import Card


# constants

NUM_PLAYERS   = 6
NUM_RANKS     = 13    # 2–A
NUM_SUITS     = 4
CARD_SIZE     = NUM_RANKS * NUM_SUITS   # 52 — one-hot per card
NUM_STREETS   = 4     # preflop, flop, turn, river
NUM_ACTIONS   = 7     # matches ActionType enum (0–6)

# Vector layout (all offsets computed at import time):
#   [0   : 104)   hole cards        (2 cards × 52)
#   [104 : 364)   board cards       (5 cards × 52)
#   [364 : 365)   pot               (1 float, normalised)
#   [365 : 371)   player stacks     (6 floats, normalised)
#   [371 : 377)   current street bets (6 floats, normalised)
#   [377 : 383)   total invested    (6 floats, normalised)
#   [383 : 389)   folded flags      (6 bits)
#   [389 : 395)   all-in flags      (6 bits)
#   [395 : 401)   position one-hot  (6 bits — seat relative to dealer)
#   [401 : 405)   street one-hot    (4 bits)
#   [405 : 406)   current bet       (1 float, normalised)
#   [406 : 407)   amount to call    (1 float, normalised)
#   [407 : 414)   legal action mask (7 bits)

_HOLE_START    = 0
_HOLE_END      = 104
_BOARD_START   = 104
_BOARD_END     = 364
_POT           = 364
_STACKS_START  = 365
_STACKS_END    = 371
_BETS_START    = 371
_BETS_END      = 377
_INVESTED_START= 377
_INVESTED_END  = 383
_FOLDED_START  = 383
_FOLDED_END    = 389
_ALLIN_START   = 389
_ALLIN_END     = 395
_POS_START     = 395
_POS_END       = 401
_STREET_START  = 401
_STREET_END    = 405
_CURRENT_BET   = 405
_TO_CALL       = 406
_MASK_START    = 407
_MASK_END      = 414

OBS_SIZE = _MASK_END   # 414

_STREET_INDEX = {
    BettingRound.PREFLOP: 0,
    BettingRound.FLOP:    1,
    BettingRound.TURN:    2,
    BettingRound.RIVER:   3,
}

_ACTION_INDEX = {a: a.value for a in ActionType}


# CARD ENCODING

def _card_index(card: Card) -> int:
    """Map a Card to an index 0–51. rank 2=0 … A=12, suit 0–3."""
    return (card.rank - 2) * NUM_SUITS + card.suit


def _encode_cards(cards: List[Card], vec: np.ndarray, offset: int, max_cards: int):
    """
    Write up to max_cards cards as one-hot blocks into vec starting at offset.
    Cards are sorted for order-invariance before encoding.
    Unused slots remain zero-padded.
    """
    sorted_cards = sorted(cards, key=lambda c: (c.rank, c.suit))
    for i, card in enumerate(sorted_cards[:max_cards]):
        slot = offset + i * CARD_SIZE
        vec[slot + _card_index(card)] = 1.0


# API

def encode_observation(state: GameState,
                        player_id: int,
                        legal_actions: List[Action],
                        starting_stack: float) -> np.ndarray:
    """
    Return a float32 numpy array of shape (OBS_SIZE,) representing the game
    from player_id's perspective.

    Args:
        state:          Current GameState.
        player_id:      The observing player's seat index (0–5).
        legal_actions:  Output of game.get_legal_actions() for this player.
        starting_stack: The game's starting stack (used for normalisation).

    Returns:
        obs: np.ndarray, dtype=float32, shape=(OBS_SIZE,)
    """
    obs  = np.zeros(OBS_SIZE, dtype=np.float32)
    norm = starting_stack   # normalisation denominator

    me = state.players[player_id]

    # ---- Hole cards (only own cards visible) ----
    _encode_cards(me.hand, obs, _HOLE_START, 2)

    # ---- Board cards ----
    _encode_cards(state.board, obs, _BOARD_START, 5)

    # ---- Pot ----
    obs[_POT] = state.pot / norm

    # ---- Per-player features (indexed by seat, NOT relative to observer) ----
    # Keeping absolute seat order is simplest for a fixed-size network.
    # If you want permutation-invariant inputs later, reindex relative to player_id.
    for p in state.players:
        i = p.id
        obs[_STACKS_START   + i] = p.stack        / norm
        obs[_BETS_START     + i] = p.current_bet  / norm
        obs[_INVESTED_START + i] = p.total_invested / norm
        obs[_FOLDED_START   + i] = float(p.has_folded)
        obs[_ALLIN_START    + i] = float(p.is_all_in)

    # ---- Position (one-hot, relative to dealer) ----
    # 0 = dealer, 1 = SB, 2 = BB, 3 = UTG, 4 = UTG+1, 5 = UTG+2
    relative_pos = (player_id - state.dealer_index) % NUM_PLAYERS
    obs[_POS_START + relative_pos] = 1.0

    # ---- Street (one-hot) ----
    street_idx = _STREET_INDEX.get(state.betting_round, 0)
    obs[_STREET_START + street_idx] = 1.0

    # ---- Current bet and amount to call ----
    obs[_CURRENT_BET] = state.current_bet / norm
    to_call = max(0.0, state.current_bet - me.current_bet)
    obs[_TO_CALL] = to_call / norm

    # ---- Legal action mask ----
    for action in legal_actions:
        obs[_MASK_START + action.type.value] = 1.0

    return obs


def decode_action_mask(obs: np.ndarray) -> List[ActionType]:
    """
    Extract the list of legal ActionTypes from an observation vector.
    Useful for policy networks that output logits over all 7 actions.
    """
    mask = obs[_MASK_START:_MASK_END]
    return [ActionType(i) for i, v in enumerate(mask) if v > 0.5]


def obs_size() -> int:
    """Return the observation vector length. Use this for your input layer size."""
    return OBS_SIZE