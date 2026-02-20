"""
snapshot.py — Lightweight cloning for tree search and CFR.

The core problem: PokerGame mutates state in place. For any algorithm that
needs to branch (CFR, MCTS, rollouts), you need to snapshot the game at a
decision point, then restore or branch from it cheaply.

Two tools are provided:

  GameSnapshot     — a frozen, serialisable record of everything needed to
                     reconstruct the game at any point mid-hand. Taking a
                     snapshot is O(n_players). Cards are frozen dataclasses
                     so they are shared by reference safely.

  take_snapshot()  — capture current game into a GameSnapshot
  restore_snapshot() — reconstruct an independent PokerGame from a snapshot
  clone_game()     — convenience wrapper: snapshot + restore in one call

Usage
-----
    snap = take_snapshot(game)

    # Explore one line
    branch_a = restore_snapshot(snap)
    branch_a.apply_action(Action(ActionType.FOLD))

    # Explore another line from the same point
    branch_b = restore_snapshot(snap)
    branch_b.apply_action(Action(ActionType.CALL))

    # Original game is untouched
    assert game.state.pot == original_pot
"""

from dataclasses import dataclass
from typing import Tuple
import random as _random

from .player import Player
from .card import Card
from .state import GameState, BettingRound


# --------------------------------------------------------------------------- #
# Player snapshot                                                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PlayerSnapshot:
    id: int
    stack: float
    hand: Tuple[Card, ...]        # Card is a frozen dataclass — safe to share
    has_folded: bool
    is_all_in: bool
    current_bet: float
    total_invested: float
    human_player: bool

    @staticmethod
    def from_player(p: Player) -> "PlayerSnapshot":
        return PlayerSnapshot(
            id             = p.id,
            stack          = p.stack,
            hand           = tuple(p.hand),
            has_folded     = p.has_folded,
            is_all_in      = p.is_all_in,
            current_bet    = p.current_bet,
            total_invested = p.total_invested,
            human_player   = p.human_player,
        )

    def to_player(self) -> Player:
        p = Player(id=self.id, stack=self.stack)
        p.hand           = list(self.hand)
        p.has_folded     = self.has_folded
        p.is_all_in      = self.is_all_in
        p.current_bet    = self.current_bet
        p.total_invested = self.total_invested
        p.human_player   = self.human_player
        return p


# --------------------------------------------------------------------------- #
# Game snapshot                                                                #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GameSnapshot:
    """
    Immutable record of a PokerGame at a single decision point.
    All fields are primitives or frozen dataclasses — no mutable references.
    """
    # Table-level state
    dealer_index:          int
    current_player_index:  int
    pot:                   float
    board:                 Tuple[Card, ...]
    betting_round:         str
    current_bet:           float
    last_raise_size:       float
    raise_count:           int
    players_acted:         frozenset          # frozenset[int]
    terminal:              bool
    winners:               Tuple[int, ...]

    # Per-player state (ordered by seat index)
    player_snapshots:      Tuple[PlayerSnapshot, ...]

    # Remaining deck cards in order (not yet dealt)
    remaining_deck:        Tuple[Card, ...]

    # Game config — needed to reconstruct a valid PokerGame
    starting_stack:        float
    big_blind:             float
    small_blind:           float
    raise_limit:           int


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def take_snapshot(game) -> GameSnapshot:
    """
    Capture the full mid-hand state of a PokerGame.
    A hand must be in progress (game.state is not None).
    Cost: O(n_players).  No deep copies — Cards are shared by reference.
    """
    if game.state is None:
        raise RuntimeError("Cannot snapshot: no hand in progress. Call start_new_hand() first.")

    s = game.state
    return GameSnapshot(
        dealer_index         = s.dealer_index,
        current_player_index = s.current_player_index,
        pot                  = s.pot,
        board                = tuple(s.board),
        betting_round        = s.betting_round,
        current_bet          = s.current_bet,
        last_raise_size      = s.last_raise_size,
        raise_count          = s.raise_count,
        players_acted        = frozenset(s.players_acted),
        terminal             = s.terminal,
        winners              = tuple(s.winners),
        player_snapshots     = tuple(PlayerSnapshot.from_player(p) for p in game.players),
        remaining_deck       = tuple(game.deck.cards),
        starting_stack       = game.starting_stack,
        big_blind            = game.big_blind,
        small_blind          = game.small_blind,
        raise_limit          = game.raise_limit,
    )


def restore_snapshot(snap: GameSnapshot) -> "PokerGame":
    """
    Reconstruct a fully playable PokerGame from a GameSnapshot.
    The returned game is completely independent of the snapshot and of any
    other game — stepping it forward does not affect anything else.
    Cost: O(n_players + deck_size).
    """
    from .game import PokerGame
    from .deck import Deck

    # Build game object without calling __init__ (avoids re-creating players/deck)
    game = PokerGame.__new__(PokerGame)
    game.starting_stack = snap.starting_stack
    game.big_blind      = snap.big_blind
    game.small_blind    = snap.small_blind
    game.raise_limit    = snap.raise_limit
    game.dealer_index   = snap.dealer_index
    game.seed           = None

    # Reconstruct players
    game.players = [ps.to_player() for ps in snap.player_snapshots]

    # Reconstruct deck — remaining cards only, fresh RNG
    deck        = Deck.__new__(Deck)
    deck._seed  = None
    deck._rng   = _random.Random()
    deck.cards  = list(snap.remaining_deck)
    game.deck   = deck

    # Reconstruct GameState
    game.state = GameState(
        players              = game.players,
        dealer_index         = snap.dealer_index,
        current_player_index = snap.current_player_index,
        board                = list(snap.board),
        pot                  = snap.pot,
        betting_round        = snap.betting_round,
        current_bet          = snap.current_bet,
        last_raise_size      = snap.last_raise_size,
        raise_count          = snap.raise_count,
        players_acted        = set(snap.players_acted),
        terminal             = snap.terminal,
        winners              = list(snap.winners),
    )

    return game


def clone_game(game) -> "PokerGame":
    """Convenience: snapshot + restore in one call."""
    return restore_snapshot(take_snapshot(game))