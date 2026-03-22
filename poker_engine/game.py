"""
6-player No-Limit Texas Hold'em engine.

Blind structure:
  - Small blind: player left of dealer  (dealer_index + 1)
  - Big blind:   player two left        (dealer_index + 2)
  - UTG (first to act preflop):         (dealer_index + 3)

Post-flop, first to act is first active player left of dealer.
"""

from typing import List, Optional
from .deck import Deck
from .state import GameState, BettingRound
from .player import Player
from .action import ActionType, Action
from .evaluator import evaluate_hand, hand_rank_name
from .side_pots import calculate_side_pots


NUM_PLAYERS = 6
ALL_IN_FRACTION = 0.35


class PokerGame:

    def __init__(self, starting_stack: float = 1000.0, seed: int | None = None, training_mode: bool = False):
        self.starting_stack = starting_stack
        self.seed = seed
        self.training_mode = training_mode
        # Big blind = starting_stack / 100; small blind = half that.
        self.big_blind = starting_stack / 100
        self.small_blind = self.big_blind / 2
        self.raise_limit = 4        # max raises per street (cap)

        self.players = [
            Player(id=i, stack=starting_stack)
            for i in range(NUM_PLAYERS)
        ]

        self.deck = Deck(seed=seed)
        self.dealer_index = 0       # moves each hand
        self.state: Optional[GameState] = None


    def start_new_hand(self) -> GameState:
        """Reset everything and deal a new hand. Returns the initial state."""
        # Rebuild and shuffle deck
        self.deck.reset()
        self.deck.shuffle()

        # Reset players
        for p in self.players:
            p.reset_for_new_hand()

        # Post blinds
        sb_index = self._next_active_index(self.dealer_index, skip=1)
        bb_index = self._next_active_index(self.dealer_index, skip=2)
        utg_index = self._next_active_index(self.dealer_index, skip=3)

        self._post_blind(self.players[sb_index], self.small_blind)
        self._post_blind(self.players[bb_index], self.big_blind)

        # Pot = what was actually posted (short-stacked players may post less)
        pot = (self.players[sb_index].current_bet
               + self.players[bb_index].current_bet)
        # Effective BB is what the BB actually put in
        effective_bb = self.players[bb_index].current_bet

        # Deal hole cards only to players who are in the hand
        for p in self.players:
            if p.stack > 0 or p.is_all_in:
                p.hand = [self.deck.draw_one(), self.deck.draw_one()]

        # If UTG is all-in (posted blind with < BB), skip them to first actable player
        utg_index = self._first_actable_preflop(utg_index)

        self.state = GameState(
            players=self.players,
            dealer_index=self.dealer_index,
            current_player_index=utg_index,
            board=[],
            pot=pot,
            betting_round=BettingRound.PREFLOP,
            current_bet=effective_bb,
            last_raise_size=effective_bb,
            raise_count=0,
            players_acted={self.players[sb_index].id, self.players[bb_index].id},
            terminal=False,
            winners=[],
            last_actions={},
        )
        return self.state

    def apply_action(self, action: Action) -> GameState:
        """Apply an action for the current player and advance the state."""
        if self.state is None:
            raise RuntimeError("Call start_new_hand() first.")
        if self.state.terminal:
            raise RuntimeError("Hand is over. Call start_new_hand().")

        player = self.state.current_player
        self._execute_action(player, action)

        # Check if only one player remains (everyone else folded)
        if len(self.state.active_players) == 1:
            self._award_pot_to_last_player()
            return self.state

        # Check if betting round is over
        if self._is_betting_round_over():
            self._advance_street()
        else:
            self._move_to_next_player()

        return self.state

    def get_legal_actions(self) -> List[Action]:
        """Return legal actions for the current player."""
        player = self.state.current_player
        actions = []

        call_amount = self.state.current_bet - player.current_bet

        # FOLD is always legal
        actions.append(Action(ActionType.FOLD))

        # CHECK if no bet to call
        if call_amount <= 0:
            actions.append(Action(ActionType.CHECK))
        else:
            # CALL (capped at player's stack)
            actual_call = min(call_amount, player.stack)
            actions.append(Action(ActionType.CALL, actual_call))

        # Bet/raise actions — only if raise limit not hit
        if self.state.raise_count < self.raise_limit:
            pot = self.state.pot
            min_raise = self.state.last_raise_size
            can_bet = player.stack > call_amount  # has chips beyond a call

            if can_bet:
                half = self._make_bet_action(player, ActionType.BET_HALF_POT, pot * 0.5, call_amount)
                full = self._make_bet_action(player, ActionType.BET_POT, pot, call_amount)
                double = self._make_bet_action(player, ActionType.BET_DOUBLE_POT, pot * 2, call_amount)
                if half:
                    actions.append(half)
                if full and full.amount != (half.amount if half else -1):
                    actions.append(full)
                if double and double.amount != (full.amount if full else -1):
                    actions.append(double)

        # ALL_IN: fractional in training to extend session life, true all-in in play
        if player.stack > 0:
            all_in_amount = (
                min(player.stack, self.starting_stack * ALL_IN_FRACTION)
                if self.training_mode else player.stack
            )
            actions.append(Action(ActionType.ALL_IN, all_in_amount))

        return actions


    def _execute_action(self, player: Player, action: Action):
        s = self.state
        s.players_acted.add(player.id)   # mark this player as having acted
        call_amount = s.current_bet - player.current_bet

        if action.type == ActionType.FOLD:
            player.has_folded = True

        elif action.type == ActionType.CHECK:
            if call_amount > 0:
                raise ValueError("Cannot check when there is a bet to call.")

        elif action.type == ActionType.CALL:
            amount = min(call_amount, player.stack)
            self._commit(player, amount)

        elif action.type in (ActionType.BET_HALF_POT,
                              ActionType.BET_POT,
                              ActionType.BET_DOUBLE_POT):
            # action.amount is the TOTAL new bet amount (raise above current)
            self._place_bet_or_raise(player, action.amount)

        elif action.type == ActionType.ALL_IN:
            amount = action.amount   # pre-computed in get_legal_actions (capped at stack)
            total_player_bet = player.current_bet + amount
            if total_player_bet > s.current_bet:
                raise_size = total_player_bet - s.current_bet
                s.last_raise_size = raise_size
                s.current_bet = total_player_bet
                s.raise_count += 1
                s.players_acted = {player.id}
            self._commit(player, amount)
            if player.stack == 0:
                player.is_all_in = True

        else:
            raise ValueError(f"Unknown action type: {action.type}")

        self.state.last_actions[player.id] = action.type.value

    def _place_bet_or_raise(self, player: Player, raise_amount: float):
        """
        raise_amount = the amount ABOVE the current bet the player wants to raise to.
        Total chips player puts in = call_amount + raise_amount (capped at stack).
        """
        s = self.state
        call_amount = s.current_bet - player.current_bet
        min_raise = max(s.last_raise_size, self.big_blind)
        actual_raise = max(raise_amount, min_raise)
        total_new_bet = s.current_bet + actual_raise
        chips_needed = total_new_bet - player.current_bet
        chips_needed = min(chips_needed, player.stack)  # cap at stack

        new_total_bet = player.current_bet + chips_needed
        if new_total_bet > s.current_bet:
            s.last_raise_size = new_total_bet - s.current_bet
            s.current_bet = new_total_bet
            s.raise_count += 1
            s.players_acted = {player.id}  # everyone else must act again

        self._commit(player, chips_needed)
        if player.stack == 0:
            player.is_all_in = True

    def _commit(self, player: Player, amount: float):
        """Move chips from player to pot."""
        amount = min(amount, player.stack)
        player.stack -= amount
        player.current_bet += amount
        player.total_invested += amount
        self.state.pot += amount

    def _make_bet_action(self, player: Player, atype: ActionType,
                         raise_size: float, call_amount: float) -> Optional[Action]:
        """Build a bet action, ensuring the raise is at least min-raise and within stack."""
        min_raise = max(self.state.last_raise_size, self.big_blind)
        actual_raise = max(raise_size, min_raise)
        chips_needed = call_amount + actual_raise
        if chips_needed >= player.stack:
            return None  # would be all-in; skip (ALL_IN covers it)
        return Action(atype, actual_raise)


    def _is_betting_round_over(self) -> bool:
        """
        Round ends when every active (non-folded, non-all-in) player has
        acted at least once this street AND everyone has matched the current bet.
        """
        actable = self.state.players_who_can_act
        if not actable:
            return True  # everyone is all-in or folded
        all_matched = all(p.current_bet >= self.state.current_bet for p in actable)
        all_acted   = all(p.id in self.state.players_acted for p in actable)
        return all_matched and all_acted

    def _move_to_next_player(self):
        s = self.state
        start = s.current_player_index
        idx = start
        for _ in range(NUM_PLAYERS):
            idx = (idx + 1) % NUM_PLAYERS
            p = s.players[idx]
            if p.is_active:
                s.current_player_index = idx
                return
        # No active player found — betting is over, advance the street
        self._advance_street()

    def _advance_street(self):
        s = self.state
        round_order = [
            BettingRound.PREFLOP,
            BettingRound.FLOP,
            BettingRound.TURN,
            BettingRound.RIVER,
            BettingRound.SHOWDOWN,
        ]
        current_idx = round_order.index(s.betting_round)
        next_round = round_order[current_idx + 1]

        # Deal community cards
        if next_round == BettingRound.FLOP:
            s.board.extend(self.deck.draw(3))
        elif next_round in (BettingRound.TURN, BettingRound.RIVER):
            s.board.extend(self.deck.draw(1))
        elif next_round == BettingRound.SHOWDOWN:
            self._resolve_showdown()
            return

        # Reset betting for new street
        s.betting_round = next_round
        s.current_bet = 0.0
        s.last_raise_size = self.big_blind
        s.raise_count = 0
        s.players_acted = set()
        for p in s.players:
            p.reset_for_new_street()

        # First to act post-flop: first active (non-folded, non-all-in) player left of dealer
        first = self._first_actable_postflop(s.dealer_index)
        if first is None:
            # Everyone remaining is all-in — run out the board
            self._advance_street()
            return
        s.current_player_index = first
        # Skip to first player who can actually act
        if not s.current_player.is_active:
            self._move_to_next_player()


    def _resolve_showdown(self):
        s = self.state
        active = s.active_players

        if len(active) == 1:
            self._award_pot_to_last_player()
            return

        # Pre-compute hand scores for all active players
        scores = {p.id: evaluate_hand(p.hand, s.board) for p in active}

        # Use side pot structure so all-in players only win what they covered
        pots = calculate_side_pots(self.players)
        all_winner_ids: set = set()

        for pot in pots:
            eligible_scores = {
                pid: scores[pid]
                for pid in pot.eligible_player_ids
                if pid in scores
            }
            if not eligible_scores:
                continue
            best = max(eligible_scores.values())
            pot_winners = [pid for pid, sc in eligible_scores.items() if sc == best]
            share = pot.amount / len(pot_winners)
            for pid in pot_winners:
                self.players[pid].stack += share
                all_winner_ids.add(pid)

        s.winners = sorted(all_winner_ids)
        s.terminal = True
        s.betting_round = BettingRound.SHOWDOWN

    def _award_pot_to_last_player(self):
        """
        One non-folded player remains. They win the entire pot.
        We simply give them state.pot — which is the ground truth of all
        chips committed — rather than recomputing via side pots.
        (Side pots only matter at showdown when multiple players show hands.)
        """
        s = self.state
        winner = s.active_players[0]
        winner.stack += s.pot
        s.winners = [winner.id]
        s.terminal = True


    def _post_blind(self, player: Player, amount: float):
        amount = min(amount, player.stack)
        player.stack -= amount
        player.current_bet += amount
        player.total_invested += amount
        # If posting the blind emptied the player's stack, mark them all-in
        if player.stack == 0 and player.current_bet > 0:
            player.is_all_in = True

    def _next_active_index(self, from_index: int, skip: int = 1) -> int:
        """Return the index of the nth active (non-busted) player after from_index.
        If fewer than skip players have chips, wraps around to the best available."""
        idx = from_index
        found = 0
        last_found = None
        for _ in range(NUM_PLAYERS * 2):
            idx = (idx + 1) % NUM_PLAYERS
            if self.players[idx].stack > 0:
                found += 1
                last_found = idx
                if found == skip:
                    return idx
        if last_found is not None:
            return last_found  # fewer players than requested — return last one found
        raise RuntimeError("No players with chips remaining.")

    def _next_nonfolded_index(self, from_index: int, skip: int = 1) -> int:
        """Next player who has NOT folded (may be all-in) — used for street ordering."""
        idx = from_index
        found = 0
        for _ in range(NUM_PLAYERS * 2):
            idx = (idx + 1) % NUM_PLAYERS
            p = self.state.players[idx]
            if not p.has_folded:
                found += 1
                if found == skip:
                    return idx
        # Fall back: if everyone is all-in/done, return dealer+1
        return (from_index + 1) % NUM_PLAYERS

    def _first_actable_postflop(self, from_dealer: int):
        """
        Find first active (non-folded, non-all-in) player left of dealer.
        Returns None if no such player exists.
        """
        idx = from_dealer
        for _ in range(NUM_PLAYERS):
            idx = (idx + 1) % NUM_PLAYERS
            p = self.players[idx]
            if p.is_active:
                return idx
        return None

    def _first_actable_preflop(self, start_index: int) -> int:
        """
        Find the first player from start_index who can actually act preflop
        (not all-in, not busted). Falls back to start_index if none found.
        """
        idx = start_index
        for _ in range(NUM_PLAYERS):
            p = self.players[idx]
            if not p.has_folded and not p.is_all_in and p.stack > 0:
                return idx
            idx = (idx + 1) % NUM_PLAYERS
        return start_index  # everyone is all-in, state will resolve immediately

    def rotate_dealer(self):
        """Call between hands to move the dealer button."""
        self.dealer_index = self._next_active_index(self.dealer_index, skip=1)

    def print_state(self):
        s = self.state
        print(f"\n{'='*50}")
        print(f"  Round: {s.betting_round}  |  Pot: {s.pot:.1f}")
        board_str = " ".join(str(c) for c in s.board) or "(none)"
        print(f"  Board: {board_str}")
        print(f"  Dealer: Player {s.dealer_index}")
        print()
        for p in s.players:
            hand_str = " ".join(str(c) for c in p.hand) if p.hand else "[]"
            status = "FOLDED" if p.has_folded else ("ALL-IN" if p.is_all_in else "active")
            marker = " <--" if (not s.terminal and p.id == s.current_player_index) else ""
            print(f"  Player {p.id}: stack={p.stack:.1f}  bet={p.current_bet:.1f}"
                  f"  hand={hand_str}  [{status}]{marker}")
        if s.terminal:
            winners_str = ", ".join(f"Player {w}" for w in s.winners)
            print(f"\n  ** Hand over — Winner(s): {winners_str} **")
        print(f"{'='*50}\n")