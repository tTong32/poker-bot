"""
cli.py — Terminal UI for humans vs bots at one table.

Run::

    python -m poker_engine.cli

PyTorch / ``training_bot`` is imported lazily so scripted bots work without
Torch installed; RLBot requires a checkpoint path (typically
``runs/checkpoints/<run>/latest.pt``).
"""

import os
import sys
import time
from typing import List

from .action import Action, ActionType
from .state import GameState, BettingRound
from .evaluator import hand_rank_name, evaluate_hand
from .session import GameSession, HandResult
from .bots import TightBot, PositionBot, RandomBot, CallBot

# Lazy import — only available if PyTorch is installed
try:
    from training_bot.rl_bot import RLBot
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


SUIT_COLORS = {
    "♦": "\033[91m",
    "♣": "\033[92m",
    "♥": "\033[91m",
    "♠": "\033[97m",
}
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"


def _color_card(card_str: str) -> str:
    for suit, color in SUIT_COLORS.items():
        if suit in card_str:
            return f"{color}{BOLD}{card_str}{RESET}"
    return card_str


def _fmt_cards(cards) -> str:
    return "  ".join(_color_card(str(c)) for c in cards)


def _fmt_stack(amount: float) -> str:
    return f"{YELLOW}{amount:>8.1f}{RESET}"


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _hr(char="─", width=60):
    print(f"{DIM}{char * width}{RESET}")


def _header(text: str):
    print(f"\n{BOLD}{CYAN}{text}{RESET}")


# STATE DISPLAY
def display_state(state: GameState, human_seat: int, show_all_hands: bool = False):
    _clear()
    _hr("═")
    print(f"  {BOLD}Texas Hold'em{RESET}   "
          f"Round: {CYAN}{state.betting_round}{RESET}   "
          f"Pot: {YELLOW}{state.pot:.1f}{RESET}")
    _hr("═")

    board_str = _fmt_cards(state.board) if state.board else f"{DIM}(waiting for cards){RESET}"
    print(f"\n  Board:  {board_str}\n")
    _hr()

    for p in state.players:
        is_current = (not state.terminal and p.id == state.current_player_index)
        is_human   = (p.id == human_seat)

        if p.has_folded:
            status = f"{DIM}FOLDED {RESET}"
        elif p.is_all_in:
            status = f"{RED}ALL-IN{RESET}"
        elif p.stack == 0:
            status = f"{RED}OUT{RESET}"
        else:
            status = f"{GREEN}active{RESET}"

        arrow = f" {BOLD}◄ YOUR TURN{RESET}" if (is_current and is_human) else \
                f" {DIM}◄ thinking...{RESET}" if is_current else ""

        if is_human or show_all_hands:
            hand_str = _fmt_cards(p.hand) if p.hand else "—"
        else:
            hand_str = f"{DIM}🂠  🂠{RESET}" if p.hand else "—"

        label = f"  {'YOU' if is_human else f'Bot{p.id}':>5}"
        print(f"{label}  stack={_fmt_stack(p.stack)}  "
              f"bet={p.current_bet:>6.1f}  [{status}]  {hand_str}{arrow}")

    _hr()
    print(f"  {DIM}Dealer button: Player {state.dealer_index}{RESET}\n")


def display_hand_result(state: GameState, human_seat: int):
    display_state(state, human_seat, show_all_hands=True)

    if state.winners:
        names = [
            "You" if w == human_seat else f"Bot{w}"
            for w in state.winners
        ]
        if human_seat in state.winners:
            print(f"  {BOLD}{GREEN}🏆  You win!  ({', '.join(names)}){RESET}\n")
        else:
            print(f"  {BOLD}{RED}💀  You lose.  Winner: {', '.join(names)}{RESET}\n")

    active = [p for p in state.players if not p.has_folded]
    if len(active) > 1 and state.board:
        print(f"  {DIM}Hand rankings:{RESET}")
        for p in active:
            score = evaluate_hand(p.hand, state.board)
            name  = "You" if p.id == human_seat else f"Bot{p.id}"
            cards = "  ".join(_color_card(str(c)) for c in p.hand)
            print(f"    {name:>5}: {cards}  →  {hand_rank_name(score)}")
        print()


# ACTION INPUTS
_ACTION_LABELS = {
    ActionType.FOLD:           ("F", "Fold"),
    ActionType.CHECK:          ("K", "Check"),
    ActionType.CALL:           ("C", "Call"),
    ActionType.BET_HALF_POT:   ("H", "Bet ½ pot"),
    ActionType.BET_POT:        ("P", "Bet pot"),
    ActionType.BET_DOUBLE_POT: ("D", "Bet 2× pot"),
    ActionType.ALL_IN:         ("A", "All-in"),
}


def prompt_action(state: GameState, legal_actions: List[Action]) -> Action:
    player      = state.current_player
    call_amount = state.current_bet - player.current_bet

    print(f"  {BOLD}Your hand:{RESET}  {_fmt_cards(player.hand)}")
    print(f"  Stack: {_fmt_stack(player.stack)}   "
          f"To call: {YELLOW}{call_amount:.1f}{RESET}\n")

    print(f"  {BOLD}Actions:{RESET}")
    key_map = {}
    for i, action in enumerate(legal_actions):
        key, label = _ACTION_LABELS[action.type]
        amount_str = f"  ({action.amount:.1f})" if action.amount > 0 else ""
        print(f"    [{key}] {label}{amount_str}")
        key_map[key.lower()] = action
        key_map[str(i)]      = action

    print()

    while True:
        try:
            raw = input("  Your choice: ").strip().lower()
            if raw in key_map:
                chosen = key_map[raw]
                print(f"\n  → {_ACTION_LABELS[chosen.type][1]}"
                      f"{f'  ({chosen.amount:.1f})' if chosen.amount else ''}")
                time.sleep(0.4)
                return chosen
            print(f"  {RED}Invalid. Enter one of: {', '.join(key_map.keys())}{RESET}")
        except (KeyboardInterrupt, EOFError):
            print("\n  (folding)")
            for a in legal_actions:
                if a.type == ActionType.FOLD:
                    return a
            return legal_actions[0]


# Session Summary
def display_session_summary(session: GameSession, human_seat: int):
    _clear()
    _hr("═")
    print(f"  {BOLD}{CYAN}Session Over — {session.hand_number} hands played{RESET}")
    _hr("═")

    stacks         = {p.id: p.stack for p in session.game.players}
    sorted_players = sorted(stacks.items(), key=lambda x: -x[1])

    print(f"\n  {BOLD}Final standings:{RESET}")
    for rank, (pid, stack) in enumerate(sorted_players, 1):
        name = "You" if pid == human_seat else f"Bot{pid}"
        bar  = "█" * int(stack / 100)
        print(f"  {rank}. {name:>5}  {_fmt_stack(stack)}  {DIM}{bar}{RESET}")

    print()
    winner_id, _ = sorted_players[0]
    winner_name  = "You" if winner_id == human_seat else f"Bot{winner_id}"
    if winner_id == human_seat:
        print(f"  {BOLD}{GREEN}🏆  Congratulations! You win the session!{RESET}\n")
    else:
        print(f"  {RED}Better luck next time. {winner_name} wins the session.{RESET}\n")

BOT_TYPES = {
    "1": ("TightBot",    lambda seat: TightBot(seat)),
    "2": ("PositionBot", lambda seat: PositionBot(seat)),
    "3": ("CallBot",     lambda seat: CallBot(seat)),
    "4": ("RandomBot",   lambda seat: RandomBot(seat)),
}

if _RL_AVAILABLE:
    BOT_TYPES["5"] = ("RLBot", None)   # factory set later after checkpoint prompt


def _prompt_rl_factory(stack: float):
    """Prompt for ``runs/checkpoints/...`` path and return ``(label, factory)``."""
    print(f"\n  {BOLD}RLBot checkpoint path{RESET}")
    print(f"  (e.g. runs/checkpoints/my_run/latest.pt): ", end="", flush=True)
    path = input().strip()
    if not path:
        print(f"  {RED}No path given — falling back to PositionBot.{RESET}")
        return "PositionBot", lambda seat: PositionBot(seat)
    if not os.path.exists(path):
        print(f"  {RED}File not found: {path}  — falling back to PositionBot.{RESET}")
        return "PositionBot", lambda seat: PositionBot(seat)
    try:
        factory = RLBot.make_factory(path, starting_stack=stack)
        print(f"  {GREEN}Loaded RLBot from {path}{RESET}")
        return "RLBot", factory
    except Exception as e:
        print(f"  {RED}Failed to load checkpoint: {e}  — falling back to PositionBot.{RESET}")
        return "PositionBot", lambda seat: PositionBot(seat)

def _setup_wizard():
    _clear()
    _hr("═")
    print(f"  {BOLD}{CYAN}Texas Hold'em — Setup{RESET}")
    _hr("═")

    # Starting stack
    print("\n  Starting stack per player (default 1000): ", end="")
    try:
        raw = input().strip()
        stack = float(raw) if raw else 1000.0
    except ValueError:
        stack = 1000.0

    # Human seat
    print(f"  Your seat (0–5, default 0): ", end="")
    try:
        raw        = input().strip()
        human_seat = int(raw) if raw else 0
        human_seat = max(0, min(5, human_seat))
    except ValueError:
        human_seat = 0

    # Bot type
    print(f"\n  Bot type for opponents:")
    for k, (name, _) in BOT_TYPES.items():
        print(f"    [{k}] {name}")
    if not _RL_AVAILABLE:
        print(f"    {DIM}[5] RLBot  (unavailable — install PyTorch){RESET}")

    default_key = "2"
    print(f"  Choose (default {default_key} = PositionBot): ", end="")
    raw     = input().strip()
    bot_key = raw if raw in BOT_TYPES else default_key

    if bot_key == "5":
        # RLBot — prompt for checkpoint
        bot_name, bot_factory = _prompt_rl_factory(stack)
    else:
        bot_name, bot_factory = BOT_TYPES[bot_key]

    # Max hands
    print(f"\n  Max hands to play (default 50): ", end="")
    try:
        raw       = input().strip()
        max_hands = int(raw) if raw else 50
    except ValueError:
        max_hands = 50

    return stack, human_seat, bot_factory, bot_name, max_hands


def run_cli():
    stack, human_seat, bot_factory, bot_name, max_hands = _setup_wizard()

    session = GameSession(starting_stack=stack)
    session.set_human_seat(human_seat)
    session.game.players[human_seat].human_player = True
    session.fill_empty_seats_with(bot_factory)

    print(f"\n  Opponents: {bot_name} × 5   |   Max hands: {max_hands}")
    print(f"  {DIM}Press Enter to start...{RESET}", end="")
    input()

    def on_state(state: GameState):
        if state.terminal:
            display_hand_result(state, human_seat)
            input(f"  {DIM}Press Enter for next hand...{RESET}")
        else:
            display_state(state, human_seat)
            if state.current_player_index != human_seat:
                time.sleep(0.6)

    session.on_state_change = on_state
    session.human_input_fn  = prompt_action

    if _RL_AVAILABLE:
        rl_bots = [b for b in session.bots.values() if isinstance(b, RLBot)]
        if rl_bots:
            def on_hand_end(result):
                for bot in rl_bots:
                    bot.notify_hand_end(result)
            session.on_hand_end = on_hand_end

    session.run(max_hands=max_hands)

    session.run(max_hands=max_hands)
    display_session_summary(session, human_seat)


if __name__ == "__main__":
    run_cli()