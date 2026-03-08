"""
simulate.py — Headless simulation of bots playing against each other

Run with:
    python -m poker_engine.simulate

"""

import os
import time
from collections import Counter

from .game import NUM_PLAYERS

from .session import GameSession, HandResult
from .bots import TightBot, PositionBot, RandomBot, CallBot

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"

def _fmt_stack(amount: float) -> str:
    return f"{YELLOW}{amount:>8.1f}{RESET}"


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _hr(char="─", width=60):
    print(f"{DIM}{char * width}{RESET}")


# Session Summary
def display_session_summary(session: GameSession, start: float):
    _clear()
    _hr("═")
    elapsed = time.time() - start
    print(f"  {BOLD}{CYAN}Session Over — {session.hand_number} hands played{RESET}")
    print(f"  {BOLD}{CYAN}Time elapsed: {elapsed:.2f}s{RESET}")
    print(f"  {BOLD}{CYAN}Hands per sec: {session.hand_number/elapsed:.0f}{RESET}")
    _hr("═")

    win_counts = Counter()
    for hand in session.hand_results:
        for winner_id in hand.winner_ids:
            win_counts[winner_id] += 1

    for pid in range(NUM_PLAYERS):
        wins = win_counts[pid]
        win_rate = wins / session.hand_number * 100
        print(f"  Bot{pid}: {wins} wins ({win_rate:.1f}%)")

    stacks = {p.id: p.stack for p in session.game.players}
    sorted_players = sorted(stacks.items(), key=lambda x: -x[1])

    print(f"\n  {BOLD}Final standings:{RESET}")
    for rank, (pid, stack) in enumerate(sorted_players, 1):
        name = f"Bot{pid}"
        bar  = "█" * int(stack / 100)
        print(f"  {rank}. {name:>5}  {_fmt_stack(stack)}  {DIM}{bar}{RESET}")

    print()
    winner_id, winner_stack = sorted_players[0]
    winner_name = f"Bot{winner_id}"
    print(f"  {RED}{winner_name} wins the session.{RESET}\n")


# SETUP
BOT_TYPES = {
    "1": ("TightBot",    lambda seat: TightBot(seat)),
    "2": ("PositionBot", lambda seat: PositionBot(seat)),
    "3": ("CallBot",     lambda seat: CallBot(seat)),
    "4": ("RandomBot",   lambda seat: RandomBot(seat)),
}


def _setup_wizard():
    """Interactive setup: choose seat, stack, bots, max hands."""
    _clear()
    _hr("═")
    print(f"  {BOLD}{CYAN}Texas Hold'em Simulation — Setup{RESET}")
    _hr("═")

    # Starting stack
    print("\n  Starting stack per player (default 1000): ", end="")
    try:
        raw = input().strip()
        stack = float(raw) if raw else 1000.0
    except ValueError:
        stack = 1000.0

    # Bot types
    bot_assignments = {}
    for seat in range(NUM_PLAYERS):
        print(f"\n  Bot for seat {seat}:")
        for k, (name, _) in BOT_TYPES.items():
            print(f"    [{k}] {name}")
        print(f"  Choose (default 2 = PositionBot): ", end="")
        raw = input().strip()
        bot_key = raw if raw in BOT_TYPES else "2"
        bot_assignments[seat] = BOT_TYPES[bot_key]

    # Max hands
    print(f"\n  Max hands to play (default 50): ", end="")
    try:
        raw = input().strip()
        max_hands = int(raw) if raw else 50
    except ValueError:
        max_hands = 50

    return stack, bot_assignments, max_hands


def run_sim():
    stack, bot_assignments, max_hands = _setup_wizard()
    session = GameSession(starting_stack=stack)
    for seat, (bot_name, bot_factory) in bot_assignments.items():
        session.assign_bot(seat, bot_factory(seat))
    for seat, (bot_name, _) in bot_assignments.items():
        print(f"  Seat {seat}: {bot_name}")
    print(f"  Max hands: {max_hands}")
    print(f"  {DIM}Press Enter to start...{RESET}", end="")
    input()

    start = time.time()
    result = session.run(max_hands=max_hands)
    display_session_summary(session, start)


if __name__ == "__main__":
    run_sim()