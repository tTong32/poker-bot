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
def display_session_summary(all_results: list, session_num: int, bot_assignments: dict):
    result = all_results[session_num]
    
    _hr("═")
    print(f"  {BOLD}{CYAN}Session {session_num+1} — {result.total_hands} hands played{RESET}")
    _hr("═")

    win_counts = Counter()
    for hand in result.hand_results:
        for winner_id in hand.winner_ids:
            win_counts[winner_id] += 1

    for pid in range(NUM_PLAYERS):
        wins = win_counts[pid]
        win_rate = wins / result.total_hands * 100
        print(f"  Bot{pid}: {wins} wins ({win_rate:.1f}%)")

    sorted_players = sorted(result.final_stacks.items(), key=lambda x: -x[1])

    print(f"\n  {BOLD}Final standings:{RESET}")
    for rank, (pid, stack) in enumerate(sorted_players, 1):
        name = f"Bot{pid}"
        bar  = "█" * int(stack / 100)
        print(f"  {rank}. {name:>5}  {_fmt_stack(stack)}  {DIM}{bar}{RESET}")

    print()
    winner_id, winner_stack = sorted_players[0]
    winner_name = f"Bot{winner_id}"
    print(f"  {RED}{winner_name} wins the session.{RESET}\n")

def display_overall_summary(all_results, win_counts, total_hands, elapsed, bot_assignments):
    _hr("═")
    print(f"  {BOLD}{CYAN}Overall Summary — {len(all_results)} sessions{RESET}")
    _hr("═")
    print(f"  {BOLD}{CYAN}Total hands played: {total_hands}{RESET}")
    print(f"  {BOLD}{CYAN}Time elapsed: {elapsed:.2f}s{RESET}")
    print(f"  {BOLD}{CYAN}Hands per sec: {total_hands/elapsed:.0f}{RESET}")
    _hr()

    print(f"\n  {BOLD}Win counts:{RESET}")
    for pid in range(NUM_PLAYERS):
        bot_name, _ = bot_assignments[pid]
        wins = win_counts[pid]
        win_rate = wins / total_hands * 100
        print(f"  Seat {pid} ({bot_name}): {wins} wins ({win_rate:.1f}%)")

    print(f"\n  {BOLD}Average final stack:{RESET}")
    for pid in range(NUM_PLAYERS):
        bot_name, _ = bot_assignments[pid]
        avg_stack = sum(r.final_stacks[pid] for r in all_results) / len(all_results)
        print(f"  Seat {pid} ({bot_name}): {_fmt_stack(avg_stack)}")

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
    
    # Number of sessions
    print(f"\n  Number of sessions (games) to play: ", end="")
    try:
        raw = input().strip()
        num_sessions = int(raw) if raw else 1
    except ValueError:
        num_sessions = 1

    return stack, num_sessions, bot_assignments, max_hands


def run_sim():
    stack, num_sessions, bot_assignments, max_hands = _setup_wizard()

    for seat, (bot_name, _) in bot_assignments.items():
        print(f"  Seat {seat}: {bot_name}")
    print(f"  Max hands: {max_hands}  |  Sessions: {num_sessions}")
    print(f"  {DIM}Press Enter to start...{RESET}", end="")
    input()

    all_results = []
    win_counts = Counter()
    total_hands = 0
    start = time.time()

    for session_num in range(num_sessions):
        session = GameSession(starting_stack=stack)
        for seat, (bot_name, bot_factory) in bot_assignments.items():
            session.assign_bot(seat, bot_factory(seat))

        result = session.run(max_hands=max_hands)
        all_results.append(result)
        total_hands += session.hand_number

        for hand in session.hand_results:
            for winner_id in hand.winner_ids:
                win_counts[winner_id] += 1
        
        print(f"\r  Running... session {session_num + 1}/{num_sessions}", end="", flush=True)

    elapsed = time.time() - start
    display_overall_summary(all_results, win_counts, total_hands, elapsed, bot_assignments)
    while True:
        print(f"\n  View a game summary? (1-{num_sessions}, or q to quit): ", end="")
        raw = input().strip().lower()
        if raw == "q":
            break
        try:
            game_num = int(raw)
            if 1 <= game_num <= num_sessions:
                display_session_summary(all_results, game_num - 1, bot_assignments)
            else:
                print(f"  {RED}Enter a number between 1 and {num_sessions}{RESET}")
        except ValueError:
            print(f"  {RED}Invalid input{RESET}")


if __name__ == "__main__":
    run_sim()