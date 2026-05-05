"""
simulate.py — Interactive wizard for headless bot-vs-bot sessions.

Run::

    python -m poker_engine.simulate

Assign scripted bots or RLBot per seat (checkpoint paths usually live under
``runs/checkpoints/<run>/``).
"""

import os
import time
from collections import Counter

from .game import NUM_PLAYERS
from .session import GameSession, HandResult
from .bots import TightBot, PositionBot, RandomBot, CallBot

# Lazy import — only available if PyTorch is installed
try:
    from training_bot.rl_bot import RLBot
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"


def _fmt_stack(amount: float) -> str:
    return f"{YELLOW}{amount:>8.1f}{RESET}"


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _hr(char="─", width=60):
    print(f"{DIM}{char * width}{RESET}")


def display_session_summary(all_results: list, session_num: int, bot_assignments: dict):
    result = all_results[session_num]

    _hr("═")
    print(f"  {BOLD}{CYAN}Session {session_num + 1} — {result.total_hands} hands played{RESET}")
    _hr("═")

    win_counts = Counter()
    for hand in result.hand_results:
        for winner_id in hand.winner_ids:
            win_counts[winner_id] += 1

    for pid in range(NUM_PLAYERS):
        wins     = win_counts[pid]
        win_rate = wins / result.total_hands * 100 if result.total_hands else 0
        print(f"  Seat {pid} ({bot_assignments[pid][0]}): {wins} wins ({win_rate:.1f}%)")

    sorted_players = sorted(result.final_stacks.items(), key=lambda x: -x[1])

    print(f"\n  {BOLD}Final standings:{RESET}")
    for rank, (pid, stack) in enumerate(sorted_players, 1):
        name = bot_assignments[pid][0]
        bar  = "█" * int(stack / 100)
        print(f"  {rank}. Seat {pid} ({name:>12})  {_fmt_stack(stack)}  {DIM}{bar}{RESET}")

    print()
    winner_id, _ = sorted_players[0]
    print(f"  {RED}Seat {winner_id} ({bot_assignments[winner_id][0]}) wins the session.{RESET}\n")


def display_overall_summary(all_results, win_counts, total_hands, elapsed, bot_assignments):
    _hr("═")
    print(f"  {BOLD}{CYAN}Overall Summary — {len(all_results)} sessions{RESET}")
    _hr("═")
    print(f"  {BOLD}{CYAN}Total hands played : {total_hands}{RESET}")
    print(f"  {BOLD}{CYAN}Time elapsed       : {elapsed:.2f}s{RESET}")
    print(f"  {BOLD}{CYAN}Hands per second   : {total_hands / max(elapsed, 0.001):.0f}{RESET}")
    _hr()

    print(f"\n  {BOLD}Win counts:{RESET}")
    for pid in range(NUM_PLAYERS):
        bot_name, _ = bot_assignments[pid]
        wins     = win_counts[pid]
        win_rate = wins / total_hands * 100 if total_hands else 0
        print(f"  Seat {pid} ({bot_name:<15}): {wins} wins ({win_rate:.1f}%)")

    print(f"\n  {BOLD}Average final stack:{RESET}")
    for pid in range(NUM_PLAYERS):
        bot_name, _ = bot_assignments[pid]
        avg = sum(r.final_stacks[pid] for r in all_results) / len(all_results)
        print(f"  Seat {pid} ({bot_name:<15}): {_fmt_stack(avg)}")

BOT_TYPES = {
    "1": ("TightBot",    lambda seat: TightBot(seat)),
    "2": ("PositionBot", lambda seat: PositionBot(seat)),
    "3": ("CallBot",     lambda seat: CallBot(seat)),
    "4": ("RandomBot",   lambda seat: RandomBot(seat)),
}

if _RL_AVAILABLE:
    BOT_TYPES["5"] = ("RLBot", None)   # factory resolved after checkpoint prompt

def _prompt_rl_factory(stack: float, seat: int) -> tuple:
    """Prompt for a checkpoint path; return ``(display_name, seat_factory)``."""
    print(f"\n  {BOLD}RLBot checkpoint for seat {seat}:{RESET}")
    print(f"  Path (e.g. runs/checkpoints/my_run/latest.pt): ", end="", flush=True)
    path = input().strip()

    if not path or not os.path.exists(path):
        if path:
            print(f"  {RED}File not found — falling back to PositionBot.{RESET}")
        else:
            print(f"  {DIM}(no path given — falling back to PositionBot){RESET}")
        return "PositionBot", lambda s: PositionBot(s)

    try:
        factory = RLBot.make_factory(path, starting_stack=stack)
        print(f"  {GREEN}Loaded: {path}{RESET}")
        return f"RLBot@{os.path.basename(path)}", factory
    except Exception as e:
        print(f"  {RED}Load failed: {e}  — falling back to PositionBot.{RESET}")
        return "PositionBot", lambda s: PositionBot(s)

def _setup_wizard():
    _clear()
    _hr("═")
    print(f"  {BOLD}{CYAN}Texas Hold'em Simulation — Setup{RESET}")
    _hr("═")

    # Starting stack
    print("\n  Starting stack per player (default 1000): ", end="")
    try:
        raw   = input().strip()
        stack = float(raw) if raw else 1000.0
    except ValueError:
        stack = 1000.0

    # Bot types per seat
    print(f"\n  Available bot types:")
    for k, (name, _) in BOT_TYPES.items():
        print(f"    [{k}] {name}")
    if not _RL_AVAILABLE:
        print(f"    {DIM}[5] RLBot  (unavailable — install PyTorch){RESET}")

    bot_assignments = {}
    for seat in range(NUM_PLAYERS):
        print(f"\n  Bot for seat {seat} (default 2 = PositionBot): ", end="")
        raw     = input().strip()
        bot_key = raw if raw in BOT_TYPES else "2"

        if bot_key == "5":
            name, factory = _prompt_rl_factory(stack, seat)
        else:
            name, factory = BOT_TYPES[bot_key]

        bot_assignments[seat] = (name, factory)

    # Max hands
    print(f"\n  Max hands to play (default 50): ", end="")
    try:
        raw       = input().strip()
        max_hands = int(raw) if raw else 50
    except ValueError:
        max_hands = 50

    # Number of sessions
    print(f"\n  Number of sessions to run (default 1): ", end="")
    try:
        raw          = input().strip()
        num_sessions = int(raw) if raw else 1
    except ValueError:
        num_sessions = 1

    return stack, num_sessions, bot_assignments, max_hands


def run_sim():
    stack, num_sessions, bot_assignments, max_hands = _setup_wizard()

    print(f"\n  Seat assignments:")
    for seat, (name, _) in bot_assignments.items():
        print(f"    Seat {seat}: {name}")
    print(f"  Max hands: {max_hands}  |  Sessions: {num_sessions}")
    print(f"  {DIM}Press Enter to start...{RESET}", end="")
    input()

    all_results  = []
    win_counts   = Counter()
    total_hands  = 0
    start        = time.time()

    for session_num in range(num_sessions):
        session = GameSession(starting_stack=stack)
        for seat, (_, factory) in bot_assignments.items():
            session.assign_bot(seat, factory(seat))

        if _RL_AVAILABLE:
            rl_bots = [b for b in session.bots.values() if isinstance(b, RLBot)]
            if rl_bots:
                def on_hand_end(result, _bots=rl_bots):
                    for bot in _bots:
                        bot.notify_hand_end(result)
                session.on_hand_end = on_hand_end

        result = session.run(max_hands=max_hands)
        all_results.append(result)
        total_hands += session.hand_number

        for hand in session.hand_results:
            for winner_id in hand.winner_ids:
                win_counts[winner_id] += 1

        print(f"\r  Running... session {session_num + 1}/{num_sessions}", end="", flush=True)

    elapsed = time.time() - start
    print()
    display_overall_summary(all_results, win_counts, total_hands, elapsed, bot_assignments)

    while True:
        print(f"\n  View a session summary? (1–{num_sessions}, or q to quit): ", end="")
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