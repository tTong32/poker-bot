"""
eval.py — Standalone evaluation for a trained RL bot.

Run with:
    python -m training_bot.eval
    python -m training_bot.eval --checkpoint runs/checkpoints/myrun/latest.pt
    python -m training_bot.eval --checkpoint runs/checkpoints/myrun/update_500.pt --hands 5000

Evaluates against each of the four built-in bot types separately, then prints:
  • Avg chip delta per hand (in BB) against each opponent
  • Overall action distribution with red-flag warnings
  • Win rate per opponent
  • Pass/fail checklist against known red flags
"""

import os
import argparse
from collections import Counter
from typing import Dict, Tuple

import torch

from poker_engine.game import NUM_PLAYERS
from poker_engine.session import GameSession
from poker_engine.bots import RandomBot, CallBot, TightBot, PositionBot
from poker_engine.action import ActionType
from .network import PokerNetwork
from .training_bot import TrainingBot

# Config
DEFAULT_EVAL_HANDS  = 10_000
STARTING_STACK      = 1000.0
BIG_BLIND           = STARTING_STACK / 100    # 10.0
HANDS_PER_SESSION   = 1000                   # cap per GameSession

# Red-flag thresholds
RF_FOLD_RATE     = 0.60   # folding > 60% of decisions is a red flag
RF_ALLIN_RATE    = 0.30   # going all-in > 30% is a red flag
RF_PASSIVE_RATE  = 0.80   # check+call > 80% is passive/weak

OPPONENT_BOTS = {
    "RandomBot":   RandomBot,
    "CallBot":     CallBot,
    "TightBot":    TightBot,
    "PositionBot": PositionBot,
}

# Terminal colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"


# Checkpoint discovery

def _find_latest_checkpoint() -> str:
    """Search for latest.pt in runs/checkpoints/<run>/ subdirectories."""
    ck_root = "runs/checkpoints"
    if not os.path.isdir(ck_root):
        raise FileNotFoundError(
            "No runs/checkpoints/ directory found.  Run train.py first, "
            "or pass --checkpoint explicitly."
        )
    candidates = []
    for entry in os.listdir(ck_root):
        candidate = os.path.join(ck_root, entry, "latest.pt")
        if os.path.exists(candidate):
            candidates.append((os.path.getmtime(candidate), candidate))
    if not candidates:
        raise FileNotFoundError(
            "No latest.pt found under runs/checkpoints/.  "
            "Pass --checkpoint <path> explicitly."
        )
    # Most recently modified
    candidates.sort(reverse=True)
    return candidates[0][1]


def _load_network(path: str) -> Tuple[PokerNetwork, dict]:
    ck = torch.load(path, weights_only=False)
    net = PokerNetwork()
    if isinstance(ck, dict) and "network_state" in ck:
        net.load_state_dict(ck["network_state"])
        meta = {k: ck.get(k, "?") for k in ("update_num", "stage", "avg_delta")}
    else:
        net.load_state_dict(ck)
        meta = {}
    net.eval()
    return net, meta


# Core evaluation runner

def _run_vs_opponent(
    network: PokerNetwork,
    opponent_cls,
    num_hands: int,
) -> Tuple[float, Counter, float, int]:
    """
    Run the RL agent (seat 0) against one opponent type (seats 1–5).

    Returns:
        avg_delta_bb  — mean chip delta per hand in BB
        action_counts — Counter{ActionType: count}
        win_rate      — fraction of hands won
        total_hands   — actual hands played
    """
    total_delta  = 0.0
    total_wins   = 0
    total_hands  = 0
    shared_buf   = []

    hands_left = num_hands

    while hands_left > 0:
        # Fresh stacks each session; we track delta relative to session start
        rl_bot = TrainingBot(
            player_id    = 0,
            network      = network,
            shared_buffer= shared_buf,
            starting_stack = STARTING_STACK,
            big_blind    = BIG_BLIND,
        )
        session = GameSession(starting_stack=STARTING_STACK, training_mode=True)
        session.assign_bot(0, rl_bot)
        for seat in range(1, NUM_PLAYERS):
            session.assign_bot(seat, opponent_cls(seat))

        def _on_hand(result, _bot=rl_bot):
            nonlocal total_delta, total_wins
            after  = result.stacks_after.get(0, _bot.starting_stack)
            delta  = (after - _bot.starting_stack) / BIG_BLIND
            total_delta += delta
            if 0 in result.winner_ids:
                total_wins += 1
            # finish_hand must be called BEFORE updating starting_stack
            _bot.finish_hand(result)
            _bot.starting_stack = after

        session.on_hand_end = _on_hand
        result = session.run(max_hands=min(hands_left, HANDS_PER_SESSION))

        hands_played  = session.hand_number
        total_hands  += hands_played
        hands_left   -= hands_played

    action_counts = Counter(ActionType(e.action) for e in shared_buf)
    avg_delta     = total_delta / max(total_hands, 1)
    win_rate      = total_wins  / max(total_hands, 1)
    return avg_delta, action_counts, win_rate, total_hands


# Display helpers

def _print_action_dist(counts: Counter):
    total = sum(counts.values()) or 1
    print(f"\n  {BOLD}Action distribution (all opponents combined):{RESET}")
    for atype in ActionType:
        n   = counts.get(atype, 0)
        pct = n / total * 100
        bar = "█" * max(0, int(pct / 2))
        flags = []
        if atype == ActionType.FOLD   and pct / 100 > RF_FOLD_RATE:
            flags.append(f"{RED}⚠ HIGH FOLD{RESET}")
        if atype == ActionType.ALL_IN and pct / 100 > RF_ALLIN_RATE:
            flags.append(f"{RED}⚠ HIGH ALL-IN{RESET}")
        flag_str = "  " + "  ".join(flags) if flags else ""
        print(f"    {atype.name:<20} {pct:>5.1f}%  {DIM}{bar}{RESET}{flag_str}")

    passive = (counts.get(ActionType.CHECK, 0) + counts.get(ActionType.CALL, 0)) / total
    if passive > RF_PASSIVE_RATE:
        print(f"\n  {RED}⚠ Combined passive rate {passive:.1%} exceeds {RF_PASSIVE_RATE:.0%}{RESET}")


def _print_red_flags(results: dict, all_counts: Counter):
    total = sum(all_counts.values()) or 1
    fold_r    = all_counts.get(ActionType.FOLD,  0) / total
    allin_r   = all_counts.get(ActionType.ALL_IN, 0) / total
    passive_r = (all_counts.get(ActionType.CHECK, 0)
               + all_counts.get(ActionType.CALL,  0)) / total

    checks = [
        ("Fold rate  < 60 %",       fold_r    < RF_FOLD_RATE,
         f"{fold_r:.1%}"),
        ("All-in rate < 30 %",      allin_r   < RF_ALLIN_RATE,
         f"{allin_r:.1%}"),
        ("Passive rate < 80 %",     passive_r < RF_PASSIVE_RATE,
         f"{passive_r:.1%}"),
        ("Positive delta vs Random", results["RandomBot"][0] > 0,
         f"{results['RandomBot'][0]:+.3f} BB/hand"),
        ("Positive delta vs Call",   results["CallBot"][0]   > 0,
         f"{results['CallBot'][0]:+.3f} BB/hand"),
    ]

    all_pass = True
    print(f"\n  {BOLD}Red-flag checklist:{RESET}")
    for label, ok, value in checks:
        sym      = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        all_pass = all_pass and ok
        print(f"    [{sym}]  {label:<35}  {value}")

    if all_pass:
        print(f"\n  {BOLD}{GREEN}✓  All checks passed.{RESET}\n")
    else:
        print(f"\n  {BOLD}{RED}✗  One or more checks failed.{RESET}\n")


# Public entry point

def run_eval(checkpoint_path: str | None = None, num_hands: int = DEFAULT_EVAL_HANDS):
    path = checkpoint_path or _find_latest_checkpoint()

    print(f"\n  {BOLD}{CYAN}{'=' * 56}{RESET}")
    print(f"  {BOLD}{CYAN}  RL Bot Evaluation{RESET}")
    print(f"  {BOLD}{CYAN}{'=' * 56}{RESET}")
    print(f"  Checkpoint : {path}")

    network, meta = _load_network(path)
    if meta:
        print(
            f"  Update     : {meta.get('update_num', '?')}   "
            f"Stage : {meta.get('stage', '?')}   "
            f"Avg delta : {meta.get('avg_delta', 0.0):.3f} BB/hand"
        )
    print(f"  Hands      : {num_hands:,} per opponent type\n")

    all_counts = Counter()
    results: Dict[str, Tuple[float, float, int]] = {}

    for bot_name, bot_cls in OPPONENT_BOTS.items():
        print(f"  vs {bot_name:<15} …", end="", flush=True)
        avg, counts, win_r, n = _run_vs_opponent(network, bot_cls, num_hands)
        all_counts.update(counts)
        results[bot_name] = (avg, win_r, n)
        delta_col = (GREEN if avg > 0 else RED) + f"{avg:>+8.3f}" + RESET
        print(f"  {delta_col} BB/hand   win={win_r:.1%}  ({n:,} hands)")

    # Summary table
    print(f"\n  {BOLD}Results summary:{RESET}")
    print(f"  {'Opponent':<15}  {'Avg Δ (BB/hand)':>16}  {'Win Rate':>9}")
    print(f"  {'-' * 45}")
    for bname, (avg, win_r, _) in results.items():
        col = GREEN if avg > 0 else RED
        print(f"  {bname:<15}  {col}{avg:>+16.3f}{RESET}  {win_r:>9.1%}")

    _print_action_dist(all_counts)
    _print_red_flags(results, all_counts)


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL poker bot")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str, default=None,
        help="Path to checkpoint .pt file (default: auto-detect latest)",
    )
    parser.add_argument(
        "--hands", "-n",
        type=int, default=DEFAULT_EVAL_HANDS,
        help=f"Hands per opponent type (default {DEFAULT_EVAL_HANDS:,})",
    )
    args = parser.parse_args()
    run_eval(args.checkpoint, args.hands)