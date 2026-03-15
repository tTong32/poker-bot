"""
train.py — Training environment for the poker RL bot.

Run with:
    python -m poker_engine.train

Curriculum stages:
    1 — RandomBot opponents              (graduate at +5.0 avg chip delta/hand)
    2 — CallBot/TightBot/PositionBot     (graduate at +2.0 avg chip delta/hand)
    3 — Self-play pool                   (runs indefinitely)

Chip delta is averaged across all 3 training bots and normalised by big blind.
Graduation requires DELTA_HISTORY_SIZE hands of data before checking.
"""

import os
import time
import random
import torch
from collections import deque

from poker_engine.game import NUM_PLAYERS
from poker_engine.session import GameSession
from poker_engine.bots import TightBot, PositionBot, RandomBot, CallBot
from .network import PokerNetwork
from .ppo import PPOTrainer
from .training_bot import TrainingBot

# Config

STARTING_STACK      = 1000.0
BIG_BLIND           = STARTING_STACK / 100   # 10.0
ROLLOUT_HANDS       = 200        # hands before each PPO update
DELTA_HISTORY_SIZE  = 10000      # hands tracked for chip delta average
CHECKPOINT_INTERVAL = 50         # PPO updates between checkpoints
CHECKPOINT_DIR      = "checkpoints"

# Curriculum thresholds (avg chip delta per hand, in big blinds)
STAGE_1_THRESHOLD   = 5.0        # +5 BB/hand avg to leave RandomBot
STAGE_2_THRESHOLD   = 2.0        # +2 BB/hand avg to leave mixed bots

# Seats assigned to TrainingBot (rest are opponents)
TRAINING_SEATS      = [0, 1, 2]


# Logging

def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}]  {msg}", flush=True)


def _log_stats(update_num: int, stage: int, avg_delta: float, stats: dict):
    print(
        f"  update={update_num:>5}  "
        f"stage={stage}  "
        f"avg_delta={avg_delta:>+7.3f} BB/hand  "
        f"policy_loss={stats['policy_loss']:>7.4f}  "
        f"value_loss={stats['value_loss']:>7.4f}  "
        f"entropy={stats['entropy']:>6.4f}",
        flush=True,
    )


# Checkpointing

def save_checkpoint(network, optimizer, update_num, stage, delta_history, path):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        'network_state':    network.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
        'update_num':       update_num,
        'stage':            stage,
        'delta_history':    list(delta_history),
    }, path)
    _log(f"Checkpoint saved -> {path}")


def load_checkpoint(network, optimizer, path):
    if not os.path.exists(path):
        return 0, 1, deque(maxlen=DELTA_HISTORY_SIZE)
    checkpoint = torch.load(path)
    network.load_state_dict(checkpoint['network_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    delta_history = deque(checkpoint['delta_history'], maxlen=DELTA_HISTORY_SIZE)
    _log(
        f"Checkpoint loaded <- {path}  "
        f"(update={checkpoint['update_num']}, stage={checkpoint['stage']})"
    )
    return checkpoint['update_num'], checkpoint['stage'], delta_history


# Session Setup

def _make_opponent(stage: int, seat: int):
    if stage == 1:
        return RandomBot(seat)
    elif stage == 2:
        return random.choice([CallBot, TightBot, PositionBot])(seat)
    else:
        return RandomBot(seat)  # placeholder until self-play pool is implemented


def _setup_session(stage, training_bots, delta_history):
    session = GameSession(starting_stack=STARTING_STACK)

    for seat in range(NUM_PLAYERS):
        if seat in TRAINING_SEATS:
            session.assign_bot(seat, training_bots[seat])
        else:
            session.assign_bot(seat, _make_opponent(stage, seat))

    def on_hand_end(result):
        # Capture delta BEFORE finish_hand() updates starting_stack
        deltas = []
        for seat in TRAINING_SEATS:
            before = training_bots[seat].starting_stack
            after  = result.stacks_after.get(seat, before)
            deltas.append((after - before) / BIG_BLIND)
        delta_history.append(sum(deltas) / len(deltas))

        # Now update starting_stack
        for seat in TRAINING_SEATS:
            training_bots[seat].finish_hand(result)

    session.on_hand_end = on_hand_end
    return session

# Chip Delta tracking

def _record_hand_deltas(hand_result, training_bots, delta_history):
    """
    After each hand, compute the average chip delta across all training bots
    and append one value per hand to the history deque.

    Delta is normalised by big blind so it's in BB/hand units.
    """
    deltas = []
    for seat in TRAINING_SEATS:
        before = training_bots[seat].starting_stack
        after  = hand_result.stacks_after.get(seat, before)
        delta  = (after - before) / BIG_BLIND
        deltas.append(delta)
    delta_history.append(sum(deltas) / len(deltas))


def _avg_delta(delta_history):
    if not delta_history:
        return 0.0
    return sum(delta_history) / len(delta_history)


# Main Training Loop

def train(resume: bool = False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    network       = PokerNetwork()
    trainer       = PPOTrainer(network)
    shared_buffer = []

    training_bots = {
        seat: TrainingBot(
            player_id=seat,
            network=network,
            shared_buffer=shared_buffer,
            starting_stack=STARTING_STACK,
            big_blind=BIG_BLIND,
        )
        for seat in TRAINING_SEATS
    }

    update_num    = 0
    stage         = 1
    delta_history = deque(maxlen=DELTA_HISTORY_SIZE)

    if resume:
        latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
        update_num, stage, delta_history = load_checkpoint(
            network, trainer.optimizer, latest
        )

    _log(f"Training started -- stage {stage}")

    hands_since_update = 0
    session_count      = 0

    while True:

        # --- Run one session ---
        session = _setup_session(stage, training_bots, delta_history)
        result  = session.run(max_hands=ROLLOUT_HANDS)

        session_count      += 1
        hands_since_update += session.hand_number

        # Reset training bot stacks for next session
        for seat in TRAINING_SEATS:
            training_bots[seat].reset_starting_stack(STARTING_STACK)

        avg = _avg_delta(delta_history)
        print(
            f"\r  session={session_count}  "
            f"hands_this_rollout={hands_since_update}/{ROLLOUT_HANDS}  "
            f"stage={stage}  "
            f"avg_delta={avg:>+7.3f} BB/hand",
            end="", flush=True,
        )

        # --- PPO update ---
        if hands_since_update >= ROLLOUT_HANDS and shared_buffer:
            print()
            stats              = trainer.update(shared_buffer)
            shared_buffer.clear()
            hands_since_update = 0
            update_num        += 1

            avg = _avg_delta(delta_history)
            _log_stats(update_num, stage, avg, stats)

            # Periodic checkpoint
            if update_num % CHECKPOINT_INTERVAL == 0:
                path = os.path.join(CHECKPOINT_DIR, f"update_{update_num}.pt")
                save_checkpoint(network, trainer.optimizer, update_num, stage, delta_history, path)
                save_checkpoint(network, trainer.optimizer, update_num, stage, delta_history,
                                os.path.join(CHECKPOINT_DIR, "latest.pt"))

            # Curriculum graduation -- only check once we have enough data
            if len(delta_history) >= DELTA_HISTORY_SIZE:
                if stage == 1 and avg >= STAGE_1_THRESHOLD:
                    stage = 2
                    delta_history.clear()
                    _log("Graduated to stage 2 -- mixed opponents")
                    save_checkpoint(network, trainer.optimizer, update_num, stage,
                                    delta_history, os.path.join(CHECKPOINT_DIR, "stage2_start.pt"))

                elif stage == 2 and avg >= STAGE_2_THRESHOLD:
                    stage = 3
                    delta_history.clear()
                    _log("Graduated to stage 3 -- self play")
                    save_checkpoint(network, trainer.optimizer, update_num, stage,
                                    delta_history, os.path.join(CHECKPOINT_DIR, "stage3_start.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)