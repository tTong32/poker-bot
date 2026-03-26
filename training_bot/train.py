"""
train.py — Training environment for the poker RL bot.

Run with:
    python -m training_bot.train
    python -m training_bot.train --name my_experiment

Named runs isolate all artifacts under checkpoints/<name>/, so multiple
experiments never overwrite each other.  On startup you are prompted to
resume an existing run or start a new one.

Directory layout per run
------------------------
    runs/checkpoints/<run_name>/
        latest.pt            ← always points to the most recent update
        update_<N>.pt        ← periodic snapshots
        stage2_start.pt
        stage3_start.pt
    runs/snapshots/<run_name>/
        snapshot_<N>.pt      ← self-play pool entries
    runs/logs/<run_name>/
        training_log.csv     ← one row per PPO update
        hands_log.csv        ← one row per hand
        eval_log.csv         ← one row per ladder evaluation
    runs/tensorboard/<run_name>/
        events.out.*         ← live TensorBoard stream

Curriculum stages
-----------------
    1 — RandomBot opponents          graduate at avg_delta > +5.0 BB/hand
    2 — Mixed (Call/Tight/Position)  graduate at avg_delta > +2.0 BB/hand
    3 — Self-play pool               runs indefinitely
"""

import os
import csv
import time
import random
import torch
import numpy as np
from collections import deque, Counter
from typing import Dict, List, Optional

from poker_engine.game import NUM_PLAYERS
from poker_engine.session import GameSession
from poker_engine.bots import TightBot, PositionBot, RandomBot, CallBot
from poker_engine.action import ActionType
from .network import PokerNetwork
from .ppo import PPOTrainer
from .training_bot import TrainingBot, Experience
from .self_play_pool import SelfPlayPool, SNAPSHOT_INTERVAL
from .rl_bot import RLBot

# Config
STARTING_STACK       = 1000.0
BIG_BLIND            = STARTING_STACK/100.0
ROLLOUT_HANDS        = 5000     # hands collected before each PPO update
DELTA_HISTORY_SIZE   = 50000    # rolling window for avg chip delta
CHECKPOINT_INTERVAL  = 50        # PPO updates between checkpoints
LADDER_EVAL_INTERVAL = 100       # PPO updates between ladder evaluations
LADDER_EVAL_HANDS    = 1_000     # hands per opponent in the mini eval

# unused
STAGE_1_THRESHOLD    = 1.5       # BB/hand avg to leave stage 1
STAGE_2_THRESHOLD    = 0.5       # BB/hand avg to leave stage 2

MIN_UPDATES_PER_STAGE = {    
    1: 150,   # ~375k hands at 2500/session
    2: 150,
    3: 200,
    4: None,  # runs indefinitely
}

#unused now
REWARD_CLIP_INITIAL  = 2.0
REWARD_CLIP_MAX      = 10.0
REWARD_CLIP_STEP     = 1.0
EV_CLIP_THRESHOLD    = 0.3    # EV must exceed this to widen
EV_STABLE_UPDATES    = 20     # must hold above threshold for this many updates

#this reward clip is not going to be used for now
REWARD_CLIP_PER_STAGE = {1: 2.0, 2: 5.0, 3: 8.0, 4: 12.0}

TIGHTBOT_SEED_POOL_SIZE = 10   # stop seeding TightBots once pool reaches this size
TIGHTBOT_SEED_RATE      = 0.4
TRAINING_SEATS       = [0, 1, 2]

# Red-flag thresholds (warnings printed to console)
RF_FOLD_RATE      = 0.60
RF_ALLIN_RATE     = 0.30
RF_PASSIVE_RATE   = 0.80
RF_EXP_VAR        = 0.30   # explained variance below this is concerning
RF_ALLIN_COLLAPSE = 0.50   # check if the bot is going all in a lot

STAGE1_RESTRICTED = {ActionType.BET_DOUBLE_POT, ActionType.BET_POT, ActionType.ALL_IN}
STAGE2_RESTRICTED = {ActionType.BET_DOUBLE_POT, ActionType.ALL_IN}
STAGE3_RESTRICTED = {ActionType.ALL_IN}
STAGE4_RESTRICTED: set = set()

ENTROPY_START = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.05}
ENTROPY_END   = {1: 0.03, 2: 0.02, 3: 0.02, 4: 0.02}

# Terminal colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"

# Logging helpers

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"  [{time.strftime('%H:%M:%S')}]  {msg}", flush=True)


def _ensure(path: str):
    os.makedirs(path, exist_ok=True)


def _init_csv(path: str, headers: List[str]):
    """Write headers only if the file does not already exist (append-safe)."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def _write_csv(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _action_pcts(dist: Counter) -> List[str]:
    """Return per-ActionType percentage strings in enum order."""
    total = sum(dist.values()) or 1
    return [f"{dist.get(atype, 0) / total * 100:.2f}" for atype in ActionType]


def _log_update(update_num, stage, avg, stats, dist):
    ev       = stats.get("explained_variance", 0.0)
    ev_warn  = f"  {RED}⚠ LOW EV{RESET}" if ev < RF_EXP_VAR else ""
    total    = sum(dist.values()) or 1

    print(
        f"  update={update_num:>5}  stage={stage}  "
        f"avg_delta={avg:>+7.3f} BB/hand  "
        f"policy_loss={stats['policy_loss']:>7.4f}  "
        f"value_loss={stats['value_loss']:>7.4f}  "
        f"entropy={stats['entropy']:>6.4f}  "
        f"ev={ev:>5.3f}{ev_warn}",
        flush=True,
    )

    # Compact action distribution
    dist_str = "  ".join(
        f"{atype.name[:4]}={dist.get(atype, 0) / total * 100:.0f}%"
        for atype in ActionType
    )
    print(f"           {dist_str}", flush=True)

    # Red-flag console warnings
    fold_r    = dist.get(ActionType.FOLD,  0) / total
    allin_r   = dist.get(ActionType.ALL_IN, 0) / total
    passive_r = (dist.get(ActionType.CHECK, 0) + dist.get(ActionType.CALL, 0)) / total

    if fold_r    > RF_FOLD_RATE:
        print(f"  {RED}⚠ FOLD RATE {fold_r:.1%} > {RF_FOLD_RATE:.0%}{RESET}")
    if allin_r   > RF_ALLIN_RATE:
        print(f"  {RED}⚠ ALL-IN RATE {allin_r:.1%} > {RF_ALLIN_RATE:.0%}{RESET}")
    if allin_r   > RF_ALLIN_COLLAPSE:
        print(f"  {RED}⚠ ALL-IN COLLAPSE — bot is degenerating. Consider resetting or lowering REWARD_CLIP.{RESET}")
    if passive_r > RF_PASSIVE_RATE:
        print(f"  {RED}⚠ PASSIVE RATE {passive_r:.1%} > {RF_PASSIVE_RATE:.0%}{RESET}")

# Checkpointing

def _save(network, trainer, update_num, stage, delta_history, path, reward_clip=3.0, ev_stable_count=0, stage_start_update=0):
    _ensure(os.path.dirname(path))
    avg = sum(delta_history) / len(delta_history) if delta_history else 0.0
    torch.save({
        "network_state":        network.state_dict(),
        "policy_optimizer":     trainer.policy_optimizer.state_dict(),
        "value_optimizer":      trainer.value_optimizer.state_dict(),
        "update_num":           update_num,
        "stage":                stage,
        "delta_history":        list(delta_history),
        "avg_delta":            avg,
        "reward_clip":          reward_clip,
        "ev_stable_count":      ev_stable_count,
        "stage_start_update":   stage_start_update,
    }, path)
    _log(f"Checkpoint saved → {path}")


def _load(network, trainer, path):
    ck = torch.load(path, weights_only=False)
    network.load_state_dict(ck["network_state"])
    if "policy_optimizer" in ck:
        trainer.policy_optimizer.load_state_dict(ck["policy_optimizer"])
        trainer.value_optimizer.load_state_dict(ck["value_optimizer"])
    history            = deque(ck.get("delta_history", []), maxlen=DELTA_HISTORY_SIZE)
    reward_clip        = ck.get("reward_clip", 3.0)
    ev_stable_count    = ck.get("ev_stable_count", 0)
    stage_start_update = ck.get("stage_start_update", 0)
    _log(
        f"Checkpoint loaded ← {path}  "
        f"(update={ck['update_num']}, stage={ck['stage']}, "
        f"avg_delta={ck.get('avg_delta', 0.0):+.3f})"
    )
    return ck["update_num"], ck["stage"], history, reward_clip, ev_stable_count, stage_start_update


def _read_meta(path: str) -> dict:
    """Read checkpoint metadata without allocating model weights."""
    try:
        ck = torch.load(path, weights_only=False)
        return {k: ck.get(k, "?") for k in ("update_num", "stage", "avg_delta")}
    except Exception:
        return {}


# Interactive startup prompt

def _startup(run_name: Optional[str]) -> tuple:
    """
    Returns (run_dir, should_resume, resolved_run_name).

    If run_name is given:
      - If checkpoints/<run_name>/latest.pt exists → ask to resume.
      - Otherwise → start fresh under that name.

    If run_name is None:
      - Scan checkpoints/ for existing runs and present a menu.
      - User can pick a run to resume, or type a name / press Enter for new.
    """
    ck_root = "runs/checkpoints"

    def _run_dir(name): return os.path.join(ck_root, name)
    def _latest(name):  return os.path.join(_run_dir(name), "latest.pt")

    # Named run
    if run_name is not None:
        run_dir = _run_dir(run_name)
        if os.path.exists(_latest(run_name)):
            meta = _read_meta(_latest(run_name))
            print(
                f"\n  {BOLD}Existing run '{run_name}':{RESET}  "
                f"update={meta.get('update_num','?')}  "
                f"stage={meta.get('stage','?')}  "
                f"avg_delta={meta.get('avg_delta', 0.0):.3f}"
            )
            print(f"  Resume? (y/n): ", end="", flush=True)
            resume = input().strip().lower() in ("y", "yes", "")
        else:
            resume = False
            print(f"\n  Starting new run: {BOLD}{run_name}{RESET}")
        return run_dir, resume, run_name

    # Auto-discover existing runs
    existing = []
    if os.path.isdir(ck_root):
        for entry in sorted(os.listdir(ck_root)):
            lp = _latest(entry)
            if os.path.exists(lp):
                existing.append((entry, _read_meta(lp)))

    if not existing:
        # First ever run
        name    = time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = _run_dir(name)
        print(f"\n  No existing runs found.  Starting: {BOLD}{name}{RESET}")
        return run_dir, False, name

    print(f"\n  {BOLD}{CYAN}Existing training runs:{RESET}")
    for i, (name, meta) in enumerate(existing, 1):
        print(
            f"    [{i}] {name:<30}  "
            f"update={meta.get('update_num','?'):>5}  "
            f"stage={meta.get('stage','?')}  "
            f"avg_delta={meta.get('avg_delta', 0.0):>+7.3f}"
        )
    print(f"    [N] Start a new run\n")
    print(f"  Choice (number, name for new run, or Enter for latest): ", end="", flush=True)
    raw = input().strip()

    if raw.upper() == "N":
        new_name = time.strftime("run_%Y%m%d_%H%M%S")
        print(f"  Starting new run: {BOLD}{new_name}{RESET}")
        return _run_dir(new_name), False, new_name

    if raw == "":
        # Default: resume the most recently modified run
        name, meta = existing[-1]
        print(f"  Resuming latest: {BOLD}{name}{RESET}")
        return _run_dir(name), True, name

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(existing):
            name, meta = existing[idx]
            print(f"  Resuming: {BOLD}{name}{RESET}")
            return _run_dir(name), True, name

    # Treat as a new run name
    new_name = raw
    run_dir  = _run_dir(new_name)
    lp       = _latest(new_name)
    if os.path.exists(lp):
        meta = _read_meta(lp)
        print(
            f"  Found existing run '{new_name}': "
            f"update={meta.get('update_num','?')}, "
            f"stage={meta.get('stage','?')}"
        )
        print(f"  Resume? (y/n): ", end="", flush=True)
        resume = input().strip().lower() in ("y", "yes", "")
    else:
        resume = False
        print(f"  Starting new run: {BOLD}{new_name}{RESET}")
    return run_dir, resume, new_name


# Opponent factories

_SEED_BOTS = [RandomBot, CallBot, TightBot, PositionBot]

def _make_opponent(stage: int, seat: int,
                   pool: SelfPlayPool, current_net: PokerNetwork):
    #if stage == 1:
    #    return random.choice([RandomBot, CallBot])(seat)
    #if stage == 2:
    #    return random.choice([TightBot, PositionBot])(seat)
    # Stage 3 — self-play pool

    if len(pool) < TIGHTBOT_SEED_POOL_SIZE and random.random() < TIGHTBOT_SEED_RATE:
            return random.choice(_SEED_BOTS)(seat)

    net = pool.sample_opponent_network(current_net)
    tag = "Cur" if net is current_net else "Old"
    return RLBot(seat, net, starting_stack=STARTING_STACK, name=f"SelfPlay({tag})")


# Session setup

def _setup_session(
    stage:          int,
    training_bots:  Dict[int, TrainingBot],
    delta_history,
    hands_csv:      str,
    session_num:    int,
    pool:           SelfPlayPool,
    current_net:    PokerNetwork,
) -> GameSession:

    session = GameSession(starting_stack=STARTING_STACK, training_mode=True)

    for seat in range(NUM_PLAYERS):
        if seat in TRAINING_SEATS:
            session.assign_bot(seat, training_bots[seat])
        else:
            session.assign_bot(seat, _make_opponent(stage, seat, pool, current_net))

    def on_hand_end(result):
        row = [session_num, result.hand_number, stage]
        for seat in TRAINING_SEATS:
            before = training_bots[seat].starting_stack
            after  = result.stacks_after.get(seat, before)
            delta  = (after - before) / BIG_BLIND
            row.append(f"{delta:.4f}")
            training_bots[seat].finish_hand(result)
            training_bots[seat].starting_stack = after
            if seat == 0:
                delta_history.append(delta)
        _write_csv(hands_csv, row)

    session.on_hand_end = on_hand_end
    return session


# Mini ladder evaluation (every LADDER_EVAL_INTERVAL updates)

_LADDER_OPPONENTS = {
    "RandomBot":   RandomBot,
    "CallBot":     CallBot,
    "TightBot":    TightBot,
    "PositionBot": PositionBot,
}


def _mini_eval(network: PokerNetwork) -> Dict[str, float]:
    """
    Run exactly LADDER_EVAL_HANDS hands against each built-in bot type.
    Uses multiple sessions if one ends early due to a bust, so the
    per-hand average is always computed over the correct number of hands.
    """
    results = {}
    for bot_name, bot_cls in _LADDER_OPPONENTS.items():
        total_delta = 0.0
        total_hands = 0
        hands_left  = LADDER_EVAL_HANDS
        buf         = []

        while hands_left > 0:
            # Fresh session and fresh rl bot each time stacks reset to
            # STARTING_STACK so busts don't carry over between sessions
            rl = TrainingBot(
                player_id      = 0,
                network        = network,
                shared_buffer  = buf,
                starting_stack = STARTING_STACK,
                big_blind      = BIG_BLIND,
            )
            session = GameSession(starting_stack=STARTING_STACK, training_mode=True)
            session.assign_bot(0, rl)
            for seat in range(1, NUM_PLAYERS):
                session.assign_bot(seat, bot_cls(seat))

            def _on_hand(res, _rl=rl):
                nonlocal total_delta
                after = res.stacks_after.get(0, _rl.starting_stack)
                total_delta += (after - _rl.starting_stack) / BIG_BLIND
                _rl.finish_hand(res)
                _rl.starting_stack = after

            session.on_hand_end = _on_hand
            session.run(max_hands=hands_left)

            played      = session.hand_number
            total_hands += played
            hands_left  -= played

        results[bot_name] = total_delta / max(total_hands, 1)

    return results


# Average chip delta helper

def _avg(history) -> float:
    return sum(history) / len(history) if history else 0.0


def _update_restricted_actions(training_bots: dict, stage: int):
    restrictions = {1: STAGE1_RESTRICTED, 2: STAGE2_RESTRICTED, 3: STAGE3_RESTRICTED, 4: STAGE4_RESTRICTED}
    r = restrictions.get(stage, set())
    for bot in training_bots.values():
        bot.restricted_actions = r


# Main Training Loop

def train(run_name: Optional[str] = None):
    # Startup
    run_dir, resume, run_name = _startup(run_name)

    ck_dir  = run_dir # checkpoints/<name>/
    snap_dir= os.path.join("runs/snapshots",   run_name)
    log_dir = os.path.join("runs/logs",        run_name)
    tb_dir  = os.path.join("runs/tensorboard", run_name)

    for d in (ck_dir, snap_dir, log_dir, tb_dir):
        _ensure(d)

    # CSV init 
    train_csv = os.path.join(log_dir, "training_log.csv")
    hands_csv = os.path.join(log_dir, "hands_log.csv")
    eval_csv  = os.path.join(log_dir, "eval_log.csv")

    _init_csv(train_csv, [
        "timestamp", "update_num", "stage", "avg_delta",
        "policy_loss", "value_loss", "entropy", "explained_variance",
        "fold%", "check%", "call%", "bet_half%", "bet_pot%", "bet_double%", "allin%",
    ])
    _init_csv(hands_csv, [
        "session_num", "hand_num", "stage",
        *[f"delta_seat{s}" for s in TRAINING_SEATS],
    ])
    _init_csv(eval_csv, [
        "update_num", "vs_RandomBot", "vs_CallBot", "vs_TightBot", "vs_PositionBot",
    ])

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_dir)
        _log(f"TensorBoard → {tb_dir}   (tensorboard --logdir {tb_dir})")
    except ImportError:
        writer = None
        _log("tensorboard not installed — skipping (pip install tensorboard)")

    # Network, trainer, shared experience buffer
    network       = PokerNetwork()
    trainer       = PPOTrainer(network)
    shared_buffer : List[Experience] = []

    training_bots = {
        seat: TrainingBot(
            player_id     = seat,
            network       = network,
            shared_buffer = shared_buffer,
            starting_stack= STARTING_STACK,
            big_blind     = BIG_BLIND,
        )
        for seat in TRAINING_SEATS
    }

    pool           = SelfPlayPool(snapshot_dir=snap_dir)
    update_num     = 0
    stage          = 1 # change this back to 1 if needed
    delta_history  = deque(maxlen=DELTA_HISTORY_SIZE)
    session_count  = 0
    hands_pending  = 0    # hands accumulated since last PPO update
    ev_stable_count = 0    # consecutive updates where EV >= EV_CLIP_THRESHOLD
    reward_clip    = REWARD_CLIP_INITIAL
    stage_start_update = 0

    # Resume
    if resume:
        latest = os.path.join(ck_dir, "latest.pt")
        update_num, stage, delta_history, reward_clip, ev_stable_count, stage_start_update = (
            _load(network, trainer, latest)
        )

    trainer.set_reward_clip(REWARD_CLIP_PER_STAGE[stage])
    _log(f"Reward clip set to ±{REWARD_CLIP_PER_STAGE[stage]:.1f} BB  (stage {stage})")

    _log(
        f"Run '{run_name}'  stage={stage}  "
        f"{'resumed' if resume else 'fresh start'}"
    )
    _update_restricted_actions(training_bots, stage)
    if writer:
        writer.add_text(
            "run_info",
            f"name={run_name}  stage={stage}  resumed={resume}",
            global_step=update_num,
        )

    # Training Loop
    while True:

        # Collect one session
        session_count += 1
        session = _setup_session(
            stage, training_bots, delta_history,
            hands_csv, session_count, pool, network,
        )
        session.run(max_hands=ROLLOUT_HANDS)
        hands_pending += session.hand_number

        # Reset bot stacks to STARTING_STACK so delta is measured per-session
        for seat in TRAINING_SEATS:
            training_bots[seat].reset_starting_stack(STARTING_STACK)

        avg = _avg(delta_history)
        print(
            f"\r  session={session_count}  "
            f"pending={hands_pending}/{ROLLOUT_HANDS}  "
            f"stage={stage}  avg_delta={avg:>+7.3f} BB/hand",
            end="", flush=True,
        )

        # PPO update
        if hands_pending < ROLLOUT_HANDS or not shared_buffer:
            continue

        print()   # end the \r line

        # Compute action distribution BEFORE clearing the buffer
        action_dist = Counter(ActionType(e.action) for e in shared_buffer)

        stats          = trainer.update(shared_buffer)
        shared_buffer.clear()
        hands_pending  = 0
        update_num    += 1
        avg            = _avg(delta_history)

        _log_update(update_num, stage, avg, stats, action_dist)

        # Entropy decay within stage
        updates_in_stage = update_num - stage_start_update
        min_updates = MIN_UPDATES_PER_STAGE.get(stage)
        progress = min(1.0, updates_in_stage / min_updates) if min_updates is not None else 1.0
        entropy_coeff = ENTROPY_START[stage] + (ENTROPY_END[stage] - ENTROPY_START[stage]) * progress
        entropy_coeff = max(entropy_coeff, 0.05)
        trainer.set_entropy_coeff(entropy_coeff)
        if writer:
            writer.add_scalar("train/entropy_coeff", entropy_coeff, update_num)


        # CSV append
        _write_csv(train_csv, [
            _ts(), update_num, stage, f"{avg:.4f}",
            f"{stats['policy_loss']:.6f}",
            f"{stats['value_loss']:.6f}",
            f"{stats['entropy']:.6f}",
            f"{stats.get('explained_variance', 0.0):.4f}",
            *_action_pcts(action_dist),
        ])

        # TensorBoard 
        if writer:
            writer.add_scalar("train/avg_delta",         avg,                               update_num)
            writer.add_scalar("train/policy_loss",       stats["policy_loss"],              update_num)
            writer.add_scalar("train/value_loss",        stats["value_loss"],               update_num)
            writer.add_scalar("train/entropy",           stats["entropy"],                  update_num)
            writer.add_scalar("train/explained_variance",stats.get("explained_variance", 0),update_num)
            writer.add_scalar("train/stage",             stage,                             update_num)
            total_a = sum(action_dist.values()) or 1
            for atype in ActionType:
                pct = action_dist.get(atype, 0) / total_a * 100
                writer.add_scalar(f"actions/{atype.name}", pct, update_num)

        # Explained-variance warning
        ev = stats.get("explained_variance", 1.0)
        if ev < RF_EXP_VAR:
            _log(f"{RED}⚠  Explained variance {ev:.3f} < {RF_EXP_VAR} — "
                 f"value head may be under-fitting{RESET}")

        

        # Periodic checkpoint
        if update_num % CHECKPOINT_INTERVAL == 0:
            _save(network, trainer, update_num, stage, delta_history,
                os.path.join(ck_dir, f"update_{update_num}.pt"),
                reward_clip=reward_clip, ev_stable_count=ev_stable_count, stage_start_update=stage_start_update)
        # Always refresh latest.pt
        _save(network, trainer, update_num, stage, delta_history,
              os.path.join(ck_dir, "latest.pt"),
              reward_clip=reward_clip, ev_stable_count=ev_stable_count, stage_start_update=stage_start_update)

        # Self-play snapshot
        if update_num % SNAPSHOT_INTERVAL == 0:
            pool.save_snapshot(network, update_num)
            _log(f"Self-play snapshot saved  (pool size: {len(pool)})")
            if writer:
                writer.add_scalar("pool/size", len(pool), update_num)

        # Bot-ladder evaluation
        if update_num % LADDER_EVAL_INTERVAL == 0:
            _log("Running ladder evaluation…")
            eval_res = _mini_eval(network)
            line = "  eval:  " + "   ".join(
                f"vs {n}={d:>+.1f}" for n, d in eval_res.items()
            )
            print(line, flush=True)
            _write_csv(eval_csv, [update_num,
                *[f"{eval_res.get(b, 0.0):.4f}"
                  for b in ("RandomBot", "CallBot", "TightBot", "PositionBot")]])
            if writer:
                for bot_name, delta in eval_res.items():
                    writer.add_scalar(f"eval/vs_{bot_name}", delta, update_num)

        # Curriculum graduation
        updates_in_stage = update_num - stage_start_update
        min_updates = MIN_UPDATES_PER_STAGE.get(stage)

        if min_updates is not None and updates_in_stage >= min_updates:
            next_stage = stage + 1
            stage = next_stage
            trainer.set_reward_clip(REWARD_CLIP_PER_STAGE[stage])
            _log(f"Reward clip set to ±{REWARD_CLIP_PER_STAGE[stage]:.1f} BB  (stage {stage})")
            stage_start_update = update_num
            delta_history.clear()
            _update_restricted_actions(training_bots, stage)
            _log(f"{CYAN}Graduated to stage {stage}{RESET}")
            if writer:
                writer.add_text("graduation", f"Stage {stage} at update {update_num}", update_num)
            _save(network, trainer, update_num, stage, delta_history,
                os.path.join(ck_dir, f"stage{stage}_start.pt"),
                reward_clip=reward_clip, ev_stable_count=ev_stable_count, stage_start_update=stage_start_update)

# Entry Point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a poker RL bot.  Named runs keep experiments isolated."
    )
    parser.add_argument(
        "--name", "-n",
        type=str, default=None,
        metavar="RUN_NAME",
        help=(
            "Name for this training run.  All artifacts are stored under "
            "checkpoints/<name>/, logs/<name>/, etc.  "
            "If omitted you are shown an interactive menu."
        ),
    )
    args = parser.parse_args()
    train(run_name=args.name)