"""
train.py — Training environment for the poker RL bot.

Run with:
    python -m training_bot.train
    python -m training_bot.train --name my_experiment

Directory layout
----------------------------
    runs/checkpoints/<run_name>/  latest.pt, update_N.pt, stageN_start.pt
    runs/snapshots/<run_name>/    snapshot_N.pt  (self-play pool)
    runs/logs/<run_name>/         training_log.csv, hands_log.csv, eval_log.csv
    runs/tensorboard/<run_name>/  TensorBoard event files

Curriculum stages
-----------------
    1 — Action-restricted (no large bets/all-in)  → ladder: RandomBot > 2 BB/hand
    2 — Moderate restriction (no all-in)           → ladder: CallBot > 0.5, TightBot > -0.5
    3 — Light restriction (no all-in)              → ladder: TightBot > 0.3, PositionBot > -0.5
    4 — Unrestricted self-play                     → runs indefinitely
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

# Configuration

STARTING_STACK        = 1000.0
BIG_BLIND             = STARTING_STACK / 100.0
ROLLOUT_HANDS         = 1000       # hands per PPO update (was 5000 — reduced to cut staleness)
DELTA_HISTORY_SIZE    = 10_000
CHECKPOINT_INTERVAL   = 50
LADDER_EVAL_INTERVAL  = 100
LADDER_EVAL_HANDS     = 1_000

# Minimum PPO updates before a stage can graduate (prevents premature promotion)
MIN_UPDATES_PER_STAGE = {1: 125, 2: 175, 3: 225, 4: None}

# Performance thresholds checked at each ladder evaluation.
# ALL listed conditions must be met simultaneously.
# Keys match the _LADDER_OPPONENTS dict in _mini_eval().
STAGE_GRAD_THRESHOLDS = {
    1: {},#{"RandomBot":   2.0},                                   # comfortably beat random
    2: {},#{"CallBot":     0.5, "TightBot":    -0.5},              # positive vs call, not crushed by tight
    3: {},#{"TightBot":    0.3, "PositionBot": -0.5},              # slightly positive vs tight
    4: None,                                                   # indefinite
}

# Reward clip applied to terminal chip delta (BB) before GAE
REWARD_CLIP_PER_STAGE = {1: 2.0, 2: 5.0, 3: 8.0, 4: 12.0}

# Action restrictions per stage (enforced in TrainingBot.choose_action)
STAGE1_RESTRICTED = {ActionType.BET_DOUBLE_POT, ActionType.BET_POT, ActionType.ALL_IN}
STAGE2_RESTRICTED = {ActionType.BET_DOUBLE_POT, ActionType.ALL_IN}
STAGE3_RESTRICTED = {ActionType.ALL_IN}
STAGE4_RESTRICTED: set = set()

# Entropy coefficient schedule per stage (decayed linearly over stage duration)
# Floor is enforced in PPOTrainer.set_entropy_coeff() at ENTROPY_FLOOR = 0.01
ENTROPY_START = {1: 0.08, 2: 0.06, 3: 0.04, 4: 0.03}
ENTROPY_END   = {1: 0.03, 2: 0.02, 3: 0.015, 4: 0.01}

# Self-play pool seeding parameters
TIGHTBOT_SEED_POOL_SIZE = 10
TIGHTBOT_SEED_RATE      = 0.4
TRAINING_SEATS          = [0, 1, 2, 3, 4, 5]

# Red-flag warning thresholds (console warnings only)
RF_FOLD_RATE      = 0.60
RF_ALLIN_RATE     = 0.30
RF_PASSIVE_RATE   = 0.80
RF_EXP_VAR        = 0.30
RF_ALLIN_COLLAPSE = 0.50

# Terminal colours

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"


# Logging helpers (unchanged)

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str):
    print(f"  [{time.strftime('%H:%M:%S')}]  {msg}", flush=True)

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)

def _init_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)

def _write_csv(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def _action_pcts(dist: Counter) -> List[str]:
    total = sum(dist.values()) or 1
    return [f"{dist.get(atype, 0) / total * 100:.2f}" for atype in ActionType]

def _log_update(update_num, stage, avg, stats, dist):
    ev      = stats.get("explained_variance", 0.0)
    ev_warn = f"  {RED}⚠ LOW EV{RESET}" if ev < RF_EXP_VAR else ""
    total   = sum(dist.values()) or 1
    print(
        f"  update={update_num:>5}  stage={stage}  "
        f"avg_delta={avg:>+7.3f} BB/hand  "
        f"policy_loss={stats['policy_loss']:>7.4f}  "
        f"value_loss={stats['value_loss']:>7.4f}  "
        f"entropy={stats['entropy']:>6.4f}  "
        f"ev={ev:>5.3f}{ev_warn}",
        flush=True,
    )
    dist_str = "  ".join(
        f"{atype.name[:4]}={dist.get(atype, 0) / total * 100:.0f}%"
        for atype in ActionType
    )
    print(f"           {dist_str}", flush=True)

    fold_r    = dist.get(ActionType.FOLD,  0) / total
    allin_r   = dist.get(ActionType.ALL_IN, 0) / total
    passive_r = (dist.get(ActionType.CHECK, 0) + dist.get(ActionType.CALL, 0)) / total

    if fold_r    > RF_FOLD_RATE:
        print(f"  {RED}⚠ FOLD RATE {fold_r:.1%} > {RF_FOLD_RATE:.0%}{RESET}")
    if allin_r   > RF_ALLIN_RATE:
        print(f"  {RED}⚠ ALL-IN RATE {allin_r:.1%} > {RF_ALLIN_RATE:.0%}{RESET}")
    if allin_r   > RF_ALLIN_COLLAPSE:
        print(f"  {RED}⚠ ALL-IN COLLAPSE{RESET}")
    if passive_r > RF_PASSIVE_RATE:
        print(f"  {RED}⚠ PASSIVE RATE {passive_r:.1%} > {RF_PASSIVE_RATE:.0%}{RESET}")


# Checkpointing

def _save(network, trainer, update_num, stage, delta_history, path,
          reward_clip=3.0, stage_start_update=0):
    _ensure(os.path.dirname(path))
    avg = sum(delta_history) / len(delta_history) if delta_history else 0.0
    torch.save({
        # Network
        "network_state":        network.state_dict(),
        # Optimizers (new single-optimizer structure)
        "optimizer":            trainer.optimizer.state_dict(),
        "value_optimizer":      trainer.value_optimizer.state_dict(),
        # Normalizer state
        "normalizer":           trainer.normalizer.state_dict(),
        # Training progress
        "update_num":           update_num,
        "stage":                stage,
        "delta_history":        list(delta_history),
        "avg_delta":            avg,
        "reward_clip":          reward_clip,
        "stage_start_update":   stage_start_update,
    }, path)
    _log(f"Checkpoint saved → {path}")


def _load(network, trainer, path):
    ck = torch.load(path, weights_only=False)
    network.load_state_dict(ck["network_state"])

    # Support both new (single optimizer) and old (dual optimizer) checkpoints
    if "optimizer" in ck:
        trainer.optimizer.load_state_dict(ck["optimizer"])
    elif "policy_optimizer" in ck:
        # Attempt partial load from old format — backbone + policy head weights
        # will be loaded; value optimizer will start fresh.
        _log(f"{YELLOW}⚠ Old checkpoint format: loading policy_optimizer only{RESET}")
        try:
            trainer.optimizer.load_state_dict(ck["policy_optimizer"])
        except Exception:
            _log(f"{YELLOW}  Could not load old optimizer — starting optimizer fresh{RESET}")

    if "value_optimizer" in ck:
        try:
            trainer.value_optimizer.load_state_dict(ck["value_optimizer"])
        except Exception:
            _log(f"{YELLOW}  Could not load value_optimizer — starting fresh{RESET}")

    if "normalizer" in ck:
        trainer.normalizer.load_state_dict(ck["normalizer"])

    history            = deque(ck.get("delta_history", []), maxlen=DELTA_HISTORY_SIZE)
    reward_clip        = ck.get("reward_clip", 3.0)
    stage_start_update = ck.get("stage_start_update", 0)

    _log(
        f"Checkpoint loaded ← {path}  "
        f"(update={ck['update_num']}, stage={ck['stage']}, "
        f"avg_delta={ck.get('avg_delta', 0.0):+.3f})"
    )
    return ck["update_num"], ck["stage"], history, reward_clip, stage_start_update


def _read_meta(path: str) -> dict:
    try:
        ck = torch.load(path, weights_only=False)
        return {k: ck.get(k, "?") for k in ("update_num", "stage", "avg_delta")}
    except Exception:
        return {}


# Interactive startup prompt (unchanged from previous version)

def _startup(run_name: Optional[str]) -> tuple:
    ck_root = "runs/checkpoints"

    def _run_dir(name): return os.path.join(ck_root, name)
    def _latest(name):  return os.path.join(_run_dir(name), "latest.pt")

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

    existing = []
    if os.path.isdir(ck_root):
        for entry in sorted(os.listdir(ck_root)):
            lp = _latest(entry)
            if os.path.exists(lp):
                existing.append((entry, _read_meta(lp)))

    if not existing:
        name = time.strftime("run_%Y%m%d_%H%M%S")
        print(f"\n  No existing runs found.  Starting: {BOLD}{name}{RESET}")
        return _run_dir(name), False, name

    print(f"\n  {BOLD}{CYAN}Existing training runs:{RESET}")
    for i, (name, meta) in enumerate(existing, 1):
        print(
            f"    [{i}] {name:<30}  "
            f"update={meta.get('update_num','?'):>5}  "
            f"stage={meta.get('stage','?')}  "
            f"avg_delta={meta.get('avg_delta', 0.0):>+7.3f}"
        )
    print(f"    [N] Start a new run\n")
    print(f"  Choice (number, name, or Enter for latest): ", end="", flush=True)
    raw = input().strip()

    if raw.upper() == "N":
        new_name = time.strftime("run_%Y%m%d_%H%M%S")
        print(f"  Starting new run: {BOLD}{new_name}{RESET}")
        return _run_dir(new_name), False, new_name

    if raw == "":
        name, _ = existing[-1]
        print(f"  Resuming latest: {BOLD}{name}{RESET}")
        return _run_dir(name), True, name

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(existing):
            name, _ = existing[idx]
            print(f"  Resuming: {BOLD}{name}{RESET}")
            return _run_dir(name), True, name

    new_name = raw
    run_dir  = _run_dir(new_name)
    if os.path.exists(_latest(new_name)):
        meta = _read_meta(_latest(new_name))
        print(f"  Found existing run '{new_name}': update={meta.get('update_num','?')}")
        print(f"  Resume? (y/n): ", end="", flush=True)
        resume = input().strip().lower() in ("y", "yes", "")
    else:
        resume = False
        print(f"  Starting new run: {BOLD}{new_name}{RESET}")
    return run_dir, resume, new_name


# Opponent factories

_SEED_BOTS = [RandomBot, CallBot, TightBot, PositionBot]


def _make_opponent(stage: int, seat: int, pool: SelfPlayPool, current_net: PokerNetwork):
    """
    Build an opponent for one training session seat.

    While the pool is small (< TIGHTBOT_SEED_POOL_SIZE), mix in seed bots at
    TIGHTBOT_SEED_RATE to prevent pure-noise early self-play.  Once the pool
    is mature, the 50/30/20 sampling inside pool.sample_opponent_network()
    provides appropriate diversity.
    """
    if len(pool) < TIGHTBOT_SEED_POOL_SIZE and random.random() < TIGHTBOT_SEED_RATE:
        return random.choice(_SEED_BOTS)(seat)
    net = pool.sample_opponent_network(current_net)
    tag = "Cur" if net is current_net else "Old"
    return RLBot(seat, net, starting_stack=STARTING_STACK, name=f"SelfPlay({tag})")


# Session setup

def _update_restricted_actions(training_bots: dict, stage: int):
    restrictions = {
        1: STAGE1_RESTRICTED,
        2: STAGE2_RESTRICTED,
        3: STAGE3_RESTRICTED,
        4: STAGE4_RESTRICTED,
    }
    r = restrictions.get(stage, set())
    for bot in training_bots.values():
        bot.restricted_actions = r


def _setup_session(
    stage: int,
    training_bots: Dict[int, TrainingBot],
    delta_history,
    hands_csv: str,
    session_num: int,
    pool: SelfPlayPool,
    current_net: PokerNetwork,
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


# Ladder evaluation

_LADDER_OPPONENTS = {
    "RandomBot":   RandomBot,
    "CallBot":     CallBot,
    "TightBot":    TightBot,
    "PositionBot": PositionBot,
}


def _mini_eval(network: PokerNetwork) -> Dict[str, float]:
    """
    Run LADDER_EVAL_HANDS hands against each built-in bot type.
    Returns avg chip delta per hand in BB for each opponent name.
    """
    results = {}
    for bot_name, bot_cls in _LADDER_OPPONENTS.items():
        total_delta = 0.0
        total_hands = 0
        hands_left  = LADDER_EVAL_HANDS
        buf         = []

        while hands_left > 0:
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


# Graduation check

def _check_graduation(
    stage: int,
    update_num: int,
    stage_start_update: int,
    last_eval: Optional[Dict[str, float]],
) -> bool:
    """
    Return True if the bot should graduate to the next stage.

    Two gates must both be satisfied:
      1. Minimum update count for this stage has been served.
      2. The most recent ladder evaluation meets all performance thresholds.

    Gate 2 is skipped until the first ladder eval has been run.
    """
    thresholds = STAGE_GRAD_THRESHOLDS.get(stage)
    if thresholds is None:
        return False   # stage 4 runs forever

    min_updates = MIN_UPDATES_PER_STAGE.get(stage)
    if min_updates is not None and (update_num - stage_start_update) < min_updates:
        return False   # haven't served minimum time yet

    if last_eval is None:
        return False   # no eval data yet

    for bot_name, threshold in thresholds.items():
        if last_eval.get(bot_name, -999.0) < threshold:
            return False

    return True


# Average delta helper

def _avg(history) -> float:
    return sum(history) / len(history) if history else 0.0


# Main training loop

def train(run_name: Optional[str] = None):

    # Startup
    run_dir, resume, run_name = _startup(run_name)

    ck_dir   = run_dir
    snap_dir = os.path.join("runs/snapshots",   run_name)
    log_dir  = os.path.join("runs/logs",        run_name)
    tb_dir   = os.path.join("runs/tensorboard", run_name)

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
        _log(f"TensorBoard → {tb_dir}")
    except ImportError:
        writer = None
        _log("tensorboard not installed — skipping")

    # Network, trainer, shared buffer
    network       = PokerNetwork()
    trainer       = PPOTrainer(network)
    shared_buffer : List[Experience] = []

    training_bots = {
        seat: TrainingBot(
            player_id      = seat,
            network        = network,
            shared_buffer  = shared_buffer,
            starting_stack = STARTING_STACK,
            big_blind      = BIG_BLIND,
        )
        for seat in TRAINING_SEATS
    }

    pool               = SelfPlayPool(snapshot_dir=snap_dir)
    update_num         = 0
    stage              = 1
    delta_history      = deque(maxlen=DELTA_HISTORY_SIZE)
    session_count      = 0
    hands_pending      = 0
    stage_start_update = 0
    last_eval_results: Optional[Dict[str, float]] = None  # tracks most recent ladder eval

    # Resume
    if resume:
        latest = os.path.join(ck_dir, "latest.pt")
        update_num, stage, delta_history, reward_clip, stage_start_update = (
            _load(network, trainer, latest)
        )

    trainer.set_reward_clip(REWARD_CLIP_PER_STAGE[stage])
    _log(f"Reward clip ±{REWARD_CLIP_PER_STAGE[stage]:.1f} BB  (stage {stage})")
    _log(f"Run '{run_name}'  stage={stage}  {'resumed' if resume else 'fresh start'}")
    _update_restricted_actions(training_bots, stage)

    if writer:
        writer.add_text("run_info",
                        f"name={run_name}  stage={stage}  resumed={resume}",
                        global_step=update_num)

    # Training loop
    while True:

        # Collect one session
        session_count += 1
        session = _setup_session(
            stage, training_bots, delta_history,
            hands_csv, session_count, pool, network,
        )
        session.run(max_hands=ROLLOUT_HANDS)
        hands_pending += session.hand_number

        for seat in TRAINING_SEATS:
            training_bots[seat].reset_starting_stack(STARTING_STACK)

        avg = _avg(delta_history)
        print(
            f"\r  session={session_count}  "
            f"pending={hands_pending}/{ROLLOUT_HANDS}  "
            f"stage={stage}  avg_delta={avg:>+7.3f} BB/hand",
            end="", flush=True,
        )

        if hands_pending < ROLLOUT_HANDS or not shared_buffer:
            continue

        print()   # end the \r line

        # PPO update
        action_dist   = Counter(ActionType(e.action) for e in shared_buffer)
        stats         = trainer.update(shared_buffer)
        shared_buffer.clear()
        hands_pending = 0
        update_num   += 1
        avg           = _avg(delta_history)

        _log_update(update_num, stage, avg, stats, action_dist)

        # Entropy decay within stage
        updates_in_stage = update_num - stage_start_update
        min_u  = MIN_UPDATES_PER_STAGE.get(stage) or 200
        progress = min(1.0, updates_in_stage / min_u)
        entropy_coeff = (ENTROPY_START[stage]
                         + (ENTROPY_END[stage] - ENTROPY_START[stage]) * progress)
        trainer.set_entropy_coeff(entropy_coeff)

        # CSV 
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
            writer.add_scalar("train/avg_delta",          avg,                                update_num)
            writer.add_scalar("train/policy_loss",        stats["policy_loss"],               update_num)
            writer.add_scalar("train/value_loss",         stats["value_loss"],                update_num)
            writer.add_scalar("train/entropy",            stats["entropy"],                   update_num)
            writer.add_scalar("train/explained_variance", stats.get("explained_variance", 0), update_num)
            writer.add_scalar("train/entropy_coeff",      entropy_coeff,                      update_num)
            writer.add_scalar("train/stage",              stage,                              update_num)
            total_a = sum(action_dist.values()) or 1
            for atype in ActionType:
                writer.add_scalar(f"actions/{atype.name}",
                                  action_dist.get(atype, 0) / total_a * 100,
                                  update_num)

        # EV warning
        ev = stats.get("explained_variance", 1.0)
        if ev < RF_EXP_VAR:
            _log(f"{RED}⚠  EV={ev:.3f} < {RF_EXP_VAR} — value head under-fitting{RESET}")

        # --- Periodic checkpoint ---
        if update_num % CHECKPOINT_INTERVAL == 0:
            _save(network, trainer, update_num, stage, delta_history,
                  os.path.join(ck_dir, f"update_{update_num}.pt"),
                  stage_start_update=stage_start_update)
        _save(network, trainer, update_num, stage, delta_history,
              os.path.join(ck_dir, "latest.pt"),
              stage_start_update=stage_start_update)

        # Self-play snapshot
        if update_num % SNAPSHOT_INTERVAL == 0:
            pool.save_snapshot(network, update_num)
            _log(f"Self-play snapshot saved  (pool size: {len(pool)})")
            if writer:
                writer.add_scalar("pool/size", len(pool), update_num)

        # Ladder evaluation
        if update_num % LADDER_EVAL_INTERVAL == 0:
            _log("Running ladder evaluation…")
            last_eval_results = _mini_eval(network)
            line = "  eval:  " + "   ".join(
                f"vs {n}={d:>+.1f}" for n, d in last_eval_results.items()
            )
            print(line, flush=True)
            _write_csv(eval_csv, [update_num,
                *[f"{last_eval_results.get(b, 0.0):.4f}"
                  for b in ("RandomBot", "CallBot", "TightBot", "PositionBot")]])
            if writer:
                for bot_name, delta in last_eval_results.items():
                    writer.add_scalar(f"eval/vs_{bot_name}", delta, update_num)

        # Curriculum graduation (dual-gate: time + performance)
        if _check_graduation(stage, update_num, stage_start_update, last_eval_results):
            next_stage = stage + 1
            _log(f"{CYAN}━━━━  Graduated to stage {next_stage}  ━━━━{RESET}")
            if last_eval_results:
                for bot, val in last_eval_results.items():
                    _log(f"  ladder: vs {bot} = {val:>+.2f} BB/hand")

            _save(network, trainer, update_num, stage, delta_history,
                  os.path.join(ck_dir, f"stage{next_stage}_start.pt"),
                  stage_start_update=stage_start_update)

            stage              = next_stage
            stage_start_update = update_num
            delta_history.clear()
            last_eval_results  = None   # reset so graduation can't fire immediately again
            trainer.set_reward_clip(REWARD_CLIP_PER_STAGE.get(stage, 12.0))
            _update_restricted_actions(training_bots, stage)

            if writer:
                writer.add_text("graduation",
                                f"Stage {stage} at update {update_num}",
                                update_num)


# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a poker RL bot with named, isolated experiment runs."
    )
    parser.add_argument(
        "--name", "-n", type=str, default=None, metavar="RUN_NAME",
        help=(
            "Name for this training run.  All artifacts are stored under "
            "runs/checkpoints/<name>/ etc.  "
            "Omit to see an interactive resume menu."
        ),
    )
    args = parser.parse_args()
    train(run_name=args.name)