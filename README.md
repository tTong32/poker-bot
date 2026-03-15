# Poker Engine — 6-Player No-Limit Texas Hold'em

A complete Python framework for playing, simulating, and training reinforcement-learning agents in 6-player No-Limit Texas Hold'em. Includes a rule engine, several hand-coded bots, a PPO-trained neural network bot, a CLI for human play, and a full training pipeline with logging, evaluation, and TensorBoard support.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Playing Against Bots (CLI)](#playing-against-bots-cli)
4. [Running Bot Simulations](#running-bot-simulations)
5. [Training the RL Bot](#training-the-rl-bot)
   - [Named Runs](#named-runs)
   - [Curriculum Stages](#curriculum-stages)
   - [Artifacts & Directory Layout](#artifacts--directory-layout)
   - [Resuming Training](#resuming-training)
   - [TensorBoard](#tensorboard)
6. [Evaluating a Trained Bot](#evaluating-a-trained-bot)
7. [Using a Trained Bot in CLI or Simulation](#using-a-trained-bot-in-cli-or-simulation)
8. [Architecture Reference](#architecture-reference)
   - [Game Engine](#game-engine)
   - [Built-in Bots](#built-in-bots)
   - [Observation Vector](#observation-vector)
   - [Neural Network](#neural-network)
   - [PPO Trainer](#ppo-trainer)
9. [Logging & Metrics](#logging--metrics)
   - [CSV Logs](#csv-logs)
   - [Red-Flag Warnings](#red-flag-warnings)
10. [Configuration Reference](#configuration-reference)
11. [Extending the Project](#extending-the-project)

---

## Project Structure

```
poker_engine/           ← game engine (rule engine, bots, CLI, simulation)
│
├── action.py           Action / ActionType dataclasses
├── bots.py             RandomBot, CallBot, TightBot, PositionBot
├── card.py             Card dataclass (rank 2–14, suit 0–3)
├── cli.py              Human vs bots terminal interface
├── deck.py             Deck with seeded shuffle
├── evaluator.py        7-card hand evaluator (returns comparable tuple)
├── game.py             PokerGame — applies actions, manages streets
├── observation.py      Encodes GameState → 414-float numpy vector
├── player.py           Player dataclass
├── session.py          GameSession — runs multi-hand sessions, wires bots
├── side_pots.py        Side pot calculator for all-in situations
├── simulate.py         Headless bot-vs-bot simulation with setup wizard
├── snapshot.py         Game cloning for tree search / CFR
└── state.py            GameState / BettingRound dataclasses

training_bot/           ← RL training pipeline
│
├── network.py          PokerNetwork — actor-critic neural network
├── ppo.py              PPOTrainer — GAE + clipped surrogate loss
├── training_bot.py     TrainingBot — collects experiences during play
├── rl_bot.py           RLBot — inference-only bot for play / eval
├── self_play_pool.py   SelfPlayPool — rolling snapshot pool (stage 3)
├── train.py            Main training loop with logging and curriculum
└── eval.py             Standalone evaluation script

requirements.txt
README.md
```

---

## Installation

**Python 3.11+** is required.

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:
```
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.13.0
```

TensorBoard is optional — training works without it and prints a notice if it is not installed.

---

## Playing Against Bots (CLI)

```bash
python -m poker_engine.cli
```

A setup wizard asks for your seat (0–5), starting stack, and what type of bot you want to face. Bot options:

| Key | Bot |
|-----|-----|
| 1 | TightBot |
| 2 | PositionBot *(default)* |
| 3 | CallBot |
| 4 | RandomBot |
| 5 | RLBot — loads a trained checkpoint *(requires PyTorch)* |

If you choose **RLBot** you will be prompted for a checkpoint path, for example:

```
checkpoints/my_experiment/latest.pt
```

If the file is not found, it falls back to PositionBot.

**In-game controls** (shown on screen each action):

| Key | Action |
|-----|--------|
| F | Fold |
| K | Check |
| C | Call |
| H | Bet ½ pot |
| P | Bet pot |
| D | Bet 2× pot |
| A | All-in |

---

## Running Bot Simulations

```bash
python -m poker_engine.simulate
```

The setup wizard lets you assign any bot type to each of the 6 seats independently. You then set the number of hands per session and how many sessions to run. After all sessions complete, a summary is printed showing win counts, win rates, and average final stacks across all sessions.

---

## Training the RL Bot

```bash
# Start a new named run
python -m training_bot.train --name my_experiment

# Interactive run selection (no flag needed)
python -m training_bot.train
```

### Named Runs

Every training run lives under its own directory. All artifacts for `my_experiment` are stored under:

```
checkpoints/my_experiment/
snapshots/my_experiment/
logs/my_experiment/
tensorboard/my_experiment/
```

Starting a second experiment with `--name second_run` is completely isolated.

### Startup Prompt

If you run without `--name`, the trainer scans `checkpoints/` and presents a numbered menu of existing runs with their update count, stage, and average chip delta. You can:

- Type the number of a run to resume it
- Press **Enter** to resume the most recently modified run
- Type **N** or a new name to start a fresh run

If no runs exist yet, one is created automatically with a timestamped name.

### Curriculum Stages

Training progresses through three stages automatically:

| Stage | Opponents | Graduation threshold |
|-------|-----------|----------------------|
| 1 | RandomBot | avg chip delta > **+5.0 BB/hand** over 10,000 hands |
| 2 | Random mix of CallBot, TightBot, PositionBot | avg chip delta > **+2.0 BB/hand** over 10,000 hands |
| 3 | Self-play pool (see below) | runs indefinitely |

The threshold window requires at least `DELTA_HISTORY_SIZE = 10,000` hands before graduation is even checked, so early noise does not cause premature promotion. 

* Note that BB stands for the Big Blind Value (meaning the bot must win by the equivalent of 5 big blinds on average)

**Stage 3 — Self-Play Pool**

Every 50 PPO updates the current network weights are saved as a frozen snapshot. A rolling pool of up to 20 snapshots is maintained (oldest evicted when full). When building a stage-3 session, opponents are sampled at:

- **50%** current network (sharpens against the latest strategy)
- **30%** recent snapshots (newest third of pool)
- **20%** older snapshots (prevents forgetting early strategies)

### Artifacts & Directory Layout

```
checkpoints/<run>/
    latest.pt          ← always the most recent update
    update_50.pt       ← saved every 50 PPO updates
    update_100.pt
    stage2_start.pt    ← saved at stage 2 graduation
    stage3_start.pt    ← saved at stage 3 graduation

snapshots/<run>/
    snapshot_50.pt     ← self-play pool (one per 50 updates, max 20)
    snapshot_100.pt
    ...

logs/<run>/
    training_log.csv   ← one row per PPO update
    hands_log.csv      ← one row per hand
    eval_log.csv       ← one row per ladder evaluation

tensorboard/<run>/
    events.out.*
```

### Resuming Training

At the interactive startup prompt, select an existing run number (or just press Enter for the latest). The trainer reloads network weights, optimizer state, update counter, stage, and the rolling delta history — training continues exactly where it left off. CSV and TensorBoard logs are **appended to**, never overwritten.

### TensorBoard

```bash
# In a separate terminal while training is running:
tensorboard --logdir tensorboard/my_experiment
```

Or to watch all runs at once:

```bash
tensorboard --logdir tensorboard
```

Metrics logged per update: `avg_delta`, `policy_loss`, `value_loss`, `entropy`, `explained_variance`, `stage`, self-play pool size, action distribution percentages, and ladder evaluation results.

---

## Evaluating a Trained Bot

```bash
# Auto-detect the most recent checkpoint
python -m training_bot.eval

# Explicit checkpoint path
python -m training_bot.eval --checkpoint checkpoints/my_experiment/latest.pt

# Fewer hands for a quick check
python -m training_bot.eval --checkpoint checkpoints/my_experiment/update_500.pt --hands 2000
```

The evaluation script runs the bot as seat 0 against 5 opponents of each type separately (default 10,000 hands per opponent type) and reports:

- **Avg chip delta per hand** (in big blinds) vs each opponent
- **Win rate** vs each opponent
- **Action distribution** across all hands combined, with ⚠ warnings for red flags
- **Pass/fail checklist** against the red-flag thresholds

Example output:

```
  vs RandomBot        ...   +8.213 BB/hand   win=34.2%  (10,000 hands)
  vs CallBot          ...   +3.881 BB/hand   win=29.7%  (10,000 hands)
  vs TightBot         ...   +1.342 BB/hand   win=22.1%  (10,000 hands)
  vs PositionBot      ...   +0.773 BB/hand   win=21.4%  (10,000 hands)

  Action distribution:
    FOLD                  28.3%  ████████████████
    CHECK                 19.1%  ████████████
    CALL                  21.4%  ████████████
    BET_HALF_POT          15.2%  ████████
    BET_POT               10.8%  █████
    BET_DOUBLE_POT         2.9%  █
    ALL_IN                 2.3%  █

  Red-flag checklist:
    [PASS]  Fold rate  < 60 %                     28.3%
    [PASS]  All-in rate < 30 %                     2.3%
    [PASS]  Passive rate < 80 %                   40.5%
    [PASS]  Positive delta vs Random              +8.213 BB/hand
    [PASS]  Positive delta vs Call                +3.881 BB/hand
```

---

## Using a Trained Bot in CLI or Simulation

**In the CLI wizard**, select **[5] RLBot** and enter the path to a checkpoint. The bot plays using greedy action selection (argmax) by default, which is recommended for human-facing play.

**In the simulation wizard**, assign **[5] RLBot** to any seat(s). Multiple seats can load from the same checkpoint — the network weights are loaded once and shared across all seats using `RLBot.make_factory()`.

**Programmatically:**

```python
from training_bot.rl_bot import RLBot
from poker_engine.session import GameSession

# Load a bot for a single seat
bot = RLBot.from_checkpoint(player_id=0, checkpoint_path="checkpoints/my_run/latest.pt")

# Or create a factory that shares one network across multiple seats
factory = RLBot.make_factory("checkpoints/my_run/latest.pt", starting_stack=1000.0)
session = GameSession(starting_stack=1000.0)
session.fill_empty_seats_with(factory)
result = session.run(max_hands=100)
```

---

## Architecture Reference

### Game Engine

**`PokerGame`** (`game.py`) is the core rule engine. It holds mutable state and exposes three methods that form the complete public API:

```python
state  = game.start_new_hand()        # deal cards, post blinds, return initial state
legal  = game.get_legal_actions()     # list of Action objects for current player
state  = game.apply_action(action)    # mutate state, advance street if needed
```

**Blind structure:** SB = BB/2, BB = starting_stack/100. With the default stack of 1000, blinds are 5/10.

**Betting:** Raise cap of 4 per street. Available bet sizes: ½ pot, pot, 2× pot, all-in. All sizes are computed dynamically from the current pot.

**Side pots** are handled correctly — players can only win up to the amount they covered from each opponent.

**`GameSession`** (`session.py`) wraps `PokerGame` to run multi-hand sessions. Assign bots with `assign_bot(seat, bot)`, mark human seats with `set_human_seat(seat)`, then call `run(max_hands=N)`. Hooks `on_state_change` and `on_hand_end` are available for UI and logging layers.

**`GameSnapshot`** (`snapshot.py`) provides cheap, complete game cloning via `take_snapshot()` / `restore_snapshot()` / `clone_game()`. Useful for tree search, CFR, or any algorithm that needs to branch from a decision point.

### Built-in Bots

All bots subclass `Bot` from `bots.py` and implement `choose_action(state, legal_actions) → Action`.

| Bot | Strategy |
|-----|----------|
| **RandomBot** | Uniform random over legal actions. Good for stress testing. |
| **CallBot** | Always checks or calls. Never raises. Calling station baseline. |
| **TightBot** | Preflop: plays top ~40% of hands by strength, raises with top ~35%. Postflop: bets made hands (pair+), folds weak holdings to pressure. |
| **PositionBot** | Extends TightBot with positional awareness: loosens thresholds in late position, bluffs on the river with 15% probability when last to act. |

### Observation Vector

`encode_observation()` in `observation.py` converts a `GameState` into a **414-element float32 numpy array** from the perspective of a single player. The player sees only their own hole cards.

| Slice | Size | Description |
|-------|------|-------------|
| `[0:104]` | 104 | Hole cards — 2 × 52 one-hot, order-invariant |
| `[104:364]` | 260 | Board cards — 5 × 52 one-hot, zero-padded |
| `[364]` | 1 | Pot / starting_stack |
| `[365:371]` | 6 | Stack per seat, normalised |
| `[371:377]` | 6 | Current street bet per seat, normalised |
| `[377:383]` | 6 | Total invested this hand per seat, normalised |
| `[383:389]` | 6 | Folded flags |
| `[389:395]` | 6 | All-in flags |
| `[395:401]` | 6 | Position one-hot relative to dealer |
| `[401:405]` | 4 | Street one-hot (preflop/flop/turn/river) |
| `[405]` | 1 | Current bet to match, normalised |
| `[406]` | 1 | Amount to call, normalised |
| `[407:414]` | 7 | Legal action mask |

### Neural Network

**`PokerNetwork`** (`network.py`) is a shared actor-critic network:

```
Input (414)
  → Linear(414, 256) + LayerNorm + ReLU
  → Linear(256, 256) + LayerNorm + ReLU
  → Linear(256, 128) + LayerNorm + ReLU
  → Policy head: Linear(128, 7) + Softmax   →  action probabilities
  → Value head:  Linear(128, 1)              →  state value estimate
```

Weights are initialised with Kaiming Normal (ReLU variant). All 6 training seats share **one network instance** — they each see their own perspective but contribute to the same gradient updates.

### PPO Trainer

**`PPOTrainer`** (`ppo.py`) implements Proximal Policy Optimization with Generalized Advantage Estimation.

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `CLIP_EPSILON` | 0.2 | Surrogate objective clip range |
| `VALUE_COEFF` | 0.5 | Value loss weight in combined loss |
| `ENTROPY_COEFF` | 0.01 | Entropy bonus weight |
| `GAMMA` | 0.999 | Future reward discount |
| `LAMBDA` | 0.95 | GAE smoothing factor |
| `UPDATE_PASSES` | 4 | Times each rollout is reused |
| `LEARNING_RATE` | 3e-4 | Adam optimizer |
| Gradient clip | 0.5 | `max_norm` for `clip_grad_norm_` |

**Reward signal:** chip delta for the hand, normalised by big blind. A hand that ends +50 chips above the starting stack with a BB of 10 produces a reward of +5.0.

**Explained variance** is computed once per update (before weight changes) as `1 - var(returns - values) / var(returns)`. Values close to 1.0 indicate the value head is accurately predicting outcomes.

---

## Logging & Metrics

### CSV Logs

All logs live under `logs/<run_name>/` and are always **appended to** — resuming training continues the same files.

**`training_log.csv`** — one row per PPO update:

```
timestamp, update_num, stage, avg_delta, policy_loss, value_loss, entropy,
explained_variance, fold%, check%, call%, bet_half%, bet_pot%, bet_double%, allin%
```

**`hands_log.csv`** — one row per hand played:

```
session_num, hand_num, stage, delta_seat0, delta_seat1, delta_seat2
```

**`eval_log.csv`** — one row per ladder evaluation (every 100 updates):

```
update_num, vs_RandomBot, vs_CallBot, vs_TightBot, vs_PositionBot
```

### Red-Flag Warnings

Warnings are printed to the console during training if any threshold is crossed:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Fold rate | > 60% | Bot is folding excessively — likely not learning to contest pots |
| All-in rate | > 30% | Bot is over-shoving — likely exploiting short-term variance |
| Passive rate (check+call) | > 80% | Bot is a calling station — not building pots or applying pressure |
| Explained variance | < 0.30 | Value head is unreliable — GAE estimates will be noisy |

The same thresholds are used in the pass/fail checklist printed by `eval.py`.

---

## Configuration Reference

All constants live at the top of their respective files and can be edited directly.

**`training_bot/train.py`**

| Constant | Default | Description |
|----------|---------|-------------|
| `STARTING_STACK` | 1000.0 | Chips each player starts with |
| `BIG_BLIND` | 10.0 | Derived as `STARTING_STACK / 100` |
| `ROLLOUT_HANDS` | 200 | Hands collected before each PPO update |
| `DELTA_HISTORY_SIZE` | 10,000 | Rolling window size for avg chip delta |
| `CHECKPOINT_INTERVAL` | 50 | PPO updates between numbered checkpoints |
| `LADDER_EVAL_INTERVAL` | 100 | PPO updates between mini evaluations |
| `LADDER_EVAL_HANDS` | 1,000 | Hands per opponent in mini eval |
| `STAGE_1_THRESHOLD` | 5.0 | BB/hand avg to graduate from stage 1 |
| `STAGE_2_THRESHOLD` | 2.0 | BB/hand avg to graduate from stage 2 |
| `TRAINING_SEATS` | [0, 1, 2] | Seats assigned to TrainingBot |

**`training_bot/ppo.py`** — see [PPO Trainer](#ppo-trainer) table above.

**`training_bot/self_play_pool.py`**

| Constant | Default | Description |
|----------|---------|-------------|
| `SNAPSHOT_INTERVAL` | 50 | PPO updates between pool snapshots |
| `MAX_POOL_SIZE` | 20 | Maximum snapshots retained on disk |

---

## Extending the Project

### Writing a new bot

Subclass `Bot` in `bots.py` (or any file) and implement one method:

```python
from poker_engine.bots import Bot
from poker_engine.action import Action
from poker_engine.state import GameState
from typing import List

class MyBot(Bot):
    def __init__(self, player_id: int):
        super().__init__(player_id, name="MyBot")

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        # state is read-only — do not mutate it
        # must return one of the objects in legal_actions
        return legal_actions[0]
```

Assign it to a session the same way as any other bot:

```python
session.assign_bot(seat=3, bot=MyBot(player_id=3))
```

### Using the game engine programmatically

```python
from poker_engine.game import PokerGame
from poker_engine.bots import TightBot

game = PokerGame(starting_stack=1000.0, seed=42)
state = game.start_new_hand()

while not state.terminal:
    legal = game.get_legal_actions()
    # pick any action from legal — here we just take the first one
    state = game.apply_action(legal[0])

print("Winner:", state.winners)
```

### Cloning the game for search algorithms

```python
from poker_engine.snapshot import take_snapshot, restore_snapshot

snap = take_snapshot(game)

branch_a = restore_snapshot(snap)
branch_a.apply_action(fold_action)

branch_b = restore_snapshot(snap)
branch_b.apply_action(call_action)
# original game is untouched
```

### Adding a new action type

1. Add a new variant to `ActionType` in `action.py`.
2. Handle it in `PokerGame._execute_action()` in `game.py`.
3. Add it to `get_legal_actions()` when appropriate.
4. Update `NUM_ACTIONS` in `observation.py` and extend `_MASK_END`.
5. Update the policy head output size in `network.py`.
6. Update `_ACTION_LABELS` in `cli.py`.