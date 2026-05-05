# Poker Engine — 6-Player No-Limit Texas Hold'em

Python toolkit for **rule-complete 6-max NLHE**, **human/bot play**, **batch simulations**, and **PyTorch PPO** training with named experiment runs, CSV + TensorBoard logging, and an offline evaluation harness.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Project structure](#project-structure)
3. [Installation](#installation)
4. [Playing against bots (CLI)](#playing-against-bots-cli)
5. [Running bot simulations](#running-bot-simulations)
6. [Training the RL bot](#training-the-rl-bot)
7. [Evaluating a trained bot](#evaluating-a-trained-bot)
8. [Using a trained bot in code](#using-a-trained-bot-in-code)
9. [Architecture reference](#architecture-reference)
10. [Logging & metrics](#logging--metrics)
11. [Configuration reference](#configuration-reference)
12. [Extending the project](#extending-the-project)

---

## Highlights

- **Game engine** (`poker_engine/`): betting rounds, dynamic bet sizing (½-pot, pot, 2× pot, all-in), raise caps, side pots, seeded shuffle, hand evaluator, multi-hand sessions, terminal CLI, snapshot-based cloning for search-style algorithms.
- **RL stack** (`training_bot/`): fixed-size observations (`OBS_SIZE` = **458**), shared actor–critic (`PokerNetwork`), **PPO + GAE**, running return normalization, four-stage curriculum (action masking, reward clipping, entropy schedules), artifacts under `runs/`.
- **Evaluation**: `training_bot.eval` benchmarks a checkpoint vs Random / Call / Tight / Position bots with red-flag sanity checks.

**Requirements:** Python **3.11+**. Dependencies: `requirements.txt` (`torch`, `numpy`, `tensorboard`). TensorBoard is optional; training prints a notice if it is missing.

---

## Project structure

```
poker_engine/
├── action.py           Action / ActionType
├── bots.py             RandomBot, CallBot, TightBot, PositionBot
├── card.py             Card (rank 2–14, suit 0–3)
├── cli.py              Human vs bots (terminal)
├── deck.py             Deck + shuffle
├── evaluator.py        7-card hand evaluator
├── game.py             PokerGame — rules API
├── observation.py      GameState → observation vector
├── player.py           Player model
├── session.py          GameSession — multi-hand runs
├── side_pots.py        Side-pot accounting
├── simulate.py         Headless bot-vs-bot wizard
├── snapshot.py         Clone / restore game state
└── state.py            GameState / BettingRound

training_bot/
├── network.py          PokerNetwork (actor–critic)
├── ppo.py              PPOTrainer (GAE, clipped objective)
├── training_bot.py     TrainingBot — experience collection
├── rl_bot.py           RLBot — inference / play
├── self_play_pool.py   Rolling snapshot pool (disk + sampling helpers)
├── train.py            Main loop, curriculum, logging
└── eval.py             Standalone evaluation script

requirements.txt
pyproject.toml          setuptools metadata (installable package name: poker_engine)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

---

## Playing against bots (CLI)

```bash
python -m poker_engine.cli
```

The wizard asks for seat (0–5), starting stack, and opponent type. **RLBot** loads a PyTorch checkpoint (falls back if missing). In-game keys are shown on screen (fold, check/call, bet sizes, all-in).

---

## Running bot simulations

```bash
python -m poker_engine.simulate
```

Assign a bot type per seat, set hands per session and number of sessions; prints summary statistics.

---

## Training the RL bot

```bash
python -m training_bot.train --name my_experiment
python -m training_bot.train
```

With `--name`, if `runs/checkpoints/<name>/latest.pt` exists you are prompted to resume. Without `--name`, an interactive menu lists existing runs (or starts a timestamped run).

### Artifact layout

```
runs/checkpoints/<run>/
    latest.pt              always overwritten each update
    update_<N>.pt          every CHECKPOINT_INTERVAL updates
    stage<N>_start.pt      saved when graduating to stage N

runs/snapshots/<run>/
    snapshot_<N>.pt        self-play pool (SNAPSHOT_INTERVAL, max pool size 20)

runs/logs/<run>/
    training_log.csv
    hands_log.csv
    eval_log.csv

runs/tensorboard/<run>/
    events.out.*
```

Checkpoints store network weights, optimizer state(s), return normalizer, `update_num`, `stage`, rolling chip-delta history, reward clip, and stage start update. Logs append on resume.

### Curriculum (four stages)

| Stage | Masked actions (learners) | Reward clip (±BB) |
|-------|---------------------------|-------------------|
| 1 | Pot bet, 2× pot, all-in | 2.0 |
| 2 | 2× pot, all-in | 5.0 |
| 3 | All-in only | 8.0 |
| 4 | None | 12.0 |

**Entropy:** Per-stage linear decay from `ENTROPY_START` → `ENTROPY_END` (see `train.py`), floored in `PPOTrainer` at `ENTROPY_FLOOR`.

**Graduation:** Requires minimum PPO updates in the current stage (`MIN_UPDATES_PER_STAGE`) and at least one ladder evaluation result available. Optional performance gates live in `STAGE_GRAD_THRESHOLDS` (empty `{}` means no ladder-performance requirement beyond having run eval). Stage 4 never graduates.

**Rollouts:** One PPO update runs after each batch of `ROLLOUT_HANDS` hands (default **1000**) fills the experience buffer.

**Seating:** Default `TRAINING_SEATS = [0, 1, 2, 3, 4, 5]` — all six seats use `TrainingBot` with **one shared network** (symmetric self-play). `SelfPlayPool` still saves periodic weight snapshots; **`_make_opponent` / pool sampling applies only to seats not listed in `TRAINING_SEATS`** if you change that list.

### TensorBoard

```bash
tensorboard --logdir runs/tensorboard/my_experiment
tensorboard --logdir runs/tensorboard
```

---

## Evaluating a trained bot

```bash
python -m training_bot.eval
python -m training_bot.eval --checkpoint runs/checkpoints/my_experiment/latest.pt --hands 5000
```

Without `--checkpoint`, picks the most recently modified `runs/checkpoints/*/latest.pt`. Reports avg chip delta (BB/hand) vs each scripted bot, win rate, action distribution, and pass/fail vs red-flag thresholds.

---

## Using a trained bot in code

```python
from training_bot.rl_bot import RLBot
from poker_engine.session import GameSession

bot = RLBot.from_checkpoint(player_id=0, checkpoint_path="runs/checkpoints/my_run/latest.pt")

factory = RLBot.make_factory("runs/checkpoints/my_run/latest.pt", starting_stack=1000.0)
session = GameSession(starting_stack=1000.0)
session.fill_empty_seats_with(factory)
result = session.run(max_hands=100)
```

---

## Architecture reference

### Game engine

`PokerGame` (`game.py`) core API:

```python
state = game.start_new_hand()
legal = game.get_legal_actions()
state = game.apply_action(action)
```

Blinds: SB = BB/2, BB = `starting_stack / 100` (default stack 1000 → 5/10). Side pots and multi-street betting are handled in-engine.

`GameSession` (`session.py`) runs hands, assigns bots, supports hooks (`on_hand_end`, etc.).

`GameSnapshot` (`snapshot.py`): clone/restore for branching simulations.

### Built-in bots

| Bot | Behaviour (summary) |
|-----|------------------------|
| RandomBot | Uniform random legal action |
| CallBot | Check/call only |
| TightBot | Stronger starting hands; straightforward postflop aggression |
| PositionBot | TightBot-like + positional tweaks and occasional river bluffs |

### Observation vector

`encode_observation()` (`observation.py`) produces a **458-float** `float32` vector: hole (2×52), board (5×52), pot, stacks, street bets, invested, folded/all-in flags, position, street, bet-to-call scalars, **7-bit legal mask**, **last action per seat** (6×7), **hand-strength hint**. See file comments for exact indices (`OBS_SIZE`).

### Neural network

`PokerNetwork`: trunk `414→256→256→128` with LayerNorm + ReLU; policy head → 7 logits; value head → 1. Forward returns `(probs, value, logits)` so training applies **masking before softmax**.

### PPO trainer

`PPOTrainer` (`ppo.py`) uses GAE and a clipped surrogate objective. **Authoritative hyperparameters are the constants in `ppo.py`** (e.g. `CLIP_EPSILON`, `LAMBDA`, `POLICY_LR`, `VALUE_LR`, `POLICY_PASSES`, `VALUE_EXTRA_PASSES`, `MINI_BATCH_SIZE`). Joint updates plus extra value passes; running return normalization (`RunningNormalizer`).

Rewards: terminal chip change for the hand, in **big-blind units**, clipped per stage in `train.py` before advantage estimation.

---

## Logging & metrics

CSV headers are created in `train.py`. Typical columns:

- **training_log.csv:** timestamp, update, stage, avg_delta, losses, entropy, explained variance, action percentages.
- **hands_log.csv:** session/hand indices, stage, per-training-seat deltas.
- **eval_log.csv:** ladder results vs each scripted bot.

Console warnings use thresholds at the top of `train.py` (fold / all-in / passive rates, explained variance, etc.).

---

## Configuration reference

Primary knobs live at the top of **`training_bot/train.py`** and **`training_bot/ppo.py`**. Examples from `train.py`:

| Symbol | Default | Role |
|--------|---------|------|
| `STARTING_STACK` | 1000.0 | Starting chips |
| `ROLLOUT_HANDS` | 1000 | Hands before each PPO update |
| `DELTA_HISTORY_SIZE` | 10_000 | Rolling window for avg chip delta |
| `CHECKPOINT_INTERVAL` | 50 | Saved numbered checkpoints |
| `LADDER_EVAL_INTERVAL` | 100 | Mini-eval frequency |
| `LADDER_EVAL_HANDS` | 1_000 | Hands per opponent in ladder eval |
| `MIN_UPDATES_PER_STAGE` | see file | Minimum updates before stage graduation |
| `TRAINING_SEATS` | all six | Which seats collect RL experience |

`self_play_pool.py`: `SNAPSHOT_INTERVAL`, `MAX_POOL_SIZE`.

---

## Extending the project

### New bot

Subclass `Bot` in `bots.py` (or elsewhere):

```python
from poker_engine.bots import Bot
from poker_engine.action import Action
from poker_engine.state import GameState
from typing import List

class MyBot(Bot):
    def __init__(self, player_id: int):
        super().__init__(player_id, name="MyBot")

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        return legal_actions[0]
```

### New action type

1. Extend `ActionType` in `action.py`.
2. Implement in `PokerGame` (`game.py`): legality and `_execute_action`.
3. Update observation mask width / layout in `observation.py`.
4. Resize policy head in `network.py`.
5. Update CLI labels in `cli.py` if humans use the action.

### Minimal engine loop

```python
from poker_engine.game import PokerGame

game = PokerGame(starting_stack=1000.0, seed=42)
state = game.start_new_hand()
while not state.terminal:
    legal = game.get_legal_actions()
    state = game.apply_action(legal[0])
```

---

## License

See `LICENSE` in the repository root.
