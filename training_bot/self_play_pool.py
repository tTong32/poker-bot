"""
Rolling disk-backed pool of frozen ``PokerNetwork`` weights.

``train.py`` writes a snapshot every ``SNAPSHOT_INTERVAL`` PPO updates under
``runs/snapshots/<run>/`` (path passed as ``snapshot_dir``).

When opponent seats call ``sample_opponent_network``:

  * 50% — train on the live ``current_network`` (shared weights, no extra load)
  * 30% — sample from the newest third of the pool
  * 20% — sample from older snapshots (staleness / forgetting hedge)

Oldest files are deleted once the pool exceeds ``MAX_POOL_SIZE``. The directory
is rescanned on startup so resumes rebuild the in-memory index.

Usage::

    pool = SelfPlayPool(snapshot_dir="runs/snapshots/my_run")
    pool.save_snapshot(network, update_num)
    opponent_net = pool.sample_opponent_network(current_network)
"""

import os
import math
import random
import torch
from typing import List, Optional, Tuple

from .network import PokerNetwork

SNAPSHOT_INTERVAL = 50     # PPO updates between snapshots
MAX_POOL_SIZE     = 20     # maximum snapshots retained on disk


class SelfPlayPool:
    """
    Maintains an ordered list of (path, update_num) pairs, oldest first.
    Evicts the oldest entry when the pool exceeds MAX_POOL_SIZE.
    Persists across process restarts by scanning the snapshot directory.
    """

    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        # Ordered list of (path, update_num), oldest first
        self._pool: List[Tuple[str, int]] = []
        self._load_existing()

    # Persistence

    def _load_existing(self):
        """Reconstruct pool from files already on disk (for resume support)."""
        entries = []
        for fname in os.listdir(self.snapshot_dir):
            if fname.startswith("snapshot_") and fname.endswith(".pt"):
                stem = fname[len("snapshot_"):-len(".pt")]
                try:
                    update_num = int(stem)
                    entries.append((
                        os.path.join(self.snapshot_dir, fname),
                        update_num,
                    ))
                except ValueError:
                    pass
        self._pool = sorted(entries, key=lambda x: x[1])
        # Trim to cap (in case the cap was lowered between runs)
        self._evict_to_cap()

    def _evict_to_cap(self):
        while len(self._pool) > MAX_POOL_SIZE:
            path, _ = self._pool.pop(0)
            if os.path.exists(path):
                os.remove(path)

    # Public API

    def save_snapshot(self, network: PokerNetwork, update_num: int):
        """Persist current network weights as a frozen snapshot."""
        path = os.path.join(self.snapshot_dir, f"snapshot_{update_num}.pt")
        torch.save(network.state_dict(), path)
        self._pool.append((path, update_num))
        self._evict_to_cap()

    def load_network(self, path: str) -> PokerNetwork:
        """Load a frozen PokerNetwork from a snapshot file."""
        net = PokerNetwork()
        net.load_state_dict(torch.load(path, weights_only=False))
        net.eval()
        return net

    def sample_opponent_network(self, current_network: PokerNetwork) -> PokerNetwork:
        """
        Sample an opponent network using the 50/30/20 ratio.

        Returns the current_network object directly for the 50% case
        (shared weights, inference mode — no extra memory cost).
        Returns a freshly loaded copy for pool entries.
        """
        if not self._pool:
            # No snapshots yet — always use current network
            return current_network

        r   = random.random()
        n   = len(self._pool)

        if r < 0.50:
            # 50% — current network
            return current_network

        # Split pool: "recent" = newest third, "older" = the rest
        recent_count = max(1, math.ceil(n / 3))
        recent = self._pool[-recent_count:]
        older  = self._pool[:-recent_count] if n > recent_count else self._pool

        if r < 0.80:
            # 30% — recent snapshots
            chosen_pool = recent
        else:
            # 20% — older snapshots
            chosen_pool = older if older else recent

        path, _ = random.choice(chosen_pool)
        return self.load_network(path)

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        if not self._pool:
            return "SelfPlayPool(empty)"
        oldest = self._pool[0][1]
        newest = self._pool[-1][1]
        return f"SelfPlayPool(size={len(self._pool)}, updates={oldest}–{newest})"