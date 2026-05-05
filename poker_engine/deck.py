"""Deck of 52 ``Card`` instances with deterministic shuffle via optional RNG seed."""

import random
from .card import Card


class Deck:

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in range(4)
        ]

    def shuffle(self):
        self._rng.shuffle(self.cards)

    def draw(self, n: int = 1) -> list:
        """Always returns a list of n cards."""
        if n > len(self.cards):
            raise ValueError("Not enough cards left in deck.")
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def draw_one(self) -> "Card":
        """Convenience: draw a single card and return it directly."""
        return self.draw(1)[0]

    def reset(self):
        # Rebuild the card list but keep the existing RNG state so
        # successive hands don't produce identical shuffles.
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in range(4)
        ]