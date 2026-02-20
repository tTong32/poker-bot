from dataclasses import dataclass

@dataclass(frozen=True)
class Card:
    rank: int  # 2–14 (14 = Ace)
    suit: int  # 0=♦, 1=♣, 2=♥, 3=♠

    def __post_init__(self):
        if not (2 <= self.rank <= 14):
            raise ValueError("Rank must be between 2 and 14.")
        if not (0 <= self.suit <= 3):
            raise ValueError("Suit must be between 0 and 3.")

    def __repr__(self):
        rank_map = {11: "J", 12: "Q", 13: "K", 14: "A"}
        suit_map = {0: "♦", 1: "♣", 2: "♥", 3: "♠"}
        rank_str = rank_map.get(self.rank, str(self.rank))
        suit_str = suit_map[self.suit]
        return f"{rank_str}{suit_str}"