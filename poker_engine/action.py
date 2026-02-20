from enum import Enum
from dataclasses import dataclass

class ActionType(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_HALF_POT = 3
    BET_POT = 4
    BET_DOUBLE_POT = 5
    ALL_IN = 6

@dataclass
class Action:
    type: ActionType
    amount: float = 0.0