"""
Quick smoke test: run a full hand with random actions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from poker_engine import PokerGame, ActionType

def test_full_hand():
    game = PokerGame(starting_stack=1000.0, seed=42)
    state = game.start_new_hand()
    game.print_state()

    rng = random.Random(99)
    steps = 0
    while not state.terminal and steps < 200:
        legal = game.get_legal_actions()
        action = rng.choice(legal)
        print(f"  Player {state.current_player_index} plays: {action.type.name}"
              f"  (amount={action.amount:.1f})")
        state = game.apply_action(action)
        steps += 1

    game.print_state()
    assert state.terminal, "Hand should be terminal"
    assert len(state.winners) >= 1, "Should have at least one winner"
    print(f"Test passed! Winner(s): {state.winners}. Steps: {steps}")

def test_evaluator():
    from poker_engine import Card, evaluate_hand, hand_rank_name
    # Royal flush
    hole = [Card(14, 3), Card(13, 3)]
    board = [Card(12, 3), Card(11, 3), Card(10, 3), Card(2, 0), Card(3, 1)]
    score = evaluate_hand(hole, board)
    assert score[0] == 8, f"Expected straight flush (8), got {score[0]}"
    print(f"Evaluator test passed: {hand_rank_name(score)}")

    # Two pair
    hole2 = [Card(5, 0), Card(5, 1)]
    board2 = [Card(9, 0), Card(9, 1), Card(2, 0), Card(7, 3), Card(3, 2)]
    score2 = evaluate_hand(hole2, board2)
    assert score2[0] == 2, f"Expected two pair (2), got {score2[0]}"
    print(f"Evaluator test passed: {hand_rank_name(score2)}")

if __name__ == "__main__":
    test_evaluator()
    test_full_hand()