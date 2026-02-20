from poker_engine.deck import Deck

def test_deck_has_52_unique_cards():
    deck = Deck(seed=42)
    assert len(deck.cards) == 52
    assert len(set(deck.cards)) == 52

def test_draw_reduces_size():
    deck = Deck(seed=42)
    deck.shuffle()
    deck.draw(5)
    assert len(deck.cards) == 47

    