"""Quick test script to verify the local simulator imports and works."""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    from local_simulator.core import BaseGame, GameLevel, GameObject, Renderer
    from local_simulator.core.game_object import Player, Wall, Goal, Position
    print("  ✓ Core imports successful")
    
    from local_simulator.games.simple_maze import SimpleMaze
    print("  ✓ SimpleMaze import successful")
    
    return SimpleMaze

def test_game():
    print("\nTesting SimpleMaze game...")
    SimpleMaze = test_imports()
    
    game = SimpleMaze()
    print(f"  ✓ Game created with ID: {game.game_id}")
    print(f"  ✓ Number of levels: {game.get_level_count()}")
    
    frame = game.reset()
    print(f"  ✓ Game reset, state: {frame.state}")
    print(f"  ✓ Frame dimensions: {len(frame.frame)}x{len(frame.frame[0])}")
    print(f"  ✓ Initial score: {frame.score}")
    print(f"  ✓ Available actions: {[a.name for a in frame.available_actions]}")
    
    # Take a few actions
    from agents.structs import GameAction
    
    print("\nTaking some actions...")
    for action in [GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION2]:
        frame = game.step(action)
        print(f"  → {action.name}: state={frame.state}, score={frame.score}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_game()
