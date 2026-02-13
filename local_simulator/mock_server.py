"""
Mock API Server - drop-in replacement for the ARC-AGI-3 API.

Run this locally and point your agents to localhost instead of three.arcprize.org.
"""

import json
import logging
import uuid
from typing import Optional
from flask import Flask, request, jsonify

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.structs import GameAction, GameState as AgentGameState
from local_simulator.core.base_game import BaseGame

# Import available games
from local_simulator.games.simple_maze import SimpleMaze


logger = logging.getLogger(__name__)

# Registry of available games
GAME_REGISTRY: dict[str, type[BaseGame]] = {
    "simple_maze": SimpleMaze,
}

# Active game sessions
SESSIONS: dict[str, BaseGame] = {}

# Scorecards
SCORECARDS: dict[str, dict] = {}


def create_app() -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    
    @app.route("/api/games", methods=["GET"])
    def list_games():
        """List available games."""
        return jsonify([{"game_id": gid} for gid in GAME_REGISTRY.keys()])
    
    @app.route("/api/scorecard/open", methods=["POST"])
    def open_scorecard():
        """Open a new scorecard."""
        data = request.get_json() or {}
        card_id = str(uuid.uuid4())
        SCORECARDS[card_id] = {
            "card_id": card_id,
            "tags": data.get("tags", []),
            "games": {},
            "open": True,
        }
        return jsonify({"card_id": card_id})
    
    @app.route("/api/scorecard/close", methods=["POST"])
    def close_scorecard():
        """Close a scorecard and return results."""
        data = request.get_json() or {}
        card_id = data.get("card_id", "")
        
        if card_id not in SCORECARDS:
            return jsonify({"error": "Scorecard not found"}), 404
        
        scorecard = SCORECARDS[card_id]
        scorecard["open"] = False
        
        # Calculate summary
        games = scorecard.get("games", {})
        won = sum(1 for g in games.values() if g.get("state") == "WIN")
        played = len(games)
        total_score = sum(g.get("score", 0) for g in games.values())
        
        return jsonify({
            "card_id": card_id,
            "won": won,
            "played": played,
            "score": total_score,
            "total_actions": sum(g.get("actions", 0) for g in games.values()),
            "cards": {
                gid: {
                    "game_id": gid,
                    "total_plays": 1,
                    "scores": [g.get("score", 0)],
                    "states": [g.get("state", "NOT_PLAYED")],
                    "actions": [g.get("actions", 0)],
                    "resets": [1],
                }
                for gid, g in games.items()
            }
        })
    
    @app.route("/api/cmd/RESET", methods=["POST"])
    def cmd_reset():
        """Reset/start a game."""
        data = request.get_json() or {}
        game_id = data.get("game_id", "")
        card_id = data.get("card_id", "")
        
        if game_id not in GAME_REGISTRY:
            return jsonify({"error": f"Game '{game_id}' not found"}), 404
        
        # Create new game instance
        game_class = GAME_REGISTRY[game_id]
        game = game_class()
        
        # Reset and get initial frame
        frame_data = game.reset(card_id)
        
        # Store session
        session_key = f"{card_id}:{game_id}"
        SESSIONS[session_key] = game
        
        # Update scorecard
        if card_id in SCORECARDS:
            SCORECARDS[card_id]["games"][game_id] = {
                "state": frame_data.state.value,
                "score": frame_data.score,
                "actions": 0,
            }
        
        return jsonify(frame_data.model_dump())
    
    @app.route("/api/cmd/<action_name>", methods=["POST"])
    def cmd_action(action_name: str):
        """Execute a game action."""
        data = request.get_json() or {}
        game_id = data.get("game_id", "")
        card_id = data.get("card_id", data.get("guid", "").split(":")[0] if ":" in data.get("guid", "") else "")
        
        # Find session
        session_key = f"{card_id}:{game_id}"
        
        # Try to find by guid
        if session_key not in SESSIONS:
            guid = data.get("guid", "")
            for key, game in SESSIONS.items():
                if game.guid == guid:
                    session_key = key
                    break
        
        if session_key not in SESSIONS:
            return jsonify({"error": "Game session not found"}), 404
        
        game = SESSIONS[session_key]
        
        # Parse action
        try:
            action = GameAction.from_name(action_name)
        except ValueError:
            return jsonify({"error": f"Invalid action: {action_name}"}), 400
        
        # Set action data for complex actions
        if action.is_complex():
            action.set_data({
                "x": data.get("x", 0),
                "y": data.get("y", 0),
                "game_id": game_id,
            })
        
        # Execute action
        frame_data = game.step(action)
        
        # Update scorecard
        actual_card_id = session_key.split(":")[0]
        if actual_card_id in SCORECARDS:
            SCORECARDS[actual_card_id]["games"][game_id] = {
                "state": frame_data.state.value,
                "score": frame_data.score,
                "actions": game.action_count,
            }
        
        return jsonify(frame_data.model_dump())
    
    @app.route("/api/scorecard/<card_id>/<game_id>", methods=["GET"])
    def get_scorecard_game(card_id: str, game_id: str):
        """Get scorecard for a specific game."""
        if card_id not in SCORECARDS:
            return jsonify({"error": "Scorecard not found"}), 404
        
        scorecard = SCORECARDS[card_id]
        game_data = scorecard.get("games", {}).get(game_id, {})
        
        return jsonify({
            "game_id": game_id,
            "total_plays": 1,
            "scores": [game_data.get("score", 0)],
            "states": [game_data.get("state", "NOT_PLAYED")],
            "actions": [game_data.get("actions", 0)],
            "resets": [1],
        })
    
    return app


def main():
    """Run the mock server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local ARC-AGI-3 Mock Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║         Local ARC-AGI-3 Simulator                          ║
╠════════════════════════════════════════════════════════════╣
║  Available games:                                          ║""")
    for game_id in GAME_REGISTRY.keys():
        print(f"║    - {game_id:<51} ║")
    print(f"""╠════════════════════════════════════════════════════════════╣
║  Run your agent with:                                      ║
║    export SCHEME=http                                      ║
║    export HOST={args.host:<42} ║
║    export PORT={args.port:<42} ║
║    uv run main.py --agent=random --game=simple_maze        ║
╚════════════════════════════════════════════════════════════╝
""")
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
