#!/usr/bin/env python3
"""
Test script to inspect frame metadata from the ARC-AGI-3 API.

This checks what fields are available in the FrameData response,
specifically to verify if available_actions is populated.
"""

import os
import json
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

if (SCHEME == "http" and str(PORT) == "80") or (SCHEME == "https" and str(PORT) == "443"):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"

HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}


def main():
    print(f"Connecting to: {ROOT_URL}")
    print(f"API Key: {HEADERS['X-API-Key'][:10]}..." if HEADERS['X-API-Key'] else "No API Key")
    print()

    session = requests.Session()
    session.headers.update(HEADERS)

    # Get list of games
    print("=== Getting available games ===")
    r = session.get(f"{ROOT_URL}/api/games", timeout=10)
    if r.status_code != 200:
        print(f"Failed to get games: {r.status_code} - {r.text}")
        return

    games = r.json()
    print(f"Found {len(games)} games")
    if games:
        print(f"First game: {games[0]}")
    print()

    # Pick first game
    if not games:
        print("No games available")
        return

    game_id = games[0]["game_id"] if isinstance(games[0], dict) else games[0]
    print(f"=== Testing with game: {game_id} ===")
    print()

    # Open a scorecard
    print("Opening scorecard...")
    r = session.post(f"{ROOT_URL}/api/scorecard/open", json={"tags": ["test"]})
    if r.status_code != 200:
        print(f"Failed to open scorecard: {r.status_code} - {r.text}")
        return
    card_id = r.json()["card_id"]
    print(f"Card ID: {card_id}")
    print()

    # Send RESET action to get initial frame
    print("=== Sending RESET action ===")
    action_data = {
        "card_id": card_id,
        "game_id": game_id,
        "action": {
            "id": 0,  # RESET
            "data": {"game_id": game_id},
        },
    }
    r = session.post(f"{ROOT_URL}/api/action", json=action_data)
    if r.status_code != 200:
        print(f"Failed to send action: {r.status_code} - {r.text}")
        return

    frame_data = r.json()

    print("=== RAW FRAME DATA ===")
    # Print all keys
    print(f"Keys in response: {list(frame_data.keys())}")
    print()

    # Print each field (except frame which is large)
    for key, value in frame_data.items():
        if key == "frame":
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], list):
                    if len(value[0]) > 0 and isinstance(value[0][0], list):
                        print(f"frame: 3D array with {len(value)} animation frames, each {len(value[0])}x{len(value[0][0])}")
                    else:
                        print(f"frame: 2D array {len(value)}x{len(value[0])}")
                else:
                    print(f"frame: list with {len(value)} elements")
            else:
                print(f"frame: {type(value)}")
        else:
            print(f"{key}: {value}")
    print()

    # Specifically check available_actions
    print("=== AVAILABLE_ACTIONS ANALYSIS ===")
    if "available_actions" in frame_data:
        aa = frame_data["available_actions"]
        print(f"Type: {type(aa)}")
        print(f"Value: {aa}")
        if isinstance(aa, list):
            print(f"Length: {len(aa)}")
            if aa:
                print(f"First element type: {type(aa[0])}")
                print(f"First element: {aa[0]}")
    else:
        print("available_actions NOT FOUND in response!")
        print("Available keys:", list(frame_data.keys()))
    print()

    # Try sending a regular action to see if available_actions changes
    print("=== Sending ACTION1 ===")
    action_data = {
        "card_id": card_id,
        "game_id": game_id,
        "action": {
            "id": 1,  # ACTION1
            "data": {"game_id": game_id},
        },
    }
    r = session.post(f"{ROOT_URL}/api/action", json=action_data)
    if r.status_code == 200:
        frame_data = r.json()
        print(f"Keys: {list(frame_data.keys())}")
        if "available_actions" in frame_data:
            print(f"available_actions: {frame_data['available_actions']}")
        else:
            print("available_actions NOT FOUND")
    else:
        print(f"Failed: {r.status_code}")
    print()

    # Close scorecard
    print("Closing scorecard...")
    session.post(f"{ROOT_URL}/api/scorecard/close", json={"card_id": card_id})
    print("Done!")


if __name__ == "__main__":
    main()
