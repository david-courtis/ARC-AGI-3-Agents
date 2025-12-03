#!/usr/bin/env python3
"""
Dataset Generator for ARC-3 State-Action-NextState Triples

Generates a structured dataset of state transitions by systematically exploring
the reachable state space of ARC-3 games using BFS with pruning.

Usage:
    uv run generate_dataset.py --game=<game_id> --duration=<seconds> [--output=<path>]

Output columns:
    - game_id: The game identifier
    - move_num: Move number in the sequence
    - prev_move_sequence: List of previous action IDs leading to this state
    - move_type: Action name (e.g., "ACTION1")
    - move: Action value (1-7)
    - before_board_state: Flattened grid before action (64*64 = 4096 values)
    - after_board_state: Flattened grid after action
    - board_state_changed: Boolean indicating if state changed
    - solution_boolean: Boolean indicating if this action led to WIN
    - after_board_state_hash: MD5 hash of after state for deduplication
    - score_before: Score before action
    - score_after: Score after action
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

from agents.structs import FrameData, GameAction, GameState
from agents.view_utils import create_grid_image, create_transition_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StateNode:
    """Represents a node in the state exploration tree."""
    frame: FrameData
    move_sequence: list[int] = field(default_factory=list)
    depth: int = 0
    guid: str = ""


@dataclass
class Transition:
    """A single state-action-next_state transition."""
    game_id: str
    move_num: int
    prev_move_sequence: list[int]
    move_type: str
    move: int
    before_board_state: np.ndarray
    after_board_state: np.ndarray
    board_state_changed: bool
    solution_boolean: bool
    after_board_state_hash: str
    score_before: int
    score_after: int


class DatasetGenerator:
    """Generates state-action-next_state datasets via BFS exploration."""

    # Simple actions only (excluding ACTION6 which requires coordinates)
    SIMPLE_ACTIONS = [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
        GameAction.ACTION7,
    ]

    def __init__(self, game_id: str, root_url: str, visualize: bool = False, viz_dir: Optional[str] = None):
        self.game_id = game_id
        self.root_url = root_url
        self.card_id: Optional[str] = None
        self.guid = ""
        self.visualize = visualize
        self.viz_dir = viz_dir

        # API session
        self._session = requests.Session()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self._session.headers.update(self.headers)

        # State tracking
        self.seen_hashes: set[str] = set()
        self.transitions: list[Transition] = []
        self.available_actions: list[GameAction] = []

        # Statistics
        self.stats = {
            "total_actions": 0,
            "unique_states": 0,
            "pruned_no_change": 0,
            "pruned_duplicate": 0,
            "solutions_found": 0,
            "game_overs": 0,
        }

        # Setup visualization directory
        if self.visualize and self.viz_dir:
            os.makedirs(self.viz_dir, exist_ok=True)
            os.makedirs(os.path.join(self.viz_dir, "transitions"), exist_ok=True)
            os.makedirs(os.path.join(self.viz_dir, "states"), exist_ok=True)
            logger.info(f"Visualization enabled, saving to: {self.viz_dir}")

    def _get_available_games(self) -> list[str]:
        """Fetch list of available games from the API."""
        try:
            r = self._session.get(
                f"{self.root_url}/api/games",
                headers=self.headers,
                timeout=10,
            )
            if r.ok:
                games = [g["game_id"] for g in r.json()]
                logger.info(f"Available games: {games}")
                return games
            else:
                logger.warning(f"Failed to fetch games: {r.status_code} - {r.text}")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch games list: {e}")
            return []

    def _open_scorecard(self) -> str:
        """Open a scorecard and return the card_id."""
        r = self._session.post(
            f"{self.root_url}/api/scorecard/open",
            json={"tags": ["dataset_generation", self.game_id]},
            headers=self.headers,
            timeout=10,
        )
        response = r.json()
        if not r.ok or "error" in response:
            raise RuntimeError(f"Failed to open scorecard: {response}")
        card_id = response["card_id"]
        logger.info(f"Opened scorecard: {card_id}")
        return card_id

    def _close_scorecard(self) -> None:
        """Close the scorecard."""
        if self.card_id:
            card_id = self.card_id
            self.card_id = None  # Prevent double-close
            try:
                self._session.post(
                    f"{self.root_url}/api/scorecard/close",
                    json={"card_id": card_id},
                    headers=self.headers,
                    timeout=10,
                )
                logger.info(f"Closed scorecard: {card_id}")
            except Exception as e:
                logger.warning(f"Failed to close scorecard: {e}")

    def _compute_state_hash(self, frame: np.ndarray) -> str:
        """Compute MD5 hash of a board state."""
        return hashlib.md5(frame.tobytes()).hexdigest()

    def _frame_to_array(self, frame_data: FrameData) -> np.ndarray:
        """Convert FrameData to numpy array (take last frame if animated)."""
        frame = np.array(frame_data.frame, dtype=np.int8)
        if len(frame.shape) == 3:
            frame = frame[-1]  # Take last frame if animated
        return frame

    def _save_transition_viz(
        self,
        before_array: np.ndarray,
        after_array: np.ndarray,
        action: GameAction,
        transition_idx: int,
        state_changed: bool,
        is_solution: bool,
        is_duplicate: bool,
        depth: int,
        move_sequence: list[int],
    ) -> None:
        """Save a visualization of a transition."""
        if not self.visualize or not self.viz_dir:
            return

        # Create status string
        status_parts = []
        if is_solution:
            status_parts.append("WIN")
        if not state_changed:
            status_parts.append("NO_CHANGE")
        if is_duplicate:
            status_parts.append("DUPLICATE")
        if not status_parts:
            status_parts.append("NEW_STATE")
        status = "_".join(status_parts)

        # Create action info string
        seq_str = "->".join(str(a) for a in move_sequence) if move_sequence else "start"
        action_info = f"Depth {depth} | {seq_str} -> {action.name} | {status}"

        # Create and save transition image
        img = create_transition_image(before_array, after_array, action_info, cell_size=8)
        filename = f"t{transition_idx:05d}_d{depth}_{action.name}_{status}.png"
        img.save(os.path.join(self.viz_dir, "transitions", filename))

    def _save_state_viz(self, state_array: np.ndarray, state_hash: str, depth: int) -> None:
        """Save a visualization of a unique state."""
        if not self.visualize or not self.viz_dir:
            return

        img = create_grid_image(state_array, cell_size=8)
        filename = f"state_{state_hash[:8]}_d{depth}.png"
        img.save(os.path.join(self.viz_dir, "states", filename))

    def _do_action(self, action: GameAction) -> Optional[FrameData]:
        """Execute an action and return the resulting frame."""
        data = action.action_data.model_dump()
        if action == GameAction.RESET:
            data["card_id"] = self.card_id
        if self.guid:
            data["guid"] = self.guid
        data["game_id"] = self.game_id

        url = f"{self.root_url}/api/cmd/{action.name}"
        logger.debug(f"POST {url} with data: {data}")

        try:
            r = self._session.post(
                url,
                json=data,
                headers=self.headers,
                timeout=10,
            )
            logger.debug(f"Response status: {r.status_code}")
            response = r.json()
            logger.debug(f"Response body: {response}")

            if "error" in response:
                logger.warning(
                    f"Action {action.name} failed for game '{self.game_id}': "
                    f"{response.get('error')} - {response.get('message', 'no message')}"
                )
                logger.warning(f"Request data was: {data}")
                return None
            frame = FrameData.model_validate(response)
            if frame.guid:
                self.guid = frame.guid
            return frame
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Action request failed: {type(e).__name__}: {e}")
            return None

    def _reset_game(self) -> Optional[FrameData]:
        """Reset the game and return initial frame."""
        self.guid = ""  # Clear guid for fresh session
        frame = self._do_action(GameAction.RESET)
        if frame and frame.available_actions:
            # Filter to only simple actions that are available
            self.available_actions = [
                a for a in self.SIMPLE_ACTIONS
                if a in frame.available_actions
            ]
            logger.info(f"Available actions for {self.game_id}: {[a.name for a in self.available_actions]}")
        return frame

    def _restore_state(self, move_sequence: list[int]) -> Optional[FrameData]:
        """Restore game to a specific state by replaying move sequence."""
        frame = self._reset_game()
        if frame is None:
            return None

        for action_id in move_sequence:
            action = GameAction.from_id(action_id)
            frame = self._do_action(action)
            if frame is None:
                return None
            # Update guid from frame
            if frame.guid:
                self.guid = frame.guid

        return frame

    def generate(self, duration_seconds: int, max_depth: int = 100) -> list[Transition]:
        """
        Generate dataset using BFS exploration with pruning.

        Args:
            duration_seconds: How long to run exploration
            max_depth: Maximum sequence length to explore

        Returns:
            List of Transition objects
        """
        start_time = time.time()
        end_time = start_time + duration_seconds

        # First check if game is available
        available_games = self._get_available_games()
        if available_games:
            # Check if our game_id matches any available game
            matching_games = [g for g in available_games if g.startswith(self.game_id)]
            if not matching_games:
                logger.error(
                    f"Game '{self.game_id}' not found in available games. "
                    f"Available: {available_games}"
                )
                return []
            if self.game_id not in available_games:
                # Use exact match if possible
                logger.info(f"Using game '{matching_games[0]}' (matched from '{self.game_id}')")
                self.game_id = matching_games[0]

        # Open scorecard first
        try:
            self.card_id = self._open_scorecard()
        except Exception as e:
            logger.error(f"Failed to open scorecard: {e}")
            return []

        # Initialize with reset
        logger.info(f"Initializing game '{self.game_id}' with card_id '{self.card_id}'")
        initial_frame = self._reset_game()
        if initial_frame is None:
            logger.error("Failed to initialize game")
            self._close_scorecard()
            return []

        if not self.available_actions:
            logger.error("No valid simple actions available for this game")
            self._close_scorecard()
            return []

        initial_array = self._frame_to_array(initial_frame)
        initial_hash = self._compute_state_hash(initial_array)
        self.seen_hashes.add(initial_hash)
        self.stats["unique_states"] = 1

        # BFS queue
        queue: deque[StateNode] = deque()
        queue.append(StateNode(
            frame=initial_frame,
            move_sequence=[],
            depth=0,
            guid=self.guid,
        ))

        iterations = 0
        last_log_time = start_time

        while queue and time.time() < end_time:
            iterations += 1
            current_node = queue.popleft()

            # Periodic logging
            if time.time() - last_log_time > 10:
                elapsed = time.time() - start_time
                logger.info(
                    f"Progress: {elapsed:.0f}s elapsed, "
                    f"{len(self.transitions)} transitions, "
                    f"{self.stats['unique_states']} unique states, "
                    f"queue size: {len(queue)}, "
                    f"depth: {current_node.depth}"
                )
                last_log_time = time.time()

            # Skip if too deep
            if current_node.depth >= max_depth:
                continue

            # Restore state to current node
            restored_frame = self._restore_state(current_node.move_sequence)
            if restored_frame is None:
                logger.warning(f"Failed to restore state at depth {current_node.depth}")
                continue

            before_array = self._frame_to_array(restored_frame)
            before_hash = self._compute_state_hash(before_array)

            # Try each available action
            for action in self.available_actions:
                if time.time() >= end_time:
                    break

                # Execute action
                after_frame = self._do_action(action)
                self.stats["total_actions"] += 1

                if after_frame is None:
                    continue

                after_array = self._frame_to_array(after_frame)
                after_hash = self._compute_state_hash(after_array)

                # Check if state changed
                state_changed = not np.array_equal(before_array, after_array)

                # Check for solution or game over
                is_solution = after_frame.state == GameState.WIN
                is_game_over = after_frame.state == GameState.GAME_OVER

                if is_solution:
                    self.stats["solutions_found"] += 1
                if is_game_over:
                    self.stats["game_overs"] += 1

                # Check if this is a duplicate state (before recording)
                is_duplicate = after_hash in self.seen_hashes

                # Create transition record
                transition = Transition(
                    game_id=self.game_id,
                    move_num=current_node.depth + 1,
                    prev_move_sequence=current_node.move_sequence.copy(),
                    move_type=action.name,
                    move=action.value,
                    before_board_state=before_array.flatten(),
                    after_board_state=after_array.flatten(),
                    board_state_changed=state_changed,
                    solution_boolean=is_solution,
                    after_board_state_hash=after_hash,
                    score_before=restored_frame.score,
                    score_after=after_frame.score,
                )
                self.transitions.append(transition)

                # Save visualization
                self._save_transition_viz(
                    before_array=before_array,
                    after_array=after_array,
                    action=action,
                    transition_idx=len(self.transitions),
                    state_changed=state_changed,
                    is_solution=is_solution,
                    is_duplicate=is_duplicate,
                    depth=current_node.depth,
                    move_sequence=current_node.move_sequence,
                )

                # Pruning logic
                if not state_changed:
                    self.stats["pruned_no_change"] += 1
                    continue  # Prune: action didn't change state

                if is_duplicate:
                    self.stats["pruned_duplicate"] += 1
                    continue  # Prune: already seen this state

                # New state found
                self.seen_hashes.add(after_hash)
                self.stats["unique_states"] += 1

                # Save new unique state
                self._save_state_viz(after_array, after_hash, current_node.depth + 1)

                # Don't explore further from terminal states
                if is_solution or is_game_over:
                    continue

                # Add to queue for further exploration
                new_sequence = current_node.move_sequence + [action.value]
                queue.append(StateNode(
                    frame=after_frame,
                    move_sequence=new_sequence,
                    depth=current_node.depth + 1,
                    guid=self.guid,
                ))

        elapsed = time.time() - start_time
        logger.info(f"Generation complete in {elapsed:.1f}s")
        logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")

        # Close scorecard
        self._close_scorecard()

        return self.transitions

    def save_dataset(self, output_path: str) -> None:
        """Save transitions to disk in multiple formats."""
        if not self.transitions:
            logger.warning("No transitions to save")
            return

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Convert to numpy arrays for efficient storage
        n_transitions = len(self.transitions)
        grid_size = 64 * 64  # Flattened grid

        # Create structured arrays
        data = {
            "game_id": np.array([t.game_id for t in self.transitions], dtype=object),
            "move_num": np.array([t.move_num for t in self.transitions], dtype=np.int32),
            "move_type": np.array([t.move_type for t in self.transitions], dtype=object),
            "move": np.array([t.move for t in self.transitions], dtype=np.int8),
            "board_state_changed": np.array([t.board_state_changed for t in self.transitions], dtype=bool),
            "solution_boolean": np.array([t.solution_boolean for t in self.transitions], dtype=bool),
            "after_board_state_hash": np.array([t.after_board_state_hash for t in self.transitions], dtype=object),
            "score_before": np.array([t.score_before for t in self.transitions], dtype=np.int32),
            "score_after": np.array([t.score_after for t in self.transitions], dtype=np.int32),
            "before_board_state": np.stack([t.before_board_state for t in self.transitions]),
            "after_board_state": np.stack([t.after_board_state for t in self.transitions]),
        }

        # Save move sequences separately (variable length)
        move_sequences = [t.prev_move_sequence for t in self.transitions]

        # Save as numpy archive (.npz)
        npz_path = output_path if output_path.endswith(".npz") else f"{output_path}.npz"
        np.savez_compressed(
            npz_path,
            **data,
            # Store move sequences as object array
            prev_move_sequences=np.array(move_sequences, dtype=object),
            # Metadata
            stats=np.array([json.dumps(self.stats)], dtype=object),
        )
        logger.info(f"Saved {n_transitions} transitions to {npz_path}")

        # Also save a JSON summary
        summary_path = output_path.replace(".npz", "") + "_summary.json"
        summary = {
            "game_id": self.game_id,
            "n_transitions": n_transitions,
            "stats": self.stats,
            "generated_at": datetime.now().isoformat(),
            "available_actions": [a.name for a in self.available_actions],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

    def close(self) -> None:
        """Clean up resources."""
        self._close_scorecard()
        self._session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate state-action-next_state dataset for ARC-3 games"
    )
    parser.add_argument(
        "-g", "--game",
        required=True,
        help="Game ID to explore (e.g., 'ls20')"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=60,
        help="Duration in seconds to run exploration (default: 60)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for dataset (default: datasets/<game_id>_<timestamp>.npz)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=100,
        help="Maximum sequence depth to explore (default: 100)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualizations of each transition and unique state"
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory for visualizations (default: datasets/<game_id>_<timestamp>_viz/)"
    )

    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Setup API URL
    scheme = os.environ.get("SCHEME", "http")
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", 8001)

    if (scheme == "http" and str(port) == "80") or (scheme == "https" and str(port) == "443"):
        root_url = f"{scheme}://{host}"
    else:
        root_url = f"{scheme}://{host}:{port}"

    logger.info(f"Connecting to {root_url}")
    logger.info(f"Generating dataset for game: {args.game}")
    logger.info(f"Duration: {args.duration} seconds")

    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        os.makedirs("datasets", exist_ok=True)
        output_path = f"datasets/{args.game}_{timestamp}.npz"
    else:
        output_path = args.output

    # Setup visualization directory
    viz_dir = args.viz_dir
    if args.visualize and viz_dir is None:
        os.makedirs("datasets", exist_ok=True)
        viz_dir = f"datasets/{args.game}_{timestamp}_viz"

    # Generate dataset
    generator = DatasetGenerator(
        args.game,
        root_url,
        visualize=args.visualize,
        viz_dir=viz_dir,
    )
    try:
        transitions = generator.generate(
            duration_seconds=args.duration,
            max_depth=args.max_depth
        )

        if transitions:
            generator.save_dataset(output_path)
            logger.info(f"Dataset generation complete: {len(transitions)} transitions")
        else:
            logger.warning("No transitions generated")

    finally:
        generator.close()


if __name__ == "__main__":
    main()
