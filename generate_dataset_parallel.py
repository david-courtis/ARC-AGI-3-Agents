#!/usr/bin/env python3
"""
Parallel Dataset Generator for ARC-3 State-Action-NextState Triples

Uses a pool of parallel agents to perform true BFS exploration of the state space.
Each agent maintains its own game session and explores from different frontier states.

Usage:
    uv run generate_dataset_parallel.py --game=<game_id> --duration=<seconds> --workers=<n>
"""

import argparse
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class StateNode:
    """Represents a node in the state exploration tree."""
    state_hash: str
    move_sequence: list[int] = field(default_factory=list)
    depth: int = 0


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


class ExplorationWorker:
    """A single worker that explores states from the shared queue."""

    SIMPLE_ACTIONS = [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
        GameAction.ACTION7,
    ]

    def __init__(
        self,
        worker_id: int,
        game_id: str,
        root_url: str,
        card_id: str,
        work_queue: queue.Queue,
        results: list,
        seen_hashes: set,
        stats: dict,
        lock: threading.Lock,
        stop_event: threading.Event,
        available_actions: list[GameAction],
        max_depth: int = 100,
        visualize: bool = False,
        viz_dir: Optional[str] = None,
    ):
        self.worker_id = worker_id
        self.game_id = game_id
        self.root_url = root_url
        self.card_id = card_id
        self.work_queue = work_queue
        self.results = results
        self.seen_hashes = seen_hashes
        self.stats = stats
        self.lock = lock
        self.stop_event = stop_event
        self.available_actions = available_actions
        self.max_depth = max_depth
        self.visualize = visualize
        self.viz_dir = viz_dir

        # Worker's own session
        self.guid = ""
        self._session = requests.Session()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self._session.headers.update(self.headers)

    def _do_action(self, action: GameAction) -> Optional[FrameData]:
        """Execute an action and return the resulting frame."""
        data = action.action_data.model_dump()
        if action == GameAction.RESET:
            data["card_id"] = self.card_id
        if self.guid:
            data["guid"] = self.guid
        data["game_id"] = self.game_id

        try:
            r = self._session.post(
                f"{self.root_url}/api/cmd/{action.name}",
                json=data,
                headers=self.headers,
                timeout=10,
            )
            response = r.json()
            if "error" in response:
                return None
            frame = FrameData.model_validate(response)
            if frame.guid:
                self.guid = frame.guid
            return frame
        except Exception:
            return None

    def _reset_and_replay(self, move_sequence: list[int]) -> Optional[FrameData]:
        """Reset game and replay a sequence to restore state."""
        self.guid = ""
        frame = self._do_action(GameAction.RESET)
        if frame is None:
            return None

        for action_id in move_sequence:
            action = GameAction.from_id(action_id)
            frame = self._do_action(action)
            if frame is None:
                return None

        return frame

    def _frame_to_array(self, frame_data: FrameData) -> np.ndarray:
        """Convert FrameData to numpy array."""
        frame = np.array(frame_data.frame, dtype=np.int8)
        if len(frame.shape) == 3:
            frame = frame[-1]
        return frame

    def _compute_hash(self, frame: np.ndarray) -> str:
        """Compute MD5 hash of a board state."""
        return hashlib.md5(frame.tobytes()).hexdigest()

    def _save_transition_viz(
        self, before: np.ndarray, after: np.ndarray, action: GameAction,
        idx: int, changed: bool, solution: bool, duplicate: bool,
        depth: int, seq: list[int]
    ) -> None:
        """Save visualization of a transition."""
        if not self.visualize or not self.viz_dir:
            return

        status_parts = []
        if solution:
            status_parts.append("WIN")
        if not changed:
            status_parts.append("NO_CHANGE")
        if duplicate:
            status_parts.append("DUPLICATE")
        if not status_parts:
            status_parts.append("NEW_STATE")
        status = "_".join(status_parts)

        seq_str = "->".join(str(a) for a in seq) if seq else "start"
        action_info = f"W{self.worker_id} D{depth} | {seq_str} -> {action.name} | {status}"

        img = create_transition_image(before, after, action_info, cell_size=8)
        filename = f"t{idx:05d}_w{self.worker_id}_d{depth}_{action.name}_{status}.png"
        img.save(os.path.join(self.viz_dir, "transitions", filename))

    def run(self) -> None:
        """Main worker loop - process states from queue."""
        logger.info(f"Worker {self.worker_id} started")

        while not self.stop_event.is_set():
            try:
                # Get next state to explore (with timeout to check stop_event)
                node = self.work_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if node.depth >= self.max_depth:
                self.work_queue.task_done()
                continue

            # Restore to this state
            frame = self._reset_and_replay(node.move_sequence)
            if frame is None:
                logger.warning(f"Worker {self.worker_id}: Failed to restore state at depth {node.depth}")
                self.work_queue.task_done()
                continue

            before_array = self._frame_to_array(frame)
            before_hash = self._compute_hash(before_array)

            # Try each action
            for action in self.available_actions:
                if self.stop_event.is_set():
                    break

                after_frame = self._do_action(action)

                with self.lock:
                    self.stats["total_actions"] += 1

                if after_frame is None:
                    continue

                after_array = self._frame_to_array(after_frame)
                after_hash = self._compute_hash(after_array)

                state_changed = not np.array_equal(before_array, after_array)
                is_solution = after_frame.state == GameState.WIN
                is_game_over = after_frame.state == GameState.GAME_OVER

                # Check duplicate under lock
                with self.lock:
                    is_duplicate = after_hash in self.seen_hashes

                    if is_solution:
                        self.stats["solutions_found"] += 1
                    if is_game_over:
                        self.stats["game_overs"] += 1

                    # Record transition
                    transition = Transition(
                        game_id=self.game_id,
                        move_num=node.depth + 1,
                        prev_move_sequence=node.move_sequence.copy(),
                        move_type=action.name,
                        move=action.value,
                        before_board_state=before_array.flatten(),
                        after_board_state=after_array.flatten(),
                        board_state_changed=state_changed,
                        solution_boolean=is_solution,
                        after_board_state_hash=after_hash,
                        score_before=frame.score,
                        score_after=after_frame.score,
                    )
                    self.results.append(transition)
                    transition_idx = len(self.results)

                # Save visualization (outside lock)
                self._save_transition_viz(
                    before_array, after_array, action, transition_idx,
                    state_changed, is_solution, is_duplicate, node.depth, node.move_sequence
                )

                # Pruning and queue management
                if not state_changed:
                    with self.lock:
                        self.stats["pruned_no_change"] += 1
                    continue

                if is_duplicate:
                    with self.lock:
                        self.stats["pruned_duplicate"] += 1
                    continue

                # New state found!
                with self.lock:
                    self.seen_hashes.add(after_hash)
                    self.stats["unique_states"] += 1

                # Don't explore terminal states
                if is_solution or is_game_over:
                    continue

                # Add new state to queue for exploration
                new_sequence = node.move_sequence + [action.value]
                new_node = StateNode(
                    state_hash=after_hash,
                    move_sequence=new_sequence,
                    depth=node.depth + 1,
                )
                self.work_queue.put(new_node)

            self.work_queue.task_done()

        logger.info(f"Worker {self.worker_id} stopped")
        self._session.close()


class ParallelDatasetGenerator:
    """Coordinates parallel BFS exploration of game state space."""

    SIMPLE_ACTIONS = [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
        GameAction.ACTION7,
    ]

    def __init__(
        self,
        game_id: str,
        root_url: str,
        num_workers: int = 4,
        visualize: bool = False,
        viz_dir: Optional[str] = None,
    ):
        self.game_id = game_id
        self.root_url = root_url
        self.num_workers = num_workers
        self.visualize = visualize
        self.viz_dir = viz_dir
        self.card_id: Optional[str] = None

        # Shared state (thread-safe)
        self.work_queue: queue.Queue = queue.Queue()
        self.results: list[Transition] = []
        self.seen_hashes: set[str] = set()
        self.stats = {
            "total_actions": 0,
            "unique_states": 0,
            "pruned_no_change": 0,
            "pruned_duplicate": 0,
            "solutions_found": 0,
            "game_overs": 0,
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Workers
        self.workers: list[ExplorationWorker] = []
        self.threads: list[threading.Thread] = []

        # API session for setup
        self._session = requests.Session()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self._session.headers.update(self.headers)

        # Available actions (will be set after first reset)
        self.available_actions: list[GameAction] = []

        # Setup visualization
        if self.visualize and self.viz_dir:
            os.makedirs(self.viz_dir, exist_ok=True)
            os.makedirs(os.path.join(self.viz_dir, "transitions"), exist_ok=True)
            os.makedirs(os.path.join(self.viz_dir, "states"), exist_ok=True)

    def _get_available_games(self) -> list[str]:
        """Fetch list of available games."""
        try:
            r = self._session.get(
                f"{self.root_url}/api/games",
                headers=self.headers,
                timeout=10,
            )
            if r.ok:
                return [g["game_id"] for g in r.json()]
        except Exception:
            pass
        return []

    def _open_scorecard(self) -> str:
        """Open a scorecard."""
        r = self._session.post(
            f"{self.root_url}/api/scorecard/open",
            json={"tags": ["dataset_generation_parallel", self.game_id]},
            headers=self.headers,
            timeout=10,
        )
        response = r.json()
        if not r.ok or "error" in response:
            raise RuntimeError(f"Failed to open scorecard: {response}")
        return response["card_id"]

    def _close_scorecard(self) -> None:
        """Close the scorecard."""
        if self.card_id:
            try:
                self._session.post(
                    f"{self.root_url}/api/scorecard/close",
                    json={"card_id": self.card_id},
                    headers=self.headers,
                    timeout=10,
                )
            except Exception:
                pass
            self.card_id = None

    def _init_game(self) -> Optional[tuple[np.ndarray, str]]:
        """Initialize game and get initial state."""
        # Reset to get initial frame
        data = {"card_id": self.card_id, "game_id": self.game_id}
        try:
            r = self._session.post(
                f"{self.root_url}/api/cmd/RESET",
                json=data,
                headers=self.headers,
                timeout=10,
            )
            response = r.json()
            if "error" in response:
                logger.error(f"Failed to reset game: {response}")
                return None

            frame = FrameData.model_validate(response)

            # Get available actions
            if frame.available_actions:
                self.available_actions = [
                    a for a in self.SIMPLE_ACTIONS
                    if a in frame.available_actions
                ]
            else:
                self.available_actions = self.SIMPLE_ACTIONS.copy()

            logger.info(f"Available actions: {[a.name for a in self.available_actions]}")

            # Get initial state
            grid = np.array(frame.frame, dtype=np.int8)
            if len(grid.shape) == 3:
                grid = grid[-1]

            state_hash = hashlib.md5(grid.tobytes()).hexdigest()
            return grid, state_hash

        except Exception as e:
            logger.error(f"Failed to initialize game: {e}")
            return None

    def generate(self, duration_seconds: int, max_depth: int = 100) -> list[Transition]:
        """Run parallel BFS exploration."""
        start_time = time.time()

        # Check game availability
        available_games = self._get_available_games()
        if available_games:
            matching = [g for g in available_games if g.startswith(self.game_id)]
            if not matching:
                logger.error(f"Game '{self.game_id}' not found. Available: {available_games}")
                return []
            if self.game_id not in available_games:
                self.game_id = matching[0]
                logger.info(f"Using game: {self.game_id}")

        # Open scorecard
        try:
            self.card_id = self._open_scorecard()
            logger.info(f"Opened scorecard: {self.card_id}")
        except Exception as e:
            logger.error(f"Failed to open scorecard: {e}")
            return []

        # Initialize game
        init_result = self._init_game()
        if init_result is None:
            self._close_scorecard()
            return []

        initial_grid, initial_hash = init_result
        self.seen_hashes.add(initial_hash)
        self.stats["unique_states"] = 1

        # Seed the queue with initial state
        initial_node = StateNode(
            state_hash=initial_hash,
            move_sequence=[],
            depth=0,
        )
        self.work_queue.put(initial_node)

        # Create and start workers
        logger.info(f"Starting {self.num_workers} workers...")
        for i in range(self.num_workers):
            worker = ExplorationWorker(
                worker_id=i,
                game_id=self.game_id,
                root_url=self.root_url,
                card_id=self.card_id,
                work_queue=self.work_queue,
                results=self.results,
                seen_hashes=self.seen_hashes,
                stats=self.stats,
                lock=self.lock,
                stop_event=self.stop_event,
                available_actions=self.available_actions,
                max_depth=max_depth,
                visualize=self.visualize,
                viz_dir=self.viz_dir,
            )
            self.workers.append(worker)

            thread = threading.Thread(target=worker.run, daemon=True)
            self.threads.append(thread)
            thread.start()

        # Monitor progress
        last_log_time = start_time
        try:
            while time.time() - start_time < duration_seconds:
                time.sleep(1)

                # Periodic logging
                if time.time() - last_log_time >= 10:
                    with self.lock:
                        logger.info(
                            f"Progress: {time.time() - start_time:.0f}s, "
                            f"{len(self.results)} transitions, "
                            f"{self.stats['unique_states']} unique states, "
                            f"queue: {self.work_queue.qsize()}"
                        )
                    last_log_time = time.time()

                # Check if queue is empty and all workers idle
                if self.work_queue.empty():
                    # Wait a bit to see if more work appears
                    time.sleep(2)
                    if self.work_queue.empty():
                        logger.info("Queue empty, exploration complete")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        # Stop workers
        logger.info("Stopping workers...")
        self.stop_event.set()

        for thread in self.threads:
            thread.join(timeout=5)

        elapsed = time.time() - start_time
        logger.info(f"Generation complete in {elapsed:.1f}s")
        logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")

        self._close_scorecard()
        return self.results

    def save_dataset(self, output_path: str) -> None:
        """Save transitions to disk."""
        if not self.results:
            logger.warning("No transitions to save")
            return

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        data = {
            "game_id": np.array([t.game_id for t in self.results], dtype=object),
            "move_num": np.array([t.move_num for t in self.results], dtype=np.int32),
            "move_type": np.array([t.move_type for t in self.results], dtype=object),
            "move": np.array([t.move for t in self.results], dtype=np.int8),
            "board_state_changed": np.array([t.board_state_changed for t in self.results], dtype=bool),
            "solution_boolean": np.array([t.solution_boolean for t in self.results], dtype=bool),
            "after_board_state_hash": np.array([t.after_board_state_hash for t in self.results], dtype=object),
            "score_before": np.array([t.score_before for t in self.results], dtype=np.int32),
            "score_after": np.array([t.score_after for t in self.results], dtype=np.int32),
            "before_board_state": np.stack([t.before_board_state for t in self.results]),
            "after_board_state": np.stack([t.after_board_state for t in self.results]),
        }

        move_sequences = [t.prev_move_sequence for t in self.results]

        npz_path = output_path if output_path.endswith(".npz") else f"{output_path}.npz"
        np.savez_compressed(
            npz_path,
            **data,
            prev_move_sequences=np.array(move_sequences, dtype=object),
            stats=np.array([json.dumps(self.stats)], dtype=object),
        )
        logger.info(f"Saved {len(self.results)} transitions to {npz_path}")

        # Summary JSON
        summary_path = output_path.replace(".npz", "") + "_summary.json"
        summary = {
            "game_id": self.game_id,
            "n_transitions": len(self.results),
            "stats": self.stats,
            "generated_at": datetime.now().isoformat(),
            "num_workers": self.num_workers,
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
        description="Parallel BFS dataset generator for ARC-3 games"
    )
    parser.add_argument("-g", "--game", required=True, help="Game ID")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--max-depth", type=int, default=100, help="Max exploration depth")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations")
    parser.add_argument("--viz-dir", default=None, help="Visualization directory")
    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup URL
    scheme = os.environ.get("SCHEME", "http")
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", 8001)

    if (scheme == "http" and str(port) == "80") or (scheme == "https" and str(port) == "443"):
        root_url = f"{scheme}://{host}"
    else:
        root_url = f"{scheme}://{host}:{port}"

    logger.info(f"Connecting to {root_url}")
    logger.info(f"Game: {args.game}, Duration: {args.duration}s, Workers: {args.workers}")

    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        os.makedirs("datasets", exist_ok=True)
        output_path = f"datasets/{args.game}_{timestamp}_parallel.npz"
    else:
        output_path = args.output

    viz_dir = args.viz_dir
    if args.visualize and viz_dir is None:
        viz_dir = f"datasets/{args.game}_{timestamp}_parallel_viz"

    # Generate
    generator = ParallelDatasetGenerator(
        args.game,
        root_url,
        num_workers=args.workers,
        visualize=args.visualize,
        viz_dir=viz_dir,
    )

    try:
        transitions = generator.generate(args.duration, args.max_depth)
        if transitions:
            generator.save_dataset(output_path)
            logger.info(f"Done! Generated {len(transitions)} transitions")
        else:
            logger.warning("No transitions generated")
    finally:
        generator.close()


if __name__ == "__main__":
    main()
