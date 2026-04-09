"""
Run Logger — detailed per-run logging that replicates the VLM agent's results/ structure.

Produces a directory tree like:

    results/{game_id}/{run_label}/
        state.json              # cumulative run state (updated after every step)
        frames/
            before_NNNN.png     # frame before action
            after_NNNN.png      # frame after action
            current_NNNN.png    # current frame snapshot
        calls/
            001_exploration_action/
                metadata.json
                state.json
                current_frame.png
            002_transition_observation/
                metadata.json
                before.png
                after.png
                diff.json
                state_before.json
                state_after.json
            003_synthesis_attempt/
                metadata.json
                prompt.txt
                response.txt        # synthesized code
                verification.json   # accuracy, mismatches
                full_conversation.json
                synthesized_code.py

Call types:
    exploration_action      — choosing which action to take (analogous to next_action_suggestion)
    transition_observation  — recording before/after for an action (analogous to ACTION*_analysis)
    synthesis_attempt       — LLM synthesis call + verification (NEW for synthesis agents)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _np_to_json(obj: Any) -> Any:
    """Make numpy types JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_frame_as_png(frame: np.ndarray, path: str) -> None:
    """Save a numpy frame (H,W,3 uint8 or H,W grayscale) as PNG."""
    frame = np.asarray(frame, dtype=np.uint8)
    img = Image.fromarray(frame)
    img.save(path)


class RunLogger:
    """Logs a single agent run to the results/ directory."""

    def __init__(
        self,
        game_id: str,
        agent_name: str,
        run_label: str | None = None,
        results_dir: str = "results",
    ) -> None:
        self.game_id = game_id
        self.agent_name = agent_name
        self.start_time = datetime.now(timezone.utc)

        if run_label is None:
            run_label = f"run_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.run_label = run_label

        self.run_dir = Path(results_dir) / game_id / run_label
        self.calls_dir = self.run_dir / "calls"
        self.frames_dir = self.run_dir / "frames"

        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._call_counter = 0
        self._frame_counter = 0

        # Cumulative state that gets written to state.json after every step
        self._state: dict[str, Any] = {
            "run_id": run_label,
            "agent_name": agent_name,
            "game_id": game_id,
            "start_time": self.start_time.isoformat(),
            "action_count": 0,
            "phase": "explore",
            "exploration": {
                "total_observations": 0,
                "action_counts": {},
                "available_actions": [],
            },
            "synthesis": {
                "attempts": 0,
                "best_accuracy": 0.0,
                "model_is_perfect": False,
                "history": [],
            },
            "score": {
                "current": 0,
                "max": 0,
            },
        }
        self._flush_state()

    # ── Public API ───────────────────────────────────────────────────────

    def log_exploration_action(
        self,
        action_id: int,
        phase: str,
        current_frame: np.ndarray | None = None,
        exploration_state: Any = None,
        reasoning: str = "",
    ) -> str:
        """Log the agent choosing an exploration action."""
        self._call_counter += 1
        call_name = f"{self._call_counter:03d}_exploration_action"
        call_dir = self.calls_dir / call_name
        call_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).isoformat()

        # Save current frame
        if current_frame is not None:
            frame_path = str(call_dir / "current_frame.png")
            _save_frame_as_png(current_frame, frame_path)
            # Also save to frames/
            self._save_numbered_frame(current_frame, "current")

        # Build metadata
        metadata = {
            "call_id": self._call_counter,
            "timestamp": ts,
            "call_type": "exploration_action",
            "action_id": action_id,
            "phase": phase,
            "reasoning": reasoning,
        }

        if exploration_state is not None:
            metadata["exploration_snapshot"] = {
                "total_observations": exploration_state.total_observations,
                "action_counts": dict(exploration_state.action_counts),
                "available_action_ids": sorted(exploration_state.available_action_ids),
                "has_minimum_coverage": exploration_state.has_minimum_coverage,
                "synthesis_attempts": exploration_state.synthesis_attempts,
                "best_accuracy": exploration_state.best_accuracy,
                "model_is_perfect": exploration_state.model_is_perfect,
            }

        self._write_json(call_dir / "metadata.json", metadata)

        # State snapshot
        self._update_state(
            action_count_increment=1,
            phase=phase,
            exploration_state=exploration_state,
        )
        self._write_json(call_dir / "state.json", self._state)

        return str(call_dir)

    def log_transition(
        self,
        action_id: int,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        observation_text: str,
        diff_data: dict[str, Any] | None = None,
        exploration_state: Any = None,
    ) -> str:
        """Log a transition observation (before/after an action)."""
        self._call_counter += 1
        call_name = f"{self._call_counter:03d}_ACTION{action_id}_transition"
        call_dir = self.calls_dir / call_name
        call_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).isoformat()

        # Save before/after frames
        before_path = str(call_dir / "before.png")
        after_path = str(call_dir / "after.png")
        _save_frame_as_png(before_frame, before_path)
        _save_frame_as_png(after_frame, after_path)

        # Also save to frames/ directory
        self._save_numbered_frame(before_frame, "before")
        self._save_numbered_frame(after_frame, "after")

        # Build metadata
        metadata = {
            "call_id": self._call_counter,
            "timestamp": ts,
            "call_type": "transition_observation",
            "action_id": action_id,
            "observation": observation_text,
        }
        self._write_json(call_dir / "metadata.json", metadata)

        # Diff data
        if diff_data is None:
            diff_data = self._compute_diff_data(before_frame, after_frame)
        self._write_json(call_dir / "diff.json", diff_data)

        # State snapshots
        state_before = {
            "timestamp": ts,
            "action_id": action_id,
            "phase": "before_action",
        }
        state_after = {
            "timestamp": ts,
            "action_id": action_id,
            "phase": "after_action",
            "observation": observation_text,
        }

        if exploration_state is not None:
            state_after["exploration_snapshot"] = {
                "total_observations": exploration_state.total_observations,
                "action_counts": dict(exploration_state.action_counts),
                "replay_buffer_size": len(exploration_state.replay_buffer),
            }

        self._write_json(call_dir / "state_before.json", state_before)
        self._write_json(call_dir / "state_after.json", state_after)

        # Update cumulative state
        self._update_state(exploration_state=exploration_state)

        return str(call_dir)

    def log_synthesis_attempt(
        self,
        attempt_number: int,
        prompt: str,
        synthesized_code: str | None,
        verification_result: Any | None,
        llm_messages: list[dict] | None = None,
        duration_ms: float = 0.0,
        refinement_round: int = 0,
        exploration_state: Any = None,
    ) -> str:
        """Log an LLM synthesis attempt (the NEW call type for synthesis agents)."""
        self._call_counter += 1
        call_name = f"{self._call_counter:03d}_synthesis_attempt"
        call_dir = self.calls_dir / call_name
        call_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).isoformat()

        # Save the prompt
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        # Save synthesized code
        if synthesized_code is not None:
            (call_dir / "synthesized_code.py").write_text(
                synthesized_code, encoding="utf-8"
            )
            (call_dir / "response.txt").write_text(
                synthesized_code, encoding="utf-8"
            )

        # Verification result
        verification_data: dict[str, Any] = {}
        if verification_result is not None:
            verification_data = {
                "correct": verification_result.correct,
                "incorrect": verification_result.incorrect,
                "accuracy": verification_result.accuracy,
                "is_perfect": verification_result.is_perfect,
                "errors": verification_result.errors[:10],
                "mismatches": [
                    {
                        "timestep": mm.get("timestep"),
                        "action_id": mm.get("action_id"),
                        "diff_pixel_count": mm.get("diff_pixel_count"),
                        "sample_diffs": mm.get("sample_diffs", [])[:5],
                    }
                    for mm in verification_result.mismatches[:10]
                ],
            }
        self._write_json(call_dir / "verification.json", verification_data)

        # Metadata
        metadata = {
            "call_id": self._call_counter,
            "timestamp": ts,
            "call_type": "synthesis_attempt",
            "attempt_number": attempt_number,
            "refinement_round": refinement_round,
            "duration_ms": duration_ms,
            "accuracy": verification_data.get("accuracy", 0.0),
            "is_perfect": verification_data.get("is_perfect", False),
            "correct": verification_data.get("correct", 0),
            "incorrect": verification_data.get("incorrect", 0),
            "num_errors": len(verification_data.get("errors", [])),
        }
        self._write_json(call_dir / "metadata.json", metadata)

        # Full conversation (if provided)
        if llm_messages is not None:
            # Strip base64 images to keep file size reasonable
            cleaned = self._strip_images_from_messages(llm_messages)
            self._write_json(call_dir / "full_conversation.json", cleaned)

        # Update synthesis history in cumulative state
        self._state["synthesis"]["attempts"] = attempt_number
        accuracy = verification_data.get("accuracy", 0.0)
        if accuracy > self._state["synthesis"]["best_accuracy"]:
            self._state["synthesis"]["best_accuracy"] = accuracy
        self._state["synthesis"]["model_is_perfect"] = verification_data.get(
            "is_perfect", False
        )
        self._state["synthesis"]["history"].append({
            "attempt": attempt_number,
            "refinement": refinement_round,
            "accuracy": accuracy,
            "is_perfect": verification_data.get("is_perfect", False),
            "timestamp": ts,
            "duration_ms": duration_ms,
        })

        self._update_state(exploration_state=exploration_state)

        return str(call_dir)

    def log_frame_analysis(
        self,
        frame: np.ndarray,
        analysis_text: str,
    ) -> str:
        """Log a frame analysis (analogous to environment_analysis)."""
        self._call_counter += 1
        call_name = f"{self._call_counter:03d}_frame_analysis"
        call_dir = self.calls_dir / call_name
        call_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).isoformat()

        _save_frame_as_png(frame, str(call_dir / "current_frame.png"))

        self._write_json(call_dir / "response.json", {
            "timestamp": ts,
            "analysis": analysis_text,
        })
        (call_dir / "context.txt").write_text(analysis_text, encoding="utf-8")

        return str(call_dir)

    def finalize(
        self,
        outcome: str = "COMPLETED",
        exploration_state: Any = None,
    ) -> None:
        """Write final state and optionally rename the run directory."""
        self._state["end_time"] = datetime.now(timezone.utc).isoformat()
        self._state["outcome"] = outcome
        self._state["total_calls"] = self._call_counter
        self._state["total_frames"] = self._frame_counter

        if exploration_state is not None:
            self._update_state(exploration_state=exploration_state)

        self._flush_state()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _save_numbered_frame(self, frame: np.ndarray, prefix: str) -> str:
        path = str(self.frames_dir / f"{prefix}_{self._frame_counter:04d}.png")
        _save_frame_as_png(frame, path)
        self._frame_counter += 1
        return path

    def _update_state(
        self,
        action_count_increment: int = 0,
        phase: str | None = None,
        exploration_state: Any = None,
    ) -> None:
        if action_count_increment:
            self._state["action_count"] += action_count_increment
        if phase is not None:
            self._state["phase"] = phase
        if exploration_state is not None:
            self._state["exploration"] = {
                "total_observations": exploration_state.total_observations,
                "action_counts": dict(exploration_state.action_counts),
                "available_actions": sorted(exploration_state.available_action_ids),
                "replay_buffer_size": len(exploration_state.replay_buffer),
                "has_minimum_coverage": exploration_state.has_minimum_coverage,
                "min_observations_per_action": exploration_state.min_observations_per_action,
            }
            self._state["score"] = {
                "current": exploration_state.current_score,
                "max": exploration_state.max_score,
            }
            self._state["synthesis"]["attempts"] = exploration_state.synthesis_attempts
            self._state["synthesis"]["best_accuracy"] = exploration_state.best_accuracy
            self._state["synthesis"]["model_is_perfect"] = exploration_state.model_is_perfect
        self._flush_state()

    def _flush_state(self) -> None:
        self._write_json(self.run_dir / "state.json", self._state)

    def _write_json(self, path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_np_to_json)

    def _compute_diff_data(
        self, before: np.ndarray, after: np.ndarray
    ) -> dict[str, Any]:
        """Compute a diff summary between two frames."""
        if before.shape != after.shape:
            return {
                "has_changes": True,
                "change_summary": f"Shape mismatch: {before.shape} vs {after.shape}",
                "changed_pixels": [],
            }

        if before.ndim == 3:
            diff_mask = np.any(before != after, axis=-1)
        else:
            diff_mask = before != after

        changed_count = int(np.sum(diff_mask))
        positions = list(zip(*np.where(diff_mask)))

        changed_pixels = []
        for r, c in positions[:50]:
            changed_pixels.append({
                "pos": [int(r), int(c)],
                "before": before[r, c].tolist() if before.ndim == 3 else int(before[r, c]),
                "after": after[r, c].tolist() if after.ndim == 3 else int(after[r, c]),
            })

        # Bounding box
        bbox = None
        if changed_count > 0:
            rows, cols = np.where(diff_mask)
            bbox = [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())]

        return {
            "has_changes": changed_count > 0,
            "changed_pixel_count": changed_count,
            "change_summary": (
                f"{changed_count} pixels changed"
                if changed_count > 0
                else "No changes detected"
            ),
            "bounding_box": bbox,
            "changed_pixels": changed_pixels,
        }

    def _strip_images_from_messages(
        self, messages: list[dict]
    ) -> list[dict]:
        """Strip base64 image data from messages to keep logs manageable."""
        cleaned = []
        for msg in messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": "[BASE64_IMAGE_DATA_OMITTED]"},
                        })
                    else:
                        new_content.append(part)
                msg_copy["content"] = new_content
            cleaned.append(msg_copy)
        return cleaned
