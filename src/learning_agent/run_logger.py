"""
Comprehensive logging for the Learning Agent.

This module handles all logging including:
- LLM call logs (prompts, responses, images)
- State snapshots
- Final reports
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import (
    ActionAnalysisResult,
    ActionID,
    AgentState,
    DiffResult,
    EnvironmentAnalysisResult,
    NextActionSuggestion,
)


class LLMCallLog(BaseModel):
    """Complete log of an LLM interaction."""

    call_id: int
    timestamp: str
    call_type: str  # "action_analysis" | "next_action_suggestion" | "environment_analysis"
    action_id: str | None

    # Inputs
    prompt: str
    images_sent: list[str]  # Paths to images
    context_provided: dict[str, Any]
    full_messages: list[dict[str, Any]] | None = None  # Full conversation history sent to LLM

    # Outputs
    response: dict[str, Any]

    # Timing
    duration_ms: float | None = None


class RunLogger:
    """
    Comprehensive logging for a single exploration run.

    Creates a structured directory with all logs, images, and state snapshots.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.calls_dir = self.run_dir / "calls"
        self.frames_dir = self.run_dir / "frames"
        self.call_count = 0

        # Ensure directories exist
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def _extract_prompt_from_messages(self, messages: list[dict] | None) -> str:
        """
        Extract all text content from LLM messages to show the actual prompt.

        This extracts the TRUE prompt sent to the LLM, not the simplified logging version.
        Images are indicated as [IMAGE] placeholders since they're sent as base64.
        """
        if not messages:
            return ""

        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                parts.append(f"=== {role.upper()} ===\n{content}\n")
            elif isinstance(content, list):
                # Multi-content message (text + images)
                text_parts = []
                image_count = 0
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        image_count += 1
                        # Check if it's the sanitized placeholder
                        url = item.get("image_url", {}).get("url", "")
                        if "OMITTED" in url:
                            text_parts.append(f"[IMAGE {image_count}]")
                        else:
                            text_parts.append(f"[IMAGE {image_count} - base64 data]")

                parts.append(f"=== {role.upper()} ===\n" + "\n".join(text_parts) + "\n")

        return "\n".join(parts)

    def log_action_analysis(
        self,
        action_id: ActionID,
        before_frame_path: str,
        after_frame_path: str,
        diff: DiffResult,
        prompt: str,
        result: ActionAnalysisResult,
        state_before: AgentState,
        state_after: AgentState,
        duration_ms: float | None = None,
        full_messages: list[dict] | None = None,
    ) -> str:
        """
        Log an action analysis LLM call.

        Returns:
            Path to the call log directory
        """
        self.call_count += 1

        # Create call directory
        call_dir = self.calls_dir / f"{self.call_count:03d}_{action_id.value}_analysis"
        call_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to call directory
        shutil.copy(before_frame_path, call_dir / "before.png")
        shutil.copy(after_frame_path, call_dir / "after.png")

        # Extract and save the actual prompt from full_messages (the true message to LLM)
        actual_prompt = self._extract_prompt_from_messages(full_messages)
        with open(call_dir / "prompt.txt", "w") as f:
            f.write("=== ACTUAL FULL CONVERSATION SENT TO LLM ===\n")
            f.write("(Images are sent as base64 data, shown as [IMAGE N] placeholders)\n\n")
            f.write(actual_prompt if actual_prompt else prompt)

        # Save diff
        with open(call_dir / "diff.json", "w") as f:
            f.write(diff.model_dump_json(indent=2))

        # Save response
        with open(call_dir / "response.json", "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save state snapshots
        with open(call_dir / "state_before.json", "w") as f:
            f.write(state_before.model_dump_json(indent=2))

        with open(call_dir / "state_after.json", "w") as f:
            f.write(state_after.model_dump_json(indent=2))

        # Save call metadata
        log = LLMCallLog(
            call_id=self.call_count,
            timestamp=datetime.now().isoformat(),
            call_type="action_analysis",
            action_id=action_id.value,
            prompt=prompt,
            images_sent=["before.png", "after.png"],
            context_provided={
                "diff_summary": diff.change_summary,
                "has_changes": diff.has_changes,
            },
            full_messages=full_messages,
            response=result.model_dump(),
            duration_ms=duration_ms,
        )

        with open(call_dir / "metadata.json", "w") as f:
            f.write(log.model_dump_json(indent=2))

        # Also save full messages as separate file for easier reading
        if full_messages:
            with open(call_dir / "full_conversation.json", "w") as f:
                json.dump(full_messages, f, indent=2)

        return str(call_dir)

    def log_next_action_suggestion(
        self,
        current_frame_path: str,
        prompt: str,
        result: NextActionSuggestion,
        state: AgentState,
        duration_ms: float | None = None,
        full_messages: list[dict] | None = None,
    ) -> str:
        """
        Log a next action suggestion LLM call.

        Returns:
            Path to the call log directory
        """
        self.call_count += 1

        # Create call directory
        call_dir = self.calls_dir / f"{self.call_count:03d}_next_action_suggestion"
        call_dir.mkdir(parents=True, exist_ok=True)

        # Copy current frame
        shutil.copy(current_frame_path, call_dir / "current_frame.png")

        # Extract and save the actual prompt from full_messages
        actual_prompt = self._extract_prompt_from_messages(full_messages)
        with open(call_dir / "prompt.txt", "w") as f:
            f.write("=== ACTUAL FULL CONVERSATION SENT TO LLM ===\n")
            f.write("(Images are sent as base64 data, shown as [IMAGE N] placeholders)\n\n")
            f.write(actual_prompt if actual_prompt else prompt)

        # Save response
        with open(call_dir / "response.json", "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save current state
        with open(call_dir / "state.json", "w") as f:
            f.write(state.model_dump_json(indent=2))

        # Save call metadata
        log = LLMCallLog(
            call_id=self.call_count,
            timestamp=datetime.now().isoformat(),
            call_type="next_action_suggestion",
            action_id=None,
            prompt=prompt,
            images_sent=["current_frame.png"],
            context_provided={
                "verified_actions": [
                    k for k, v in state.action_knowledge.items() if v.is_verified
                ],
                "pending_actions": [
                    k
                    for k, v in state.action_knowledge.items()
                    if v.needs_more_observations()
                ],
            },
            full_messages=full_messages,
            response=result.model_dump(),
            duration_ms=duration_ms,
        )

        with open(call_dir / "metadata.json", "w") as f:
            f.write(log.model_dump_json(indent=2))

        # Also save full messages as separate file for easier reading
        if full_messages:
            with open(call_dir / "full_conversation.json", "w") as f:
                json.dump(full_messages, f, indent=2)

        return str(call_dir)

    def log_environment_analysis(
        self,
        analysis: EnvironmentAnalysisResult,
        action_context: str,
        duration_ms: float | None = None,
        full_messages: list[dict] | None = None,
    ) -> str:
        """
        Log an environment analysis LLM call.

        Returns:
            Path to the call log directory
        """
        self.call_count += 1

        # Create call directory
        call_dir = self.calls_dir / f"{self.call_count:03d}_environment_analysis"
        call_dir.mkdir(parents=True, exist_ok=True)

        # Save context
        with open(call_dir / "context.txt", "w") as f:
            f.write(f"Action Context: {action_context}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            if duration_ms:
                f.write(f"Duration: {duration_ms:.0f}ms\n")

        # Save response
        with open(call_dir / "response.json", "w") as f:
            f.write(analysis.model_dump_json(indent=2))

        # Save breakthroughs separately for easy review
        if analysis.breakthroughs:
            with open(call_dir / "breakthroughs.txt", "w") as f:
                for i, b in enumerate(analysis.breakthroughs, 1):
                    f.write(f"{i}. {b}\n")

        # Save movement constraints
        if analysis.movement_constraints:
            with open(call_dir / "constraints.txt", "w") as f:
                for c in analysis.movement_constraints:
                    f.write(f"- {c}\n")

        # Save suggested action updates
        if analysis.suggested_action_updates:
            with open(call_dir / "action_updates.txt", "w") as f:
                for update in analysis.suggested_action_updates:
                    f.write(f"{update.action_id}: {update.suggested_definition}\n")
                    f.write(f"  Reason: {update.reasoning}\n\n")

        # Save full messages as separate file for easier reading
        if full_messages:
            with open(call_dir / "full_conversation.json", "w") as f:
                json.dump(full_messages, f, indent=2)

        return str(call_dir)

    def log_setup_action(
        self,
        action_id: ActionID,
        frame_path: str,
        reason: str,
    ) -> None:
        """Log a setup action (not analyzed, just executed)."""
        self.call_count += 1

        call_dir = self.calls_dir / f"{self.call_count:03d}_{action_id.value}_setup"
        call_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(frame_path, call_dir / "frame.png")

        with open(call_dir / "info.txt", "w") as f:
            f.write(f"Setup action: {action_id.value}\n")
            f.write(f"Reason: {reason}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    def generate_final_report(self, state: AgentState) -> str:
        """
        Generate a human-readable final report.

        Returns:
            Path to the report file
        """
        report_path = self.run_dir / "final_report.md"

        lines = [
            "# Learning Agent Exploration Report",
            "",
            f"**Run ID:** {state.run_id}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Total Actions:** {state.action_count}",
            f"**Total LLM Calls:** {state.llm_call_count}",
            "",
            "---",
            "",
            "## Action Knowledge Summary",
            "",
        ]

        for action_id, knowledge in state.action_knowledge.items():
            if knowledge.is_verified:
                status = "VERIFIED"
            elif knowledge.is_exhausted:
                status = "EXHAUSTED"
            else:
                status = "UNVERIFIED"
            obs_count = len(knowledge.observations)
            effective_attempts = knowledge.verification_attempts

            lines.append(f"### {action_id} ({status})")
            lines.append("")
            lines.append(f"**Definition:** {knowledge.current_definition or 'None'}")
            lines.append(f"**Observations:** {obs_count}")
            lines.append(f"**Effective Attempts:** {effective_attempts}/8")
            if knowledge.is_exhausted:
                lines.append(f"**Consecutive No-Effects:** {knowledge.consecutive_no_effects} (exhausted)")
            lines.append("")

            if knowledge.observations:
                lines.append("**Recent Observations:**")
                for obs in knowledge.observations[-3:]:
                    effect = "HAD EFFECT" if obs.had_effect else "NO EFFECT"
                    lines.append(f"- [{effect}] {obs.llm_interpretation[:100]}...")
                lines.append("")

        lines.extend(
            [
                "---",
                "",
                "## Environment Understanding",
                "",
                f"**Environment Analyses Performed:** {state.environment.analysis_count}",
                "",
            ]
        )

        # Background and boundaries
        if state.environment.background_color:
            lines.append(f"**Background Color:** {state.environment.background_color}")

        if state.environment.has_border:
            lines.append(f"**Border:** Yes ({state.environment.border_color or 'unknown color'})")
            if state.environment.border_description:
                lines.append(f"  - {state.environment.border_description}")
        lines.append("")

        # Breakthroughs
        if state.environment.breakthroughs:
            lines.append("### 🔍 Key Breakthroughs")
            for breakthrough in state.environment.breakthroughs:
                lines.append(f"- {breakthrough}")
            lines.append("")

        # Movement constraints
        if state.environment.movement_constraints:
            lines.append("### Movement Constraints")
            for constraint in state.environment.movement_constraints:
                lines.append(f"- {constraint}")
            lines.append("")

        # Internal walls
        if state.environment.internal_walls:
            lines.append("### Internal Walls/Obstacles")
            for wall in state.environment.internal_walls:
                lines.append(f"- {wall}")
            lines.append("")

        # Identified objects (structured)
        if state.environment.identified_objects:
            lines.append("### Identified Objects (Structured)")
            for obj in state.environment.identified_objects:
                name = obj.get('name', 'Unknown')
                color = obj.get('color', '?')
                shape = obj.get('shape', '?')
                role = obj.get('role_hypothesis', 'Unknown')
                lines.append(f"- **{name}:** {color} {shape}")
                lines.append(f"  - Role hypothesis: {role}")
            lines.append("")

        # Spatial structure
        if state.environment.spatial_structure:
            lines.append(f"### Spatial Structure")
            lines.append(state.environment.spatial_structure)
            lines.append("")

        # Open questions
        if state.environment.open_questions:
            lines.append("### Open Questions")
            for question in state.environment.open_questions:
                lines.append(f"- {question}")
            lines.append("")

        # Legacy objects and observations
        if state.environment.objects:
            lines.append("### Objects Identified (Legacy)")
            for obj in state.environment.objects:
                lines.append(f"- **{obj.name}:** {obj.description}")
            lines.append("")

        if state.environment.spatial_rules:
            lines.append("### Spatial Rules")
            for rule in state.environment.spatial_rules:
                lines.append(f"- {rule}")
            lines.append("")

        if state.environment.general_observations:
            lines.append("### General Observations")
            for obs in state.environment.general_observations[-10:]:
                lines.append(f"- {obs}")
            lines.append("")

        # Write report
        report_content = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report_content)

        return str(report_path)

    def save_state_snapshot(self, state: AgentState, label: str) -> str:
        """Save a labeled state snapshot."""
        snapshot_path = self.run_dir / f"state_{label}.json"
        with open(snapshot_path, "w") as f:
            f.write(state.model_dump_json(indent=2))
        return str(snapshot_path)


class ConsoleLogger:
    """Simple console logging for progress updates."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def info(self, message: str) -> None:
        if self.verbose:
            print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} | {message}")

    def action(self, action_id: ActionID, message: str) -> None:
        if self.verbose:
            print(f"[ACTION] {action_id.value} | {message}")

    def llm_call(self, call_type: str, action_id: ActionID | None = None) -> None:
        if self.verbose:
            action_str = f" ({action_id.value})" if action_id else ""
            print(f"[LLM] {call_type}{action_str}")

    def result(self, message: str) -> None:
        if self.verbose:
            print(f"[RESULT] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")

    def separator(self) -> None:
        if self.verbose:
            print("-" * 60)
