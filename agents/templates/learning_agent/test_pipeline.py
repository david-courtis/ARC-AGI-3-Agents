#!/usr/bin/env python
"""
Test script for the Learning Agent pipeline.

Run with: uv run python -m agents.templates.learning_agent.test_pipeline

Tests each component:
1. Models - Data structures work correctly
2. Diff - Frame comparison works
3. Vision - Rendering and capture works
4. Knowledge - State management works
5. LLM Agents - API calls work (requires OPENROUTER_API_KEY)
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def test_models():
    """Test that all Pydantic models work correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Models")
    print("=" * 60)

    from .models import (
        ActionID,
        ActionKnowledge,
        ActionObservation,
        AgentState,
        DiffResult,
        EnvironmentKnowledge,
        PixelChange,
        ActionAnalysisResult,
        NextActionSuggestion,
    )

    # Test ActionID enum
    print("  Testing ActionID enum...")
    assert ActionID.ACTION1.value == "ACTION1"
    assert len(list(ActionID)) == 5
    print("    ✓ ActionID enum works")

    # Test ActionKnowledge
    print("  Testing ActionKnowledge...")
    knowledge = ActionKnowledge(action_id=ActionID.ACTION1)
    assert not knowledge.is_verified
    assert knowledge.needs_more_observations()
    assert knowledge.verification_attempts == 0
    print("    ✓ ActionKnowledge initialization works")

    # Test adding observations
    obs = ActionObservation(
        before_frame_path="/tmp/before.png",
        after_frame_path="/tmp/after.png",
        diff_summary="Test diff",
        llm_interpretation="Test interpretation",
        context_description="Test context",
        had_effect=True,
        was_consistent=True,
    )
    knowledge.add_observation(obs, "Test definition", is_consistent=True)
    assert len(knowledge.observations) == 1
    assert knowledge.verification_attempts == 1
    print("    ✓ Adding observations works")

    # Test verification logic
    for i in range(2):
        obs2 = ActionObservation(
            before_frame_path=f"/tmp/before{i}.png",
            after_frame_path=f"/tmp/after{i}.png",
            diff_summary="Test diff",
            llm_interpretation="Test interpretation",
            context_description="Test context",
            had_effect=True,
            was_consistent=True,
        )
        knowledge.add_observation(obs2, "Test definition", is_consistent=True)

    assert knowledge.is_verified, "Should be verified after 3 consistent observations"
    print("    ✓ Verification logic works")

    # Test AgentState
    print("  Testing AgentState...")
    state = AgentState(run_id="test_run")
    assert state.run_id == "test_run"
    assert len(state.action_knowledge) == 0  # Empty initially
    print("    ✓ AgentState works")

    # Test DiffResult
    print("  Testing DiffResult...")
    diff = DiffResult(
        has_changes=True,
        changed_pixels=[PixelChange(row=5, col=5, old_value=0, new_value=1)],
        change_summary="1 pixel changed",
    )
    assert diff.has_changes
    assert len(diff.changed_pixels) == 1
    print("    ✓ DiffResult works")

    # Test ActionAnalysisResult
    print("  Testing ActionAnalysisResult...")
    analysis = ActionAnalysisResult(
        interpretation="Something moved",
        action_definition="Moves something",
        is_consistent_with_previous=None,
        objects_involved=["player"],
        context_description="Player in center",
        had_effect=True,
        no_effect_reason=None,
        environment_updates=["Found a wall"],
        confidence=0.9,
    )
    assert analysis.had_effect
    print("    ✓ ActionAnalysisResult works")

    # Test NextActionSuggestion
    print("  Testing NextActionSuggestion...")
    suggestion = NextActionSuggestion(
        target_action=ActionID.ACTION2,
        setup_sequence=[ActionID.ACTION1],
        reasoning="Testing movement",
        expected_information_gain="Learn what ACTION2 does",
        current_board_assessment="Player in corner",
    )
    assert suggestion.target_action == ActionID.ACTION2
    print("    ✓ NextActionSuggestion works")

    print("\n  ✅ All model tests passed!")
    return True


def test_diff():
    """Test frame differencing."""
    print("\n" + "=" * 60)
    print("TEST 2: Diff Computation")
    print("=" * 60)

    from .diff import PixelDiffer, SmartDiffer, create_differ

    # Create test frames
    frame1 = np.zeros((10, 10), dtype=np.uint8)
    frame2 = np.zeros((10, 10), dtype=np.uint8)
    frame2[5, 5] = 1  # Single pixel change
    frame2[3, 3] = 2  # Another change

    # Test PixelDiffer
    print("  Testing PixelDiffer...")
    pixel_differ = PixelDiffer()
    diff = pixel_differ.compute_diff(frame1, frame2)
    assert diff.has_changes
    assert len(diff.changed_pixels) == 2
    print(f"    Found {len(diff.changed_pixels)} changed pixels")
    print("    ✓ PixelDiffer works")

    # Test SmartDiffer
    print("  Testing SmartDiffer...")
    smart_differ = SmartDiffer()
    diff = smart_differ.compute_diff(frame1, frame2)
    assert diff.has_changes
    print(f"    Summary: {diff.change_summary}")
    print("    ✓ SmartDiffer works")

    # Test create_differ factory
    print("  Testing create_differ factory...")
    d1 = create_differ("pixel")
    d2 = create_differ("smart")
    assert isinstance(d1, PixelDiffer)
    assert isinstance(d2, SmartDiffer)
    print("    ✓ Factory works")

    # Test no changes
    print("  Testing no-change detection...")
    diff_no_change = pixel_differ.compute_diff(frame1, frame1)
    assert not diff_no_change.has_changes
    print("    ✓ No-change detection works")

    print("\n  ✅ All diff tests passed!")
    return True


def test_vision():
    """Test vision utilities."""
    print("\n" + "=" * 60)
    print("TEST 3: Vision Utilities")
    print("=" * 60)

    from .vision import GridFrameRenderer, FrameCapture

    # Create test frame
    frame = np.zeros((10, 10), dtype=np.uint8)
    frame[5, 5] = 1  # Blue pixel
    frame[3, 3] = 2  # Red pixel

    # Test GridFrameRenderer
    print("  Testing GridFrameRenderer...")
    renderer = GridFrameRenderer(scale=8)
    img = renderer.render(frame)
    assert img.size == (80, 80)  # 10x10 * 8 scale
    print(f"    Rendered image size: {img.size}")
    print("    ✓ Rendering works")

    # Test render_to_base64
    print("  Testing base64 encoding...")
    b64 = renderer.render_to_base64(frame)
    assert len(b64) > 0
    # Verify it's valid base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0
    print(f"    Base64 length: {len(b64)}")
    print("    ✓ Base64 encoding works")

    # Test FrameCapture with temp directory
    print("  Testing FrameCapture...")
    with tempfile.TemporaryDirectory() as tmpdir:
        capture = FrameCapture(renderer=renderer, output_dir=tmpdir)

        # Capture single frame
        path = capture.capture(frame, "test_frame")
        assert Path(path).exists()
        print(f"    Saved to: {path}")
        print("    ✓ Frame capture works")

        # Capture pair
        before_path, after_path = capture.capture_pair(frame, frame, "test_pair")
        assert Path(before_path).exists()
        assert Path(after_path).exists()
        print("    ✓ Pair capture works")

    print("\n  ✅ All vision tests passed!")
    return True


def test_knowledge():
    """Test knowledge management."""
    print("\n" + "=" * 60)
    print("TEST 4: Knowledge Management")
    print("=" * 60)

    from .knowledge import KnowledgeManager
    from .models import ActionID, ActionKnowledge, EnvironmentKnowledge, DiffResult

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test KnowledgeManager
        print("  Testing KnowledgeManager...")
        km = KnowledgeManager(base_dir=tmpdir)

        # Create new run
        state = km.create_new_run()
        assert state.run_id is not None
        assert len(state.action_knowledge) == 5  # All 5 actions initialized
        print(f"    Created run: {state.run_id}")
        print("    ✓ Run creation works")

        # Test save/load
        print("  Testing state persistence...")
        km.save_state(state)
        loaded = km.store.load(state.run_id)
        assert loaded is not None
        assert loaded.run_id == state.run_id
        print("    ✓ Save/load works")

        # Test prompt formatting
        print("  Testing prompt formatting...")
        action_knowledge = ActionKnowledge(action_id=ActionID.ACTION1)
        environment = EnvironmentKnowledge()
        diff = DiffResult(has_changes=True, changed_pixels=[], change_summary="1 change")

        context = km.format_for_action_analysis(
            ActionID.ACTION1, action_knowledge, environment, diff
        )
        assert "observation_history" in context
        assert "environment" in context
        assert "diff_summary" in context
        print("    ✓ Action analysis formatting works")

        context2 = km.format_for_next_action(state.action_knowledge, environment)
        assert "action_status" in context2
        assert "verified_actions" in context2
        assert "pending_actions" in context2
        print("    ✓ Next action formatting works")

    print("\n  ✅ All knowledge tests passed!")
    return True


def test_llm_agent_creation():
    """Test LLM agent creation (no API calls)."""
    print("\n" + "=" * 60)
    print("TEST 5: LLM Agent Creation")
    print("=" * 60)

    from .llm_agents import create_agent, OpenRouterAgent

    # Check if API key is available
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  ⚠️  OPENROUTER_API_KEY not set, skipping agent creation test")
        print("     Set OPENROUTER_API_KEY to test LLM integration")
        return True

    print("  Testing agent creation...")
    agent = create_agent(provider="openrouter", model="google/gemini-2.5-flash")
    assert isinstance(agent, OpenRouterAgent)
    print(f"    Agent type: {type(agent).__name__}")
    print(f"    Model: {agent.model_name}")
    print("    ✓ Agent creation works")

    # Test that agent has required methods
    print("  Testing agent interface...")
    assert hasattr(agent, "analyze_action")
    assert hasattr(agent, "suggest_next_action")
    assert hasattr(agent, "client")  # OpenAI client
    print("    ✓ Agent interface correct")

    print("\n  ✅ All LLM agent creation tests passed!")
    return True


def test_llm_api_call():
    """Test actual LLM API call (requires API key)."""
    print("\n" + "=" * 60)
    print("TEST 6: LLM API Call (Live)")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  ⚠️  OPENROUTER_API_KEY not set, skipping live API test")
        print("     Set OPENROUTER_API_KEY to test LLM integration")
        return True

    from .llm_agents import create_agent
    from .models import ActionID, ActionKnowledge, EnvironmentKnowledge, DiffResult
    from .vision import GridFrameRenderer

    print("  Creating test frames...")
    # Create simple test frames
    frame1 = np.zeros((10, 10), dtype=np.uint8)
    frame2 = np.zeros((10, 10), dtype=np.uint8)
    frame2[5, 5] = 1  # Single pixel change

    renderer = GridFrameRenderer()
    before_b64 = renderer.render_to_base64(frame1)
    after_b64 = renderer.render_to_base64(frame2)
    print("    ✓ Test frames created")

    print("  Creating agent and making API call...")
    print("    (This may take a few seconds...)")

    agent = create_agent(provider="openrouter", model="google/gemini-2.5-flash")

    # Create test knowledge
    action_knowledge = ActionKnowledge(action_id=ActionID.ACTION1)
    environment = EnvironmentKnowledge()
    diff = DiffResult(
        has_changes=True,
        changed_pixels=[],
        change_summary="1 pixel changed from black to blue at position (5,5)",
    )

    try:
        result = agent.analyze_action(
            before_image_b64=before_b64,
            after_image_b64=after_b64,
            action_id=ActionID.ACTION1,
            diff=diff,
            action_knowledge=action_knowledge,
            environment=environment,
        )

        print("    ✓ API call successful!")
        print(f"    Interpretation: {result.interpretation[:100]}...")
        print(f"    Had effect: {result.had_effect}")
        print(f"    Confidence: {result.confidence}")

    except Exception as e:
        print(f"    ✗ API call failed: {e}")
        return False

    print("\n  ✅ LLM API test passed!")
    return True


def test_run_logger():
    """Test run logging."""
    print("\n" + "=" * 60)
    print("TEST 7: Run Logger")
    print("=" * 60)

    from .run_logger import RunLogger, ConsoleLogger
    from .models import (
        ActionID,
        AgentState,
        DiffResult,
        ActionAnalysisResult,
        NextActionSuggestion,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Testing RunLogger...")
        logger = RunLogger(tmpdir)

        # Create test data
        state = AgentState(run_id="test_run")
        diff = DiffResult(has_changes=True, changed_pixels=[], change_summary="test")
        analysis = ActionAnalysisResult(
            interpretation="Test",
            action_definition="Test def",
            is_consistent_with_previous=None,
            objects_involved=[],
            context_description="Test context",
            had_effect=True,
            no_effect_reason=None,
            environment_updates=[],
            confidence=0.9,
        )

        # Create dummy frame files
        frame_path = Path(tmpdir) / "test_frame.png"
        frame_path.write_bytes(b"fake image data")

        # Test action analysis logging
        print("  Testing action analysis logging...")
        call_dir = logger.log_action_analysis(
            action_id=ActionID.ACTION1,
            before_frame_path=str(frame_path),
            after_frame_path=str(frame_path),
            diff=diff,
            prompt="Test prompt",
            result=analysis,
            state_before=state,
            state_after=state,
            duration_ms=100.0,
        )
        assert Path(call_dir).exists()
        assert (Path(call_dir) / "prompt.txt").exists()
        assert (Path(call_dir) / "response.json").exists()
        print(f"    Logged to: {call_dir}")
        print("    ✓ Action analysis logging works")

        # Test next action suggestion logging
        print("  Testing suggestion logging...")
        suggestion = NextActionSuggestion(
            target_action=ActionID.ACTION2,
            setup_sequence=[],
            reasoning="Test",
            expected_information_gain="Test",
            current_board_assessment="Test",
        )
        call_dir2 = logger.log_next_action_suggestion(
            current_frame_path=str(frame_path),
            prompt="Test prompt",
            result=suggestion,
            state=state,
            duration_ms=50.0,
        )
        assert Path(call_dir2).exists()
        print("    ✓ Suggestion logging works")

        # Test final report
        print("  Testing final report generation...")
        report_path = logger.generate_final_report(state)
        assert Path(report_path).exists()
        print(f"    Report: {report_path}")
        print("    ✓ Report generation works")

    # Test ConsoleLogger
    print("  Testing ConsoleLogger...")
    console = ConsoleLogger(verbose=False)
    console.info("Test message")  # Should be silent
    console = ConsoleLogger(verbose=True)
    console.info("Test visible message")
    print("    ✓ Console logger works")

    print("\n  ✅ All logger tests passed!")
    return True


def run_all_tests():
    """Run all pipeline tests."""
    print("\n" + "=" * 60)
    print("LEARNING AGENT PIPELINE TESTS")
    print("=" * 60)

    tests = [
        ("Models", test_models),
        ("Diff", test_diff),
        ("Vision", test_vision),
        ("Knowledge", test_knowledge),
        ("LLM Agent Creation", test_llm_agent_creation),
        ("Run Logger", test_run_logger),
        ("LLM API Call (Live)", test_llm_api_call),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  {passed}/{total} test groups passed\n")

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    print()
    return all(p for _, p in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
