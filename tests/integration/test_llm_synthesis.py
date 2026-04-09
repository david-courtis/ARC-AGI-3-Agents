"""
Integration tests for LLM-driven synthesis using a local OpenAI-compatible endpoint.

These tests exercise the full CEGIS pipeline: build prompt -> call LLM ->
execute synthesized code -> verify against replay buffer.

Endpoint: http://127.0.0.1:8484/code/v1
Model: claude-opus-4-6-thinking
API Key: LkY56yDjUVYkfL9BPm7VLTpMs7kM6gbXPMp6PV2QysqcAvpr8PAdyjPYYbbgTwgH

These tests are slow (LLM calls) and require the local server to be running.
Mark them with @pytest.mark.integration so they can be skipped in CI.
"""

import numpy as np
import pytest
from openai import OpenAI

# ---------------------------------------------------------------------------
# OOP agent
# ---------------------------------------------------------------------------
from src.oop_agent.synthesis import (
    DomainSynthesizer as OOPSynthesizer,
    Transition as OOPTransition,
    VerificationResult as OOPVerificationResult,
    verify_domain as oop_verify_domain,
    execute_synthesized_code as oop_execute_code,
    build_synthesis_prompt as oop_build_prompt,
    SYSTEM_PROMPT as OOP_SYSTEM_PROMPT,
)
from src.oop_agent.agent import analyze_frame as oop_analyze

# ---------------------------------------------------------------------------
# Monolithic agent
# ---------------------------------------------------------------------------
from src.monolithic_agent.synthesis import (
    ModelSynthesizer as MonoSynthesizer,
    Transition as MonoTransition,
    VerificationResult as MonoVerificationResult,
    verify_model as mono_verify_model,
    execute_synthesized_code as mono_execute_code,
    build_synthesis_prompt as mono_build_prompt,
    SYSTEM_PROMPT as MONO_SYSTEM_PROMPT,
)
from src.monolithic_agent.agent import analyze_frame as mono_analyze

# ---------------------------------------------------------------------------
# Object-centric agent
# ---------------------------------------------------------------------------
from src.object_centric_agent.synthesis import (
    DomainSynthesizer as OCSynthesizer,
    Transition as OCTransition,
    VerificationResult as OCVerificationResult,
    verify_domain as oc_verify_domain,
    execute_synthesized_code as oc_execute_code,
    build_synthesis_prompt as oc_build_prompt,
    SYSTEM_PROMPT as OC_SYSTEM_PROMPT,
)
from src.object_centric_agent.agent import analyze_frame as oc_analyze


# =============================================================================
# Config
# =============================================================================

LOCAL_BASE_URL = "http://127.0.0.1:8484/code/v1"
LOCAL_API_KEY = "LkY56yDjUVYkfL9BPm7VLTpMs7kM6gbXPMp6PV2QysqcAvpr8PAdyjPYYbbgTwgH"
LOCAL_MODEL = "claude-opus-4-6-thinking"


def local_llm_available() -> bool:
    """Check if the local LLM server is reachable."""
    try:
        client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        client.models.list()
        return True
    except Exception:
        return False


requires_llm = pytest.mark.skipif(
    not local_llm_available(),
    reason="Local LLM server not available at http://127.0.0.1:8484/code/v1",
)


# =============================================================================
# Test game: Simple pixel-move game
#
# Rules:
# - 8x8 RGB frame, black background
# - A single red pixel at some position
# - ACTION1: move red pixel right by 1
# - ACTION2: move red pixel down by 1
# - ACTION3: move red pixel left by 1
# - ACTION4: move red pixel up by 1
# - ACTION5: no effect
# =============================================================================


def make_pixel_game_frame(row, col, size=8):
    """Create a frame with a single red pixel."""
    f = np.zeros((size, size, 3), dtype=np.uint8)
    f[row, col] = [255, 0, 0]
    return f


def pixel_game_transition(before_row, before_col, action_id, size=8):
    """Generate a transition for the pixel-move game."""
    after_row, after_col = before_row, before_col
    if action_id == 1:  # right
        after_col = min(before_col + 1, size - 1)
    elif action_id == 2:  # down
        after_row = min(before_row + 1, size - 1)
    elif action_id == 3:  # left
        after_col = max(before_col - 1, 0)
    elif action_id == 4:  # up
        after_row = max(before_row - 1, 0)
    # ACTION5: no change
    return (
        make_pixel_game_frame(before_row, before_col, size),
        action_id,
        make_pixel_game_frame(after_row, after_col, size),
    )


def make_pixel_game_buffer(TransClass, size=8):
    """Generate a replay buffer for the pixel game."""
    transitions = []
    positions = [(3, 3), (3, 4), (4, 4), (4, 3), (3, 3), (2, 3), (3, 3), (3, 2),
                 (3, 3), (3, 3), (0, 0), (0, 0), (size-1, size-1), (size-1, size-1)]
    actions =   [1,      2,      3,      4,      1,      2,      4,      1,
                 5,      3,      4,      3,      2,            1]

    for i, (pos, act) in enumerate(zip(positions, actions)):
        before, aid, after = pixel_game_transition(pos[0], pos[1], act, size)
        transitions.append(TransClass(
            before_frame=before, action_id=aid, after_frame=after, timestep=i,
        ))
    return transitions


def make_observations(buffer):
    """Build action_observations from a replay buffer."""
    from src.oop_agent.world_model import compute_diff
    obs = {1: [], 2: [], 3: [], 4: [], 5: []}
    for trans in buffer:
        diff = compute_diff(trans.before_frame, trans.after_frame)
        if diff.count == 0:
            obs[trans.action_id].append(f"ACTION{trans.action_id}: no visible change")
        else:
            obs[trans.action_id].append(f"ACTION{trans.action_id}: {diff.count} pixels changed")
    return obs


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def llm_client():
    return OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)


# =============================================================================
# Test: Raw LLM call
# =============================================================================


@requires_llm
@pytest.mark.integration
class TestLLMConnection:
    def test_basic_completion(self, llm_client):
        """Verify the local LLM endpoint responds."""
        response = llm_client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=100,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0


# =============================================================================
# Test: Monolithic agent — full synthesis
# =============================================================================


@requires_llm
@pytest.mark.integration
class TestMonolithicLLMSynthesis:
    def test_synthesize_pixel_game(self):
        """LLM should be able to synthesize a predict() for the pixel-move game."""
        buffer = make_pixel_game_buffer(MonoTransition, size=8)
        observations = make_observations(buffer)
        frame_analysis = mono_analyze(buffer[0].before_frame)

        prompt = mono_build_prompt(
            replay_buffer=buffer,
            frame_analysis=frame_analysis,
            action_observations=observations,
            frame_shape=(8, 8, 3),
        )

        client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        response = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[
                {"role": "system", "content": MONO_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=32000,
        )

        content = response.choices[0].message.content
        assert content is not None

        # Strip markdown fences
        code = content.strip()
        if "```python" in code:
            code = code.split("```python", 1)[1]
            code = code.split("```", 1)[0]
        elif "```" in code:
            code = code.split("```", 1)[1]
            code = code.split("```", 1)[0]
        code = code.strip()

        model = mono_execute_code(code)
        assert model is not None, f"Failed to execute synthesized code:\n{code[:500]}"

        result = mono_verify_model(model, buffer)
        print(f"\n[Monolithic] Accuracy: {result.accuracy:.1%} "
              f"({result.correct}/{result.correct + result.incorrect})")
        if result.errors:
            for e in result.errors[:3]:
                print(f"  ERROR: {e}")
        if result.mismatches:
            for mm in result.mismatches[:3]:
                print(f"  MISMATCH t={mm['timestep']} a={mm['action_id']}: "
                      f"{mm['diff_pixel_count']} px differ")

        # We expect at least partial accuracy
        assert result.accuracy >= 0.3, (
            f"Accuracy too low: {result.accuracy:.1%}. "
            f"Errors: {result.errors[:3]}"
        )

    def test_full_cegis_loop(self):
        """Test ModelSynthesizer with CEGIS refinement."""
        buffer = make_pixel_game_buffer(MonoTransition, size=8)
        observations = make_observations(buffer)
        frame_analysis = mono_analyze(buffer[0].before_frame)

        synthesizer = MonoSynthesizer.__new__(MonoSynthesizer)
        synthesizer.client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        synthesizer.model = LOCAL_MODEL
        synthesizer.max_refinements = 2
        synthesizer.last_code = None
        synthesizer.synthesis_count = 0

        model, code, result = synthesizer.synthesize(
            replay_buffer=buffer,
            frame_analysis=frame_analysis,
            action_observations=observations,
            frame_shape=(8, 8, 3),
        )

        print(f"\n[Monolithic CEGIS] Attempts: {synthesizer.synthesis_count}")
        if result:
            print(f"  Final accuracy: {result.accuracy:.1%}")
        else:
            print("  Synthesis returned None")

        # Should produce something
        assert model is not None or synthesizer.synthesis_count > 0


# =============================================================================
# Test: OOP agent — full synthesis
# =============================================================================


@requires_llm
@pytest.mark.integration
class TestOOPLLMSynthesis:
    def test_synthesize_pixel_game(self):
        """LLM should synthesize OOP classes for the pixel-move game."""
        buffer = make_pixel_game_buffer(OOPTransition, size=8)
        observations = make_observations(buffer)
        frame_analysis = oop_analyze(buffer[0].before_frame)

        prompt = oop_build_prompt(
            replay_buffer=buffer,
            frame_analysis=frame_analysis,
            action_observations=observations,
            frame_shape=(8, 8, 3),
        )

        client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        response = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[
                {"role": "system", "content": OOP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=32000,
        )

        content = response.choices[0].message.content
        assert content is not None

        code = content.strip()
        if "```python" in code:
            code = code.split("```python", 1)[1]
            code = code.split("```", 1)[0]
        elif "```" in code:
            code = code.split("```", 1)[1]
            code = code.split("```", 1)[0]
        code = code.strip()

        domain = oop_execute_code(code)
        assert domain is not None, f"Failed to execute synthesized code:\n{code[:500]}"

        result = oop_verify_domain(domain, buffer)
        print(f"\n[OOP] Accuracy: {result.accuracy:.1%} "
              f"({result.correct}/{result.correct + result.incorrect})")
        if result.errors:
            for e in result.errors[:3]:
                print(f"  ERROR: {e}")

        assert result.accuracy >= 0.3, (
            f"Accuracy too low: {result.accuracy:.1%}"
        )

    def test_full_cegis_loop(self):
        """Test DomainSynthesizer with CEGIS refinement."""
        buffer = make_pixel_game_buffer(OOPTransition, size=8)
        observations = make_observations(buffer)
        frame_analysis = oop_analyze(buffer[0].before_frame)

        synthesizer = OOPSynthesizer.__new__(OOPSynthesizer)
        synthesizer.client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        synthesizer.model = LOCAL_MODEL
        synthesizer.max_refinements = 2
        synthesizer.last_code = None
        synthesizer.synthesis_count = 0

        domain, code, result = synthesizer.synthesize(
            replay_buffer=buffer,
            frame_analysis=frame_analysis,
            action_observations=observations,
            frame_shape=(8, 8, 3),
        )

        print(f"\n[OOP CEGIS] Attempts: {synthesizer.synthesis_count}")
        if result:
            print(f"  Final accuracy: {result.accuracy:.1%}")

        assert domain is not None or synthesizer.synthesis_count > 0


# =============================================================================
# Test: Object-centric agent — full synthesis
# =============================================================================


@requires_llm
@pytest.mark.integration
class TestObjectCentricLLMSynthesis:
    def test_synthesize_pixel_game(self):
        """LLM should synthesize object-centric classes for the pixel-move game."""
        buffer = make_pixel_game_buffer(OCTransition, size=8)
        observations = make_observations(buffer)
        frame_analysis = oc_analyze(buffer[0].before_frame)

        prompt = oc_build_prompt(
            replay_buffer=buffer,
            frame_analysis=frame_analysis,
            action_observations=observations,
            frame_shape=(8, 8, 3),
        )

        client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        response = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[
                {"role": "system", "content": OC_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=32000,
        )

        content = response.choices[0].message.content
        assert content is not None

        code = content.strip()
        if "```python" in code:
            code = code.split("```python", 1)[1]
            code = code.split("```", 1)[0]
        elif "```" in code:
            code = code.split("```", 1)[1]
            code = code.split("```", 1)[0]
        code = code.strip()

        domain = oc_execute_code(code)
        assert domain is not None, f"Failed to execute synthesized code:\n{code[:500]}"

        result = oc_verify_domain(domain, buffer)
        print(f"\n[Object-Centric] Accuracy: {result.accuracy:.1%} "
              f"({result.correct}/{result.correct + result.incorrect})")
        if result.errors:
            for e in result.errors[:3]:
                print(f"  ERROR: {e}")

        assert result.accuracy >= 0.3, (
            f"Accuracy too low: {result.accuracy:.1%}"
        )

    def test_full_cegis_loop(self):
        """Test OC DomainSynthesizer with CEGIS refinement."""
        buffer = make_pixel_game_buffer(OCTransition, size=8)
        observations = make_observations(buffer)
        frame_analysis = oc_analyze(buffer[0].before_frame)

        synthesizer = OCSynthesizer.__new__(OCSynthesizer)
        synthesizer.client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
        synthesizer.model = LOCAL_MODEL
        synthesizer.max_refinements = 2
        synthesizer.last_code = None
        synthesizer.synthesis_count = 0

        domain, code, result = synthesizer.synthesize(
            replay_buffer=buffer,
            frame_analysis=frame_analysis,
            action_observations=observations,
            frame_shape=(8, 8, 3),
        )

        print(f"\n[Object-Centric CEGIS] Attempts: {synthesizer.synthesis_count}")
        if result:
            print(f"  Final accuracy: {result.accuracy:.1%}")

        assert domain is not None or synthesizer.synthesis_count > 0


# =============================================================================
# Test: Comparative — all 3 agents on same game
# =============================================================================


@requires_llm
@pytest.mark.integration
class TestComparativeThreeAgents:
    def test_all_three_produce_output(self):
        """Smoke test: all 3 synthesizers produce something on the same game."""
        results = {}

        for name, SynthClass, TransClass, verify_fn, exec_fn, analyze_fn, sys_prompt, build_fn in [
            ("monolithic", MonoSynthesizer, MonoTransition, mono_verify_model,
             mono_execute_code, mono_analyze, MONO_SYSTEM_PROMPT, mono_build_prompt),
            ("oop", OOPSynthesizer, OOPTransition, oop_verify_domain,
             oop_execute_code, oop_analyze, OOP_SYSTEM_PROMPT, oop_build_prompt),
            ("object_centric", OCSynthesizer, OCTransition, oc_verify_domain,
             oc_execute_code, oc_analyze, OC_SYSTEM_PROMPT, oc_build_prompt),
        ]:
            buffer = make_pixel_game_buffer(TransClass, size=8)
            observations = make_observations(buffer)
            frame_analysis = analyze_fn(buffer[0].before_frame)

            # Single LLM call (no CEGIS refinement)
            prompt = build_fn(
                replay_buffer=buffer,
                frame_analysis=frame_analysis,
                action_observations=observations,
                frame_shape=(8, 8, 3),
            )

            client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
            response = client.chat.completions.create(
                model=LOCAL_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=32000,
            )

            content = response.choices[0].message.content
            assert content is not None, f"{name}: LLM returned empty response"

            code = content.strip()
            if "```python" in code:
                code = code.split("```python", 1)[1]
                code = code.split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1]
                code = code.split("```", 1)[0]
            code = code.strip()

            model_or_domain = exec_fn(code)
            if model_or_domain is not None:
                result = verify_fn(model_or_domain, buffer)
                results[name] = result.accuracy
                print(f"  {name}: {result.accuracy:.1%} accuracy")
            else:
                results[name] = 0.0
                print(f"  {name}: exec failed")

        # At least one agent should produce a working model
        assert max(results.values()) > 0.0, (
            f"All agents failed: {results}"
        )

        print(f"\n[Comparative] Results: {results}")
