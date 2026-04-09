"""
Unit tests for the synthesis modules of all three agents.

Tests:
- Transition dataclass
- VerificationResult properties (accuracy, is_perfect)
- verify_domain / verify_model against replay buffers
- execute_synthesized_code (sandbox execution)
- build_synthesis_prompt (prompt construction)
- diagnose_persistent_errors
- Code stripping from markdown fences (_call_llm logic)
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# OOP agent synthesis
# ---------------------------------------------------------------------------
from src.oop_agent.synthesis import (
    Transition as OOPTransition,
    VerificationResult as OOPVerificationResult,
    verify_domain as oop_verify_domain,
    execute_synthesized_code as oop_execute_code,
    build_synthesis_prompt as oop_build_prompt,
    diagnose_persistent_errors as oop_diagnose,
)
from src.oop_agent.world_model import (
    GameObject as OOPGameObject,
    Action as OOPAction,
    World as OOPWorld,
    Domain as OOPDomain,
)

# ---------------------------------------------------------------------------
# Monolithic agent synthesis
# ---------------------------------------------------------------------------
from src.monolithic_agent.synthesis import (
    Transition as MonoTransition,
    VerificationResult as MonoVerificationResult,
    verify_model as mono_verify_model,
    execute_synthesized_code as mono_execute_code,
    build_synthesis_prompt as mono_build_prompt,
    diagnose_persistent_errors as mono_diagnose,
)
from src.monolithic_agent.world_model import (
    WorldModel,
    IdentityModel,
)

# ---------------------------------------------------------------------------
# Object-centric agent synthesis
# ---------------------------------------------------------------------------
from src.object_centric_agent.synthesis import (
    Transition as OCTransition,
    VerificationResult as OCVerificationResult,
    verify_domain as oc_verify_domain,
    execute_synthesized_code as oc_execute_code,
    build_synthesis_prompt as oc_build_prompt,
    diagnose_persistent_errors as oc_diagnose,
)
from src.object_centric_agent.world_model import (
    GameObject as OCGameObject,
    World as OCWorld,
    Domain as OCDomain,
)


# =============================================================================
# Helpers
# =============================================================================


def make_transition(before, action_id, after, timestep=0):
    """Create a Transition using the OOP agent's class (all are identical)."""
    return OOPTransition(
        before_frame=np.array(before, dtype=np.uint8),
        action_id=action_id,
        after_frame=np.array(after, dtype=np.uint8),
        timestep=timestep,
    )


def make_mono_transition(before, action_id, after, timestep=0):
    return MonoTransition(
        before_frame=np.array(before, dtype=np.uint8),
        action_id=action_id,
        after_frame=np.array(after, dtype=np.uint8),
        timestep=timestep,
    )


def make_oc_transition(before, action_id, after, timestep=0):
    return OCTransition(
        before_frame=np.array(before, dtype=np.uint8),
        action_id=action_id,
        after_frame=np.array(after, dtype=np.uint8),
        timestep=timestep,
    )


def identity_frame():
    """4x4 black frame."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


def shifted_frame():
    """4x4 frame with red pixel at (0,1)."""
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    f[0, 1] = [255, 0, 0]
    return f


# =============================================================================
# Test: Transition dataclass
# =============================================================================


@pytest.mark.unit
class TestTransition:
    def test_oop_transition(self):
        t = make_transition(identity_frame(), 1, shifted_frame(), timestep=5)
        assert t.action_id == 1
        assert t.timestep == 5
        assert t.before_frame.shape == (4, 4, 3)
        assert t.after_frame.shape == (4, 4, 3)

    def test_mono_transition(self):
        t = make_mono_transition(identity_frame(), 3, shifted_frame(), timestep=2)
        assert t.action_id == 3
        assert t.timestep == 2

    def test_oc_transition(self):
        t = make_oc_transition(identity_frame(), 5, shifted_frame(), timestep=0)
        assert t.action_id == 5


# =============================================================================
# Test: VerificationResult
# =============================================================================


@pytest.mark.unit
class TestVerificationResult:
    @pytest.mark.parametrize("VR", [OOPVerificationResult, MonoVerificationResult, OCVerificationResult])
    def test_empty_result(self, VR):
        r = VR()
        assert r.correct == 0
        assert r.incorrect == 0
        assert r.accuracy == 0.0
        assert not r.is_perfect

    @pytest.mark.parametrize("VR", [OOPVerificationResult, MonoVerificationResult, OCVerificationResult])
    def test_perfect_result(self, VR):
        r = VR(correct=5, incorrect=0)
        assert r.accuracy == 1.0
        assert r.is_perfect

    @pytest.mark.parametrize("VR", [OOPVerificationResult, MonoVerificationResult, OCVerificationResult])
    def test_partial_result(self, VR):
        r = VR(correct=3, incorrect=2)
        assert r.accuracy == pytest.approx(0.6)
        assert not r.is_perfect

    @pytest.mark.parametrize("VR", [OOPVerificationResult, MonoVerificationResult, OCVerificationResult])
    def test_all_wrong(self, VR):
        r = VR(correct=0, incorrect=5, errors=["error1"])
        assert r.accuracy == 0.0
        assert not r.is_perfect
        assert len(r.errors) == 1

    @pytest.mark.parametrize("VR", [OOPVerificationResult, MonoVerificationResult, OCVerificationResult])
    def test_mismatches_stored(self, VR):
        r = VR(correct=1, incorrect=1, mismatches=[{"timestep": 0, "action_id": 1}])
        assert len(r.mismatches) == 1


# =============================================================================
# Test: verify_domain / verify_model
# =============================================================================


@pytest.mark.unit
class TestVerifyDomain:
    def test_oop_perfect_domain(self):
        """Identity domain should perfectly predict no-change transitions."""
        class IdentityDomain(OOPDomain):
            def perceive(self, frame):
                return OOPWorld(frame, [])

            def get_action(self, action_id):
                class NoOp(OOPAction):
                    def apply(self, world):
                        pass
                return NoOp(action_id)

        domain = IdentityDomain()
        f = identity_frame()
        buffer = [make_transition(f, 1, f, 0), make_transition(f, 2, f, 1)]
        result = oop_verify_domain(domain, buffer)
        assert result.is_perfect
        assert result.correct == 2

    def test_oop_imperfect_domain(self):
        """Domain that always returns zeros will fail on non-zero transitions."""
        class BadDomain(OOPDomain):
            def perceive(self, frame):
                return OOPWorld(np.zeros_like(frame), [])

            def get_action(self, action_id):
                class NoOp(OOPAction):
                    def apply(self, world):
                        pass
                return NoOp(action_id)

        domain = BadDomain()
        f = identity_frame()
        sf = shifted_frame()
        buffer = [make_transition(f, 1, sf, 0)]
        result = oop_verify_domain(domain, buffer)
        assert result.incorrect == 1
        assert not result.is_perfect
        assert len(result.mismatches) == 1
        assert result.mismatches[0]["diff_pixel_count"] == 1

    def test_oop_domain_exception_counted_as_incorrect(self):
        """Exceptions during transition are counted as incorrect."""
        class BrokenDomain(OOPDomain):
            def perceive(self, frame):
                raise ValueError("broken")

            def get_action(self, action_id):
                raise ValueError("broken")

        domain = BrokenDomain()
        f = identity_frame()
        buffer = [make_transition(f, 1, f, 0)]
        result = oop_verify_domain(domain, buffer)
        assert result.incorrect == 1
        assert len(result.errors) == 1
        assert "ValueError" in result.errors[0]


@pytest.mark.unit
class TestVerifyModel:
    def test_mono_perfect_identity(self):
        model = IdentityModel()
        f = identity_frame()
        buffer = [make_mono_transition(f, 1, f, 0)]
        result = mono_verify_model(model, buffer)
        assert result.is_perfect

    def test_mono_imperfect(self):
        model = IdentityModel()
        f = identity_frame()
        sf = shifted_frame()
        buffer = [make_mono_transition(f, 1, sf, 0)]
        result = mono_verify_model(model, buffer)
        assert result.incorrect == 1
        assert not result.is_perfect


@pytest.mark.unit
class TestVerifyOCDomain:
    def test_oc_perfect_domain(self):
        class IdentityOCDomain(OCDomain):
            def perceive(self, frame):
                return OCWorld(frame, [])

        domain = IdentityOCDomain()
        f = identity_frame()
        buffer = [make_oc_transition(f, 1, f, 0)]
        result = oc_verify_domain(domain, buffer)
        assert result.is_perfect


# =============================================================================
# Test: execute_synthesized_code
# =============================================================================


@pytest.mark.unit
class TestExecuteSynthesizedCode:
    def test_oop_valid_code(self):
        code = """
class SimpleObj(GameObject):
    def respond_to_action(self, action, world):
        pass
    def render(self, frame):
        pass

class SimpleAct(Action):
    def apply(self, world):
        pass

class SynthesizedDomain(Domain):
    def perceive(self, frame):
        return World(frame, [])
    def get_action(self, action_id):
        return SimpleAct(action_id)
"""
        domain = oop_execute_code(code)
        assert domain is not None
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        result = domain.transition(f, 1)
        assert result.shape == f.shape

    def test_oop_missing_class(self):
        code = "x = 42"
        domain = oop_execute_code(code)
        assert domain is None

    def test_oop_syntax_error(self):
        code = "def broken(:\n  pass"
        domain = oop_execute_code(code)
        assert domain is None

    def test_oop_runtime_error(self):
        code = "raise RuntimeError('boom')"
        domain = oop_execute_code(code)
        assert domain is None

    def test_mono_valid_code(self):
        code = """
class SynthesizedModel(WorldModel):
    def predict(self, frame, action_id):
        return frame.copy()
"""
        model = mono_execute_code(code)
        assert model is not None
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        result = model.predict(f, 1)
        assert np.array_equal(result, f)

    def test_mono_missing_class(self):
        assert mono_execute_code("y = 99") is None

    def test_oc_valid_code(self):
        code = """
class SynthesizedDomain(Domain):
    def perceive(self, frame):
        return World(frame, [])
"""
        domain = oc_execute_code(code)
        assert domain is not None
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        result = domain.transition(f, 1)
        assert result.shape == f.shape

    def test_oc_missing_class(self):
        assert oc_execute_code("z = 0") is None

    def test_oop_code_with_numpy(self):
        """Synthesized code should have access to numpy."""
        code = """
class SynthesizedDomain(Domain):
    def perceive(self, frame):
        return World(frame, [])
    def get_action(self, action_id):
        class A(Action):
            def apply(self, world):
                world.frame[:] = np.zeros_like(world.frame)
        return A(action_id)
"""
        domain = oop_execute_code(code)
        assert domain is not None

    def test_oop_code_with_copy(self):
        """Synthesized code should have access to copy module."""
        code = """
import copy as _  # verify it's importable from namespace

class SynthesizedDomain(Domain):
    def perceive(self, frame):
        return World(copy.deepcopy(frame), [])
    def get_action(self, action_id):
        class A(Action):
            def apply(self, world):
                pass
        return A(action_id)
"""
        domain = oop_execute_code(code)
        assert domain is not None


# =============================================================================
# Test: build_synthesis_prompt
# =============================================================================


@pytest.mark.unit
class TestBuildSynthesisPrompt:
    def _make_simple_buffer(self, TransClass):
        f1 = np.zeros((4, 4, 3), dtype=np.uint8)
        f2 = f1.copy()
        f2[0, 0] = [255, 0, 0]
        return [TransClass(
            before_frame=f1, action_id=1, after_frame=f2, timestep=0,
        )]

    def test_oop_first_synthesis(self):
        buffer = self._make_simple_buffer(OOPTransition)
        prompt = oop_build_prompt(
            replay_buffer=buffer,
            frame_analysis="Frame shape: (4, 4, 3)",
            action_observations={1: ["ACTION1: 1 pixel changed"]},
            frame_shape=(4, 4, 3),
        )
        assert "FRAME ANALYSIS" in prompt
        assert "ACTION OBSERVATIONS" in prompt
        assert "SAMPLE TRANSITIONS" in prompt
        assert "SynthesizedDomain" in prompt
        assert "Frame shape: (4, 4, 3)" in prompt

    def test_oop_refinement_prompt(self):
        buffer = self._make_simple_buffer(OOPTransition)
        errors = OOPVerificationResult(correct=0, incorrect=1, errors=["test error"])
        prompt = oop_build_prompt(
            replay_buffer=buffer,
            frame_analysis="test",
            action_observations={1: []},
            previous_code="class Broken: pass",
            verification_errors=errors,
        )
        assert "PREVIOUS CODE" in prompt
        assert "VERIFICATION ERRORS" in prompt
        assert "Fix the errors" in prompt
        assert "class Broken: pass" in prompt

    def test_mono_first_synthesis(self):
        buffer = self._make_simple_buffer(MonoTransition)
        prompt = mono_build_prompt(
            replay_buffer=buffer,
            frame_analysis="test",
            action_observations={1: ["obs"]},
        )
        assert "SynthesizedModel" in prompt

    def test_oc_first_synthesis(self):
        buffer = self._make_simple_buffer(OCTransition)
        prompt = oc_build_prompt(
            replay_buffer=buffer,
            frame_analysis="test",
            action_observations={1: ["obs"]},
        )
        assert "SynthesizedDomain" in prompt

    def test_prompt_includes_observations(self):
        buffer = self._make_simple_buffer(OOPTransition)
        prompt = oop_build_prompt(
            replay_buffer=buffer,
            frame_analysis="test",
            action_observations={
                1: ["ACTION1: 5 pixels changed"],
                2: ["ACTION2: no change"],
            },
        )
        assert "ACTION1" in prompt
        assert "ACTION2" in prompt
        assert "5 pixels changed" in prompt

    def test_prompt_limits_observations_to_last_5(self):
        buffer = self._make_simple_buffer(OOPTransition)
        observations = {1: [f"obs_{i}" for i in range(10)]}
        prompt = oop_build_prompt(
            replay_buffer=buffer,
            frame_analysis="test",
            action_observations=observations,
        )
        # Should only include last 5
        assert "obs_5" in prompt
        assert "obs_9" in prompt
        assert "obs_0" not in prompt

    def test_prompt_limits_transitions_to_8(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        buffer = [OOPTransition(before_frame=f, action_id=1, after_frame=f, timestep=i)
                  for i in range(20)]
        prompt = oop_build_prompt(
            replay_buffer=buffer,
            frame_analysis="test",
            action_observations={1: []},
        )
        assert "Timestep 7" in prompt
        assert "Timestep 8" not in prompt


# =============================================================================
# Test: diagnose_persistent_errors
# =============================================================================


@pytest.mark.unit
class TestDiagnosePersistentErrors:
    def test_oop_not_stuck(self):
        class PerfectDomain(OOPDomain):
            def perceive(self, frame):
                return OOPWorld(frame, [])

            def get_action(self, action_id):
                class N(OOPAction):
                    def apply(self, world):
                        pass
                return N(action_id)

        domain = PerfectDomain()
        f = identity_frame()
        buffer = [make_transition(f, 1, f, 0)]
        diag = oop_diagnose(buffer, domain, synthesis_attempts=1)
        assert not diag.is_stuck
        assert len(diag.failed_transitions) == 0
        assert "Improving" in diag.description

    def test_oop_stuck(self):
        class BadDomain(OOPDomain):
            def perceive(self, frame):
                return OOPWorld(np.zeros_like(frame), [])

            def get_action(self, action_id):
                class N(OOPAction):
                    def apply(self, world):
                        pass
                return N(action_id)

        domain = BadDomain()
        f = identity_frame()
        sf = shifted_frame()
        buffer = [make_transition(f, 1, sf, 0)]
        diag = oop_diagnose(buffer, domain, synthesis_attempts=5)
        assert diag.is_stuck
        assert len(diag.failed_transitions) == 1
        assert "Stuck" in diag.description

    def test_mono_not_stuck(self):
        model = IdentityModel()
        f = identity_frame()
        buffer = [make_mono_transition(f, 1, f, 0)]
        diag = mono_diagnose(buffer, model, synthesis_attempts=1)
        assert not diag.is_stuck

    def test_oc_stuck(self):
        class BadOCDomain(OCDomain):
            def perceive(self, frame):
                return OCWorld(np.zeros_like(frame), [])

        domain = BadOCDomain()
        f = identity_frame()
        sf = shifted_frame()
        buffer = [make_oc_transition(f, 1, sf, 0)]
        diag = oc_diagnose(buffer, domain, synthesis_attempts=5)
        assert diag.is_stuck


# =============================================================================
# Test: Code stripping from markdown fences (testing _call_llm parse logic)
# =============================================================================


@pytest.mark.unit
class TestCodeStripping:
    """Test the code extraction logic used in _call_llm."""

    def _strip_code(self, content):
        """Replicate the stripping logic from _call_llm."""
        code = content.strip()
        if "```python" in code:
            code = code.split("```python", 1)[1]
            code = code.split("```", 1)[0]
        elif "```" in code:
            code = code.split("```", 1)[1]
            code = code.split("```", 1)[0]
        return code.strip()

    def test_plain_code(self):
        code = "class Foo:\n    pass"
        assert self._strip_code(code) == code

    def test_python_fence(self):
        content = "Here's the code:\n```python\nclass Foo:\n    pass\n```\nDone."
        assert self._strip_code(content) == "class Foo:\n    pass"

    def test_generic_fence(self):
        content = "```\nclass Foo:\n    pass\n```"
        assert self._strip_code(content) == "class Foo:\n    pass"

    def test_nested_fences_takes_first(self):
        content = "```python\nfirst\n```\n```python\nsecond\n```"
        assert self._strip_code(content) == "first"

    def test_whitespace_stripped(self):
        content = "\n\n  class Foo:\n    pass  \n\n"
        assert self._strip_code(content) == "class Foo:\n    pass"
