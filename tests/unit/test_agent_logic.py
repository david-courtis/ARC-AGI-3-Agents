"""
Unit tests for agent logic: ExplorationState, phase management, action
selection, transition summarization, and frame analysis.

Tests all three agents. Since the exploration logic is structurally identical
across agents (only the synthesized model type differs), we test the OOP agent
thoroughly and verify the others match behavior.
"""

import numpy as np
import pytest

from arcengine import FrameDataRaw, GameAction, GameState

# ---------------------------------------------------------------------------
# OOP agent
# ---------------------------------------------------------------------------
from src.oop_agent.agent import (
    ExplorationState as OOPExplorationState,
    choose_exploration_action as oop_choose_action,
    summarize_transition as oop_summarize,
    analyze_frame as oop_analyze,
    OOPAgent,
)
from src.oop_agent.synthesis import Transition as OOPTransition

# ---------------------------------------------------------------------------
# Monolithic agent
# ---------------------------------------------------------------------------
from src.monolithic_agent.agent import (
    ExplorationState as MonoExplorationState,
    choose_exploration_action as mono_choose_action,
    summarize_transition as mono_summarize,
    analyze_frame as mono_analyze,
    MonolithicAgent,
)
from src.monolithic_agent.synthesis import Transition as MonoTransition

# ---------------------------------------------------------------------------
# Object-centric agent
# ---------------------------------------------------------------------------
from src.object_centric_agent.agent import (
    ExplorationState as OCExplorationState,
    choose_exploration_action as oc_choose_action,
    summarize_transition as oc_summarize,
    analyze_frame as oc_analyze,
    ObjectCentricAgent,
)
from src.object_centric_agent.synthesis import Transition as OCTransition


def _make_obs(state=GameState.NOT_FINISHED, levels_completed=0, frame_data=None):
    """Create a FrameDataRaw for testing."""
    obs = FrameDataRaw(
        game_id="test-game",
        state=state,
        levels_completed=levels_completed,
        available_actions=[1, 2, 3, 4, 5],
    )
    if frame_data is not None:
        obs.frame = frame_data
    else:
        # Default: single 4x4 palette-indexed grid
        grid = np.zeros((4, 4), dtype=np.uint8)
        obs.frame = [grid]
    return obs


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fresh_exploration():
    """Factory for fresh ExplorationState for each agent type."""
    return {
        "oop": OOPExplorationState,
        "mono": MonoExplorationState,
        "oc": OCExplorationState,
    }


@pytest.fixture
def black_frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def game_frame_64():
    """64x64 frame with uniform borders and a few colored regions."""
    f = np.zeros((64, 64, 3), dtype=np.uint8)
    f[:] = [32, 32, 32]
    # Uniform top border
    f[0, :] = [255, 255, 255]
    # Uniform bottom border
    f[-1, :] = [255, 255, 255]
    # Left border
    f[:, 0] = [255, 255, 255]
    # Right border
    f[:, -1] = [255, 255, 255]
    # Red block
    f[10:14, 10:14] = [255, 0, 0]
    return f


# =============================================================================
# Test: ExplorationState
# =============================================================================


EXPLORATION_CLASSES = [
    pytest.param(OOPExplorationState, id="oop"),
    pytest.param(MonoExplorationState, id="mono"),
    pytest.param(OCExplorationState, id="oc"),
]


@pytest.mark.unit
class TestExplorationState:
    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_initial_state(self, ES):
        state = ES()
        assert state.total_observations == 0
        assert state.synthesis_attempts == 0
        assert state.best_accuracy == 0.0
        assert not state.model_is_perfect
        assert state.available_action_ids == {1, 2, 3, 4, 5}
        assert not state.has_minimum_coverage

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_total_observations(self, ES):
        state = ES()
        state.action_counts = {1: 3, 2: 5, 3: 0, 4: 1, 5: 2}
        assert state.total_observations == 11

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_least_observed_action(self, ES):
        state = ES()
        state.action_counts = {1: 3, 2: 5, 3: 0, 4: 1, 5: 2}
        assert state.least_observed_action == 3

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_least_observed_respects_available(self, ES):
        state = ES()
        state.action_counts = {1: 3, 2: 5, 3: 0, 4: 1, 5: 2}
        state.available_action_ids = {1, 2}
        assert state.least_observed_action == 1

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_has_minimum_coverage(self, ES):
        state = ES()
        state.min_observations_per_action = 2
        state.action_counts = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
        assert state.has_minimum_coverage

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_has_minimum_coverage_false(self, ES):
        state = ES()
        state.min_observations_per_action = 3
        state.action_counts = {1: 3, 2: 3, 3: 2, 4: 3, 5: 3}
        assert not state.has_minimum_coverage

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_record_transition(self, ES):
        state = ES()
        before = np.zeros((4, 4, 3), dtype=np.uint8)
        after = before.copy()
        after[0, 0] = [255, 0, 0]
        state.record_transition(before, 1, after, "test obs")
        assert state.action_counts[1] == 1
        assert len(state.replay_buffer) == 1
        assert state.action_observations[1] == ["test obs"]

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_record_multiple_transitions(self, ES):
        state = ES()
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(5):
            state.record_transition(f, 1, f, f"obs_{i}")
        assert state.action_counts[1] == 5
        assert len(state.replay_buffer) == 5
        assert len(state.action_observations[1]) == 5

    @pytest.mark.parametrize("ES", EXPLORATION_CLASSES)
    def test_record_transition_new_action(self, ES):
        """Recording for an action not in default dict."""
        state = ES()
        state.available_action_ids = {1, 2, 3, 4, 5, 6}
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        state.record_transition(f, 6, f, "action 6 obs")
        assert state.action_counts[6] == 1
        assert state.action_observations[6] == ["action 6 obs"]


# =============================================================================
# Test: choose_exploration_action
# =============================================================================


CHOOSE_ACTION_FUNCS = [
    pytest.param((OOPExplorationState, oop_choose_action), id="oop"),
    pytest.param((MonoExplorationState, mono_choose_action), id="mono"),
    pytest.param((OCExplorationState, oc_choose_action), id="oc"),
]


@pytest.mark.unit
class TestChooseExplorationAction:
    @pytest.mark.parametrize("ES_fn", CHOOSE_ACTION_FUNCS)
    def test_untested_first(self, ES_fn):
        ES, choose = ES_fn
        state = ES()
        # All untested — should pick lowest untested (action 1)
        assert choose(state) == 1

    @pytest.mark.parametrize("ES_fn", CHOOSE_ACTION_FUNCS)
    def test_skips_tested_actions(self, ES_fn):
        ES, choose = ES_fn
        state = ES()
        state.action_counts = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0}
        assert choose(state) == 2

    @pytest.mark.parametrize("ES_fn", CHOOSE_ACTION_FUNCS)
    def test_under_observed(self, ES_fn):
        ES, choose = ES_fn
        state = ES()
        state.min_observations_per_action = 3
        state.action_counts = {1: 3, 2: 1, 3: 3, 4: 2, 5: 3}
        # Actions 2 and 4 are under-observed, 2 has fewer
        assert choose(state) == 2

    @pytest.mark.parametrize("ES_fn", CHOOSE_ACTION_FUNCS)
    def test_all_covered_picks_least(self, ES_fn):
        ES, choose = ES_fn
        state = ES()
        state.min_observations_per_action = 1
        state.action_counts = {1: 5, 2: 3, 3: 7, 4: 1, 5: 10}
        # All >= 1, so pick least observed
        assert choose(state) == 4

    @pytest.mark.parametrize("ES_fn", CHOOSE_ACTION_FUNCS)
    def test_respects_available_actions(self, ES_fn):
        ES, choose = ES_fn
        state = ES()
        state.available_action_ids = {3, 4, 5}
        state.action_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        # Only actions 3, 4, 5 available — should pick 3
        assert choose(state) == 3


# =============================================================================
# Test: summarize_transition
# =============================================================================


SUMMARIZE_FUNCS = [
    pytest.param(oop_summarize, id="oop"),
    pytest.param(mono_summarize, id="mono"),
    pytest.param(oc_summarize, id="oc"),
]


@pytest.mark.unit
class TestSummarizeTransition:
    @pytest.mark.parametrize("summarize", SUMMARIZE_FUNCS)
    def test_no_change(self, summarize):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        result = summarize(f, f.copy(), 1)
        assert "no visible change" in result
        assert "ACTION1" in result

    @pytest.mark.parametrize("summarize", SUMMARIZE_FUNCS)
    def test_pixel_change(self, summarize):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[1, 1] = [255, 0, 0]
        result = summarize(a, b, 2)
        assert "ACTION2" in result
        assert "1 pixels changed" in result

    @pytest.mark.parametrize("summarize", SUMMARIZE_FUNCS)
    def test_color_change_reported(self, summarize):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = [255, 0, 0]
        b[0, 1] = [255, 0, 0]
        result = summarize(a, b, 3)
        assert "(0, 0, 0) -> (255, 0, 0)" in result

    @pytest.mark.parametrize("summarize", SUMMARIZE_FUNCS)
    def test_region_reported(self, summarize):
        a = np.zeros((8, 8, 3), dtype=np.uint8)
        b = a.copy()
        b[2:4, 3:5] = [128, 128, 128]
        result = summarize(a, b, 1)
        assert "region" in result


# =============================================================================
# Test: analyze_frame
# =============================================================================


ANALYZE_FUNCS = [
    pytest.param(oop_analyze, id="oop"),
    pytest.param(mono_analyze, id="mono"),
    pytest.param(oc_analyze, id="oc"),
]


@pytest.mark.unit
class TestAnalyzeFrame:
    @pytest.mark.parametrize("analyze", ANALYZE_FUNCS)
    def test_reports_shape(self, analyze, game_frame_64):
        result = analyze(game_frame_64)
        assert "64" in result
        assert "Frame shape" in result

    @pytest.mark.parametrize("analyze", ANALYZE_FUNCS)
    def test_reports_colors(self, analyze):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[0, 0] = [255, 0, 0]
        result = analyze(f)
        assert "Unique colors: 2" in result

    @pytest.mark.parametrize("analyze", ANALYZE_FUNCS)
    def test_reports_background(self, analyze):
        f = np.full((4, 4, 3), 64, dtype=np.uint8)
        f[0, 0] = [255, 0, 0]
        result = analyze(f)
        assert "Most common color" in result
        assert "(64, 64, 64)" in result

    @pytest.mark.parametrize("analyze", ANALYZE_FUNCS)
    def test_reports_borders(self, analyze, game_frame_64):
        result = analyze(game_frame_64)
        # Left and right borders should be detected as uniform
        assert "Border structure" in result

    @pytest.mark.parametrize("analyze", ANALYZE_FUNCS)
    def test_many_colors_skips_listing(self, analyze):
        """If > 20 unique colors, individual listing is skipped."""
        f = np.zeros((8, 8, 3), dtype=np.uint8)
        for i in range(25):
            row, col = divmod(i, 8)
            f[row, col] = [i * 10 % 256, i * 7 % 256, i * 3 % 256]
        result = analyze(f)
        assert "Unique colors:" in result


# =============================================================================
# Test: Agent initialization and properties
# =============================================================================


@pytest.mark.unit
class TestAgentInit:
    def _make_agent(self, AgentClass):
        return AgentClass(verbose=False)

    def test_oop_agent_init(self):
        agent = self._make_agent(OOPAgent)
        assert agent.MAX_ACTIONS == 30
        assert agent._phase == "explore"
        assert agent._previous_frame is None
        assert agent._pending_action_id is None
        assert isinstance(agent.exploration, OOPExplorationState)

    def test_mono_agent_init(self):
        agent = self._make_agent(MonolithicAgent)
        assert agent.MAX_ACTIONS == 100
        assert agent._phase == "explore"
        assert isinstance(agent.exploration, MonoExplorationState)

    def test_oc_agent_init(self):
        agent = self._make_agent(ObjectCentricAgent)
        assert agent.MAX_ACTIONS == 100
        assert agent._phase == "explore"
        assert isinstance(agent.exploration, OCExplorationState)


# =============================================================================
# Test: is_done
# =============================================================================


@pytest.mark.unit
class TestIsDone:
    def _make_agent(self, AgentClass):
        return AgentClass(verbose=False)

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_done_on_win(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.WIN)
        assert agent.is_done(obs) is True

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_not_done_when_playing(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.NOT_FINISHED)
        assert agent.is_done(obs) is False

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_done_on_max_actions(self, AgentClass):
        agent = self._make_agent(AgentClass)
        agent.action_counter = agent.MAX_ACTIONS
        obs = _make_obs(state=GameState.NOT_FINISHED)
        assert agent.is_done(obs) is True


# =============================================================================
# Test: Phase transitions (_update_phase)
# =============================================================================


@pytest.mark.unit
class TestPhaseTransitions:
    def _make_agent(self, AgentClass):
        return AgentClass(verbose=False)

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_starts_in_explore(self, AgentClass):
        agent = self._make_agent(AgentClass)
        assert agent._phase == "explore"

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_stays_explore_without_coverage(self, AgentClass):
        agent = self._make_agent(AgentClass)
        agent._update_phase()
        assert agent._phase == "explore"

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_transitions_to_synthesize(self, AgentClass):
        agent = self._make_agent(AgentClass)
        exp = agent.exploration
        # Meet all conditions for synthesis
        exp.action_counts = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(15):
            exp.replay_buffer.append(
                OOPTransition(before_frame=f, action_id=1, after_frame=f, timestep=i)
                if AgentClass is OOPAgent else
                MonoTransition(before_frame=f, action_id=1, after_frame=f, timestep=i)
                if AgentClass is MonolithicAgent else
                OCTransition(before_frame=f, action_id=1, after_frame=f, timestep=i)
            )
        agent._update_phase()
        assert agent._phase == "synthesize"

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_perfect_model_stays_phase(self, AgentClass):
        """When model is perfect, _update_phase returns early (is_done handles exit)."""
        agent = self._make_agent(AgentClass)
        agent._phase = "synthesize"
        agent.exploration.model_is_perfect = True
        agent._update_phase()
        # Phase unchanged — is_done() handles termination
        assert agent._phase == "synthesize"

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_max_synthesis_rounds_goes_to_explore(self, AgentClass):
        agent = self._make_agent(AgentClass)
        agent.exploration.synthesis_attempts = agent._max_synthesis_rounds
        agent._update_phase()
        assert agent._phase == "explore"


# =============================================================================
# Test: choose_action for edge cases
# =============================================================================


@pytest.mark.unit
class TestChooseAction:
    def _make_agent(self, AgentClass):
        return AgentClass(verbose=False)

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_reset_on_not_played(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.NOT_PLAYED)
        action = agent.choose_action(obs)
        assert action == GameAction.RESET

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_reset_on_game_over(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.GAME_OVER)
        action = agent.choose_action(obs)
        assert action == GameAction.RESET

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_explore_returns_game_action(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.NOT_FINISHED)
        action = agent.choose_action(obs)
        assert isinstance(action, GameAction)
        assert action != GameAction.RESET

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_records_transition_on_second_call(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.NOT_FINISHED)
        # First call sets _pending_action_id
        agent.choose_action(obs)
        assert agent._pending_action_id is not None
        # Second call should record a transition
        agent.choose_action(obs)
        assert len(agent.exploration.replay_buffer) == 1

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_tracks_score(self, AgentClass):
        agent = self._make_agent(AgentClass)
        obs = _make_obs(state=GameState.NOT_FINISHED, levels_completed=3)
        agent.choose_action(obs)
        assert agent.exploration.current_score == 3


# =============================================================================
# Test: _update_available_actions
# =============================================================================


@pytest.mark.unit
class TestUpdateAvailableActions:
    def _make_agent(self, AgentClass):
        return AgentClass(verbose=False)

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_updates_from_api_ints(self, AgentClass):
        """New arc-agi package sends list[int]."""
        agent = self._make_agent(AgentClass)
        agent._update_available_actions([1, 2, 3])
        assert agent.exploration.available_action_ids == {1, 2, 3}

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_updates_from_api_game_actions(self, AgentClass):
        """Also handles GameAction enums for backward compat."""
        agent = self._make_agent(AgentClass)
        agent._update_available_actions([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3])
        assert agent.exploration.available_action_ids == {1, 2, 3}

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_excludes_reset(self, AgentClass):
        agent = self._make_agent(AgentClass)
        agent._update_available_actions([0, 1])  # 0 = RESET
        assert 0 not in agent.exploration.available_action_ids

    @pytest.mark.parametrize("AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent])
    def test_initializes_counts_for_new_actions(self, AgentClass):
        agent = self._make_agent(AgentClass)
        agent.exploration.action_counts = {}
        agent.exploration.action_observations = {}
        agent._update_available_actions([1, 5])
        assert agent.exploration.action_counts[1] == 0
        assert agent.exploration.action_counts[5] == 0
        assert agent.exploration.action_observations[1] == []
        assert agent.exploration.action_observations[5] == []
