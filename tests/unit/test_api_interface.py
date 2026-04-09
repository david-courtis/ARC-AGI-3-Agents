"""
Tests for ARC-AGI-3 API interface: frame format, grid extraction, and agent
lifecycle against mock and real API responses.

Ensures our agents correctly handle the 64x64x3 RGB frame format returned
by the ARC-AGI-3 API.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from agents.structs import (
    ActionInput,
    FrameData,
    GameAction,
    GameState,
)
from src.oop_agent.agent import OOPAgent
from src.oop_agent.world_model import extract_grid as oop_extract_grid
from src.monolithic_agent.agent import MonolithicAgent
from src.monolithic_agent.world_model import extract_grid as mono_extract_grid
from src.object_centric_agent.agent import ObjectCentricAgent
from src.object_centric_agent.world_model import extract_grid as oc_extract_grid


# =============================================================================
# Helpers: build realistic API frame data
# =============================================================================


def make_palette_frame(bg_index=5):
    """Create a palette-indexed frame as the real ARC-AGI-3 API returns it.

    Format: [grids][rows][cols] where values are 0-15 color indices.
    Typically 1 grid of 64x64.
    """
    return [
        [[bg_index] * 64 for _ in range(64)]
    ]


def make_palette_frame_with_objects():
    """Create a palette-indexed 64x64 frame with distinct colored regions."""
    grid = [[5] * 64 for _ in range(64)]  # gray background (5)

    # White border (10)
    for x in range(64):
        grid[0][x] = 10
        grid[63][x] = 10
    for y in range(64):
        grid[y][0] = 10
        grid[y][63] = 10

    # Red block at (10,10) to (14,14) — color index 2
    for y in range(10, 15):
        for x in range(10, 15):
            grid[y][x] = 2

    # Green block at (30,30) to (34,34) — color index 3
    for y in range(30, 35):
        for x in range(30, 35):
            grid[y][x] = 3

    # Blue block at (50,20) to (54,24) — color index 6
    for y in range(50, 55):
        for x in range(20, 25):
            grid[y][x] = 6

    return [grid]  # Wrap in grids list


def make_64x64_rgb_frame(bg_color=(7, 7, 7)):
    """Create a 64x64x3 RGB frame list (legacy/local sim format)."""
    return [
        [[bg_color[0], bg_color[1], bg_color[2]] for _ in range(64)]
        for _ in range(64)
    ]


def make_frame_with_objects():
    """Create a 64x64x3 RGB frame with distinct colored regions."""
    frame = make_64x64_rgb_frame(bg_color=(0, 0, 0))

    # White border
    for x in range(64):
        frame[0][x] = [255, 255, 255]
        frame[63][x] = [255, 255, 255]
    for y in range(64):
        frame[y][0] = [255, 255, 255]
        frame[y][63] = [255, 255, 255]

    # Red block at (10,10) to (14,14)
    for y in range(10, 15):
        for x in range(10, 15):
            frame[y][x] = [255, 0, 0]

    # Green block at (30,30) to (34,34)
    for y in range(30, 35):
        for x in range(30, 35):
            frame[y][x] = [0, 255, 0]

    # Blue block at (50,20) to (54,24)
    for y in range(50, 55):
        for x in range(20, 25):
            frame[y][x] = [0, 0, 255]

    return frame


def make_api_frame_response(
    game_id="test_game",
    state=GameState.NOT_FINISHED,
    score=0,
    frame=None,
    guid="test-guid-123",
    available_actions=None,
):
    """Build a dict mimicking what the ARC-AGI-3 API returns for a frame."""
    if frame is None:
        frame = make_palette_frame()
    if available_actions is None:
        available_actions = [0, 1, 2, 3, 4, 5]  # RESET + ACTION1-5
    return {
        "game_id": game_id,
        "frame": frame,
        "state": state.value,
        "score": score,
        "action_input": {"id": 0, "data": {"game_id": game_id}, "reasoning": None},
        "guid": guid,
        "full_reset": False,
        "available_actions": available_actions,
    }


def make_frame_data(frame=None, state=GameState.NOT_FINISHED, score=0, game_id="test_game"):
    """Create a FrameData with a palette-indexed frame (real API format)."""
    if frame is None:
        frame = make_palette_frame()
    return FrameData(
        game_id=game_id,
        frame=frame,
        state=state,
        score=score,
        available_actions=[
            GameAction.RESET,
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ],
    )


# =============================================================================
# Test: Frame format validation
# =============================================================================


@pytest.mark.unit
class TestFrameFormat:
    """Verify frame format handling for real API (palette-indexed) frames."""

    def test_palette_frame_dimensions(self):
        """Real API frames: [1 grid][64 rows][64 cols] with values 0-15."""
        frame = make_palette_frame()
        assert len(frame) == 1, f"Expected 1 grid, got {len(frame)}"
        assert len(frame[0]) == 64, f"Expected 64 rows, got {len(frame[0])}"
        assert len(frame[0][0]) == 64, f"Expected 64 cols, got {len(frame[0][0])}"

    def test_palette_frame_as_numpy(self):
        """Palette frame converts to (1, 64, 64) numpy array."""
        frame = make_palette_frame()
        arr = np.array(frame, dtype=np.uint8)
        assert arr.shape == (1, 64, 64)
        assert arr.dtype == np.uint8

    def test_palette_values_in_range(self):
        """All values should be 0-15 palette indices."""
        frame = make_palette_frame_with_objects()
        arr = np.array(frame[0], dtype=np.uint8)
        assert arr.max() <= 15
        assert arr.min() >= 0

    def test_palette_frame_with_objects(self):
        """Verify colored objects are placed correctly with palette indices."""
        frame = make_palette_frame_with_objects()
        grid = frame[0]
        assert grid[10][10] == 2   # Red
        assert grid[30][30] == 3   # Green
        assert grid[50][20] == 6   # Blue
        assert grid[5][5] == 5     # Gray background
        assert grid[0][0] == 10    # White border

    def test_framedata_validates_palette_frame(self):
        """FrameData should accept palette-indexed frames."""
        frame = make_palette_frame()
        fd = FrameData(
            game_id="test",
            frame=frame,
            state=GameState.NOT_FINISHED,
            score=0,
        )
        assert len(fd.frame) == 1
        assert len(fd.frame[0]) == 64
        assert len(fd.frame[0][0]) == 64

    def test_api_response_parses_to_framedata(self):
        """A realistic API response with palette frame should parse into FrameData."""
        response = make_api_frame_response()
        fd = FrameData.model_validate(response)
        assert fd.game_id == "test_game"
        assert fd.state == GameState.NOT_FINISHED
        assert len(fd.frame) == 1  # 1 grid
        assert len(fd.frame[0]) == 64

    def test_rgb_frame_still_works(self):
        """Legacy RGB format [64][64][3] should still be accepted."""
        frame = make_64x64_rgb_frame()
        fd = FrameData(game_id="test", frame=frame, state=GameState.NOT_FINISHED, score=0)
        assert len(fd.frame) == 64
        assert len(fd.frame[0]) == 64
        assert len(fd.frame[0][0]) == 3


# =============================================================================
# Test: extract_grid works with 64x64x3 frames
# =============================================================================


EXTRACT_FUNCS = [
    pytest.param(oop_extract_grid, id="oop"),
    pytest.param(mono_extract_grid, id="mono"),
    pytest.param(oc_extract_grid, id="oc"),
]


@pytest.mark.unit
class TestExtractGrid:
    """Verify extract_grid correctly converts API frame data to (H, W, 3) RGB arrays."""

    @pytest.mark.parametrize("extract_grid", EXTRACT_FUNCS)
    def test_palette_frame_to_rgb(self, extract_grid):
        """Real API palette frame [1][64][64] should extract to (64, 64, 3) RGB."""
        frame = make_palette_frame(bg_index=5)
        arr = extract_grid(frame)
        assert arr is not None
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8
        # Palette index 5 = Yellow (255, 220, 0)
        assert tuple(arr[0, 0]) == (255, 220, 0)

    @pytest.mark.parametrize("extract_grid", EXTRACT_FUNCS)
    def test_palette_preserves_objects(self, extract_grid):
        """Palette objects should map to correct RGB colors."""
        frame = make_palette_frame_with_objects()
        arr = extract_grid(frame)
        assert arr is not None
        assert arr.shape == (64, 64, 3)
        # Index 2 = Red (255, 0, 0)
        assert tuple(arr[10, 10]) == (255, 0, 0)
        # Index 3 = Green (46, 204, 64)
        assert tuple(arr[30, 30]) == (46, 204, 64)
        # Index 6 = Blue (0, 0, 255)
        assert tuple(arr[50, 20]) == (0, 0, 255)
        # Index 10 = White border
        assert tuple(arr[0, 0]) == (255, 255, 255)

    @pytest.mark.parametrize("extract_grid", EXTRACT_FUNCS)
    def test_rgb_frame_passthrough(self, extract_grid):
        """Legacy 64x64x3 RGB frames should pass through unchanged."""
        frame = make_64x64_rgb_frame()
        arr = extract_grid(frame)
        assert arr is not None
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.parametrize("extract_grid", EXTRACT_FUNCS)
    def test_empty_frame(self, extract_grid):
        """Empty frame list should return None."""
        assert extract_grid([]) is None

    @pytest.mark.parametrize("extract_grid", EXTRACT_FUNCS)
    def test_2d_palette_grid(self, extract_grid):
        """A flat 2D palette grid should convert to RGB."""
        frame = [[5] * 4 for _ in range(4)]  # 4x4 all palette index 5
        arr = extract_grid(frame)
        assert arr is not None
        assert arr.shape == (4, 4, 3)
        assert tuple(arr[0, 0]) == (255, 220, 0)  # Index 5 = Yellow

    @pytest.mark.parametrize("extract_grid", EXTRACT_FUNCS)
    def test_multi_grid_takes_last(self, extract_grid):
        """Multiple grids should extract the last one."""
        grid1 = [[0] * 8 for _ in range(8)]   # all black
        grid2 = [[10] * 8 for _ in range(8)]  # all white
        frame = [grid1, grid2]
        arr = extract_grid(frame)
        assert arr is not None
        assert arr.shape == (8, 8, 3)
        # Should be last grid = white (10)
        assert tuple(arr[0, 0]) == (255, 255, 255)


# =============================================================================
# Test: Agent choose_action processes 64x64 frames correctly
# =============================================================================


@pytest.mark.unit
class TestAgentFrameProcessing:
    """Test that agents correctly handle 64x64 frames from the API."""

    def _make_agent(self, AgentClass):
        return AgentClass(
            card_id="test-card",
            game_id="test-game",
            agent_name="test",
            ROOT_URL="https://example.com",
            record=False,
            verbose=False,
        )

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_reset_then_explore_with_64x64_frame(self, AgentClass):
        """Agent should handle RESET -> explore with a proper 64x64 frame."""
        agent = self._make_agent(AgentClass)

        # First frame: NOT_PLAYED -> agent should RESET
        not_played = make_frame_data(state=GameState.NOT_PLAYED)
        action = agent.choose_action([not_played], not_played)
        assert action == GameAction.RESET

        # Second frame: NOT_FINISHED with 64x64 frame -> agent should explore
        playing = make_frame_data(state=GameState.NOT_FINISHED)
        action = agent.choose_action([not_played, playing], playing)
        assert isinstance(action, GameAction)
        assert action != GameAction.RESET

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_frame_shape_recorded(self, AgentClass):
        """Agent should record the frame shape from the first valid frame."""
        agent = self._make_agent(AgentClass)

        frame = make_frame_data(state=GameState.NOT_FINISHED)
        agent.choose_action([frame], frame)

        assert agent._frame_shape is not None
        assert agent._frame_shape == (64, 64, 3)

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_transition_recorded_with_palette_frames(self, AgentClass):
        """Transitions between palette frames should be recorded as RGB arrays."""
        agent = self._make_agent(AgentClass)

        frame1 = make_frame_data(state=GameState.NOT_FINISHED)
        agent.choose_action([frame1], frame1)

        # Second call records the transition
        frame2 = make_frame_data(
            frame=make_palette_frame_with_objects(),
            state=GameState.NOT_FINISHED,
        )
        agent.choose_action([frame1, frame2], frame2)

        assert len(agent.exploration.replay_buffer) == 1
        t = agent.exploration.replay_buffer[0]
        assert t.before_frame.shape == (64, 64, 3)
        assert t.after_frame.shape == (64, 64, 3)

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_score_tracking(self, AgentClass):
        """Score from API frames should be tracked correctly."""
        agent = self._make_agent(AgentClass)

        frame = make_frame_data(state=GameState.NOT_FINISHED, score=42)
        agent.choose_action([frame], frame)
        assert agent.exploration.current_score == 42
        assert agent.exploration.max_score == 42

        frame2 = make_frame_data(state=GameState.NOT_FINISHED, score=85)
        agent.choose_action([frame, frame2], frame2)
        assert agent.exploration.current_score == 85
        assert agent.exploration.max_score == 85

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_available_actions_from_api(self, AgentClass):
        """Agent should update available actions from API frame."""
        agent = self._make_agent(AgentClass)

        frame = make_frame_data(state=GameState.NOT_FINISHED)
        # The frame has ACTION1-5 available
        agent.choose_action([frame], frame)

        assert agent.exploration.available_action_ids == {1, 2, 3, 4, 5}


# =============================================================================
# Test: Full agent lifecycle with mocked HTTP
# =============================================================================


@pytest.mark.unit
class TestAgentLifecycleMocked:
    """Test the agent lifecycle with mocked HTTP — no LLM calls."""

    def _make_agent(self, AgentClass):
        return AgentClass(
            card_id="test-card",
            game_id="test-game",
            agent_name="test",
            ROOT_URL="https://three.arcprize.org",
            record=False,
            verbose=False,
        )

    def _mock_api_response(self, frame=None, state=GameState.NOT_FINISHED, score=0):
        """Create a mock requests.Response with palette-indexed frame data."""
        resp = MagicMock(spec=requests.Response)
        resp.ok = True
        data = make_api_frame_response(
            frame=frame or make_palette_frame(),
            state=state,
            score=score,
        )
        resp.json.return_value = data
        return resp

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_take_action_returns_valid_frame(self, AgentClass):
        """Agent.take_action should return a FrameData with palette frame."""
        agent = self._make_agent(AgentClass)
        mock_resp = self._mock_api_response()

        with patch.object(agent, "do_action_request", return_value=mock_resp):
            frame = agent.take_action(GameAction.RESET)

        assert frame is not None
        # Real API format: [1 grid][64 rows][64 cols]
        assert len(frame.frame) == 1
        assert len(frame.frame[0]) == 64
        assert len(frame.frame[0][0]) == 64

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_take_action_with_objects(self, AgentClass):
        """take_action should preserve palette-indexed objects in the frame."""
        agent = self._make_agent(AgentClass)
        mock_resp = self._mock_api_response(frame=make_palette_frame_with_objects())

        with patch.object(agent, "do_action_request", return_value=mock_resp):
            frame = agent.take_action(GameAction.ACTION1)

        assert frame is not None
        # Frame is palette-indexed: [1][64][64]
        assert len(frame.frame) == 1
        assert frame.frame[0][10][10] == 2  # Red palette index

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_append_frame_stores_guid(self, AgentClass):
        """append_frame should update the agent's guid from the frame."""
        agent = self._make_agent(AgentClass)
        fd = FrameData(
            game_id="test-game",
            frame=make_palette_frame(),
            state=GameState.NOT_FINISHED,
            score=10,
            guid="abc-123",
        )
        agent.append_frame(fd)
        assert agent.guid == "abc-123"
        assert len(agent.frames) == 2  # initial + appended

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_game_over_triggers_reset(self, AgentClass):
        """GAME_OVER state should trigger a RESET action."""
        agent = self._make_agent(AgentClass)
        game_over = make_frame_data(state=GameState.GAME_OVER, score=50)
        action = agent.choose_action([game_over], game_over)
        assert action == GameAction.RESET

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_win_triggers_done(self, AgentClass):
        """WIN state should cause is_done to return True."""
        agent = self._make_agent(AgentClass)
        win = make_frame_data(state=GameState.WIN, score=254)
        assert agent.is_done([win], win) is True

    @pytest.mark.parametrize(
        "AgentClass", [OOPAgent, MonolithicAgent, ObjectCentricAgent]
    )
    def test_state_properties(self, AgentClass):
        """Agent state/score properties should reflect latest frame."""
        agent = self._make_agent(AgentClass)
        fd = FrameData(
            game_id="test-game",
            frame=make_palette_frame(),
            state=GameState.NOT_FINISHED,
            score=42,
        )
        agent.append_frame(fd)
        assert agent.state == GameState.NOT_FINISHED
        assert agent.score == 42


# =============================================================================
# Test: API response format validation
# =============================================================================


@pytest.mark.unit
class TestAPIResponseFormat:
    """Verify our code handles various API response shapes correctly."""

    def test_score_range(self):
        """Score must be 0-254 per the API spec."""
        # Valid scores
        for s in [0, 1, 100, 254]:
            fd = FrameData(game_id="t", frame=[], state=GameState.NOT_FINISHED, score=s)
            assert fd.score == s

        # Invalid score (255+) should raise
        with pytest.raises(Exception):
            FrameData(game_id="t", frame=[], state=GameState.NOT_FINISHED, score=255)

    def test_game_state_values(self):
        """All GameState values should be parseable."""
        for state in GameState:
            fd = FrameData(game_id="t", frame=[], state=state, score=0)
            assert fd.state == state

    def test_action_round_trip(self):
        """GameAction from_id and from_name should produce consistent results."""
        for action in GameAction:
            assert GameAction.from_id(action.value) == action
            assert GameAction.from_name(action.name) == action

    def test_complex_action_data(self):
        """ACTION6 (complex) should accept x,y coordinates."""
        action = GameAction.ACTION6
        data = action.set_data({"x": 32, "y": 48})
        assert data.x == 32
        assert data.y == 48

    def test_complex_action_bounds(self):
        """ACTION6 x,y must be 0-63."""
        action = GameAction.ACTION6
        # Valid
        action.set_data({"x": 0, "y": 0})
        action.set_data({"x": 63, "y": 63})
        # Invalid
        with pytest.raises(Exception):
            action.set_data({"x": 64, "y": 0})

    def test_empty_frame_is_empty(self):
        """FrameData with no frame should report is_empty."""
        fd = FrameData(game_id="t", frame=[], state=GameState.NOT_PLAYED, score=0)
        assert fd.is_empty()

    def test_full_frame_not_empty(self):
        """FrameData with a 64x64 frame should not be empty."""
        fd = make_frame_data()
        assert not fd.is_empty()


# =============================================================================
# Test: Animation frame extraction (regression for RGB vs palette split bug)
# =============================================================================


@pytest.mark.unit
class TestAnimationFrameExtraction:
    """Ensure RGB frames are not split into rows by animation extraction."""

    def test_rgb_array_treated_as_single_frame(self):
        """A (H, W, 3) RGB array must be treated as 1 frame, not H frames."""
        from src.shared.exploration_engine import ExplorationEngine

        engine = ExplorationEngine.__new__(ExplorationEngine)
        rgb_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = engine._extract_animation_frames(rgb_frame)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_palette_animation_split_correctly(self):
        """A (N, H, W) palette array should be split into N frames."""
        from src.shared.exploration_engine import ExplorationEngine

        engine = ExplorationEngine.__new__(ExplorationEngine)
        palette_anim = np.zeros((5, 64, 64), dtype=np.uint8)
        result = engine._extract_animation_frames(palette_anim)
        assert len(result) == 5
        assert result[0].shape == (64, 64)

    def test_2d_frame_treated_as_single(self):
        """A 2D palette frame should be treated as 1 frame."""
        from src.shared.exploration_engine import ExplorationEngine

        engine = ExplorationEngine.__new__(ExplorationEngine)
        frame_2d = np.zeros((64, 64), dtype=np.uint8)
        result = engine._extract_animation_frames(frame_2d)
        assert len(result) == 1

    def test_frames_to_base64_rgb_single_frame(self):
        """frames_to_base64_list on (H,W,3) RGB must return 1 image, not H."""
        from src.shared.vision import FrameCapture

        capture = FrameCapture()
        rgb_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = capture.frames_to_base64_list(rgb_frame)
        assert len(result) == 1

    def test_frames_to_base64_palette_animation(self):
        """frames_to_base64_list on (N,H,W) palette should return N images."""
        from src.shared.vision import FrameCapture

        capture = FrameCapture()
        palette_anim = np.zeros((5, 8, 8), dtype=np.uint8)
        result = capture.frames_to_base64_list(palette_anim)
        assert len(result) == 5
