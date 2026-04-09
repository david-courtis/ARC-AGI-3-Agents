"""
Integration tests against the real ARC-AGI-3 API at three.arcprize.org.

These tests require:
- ARC_API_KEY environment variable set
- Network access to three.arcprize.org

Run with: pytest tests/integration/test_real_api.py -v
Skip automatically if ARC_API_KEY is not set or API is unreachable.
"""

import os

import numpy as np
import pytest
import requests

from agents.structs import FrameData, GameAction, GameState, Scorecard

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ARC_API_KEY") or os.environ.get("ARC_API_KEY") == "test-key",
    reason="ARC_API_KEY not set — skipping real API tests",
)

ROOT_URL = "https://three.arcprize.org"
HEADERS = {
    "X-API-Key": os.environ.get("ARC_API_KEY", ""),
    "Accept": "application/json",
}


@pytest.fixture(scope="module")
def api_session():
    """Create a requests session for API calls."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


@pytest.fixture(scope="module")
def game_list(api_session):
    """Fetch the list of available games."""
    r = api_session.get(f"{ROOT_URL}/api/games", timeout=10)
    assert r.ok, f"Failed to fetch games: {r.status_code} {r.text}"
    games = r.json()
    assert len(games) > 0, "No games available"
    return [g["game_id"] for g in games]


@pytest.fixture(scope="module")
def scorecard_id(api_session):
    """Open a scorecard for testing and close it after."""
    r = api_session.post(
        f"{ROOT_URL}/api/scorecard/open",
        json={"tags": ["test", "api_interface_test"]},
        timeout=10,
    )
    assert r.ok, f"Failed to open scorecard: {r.status_code} {r.text}"
    card_id = r.json()["card_id"]
    yield card_id
    # Cleanup: close scorecard
    api_session.post(
        f"{ROOT_URL}/api/scorecard/close",
        json={"card_id": card_id},
        timeout=10,
    )


@pytest.mark.integration
class TestRealAPIConnection:
    """Test basic connectivity to the ARC-AGI-3 API."""

    def test_games_endpoint(self, api_session):
        """GET /api/games should return a list of games."""
        r = api_session.get(f"{ROOT_URL}/api/games", timeout=10)
        assert r.ok
        games = r.json()
        assert isinstance(games, list)
        assert len(games) > 0
        assert "game_id" in games[0]

    def test_scorecard_open_close(self, api_session):
        """Opening and closing a scorecard should work."""
        # Open
        r = api_session.post(
            f"{ROOT_URL}/api/scorecard/open",
            json={"tags": ["test"]},
            timeout=10,
        )
        assert r.ok
        card_id = r.json()["card_id"]
        assert card_id

        # Close
        r = api_session.post(
            f"{ROOT_URL}/api/scorecard/close",
            json={"card_id": card_id},
            timeout=10,
        )
        assert r.ok


@pytest.mark.integration
class TestRealAPIFrameFormat:
    """Test that frames from the real API have the expected 64x64x3 format."""

    def test_reset_returns_palette_frame(self, api_session, game_list, scorecard_id):
        """RESET should return a FrameData with palette-indexed [grids][64][64] frame."""
        game_id = game_list[0]

        r = api_session.post(
            f"{ROOT_URL}/api/cmd/RESET",
            json={"card_id": scorecard_id, "game_id": game_id},
            timeout=10,
        )
        assert r.ok, f"RESET failed: {r.status_code} {r.text}"

        data = r.json()
        fd = FrameData.model_validate(data)

        assert fd.game_id == game_id
        assert fd.state in (GameState.NOT_FINISHED, GameState.NOT_PLAYED)

        # Real API: frame is [grids][rows][cols] with palette indices
        assert len(fd.frame) >= 1, f"Expected at least 1 grid, got {len(fd.frame)}"
        grid = fd.frame[-1]  # Last grid
        assert len(grid) == 64, f"Expected 64 rows, got {len(grid)}"
        assert len(grid[0]) == 64, f"Expected 64 cols, got {len(grid[0])}"

        # Values should be palette indices 0-15
        arr = np.array(grid, dtype=np.uint8)
        assert arr.max() <= 15, f"Max value {arr.max()} exceeds palette range 0-15"

    def test_frame_has_content(self, api_session, game_list, scorecard_id):
        """After RESET, the frame should have non-trivial content (multiple palette values)."""
        game_id = game_list[0]

        r = api_session.post(
            f"{ROOT_URL}/api/cmd/RESET",
            json={"card_id": scorecard_id, "game_id": game_id},
            timeout=10,
        )
        data = r.json()
        fd = FrameData.model_validate(data)
        grid = np.array(fd.frame[-1], dtype=np.uint8)

        # Frame should have at least 2 unique palette values
        unique_values = np.unique(grid)
        assert len(unique_values) >= 2, (
            f"Frame has only {len(unique_values)} unique values: {unique_values}"
        )

    def test_action_returns_frame(self, api_session, game_list, scorecard_id):
        """Sending an action after RESET should return a valid palette frame."""
        game_id = game_list[0]

        # RESET first
        r = api_session.post(
            f"{ROOT_URL}/api/cmd/RESET",
            json={"card_id": scorecard_id, "game_id": game_id},
            timeout=10,
        )
        reset_data = r.json()
        guid = reset_data.get("guid")

        # Send ACTION1
        action_payload = {"game_id": game_id}
        if guid:
            action_payload["guid"] = guid
        r = api_session.post(
            f"{ROOT_URL}/api/cmd/ACTION1",
            json=action_payload,
            timeout=10,
        )
        assert r.ok, f"ACTION1 failed: {r.status_code} {r.text}"

        data = r.json()
        fd = FrameData.model_validate(data)

        # Frame should be [grids][64][64] palette-indexed
        assert len(fd.frame) >= 1
        grid = fd.frame[-1]
        assert len(grid) == 64
        assert len(grid[0]) == 64

    def test_extract_grid_on_real_frame(self, api_session, game_list, scorecard_id):
        """extract_grid should convert real API palette frame to (64,64,3) RGB."""
        from src.oop_agent.world_model import extract_grid

        game_id = game_list[0]

        r = api_session.post(
            f"{ROOT_URL}/api/cmd/RESET",
            json={"card_id": scorecard_id, "game_id": game_id},
            timeout=10,
        )
        data = r.json()
        fd = FrameData.model_validate(data)

        arr = extract_grid(fd.frame)
        assert arr is not None
        assert arr.shape == (64, 64, 3), f"Expected (64,64,3), got {arr.shape}"
        assert arr.dtype == np.uint8
        # Should have actual colors (not all zeros)
        assert arr.sum() > 0

    def test_available_actions_populated(self, api_session, game_list, scorecard_id):
        """API should return available_actions after RESET."""
        game_id = game_list[0]

        r = api_session.post(
            f"{ROOT_URL}/api/cmd/RESET",
            json={"card_id": scorecard_id, "game_id": game_id},
            timeout=10,
        )
        data = r.json()
        fd = FrameData.model_validate(data)

        assert len(fd.available_actions) > 0, "No available actions returned"

    def test_multiple_games_all_64x64(self, api_session, game_list, scorecard_id):
        """Verify frame format across multiple different games."""
        from src.oop_agent.world_model import extract_grid

        for game_id in game_list[:3]:
            r = api_session.post(
                f"{ROOT_URL}/api/cmd/RESET",
                json={"card_id": scorecard_id, "game_id": game_id},
                timeout=10,
            )
            if not r.ok:
                continue

            data = r.json()
            fd = FrameData.model_validate(data)

            # Raw frame: [grids][64][64] palette-indexed
            assert len(fd.frame) >= 1, (
                f"Game {game_id}: expected at least 1 grid, got {len(fd.frame)}"
            )
            grid = fd.frame[-1]
            assert len(grid) == 64, (
                f"Game {game_id}: expected 64 rows, got {len(grid)}"
            )
            assert len(grid[0]) == 64, (
                f"Game {game_id}: expected 64 cols, got {len(grid[0])}"
            )

            # extract_grid should produce (64, 64, 3) RGB
            arr = extract_grid(fd.frame)
            assert arr is not None
            assert arr.shape == (64, 64, 3), (
                f"Game {game_id}: expected (64,64,3), got {arr.shape}"
            )
