import os
import shutil

import numpy as np
import pytest

from arcengine import FrameDataRaw, GameState


def get_test_recordings_dir():
    conftest_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(conftest_dir, "recordings")


@pytest.fixture(scope="session", autouse=True)
def clean_test_recordings():
    test_recordings_dir = get_test_recordings_dir()

    os.environ["RECORDINGS_DIR"] = test_recordings_dir

    if os.path.exists(test_recordings_dir):
        shutil.rmtree(test_recordings_dir)
    os.makedirs(test_recordings_dir, exist_ok=True)

    yield test_recordings_dir


@pytest.fixture
def temp_recordings_dir(clean_test_recordings):
    test_recordings_dir = get_test_recordings_dir()

    os.makedirs(test_recordings_dir, exist_ok=True)

    original_dir = os.environ.get("RECORDINGS_DIR")
    os.environ["RECORDINGS_DIR"] = test_recordings_dir

    yield test_recordings_dir

    if original_dir:
        os.environ["RECORDINGS_DIR"] = original_dir
    else:
        os.environ.pop("RECORDINGS_DIR", None)


@pytest.fixture
def sample_frame():
    """Create a sample FrameDataRaw for testing."""
    obs = FrameDataRaw(
        game_id="test-game",
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5],
    )
    # Set frame data (palette-indexed 2D grid)
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[10:20, 10:20] = 2  # Red block
    obs.frame = [grid]
    return obs


@pytest.fixture
def use_env_vars(monkeypatch):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("ARC_API_KEY"):
        monkeypatch.setenv("ARC_API_KEY", "test-key")
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
