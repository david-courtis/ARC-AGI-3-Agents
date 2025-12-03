"""Utility functions for experiment setup and logging."""

import os
import logging
from datetime import datetime
from typing import Tuple


def setup_experiment_directory(base_name: str = "experiments") -> Tuple[str, str]:
    """
    Create a timestamped experiment directory.

    Returns:
        Tuple of (experiment_dir, log_file_path)
    """
    # Create base experiments directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiments_dir = os.path.join(project_root, base_name)
    os.makedirs(experiments_dir, exist_ok=True)

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(experiments_dir, f"run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(experiment_dir, "experiment.log")

    return experiment_dir, log_file


def setup_logging_for_experiment(log_file: str, level: int = logging.INFO) -> None:
    """
    Configure logging to write to both file and console.

    Args:
        log_file: Path to the log file
        level: Logging level (default: INFO)
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_environment_directory(base_dir: str, game_id: str) -> str:
    """
    Get or create a directory for a specific game/environment.

    Args:
        base_dir: Base experiment directory
        game_id: The game identifier

    Returns:
        Path to the environment-specific directory
    """
    # Sanitize game_id for use as directory name
    safe_game_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in game_id)
    env_dir = os.path.join(base_dir, f"env_{safe_game_id}")
    os.makedirs(env_dir, exist_ok=True)
    return env_dir
