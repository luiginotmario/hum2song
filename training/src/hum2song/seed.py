"""Deterministic seeding for the Python runtime and numpy."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed hash randomization, the random module, and numpy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
