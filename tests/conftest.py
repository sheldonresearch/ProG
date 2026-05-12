"""Shared pytest fixtures."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure the repo root (which contains bench.py) is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Fix all RNG seeds before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    yield
