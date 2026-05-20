"""Global seeding for reproducibility.

Call :func:`seed_everything` once at every entrypoint (CLI, training, tests).
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> int:
    """Seed Python, NumPy, and (if installed) PyTorch.

    Returns the seed used so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional at import time
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
