from __future__ import annotations

import os
import random

import numpy as np
from loguru import logger

from src.utils import config

try:
    import torch
except ImportError:  # torch is an optional dependency for some entrypoints
    torch = None


def set_global_seed(seed: int | None = None) -> int:
    """
    Seed every source of randomness used across the project.

    This seeds Python's `random`, NumPy's global RNG, PyTorch's CPU/CUDA
    RNGs (when available), and `PYTHONHASHSEED`. It does *not* seed a
    specific `gymnasium` environment's internal RNG on its own -- pass
    the same seed to `env.reset(seed=...)` / `action_space.seed(seed)`
    (see `seed_env`) so environment-level randomness (e.g. episode start
    position, `action_space.sample()`) is reproducible too.

    Parameters
    ----------
    seed
        Seed to use. If omitted, falls back to `training.seed` in
        `config.yaml`.

    Returns
    -------
    int
        The seed that was applied, for logging/bookkeeping.
    """

    if seed is None:
        seed = int(config.get("training", {}).get("seed", 42))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Trade a little speed for reproducibility on CUDA.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Global random seed set to {}", seed)

    return seed


def seed_env(env, seed: int | None = None) -> int:
    """
    Seed a `gymnasium` environment's internal RNG and its action space.

    `env.reset(seed=...)` re-seeds `env.np_random` (used e.g. for
    `GymBitcoinEnv`'s randomized episode start step), and seeding the
    action space makes `action_space.sample()` reproducible too (used
    by the random-agent benchmark). Without this, an environment falls
    back to OS entropy the first time it's used, even if the rest of
    the process has been seeded via `set_global_seed`.

    Parameters
    ----------
    env
        A `gymnasium.Env` (or wrapper) instance.
    seed
        Seed to use. If omitted, falls back to `training.seed` in
        `config.yaml`.

    Returns
    -------
    int
        The seed that was applied.
    """

    if seed is None:
        seed = int(config.get("training", {}).get("seed", 42))

    env.action_space.seed(seed)
    env.reset(seed=seed)

    return seed