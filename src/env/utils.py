from __future__ import annotations
import numpy as np


class PositionSizer:

    def __init__(self, max_step_change: float = 0.2):
        self.max_step_change = float(max_step_change)

    def apply(self, current: float, target: float) -> float:
        delta = np.clip(
            target - current,
            -self.max_step_change,
            self.max_step_change
        )
        return current + delta


class CostModel:

    def __init__(self, rate: float = 0.0005):
        self.rate = float(rate)

    def cost(self, notional: float) -> float:
        return abs(notional) * self.rate


class FundingModel:

    def __init__(self, rate_per_step: float = 0.0):
        self.rate_per_step = rate_per_step

    def funding_cost(self, position: float, capital: float) -> float:
        return position * capital * self.rate_per_step


def wrap_vec_normalize(venv, training: bool = True):
    """
    Wraps a VecEnv with observation normalization (running mean/std).

    Only `volume_zscore`/`OBV_zscore` are rolling-normalized upstream in
    the feature pipeline; the rest of `selected_features` (price,
    momentum, volatility indicators, ...) aren't uniformly scaled. That
    mattered less when the MLP-era env fed a whole flattened window at
    once; it matters more now that the policy sees one raw timestep per
    step (see GymBitcoinEnv.get_obs), so observation normalization is
    handled here instead.

    Reward normalization is deliberately left OFF: the reward is already
    hand-shaped (see src/env/rewards.py -- step return + Sharpe term +
    drawdown/overtrade penalties, each with its own tuned scale).
    VecNormalize's running reward normalization would fight that tuning
    rather than complement it.

    Intended call site: wherever the training/eval VecEnv is
    constructed (trainer.py, evaluate.py, quick_eval.py, ...) -- wrap
    once right after building the VecEnv and before handing it to
    RecurrentPPO. Use `training=False` (and load the stats saved
    alongside the model) for evaluation/inference so eval doesn't keep
    updating the running statistics on held-out data.

    Parameters
    ----------
    venv
        A VecEnv (e.g. DummyVecEnv/SubprocVecEnv) wrapping GymBitcoinEnv.
    training
        Whether this wrapper should keep updating its running
        mean/std (True during training) or use frozen stats loaded
        from a previous run (False during evaluation/inference).
    """
    from stable_baselines3.common.vec_env import VecNormalize

    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=training,
    )