from collections import deque

import numpy as np
from src.utils import config

SHARPE_WINDOW = config['reward']['sharpe_window']
STEP_RETURN_WEIGHT = config['reward']['step_return_weight']
SHARPE_WEIGHT = config['reward']['sharpe_weight']
DRAWDOWN_PENALTY_SCALE = config['reward']['drawdown_penalty_scale']
OVERTRADE_PENALTY_SCALE = config['reward']['overtrade_penalty_scale']
MIN_BUFFER_SIZE = config['reward']['min_buffer_size']
EPSILON = config['reward']['epsilon']

class RewardCalculator:
    def __init__(
        self,
        window: int = SHARPE_WINDOW,
        step_return_weight: float = STEP_RETURN_WEIGHT,
        sharpe_weight: float = SHARPE_WEIGHT,
        drawdown_scale: float = DRAWDOWN_PENALTY_SCALE,
        overtrade_scale: float = OVERTRADE_PENALTY_SCALE,
    ):
        self.window = window
        self.step_return_weight = step_return_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_scale = drawdown_scale
        self.overtrade_scale = overtrade_scale
        self.returns_buffer: deque = deque(maxlen=window)
        self.last_components: dict = {
            "reward_return": 0.0,
            "sharpe_reward": 0.0,
            "drawdown_penalty": 0.0,
            "overtrade_penalty": 0.0,
            "total_reward": 0.0,
        }

    def reset(self) -> None:
        self.returns_buffer.clear()
        self.last_components = {
            "reward_return": 0.0,
            "sharpe_reward": 0.0,
            "drawdown_penalty": 0.0,
            "overtrade_penalty": 0.0,
            "total_reward": 0.0,
        }

    def _sharpe_reward(self, step_return: float) -> float:
        self.returns_buffer.append(step_return)

        if len(self.returns_buffer) < MIN_BUFFER_SIZE:
            return 0.0

        mean_r = np.mean(self.returns_buffer)
        std_r = np.std(self.returns_buffer) + EPSILON
        return float(np.clip(mean_r / std_r, -5.0, 5.0))

    def _drawdown_penalty(self, drawdown: float) -> float:
        return self.drawdown_scale * max(drawdown, 0.0)

    def _overtrade_penalty(self, position_change: float) -> float:
        return self.overtrade_scale * abs(position_change)

    def calculate(
    self,
    step_return: float,
    drawdown: float,
    position_change: float,
) -> float:
        

        dd_pen = self._drawdown_penalty(drawdown)
        ot_pen = self._overtrade_penalty(position_change)

    # Stabilize returns
        step_return_component = np.tanh(step_return * 100.0)

        # `_sharpe_reward` returns a rolling Sharpe-like ratio (clipped to
        # [-5, 5]), a very different scale from the tanh-squashed
        # step-return term above ([-1, 1]). Squashing it through tanh too
        # keeps both components on a comparable scale before weighting,
        # so `sharpe_weight` actually controls their relative influence
        # the way the config implies, rather than one term silently
        # dominating (or, previously, not being applied at all).
        sharpe_component = np.tanh(self._sharpe_reward(step_return))

        blended_return = (
            self.step_return_weight * step_return_component
            + self.sharpe_weight * sharpe_component
        )

        total = (
        blended_return
        - dd_pen
        - ot_pen
    )

        self.last_components = {
        "step_return": step_return,
        "reward_return": step_return_component,
        "sharpe_reward": sharpe_component,
        "drawdown_penalty": dd_pen,
        "overtrade_penalty": ot_pen,
        "total_reward": total,
    }

        return float(total)

    @property
    def buffer_mean(self) -> float:
        if not self.returns_buffer:
            return 0.0
        return float(np.mean(self.returns_buffer))

    @property
    def buffer_std(self) -> float:
        if not self.returns_buffer:
            return 0.0
        return float(np.std(self.returns_buffer))

    @property
    def annualized_sharpe(self) -> float:
        if len(self.returns_buffer) < MIN_BUFFER_SIZE:
            return 0.0
        mean_r = np.mean(self.returns_buffer)
        std_r  = np.std(self.returns_buffer) + EPSILON
        return float((mean_r / std_r) * np.sqrt(8760))