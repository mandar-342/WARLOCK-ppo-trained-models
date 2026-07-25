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
# How many steps of step_return get aggregated (summed) into one sample
# before it's pushed into the rolling Sharpe buffer. Previously every
# single step's return was pushed straight into the buffer, so with
# MIN_BUFFER_SIZE=2 the ratio was frequently estimated from 2 raw,
# single-timestep returns -- a near-meaningless mean/std estimate that
# can swing wildly before being clipped to [-5, 5]. That's a plausible
# cause of the observed pattern where higher sharpe_weight correlated
# with *worse* outcomes and higher drawdown-breaker rates: the term was
# injecting noise, not a genuine risk-adjusted signal. Aggregating
# returns over a short window first makes each buffer sample represent
# sustained performance over several steps rather than tick noise,
# closer to what "Sharpe ratio" is meant to capture. Defaults to 1
# (= old per-step behavior) if the config key isn't present, so this is
# opt-in via config/sweep rather than a silent behavior change for
# anyone not on the updated config.yaml.
SHARPE_AGGREGATION_STEPS = config['reward'].get('sharpe_aggregation_steps', 1)

class RewardCalculator:
    def __init__(
        self,
        window: int = SHARPE_WINDOW,
        step_return_weight: float = STEP_RETURN_WEIGHT,
        sharpe_weight: float = SHARPE_WEIGHT,
        drawdown_scale: float = DRAWDOWN_PENALTY_SCALE,
        overtrade_scale: float = OVERTRADE_PENALTY_SCALE,
        sharpe_aggregation_steps: int = SHARPE_AGGREGATION_STEPS,
    ):
        self.window = window
        self.step_return_weight = step_return_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_scale = drawdown_scale
        self.overtrade_scale = overtrade_scale
        self.sharpe_aggregation_steps = max(1, int(sharpe_aggregation_steps))
        self.returns_buffer: deque = deque(maxlen=window)
        self._pending_returns: list = []
        self._last_sharpe_component: float = 0.0
        self.last_components: dict = {
            "reward_return": 0.0,
            "sharpe_reward": 0.0,
            "drawdown_penalty": 0.0,
            "overtrade_penalty": 0.0,
            "total_reward": 0.0,
        }

    def reset(self) -> None:
        self.returns_buffer.clear()
        self._pending_returns = []
        self._last_sharpe_component = 0.0
        self.last_components = {
            "reward_return": 0.0,
            "sharpe_reward": 0.0,
            "drawdown_penalty": 0.0,
            "overtrade_penalty": 0.0,
            "total_reward": 0.0,
        }

    def _sharpe_reward(self, step_return: float) -> float:
        self._pending_returns.append(step_return)

        # Between aggregation boundaries, hold the last computed ratio
        # rather than returning 0.0 -- returning to 0.0 every intermediate
        # step would itself be a noisy, sawtooth signal working against
        # the whole point of aggregating in the first place.
        if len(self._pending_returns) < self.sharpe_aggregation_steps:
            return self._last_sharpe_component

        aggregated_return = float(np.sum(self._pending_returns))
        self._pending_returns = []
        self.returns_buffer.append(aggregated_return)

        if len(self.returns_buffer) < MIN_BUFFER_SIZE:
            self._last_sharpe_component = 0.0
            return 0.0

        mean_r = np.mean(self.returns_buffer)
        std_r = np.std(self.returns_buffer) + EPSILON
        self._last_sharpe_component = float(np.clip(mean_r / std_r, -5.0, 5.0))
        return self._last_sharpe_component

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
        # 8760 assumes hourly steps/year; each buffer entry now spans
        # `sharpe_aggregation_steps` steps, so the number of samples/year
        # (and hence the annualization factor) scales down accordingly.
        periods_per_year = 8760 / self.sharpe_aggregation_steps
        return float((mean_r / std_r) * np.sqrt(periods_per_year))