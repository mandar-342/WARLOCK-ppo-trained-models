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