from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from src.env.rewards import RewardCalculator
from src.env.utils import PositionSizer
from src.portfolio import Portfolio
from src.utils import config, root

class GymBitcoinEnv(gym.Env):
    """
    The environment owns market data, the action/observation spaces, and
    episode bookkeeping (drawdown tracking, termination). All cash,
    position, fee, and slippage accounting is delegated to
    (`src/portfolio`), which the env treats as an opaque accounting engine.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str | None = None,
        window_len: int | None = None,
        max_trade_step: float | None = None,
        max_drawdown: float | None = None,
    ):
        super().__init__()

        env_cfg = config.get("env", {})

        # Load config if not provided
        if data_path is None:
            data_path = str(root(config["paths"]["feature_engineered_dir"]) / "train.parquet")

        self.data_path = data_path
        self.window_len = window_len if window_len is not None else env_cfg.get("window_len", 48)
        self.max_drawdown = max_drawdown if max_drawdown is not None else env_cfg.get("max_drawdown", 0.3)

        max_trade_step = (
            max_trade_step if max_trade_step is not None else env_cfg.get("max_trade_step", 0.2)
        )
        self.position_sizer = PositionSizer(max_step_change=max_trade_step)

        # One pool backs both long (spot-style) and short (futures-style
        # margin) exposure; see src/portfolio/unified_portfolio.py.
        self.portfolio = Portfolio()
        self.initial_capital = self.portfolio.cash
        self.n_assets = len(self.portfolio.symbols)

        self.load_data()

        # Define spaces
        # obs = market window + portfolio vector
        # portfolio vector: [cash_weight, *asset_weights, unrealized_pnl_pct, holding_time_norm]
        # = 1 (cash) + n_assets (weights) + 1 (unrealized pnl) + 1 (holding time)
        portfolio_vec_dim = 3 + self.n_assets
        obs_dim = self.window_len * self.n_features + portfolio_vec_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Target *delta* per asset, in [-1, 1], applied to the current net
        # weight each step (see step()). Net weight itself can now range
        # over [-1, 1]: positive is spot long, negative is futures short.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.current_step = 0
        self.current_weights = [0.0] * self.n_assets
        self.holding_time = 0
        self.peak_value = 0.0
        self.reward_calc = RewardCalculator()
        #  Trade State
        self.position_open = False
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.entry_weight = 0.0
        self.stop_loss_multiple=config.get("risk", {}).get("stop_loss_atr_multiple", 1.5)
        self.take_profit_multiple= config.get("risk", {}).get("take_profit_atr_multiple", 3.0)
        self.target_atr_pct=config.get("risk", {}).get("target_atr_pct", 1.0)

        logger.info(
            f"GymBitcoinEnv initialized: "
            f"data={self.data_path}, window={self.window_len}, "
            f"features={self.n_features}, obs_dim={obs_dim}, "
            f"max_steps={self.max_steps}, assets={self.portfolio.symbols}"
        )

    def load_data(self) -> None:
        df = pd.read_parquet(self.data_path)

        #Identify feature columns (exclude OHLCV + timestamp)
        selected = config["features"]["selected_features"]

        self.feature_cols = [
    c for c in selected
    if c in df.columns
            ]
        logger.info(
    "Observation Features: {}",
    self.feature_cols,
            )

        self.features = df[self.feature_cols].values.astype(np.float32)
        self.prices = df["close"].values.astype(np.float32)
        self.timestamps = df["timestamp"].values
        self.n_features = len(self.feature_cols)
        if "ATR_pct" not in self.feature_cols:
            raise ValueError("ATR_pct feature is required for risk management.")
        self.atr_feature_idx = self.feature_cols.index("ATR_pct")

        #Max steps = data length - window_len - 1 (need at least one step ahead)
        self.max_steps = len(self.features) - self.window_len - 1

        if self.max_steps <= 0:
            raise ValueError(
                f"Not enough data: {len(self.features)} rows, "
                f"need at least {self.window_len + 2}"
            )

    def get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_len
        market_window = self.features[start:self.current_step].flatten()

        prices = [self.prices[self.current_step]]
        net_weights = self.portfolio.current_weights(prices)
        # cash_weight approximates the "uninvested" fraction of equity;
        # long notional is cash-backed 1:1 so this stays meaningful even
        # though short notional draws margin rather than cash directly.
        cash_weight = 1.0 - sum(max(0.0, w) for w in net_weights)
        unrealized_pnl_pct = (
            self.portfolio.unrealized_pnl(prices) / self.initial_capital
            if self.initial_capital > 0 else 0.0
        )

        portfolio_vec = np.array(
            [cash_weight, *net_weights, unrealized_pnl_pct],
            dtype=np.float32,
        )
        portfolio_vec = np.concatenate(
            [portfolio_vec, np.array([self.holding_time / 100.0], dtype=np.float32)]
        )

        return np.concatenate([market_window, portfolio_vec])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # random start for generalization
        self.current_step = self.np_random.integers(
            self.window_len,
            max(self.window_len + 1, self.max_steps // 2)
        )
        self.portfolio.reset()
        self.current_weights = [0.0] * self.n_assets
        self.holding_time = 0
        self.peak_value = self.portfolio.cash
        self.reward_calc.reset()
        self.position_open = False
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.entry_weight = 0.0

        obs = self.get_obs()
        price = float(self.prices[self.current_step])
        info = {
            "step": self.current_step,
            "price": price,
            "weights": list(self.current_weights),
            "capital": self.portfolio.total_value([price]),
        }
        return obs, info
    
    def _rebalance_portfolio(
    self,
    target_weights: list[float],
    prices: list[float],
       ) -> list:
        """Execute a rebalance toward net signed `target_weights` in
        [-1, 1]. The unified portfolio itself splits each weight into its
        long (>=0) and short (<0) leg and draws both from the same pool.
        """
        trades = self.portfolio.step(
        target_weights=target_weights,
        prices=prices,
        step=self.current_step,
        timestamp=self.timestamps[self.current_step],)

        self.current_weights = self.portfolio.current_weights(prices)

        return trades
    
    def hold_step(self):
        
        """
    Advance one timestep without rebalancing the portfolio.
    Used by passive benchmarks such as Buy & Hold.
        """

        price_prev = float(self.prices[self.current_step])

        value_before = self.portfolio.total_value([price_prev])

        self.current_step += 1

        price = float(self.prices[self.current_step])
        prices = [price]

        trades = []

        total_value = self.portfolio.total_value(prices)

        step_return = (
        (total_value - value_before) / value_before
        if value_before > 0
        else 0.0
         )

        self.holding_time += 1

        self.peak_value = max(
        self.peak_value,
        total_value,
        )

        drawdown = (
            (self.peak_value - total_value)
        / self.peak_value
        if self.peak_value > 0
        else 0.0
          )

        reward = self.reward_calc.calculate(
        step_return=step_return,
        drawdown=drawdown,
        position_change=0.0,
        )

        terminated = (
        self.current_step >= self.max_steps
        or drawdown >= self.max_drawdown
        or total_value <= 0
         )

        truncated = False

        obs = self.get_obs()

        info = {
        "step": self.current_step,
        "price": price,
        "weights": list(self.current_weights),
        "cash": self.portfolio.cash,
        "cost": 0.0,
        "capital": total_value,
        "drawdown": drawdown,
        "n_trades_this_step": 0,
        "realized_pnl": self.portfolio.realized_pnl(),
        "unrealized_pnl": self.portfolio.unrealized_pnl(prices),
        "reward_components": dict(
            self.reward_calc.last_components
        ),
        "forced_exit": False,
        "exit_reason": None,
        }

        return (
        obs,
        float(reward),
        terminated,
        truncated,
        info,
    )

    def step(self, action: np.ndarray):
        price_prev = float(self.prices[self.current_step])

        # 1. Clip raw action to the valid delta range, then rate-limit the
        # change in net allocation. target can now be negative (futures
        # short); previously it was clamped to [0, 1] (spot long only).
        delta_actions = np.clip(action, -1.0, 1.0)
        target_weights = [
       float(np.clip(current + delta * 0.50, -1.0, 1.0))
        for current, delta in zip(self.current_weights, delta_actions)
        ]
        new_weights = [
    self.position_sizer.apply(current, target)
    for current, target in zip(self.current_weights, target_weights)
       ]
        # Dynamic ATR Position Sizing.
        # A short's equity risk for a given price move is `leverage`x a
        # long's, so its ATR budget must be divided by leverage before
        # solving for the multiplier -- otherwise the sizer treats a 3x
        # leveraged short as if it carried the same equity risk as an
        # unleveraged long of the same weight magnitude.
        current_atr = float(self.features[self.current_step, self.atr_feature_idx])
        leverage = self.portfolio.leverage
        risk_multiplier_long = min(1.0, self.target_atr_pct / max(current_atr, 1e-6))
        risk_multiplier_short = min(1.0, self.target_atr_pct / max(current_atr * leverage, 1e-6))
        # Kept for logging/back-compat: the multiplier actually applied to
        # asset 0 (single-asset env today), captured before new_weights is
        # overwritten below.
        risk_multiplier = risk_multiplier_short if new_weights[0] < 0 else risk_multiplier_long
        new_weights = [
            weight * (risk_multiplier_short if weight < 0 else risk_multiplier_long)
            for weight in new_weights
        ]

        # Capture portfolio value at the *old* price, before this step's
        # trade and before advancing the market cursor. This is the
        # correct denominator for step_return below — valuing pre-trade
        # holdings at the pre-trade price.
        value_before = self.portfolio.total_value([price_prev])

        # 2. Advance market cursor to the next candle.
        self.current_step += 1
        price = float(self.prices[self.current_step])
        prices = [price]
        # ATR Stop Loss
        
        forced_exit = False
        exit_reason = None
        if self.position_open:
              unrealized_return = (price - self.entry_price) / self.entry_price
              if self.entry_weight < 0:
                  # Short leg: profit is the mirror image of a long's return.
                  unrealized_return = -unrealized_return
              # Shorts are margin-backed at `portfolio.short.leverage`, so a
              # given price move produces `leverage`x the equity impact of
              # the same move on an unleveraged (1x) long. The ATR-based
              # stop_loss_pct/take_profit_pct thresholds below are equity-risk
              # budgets (e.g. "don't lose more than ~1.5*ATR% of equity"), so
              # we must scale the raw price return by effective leverage
              # before comparing, or shorts blow through the intended budget
              # by a factor of `leverage`.
              effective_leverage = self.portfolio.leverage if self.entry_weight < 0 else 1.0
              equity_impact = unrealized_return * effective_leverage
              stop_loss_pct=(self.stop_loss_multiple * self.entry_atr/100.0)
              take_profit_pct=(self.take_profit_multiple*self.entry_atr/100.0)
              if equity_impact <= -stop_loss_pct:
                  logger.info( f"ATR Stop Loss Triggered "
                               f"| Entry={self.entry_price:.2f} "
                                f"| Current={price:.2f}")
                  new_weights = [0.0] * self.n_assets  # Force exit
                  forced_exit = True
                  exit_reason = "stop_loss"
              elif equity_impact >= take_profit_pct:
                  
                    logger.info( f"ATR Take Profit Triggered "
                               f"| Entry={self.entry_price:.2f} "
                                f"| Current={price:.2f}")
                    new_weights = [0.0] * self.n_assets  # Force exit
                    forced_exit = True
                    exit_reason = "take_profit"

        # 3. Execute the rebalance at the new candle's close.
        weight_delta_before = sum(abs(a - b) for a, b in zip(new_weights, self.current_weights))
        trades = self._rebalance_portfolio(
         new_weights,
         prices,
                         )

        #Detects a new position opening
        if(not self.position_open and any(abs(weight)>1e-6 for weight in self.current_weights)):
            self.position_open = True
            self.entry_price = price
            self.entry_atr = float(self.features[self.current_step, self.atr_feature_idx])
            self.entry_weight = self.current_weights[0]
        # Detect position fully closed
        if (self.position_open and all(abs(weight) < 1e-6 for weight in self.current_weights)):
            self.position_open = False
            self.entry_price = 0.0
            self.entry_atr = 0.0
            self.entry_weight = 0.0

        total_value = self.portfolio.total_value(prices)
        cost = sum(t.fee + t.slippage_cost for t in trades)

        # 4. Step return, for the reward calculator
        step_return = (total_value - value_before) / value_before if value_before > 0 else 0.0

        # 5. Holding time bookkeeping
        self.holding_time = 0 if weight_delta_before > 1e-9 else self.holding_time + 1

        # 6. Drawdown tracking
        self.peak_value = max(self.peak_value, total_value)
        drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0.0

        # 7. Reward
        reward = self.reward_calc.calculate(
            step_return=step_return,
            drawdown=drawdown,
            position_change=weight_delta_before,
        )

        # 8. Termination conditions
        terminated = (
            self.current_step >= self.max_steps
            or drawdown >= self.max_drawdown
            or total_value <= 0
        )
        truncated = False

        # 9. Observation & info
        obs = self.get_obs()
        info = {
            "step": self.current_step,
            "price": price,
            "weights": list(self.current_weights),
            "cash": self.portfolio.cash,
            "cost": cost,
            "capital": total_value,
            "drawdown": drawdown,
            "n_trades_this_step": len(trades),
            "realized_pnl": self.portfolio.realized_pnl(),
            "unrealized_pnl": self.portfolio.unrealized_pnl(prices),
            "reward_components": dict(self.reward_calc.last_components),
            "raw_action": action.tolist(),
            "target_weights": target_weights,
            "position_sized_weights": list(new_weights),
            "risk_multiplier": float(risk_multiplier),
            "risk_multiplier_long": float(risk_multiplier_long),
            "risk_multiplier_short": float(risk_multiplier_short),
            "forced_exit": forced_exit,
            "exit_reason": exit_reason,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Optional render - not implemented."""
        pass

    def close(self):
        """Cleanup."""
        pass