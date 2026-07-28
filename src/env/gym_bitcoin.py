from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from src.env.rewards import RewardCalculator
from src.env.utils import PositionSizer
from src.portfolio import Portfolio
from src.portfolio.utils import base_asset
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
        deterministic_start: bool = False,
    ):
        super().__init__()

        env_cfg = config.get("env", {})

        # Load config if not provided
        if data_path is None:
            data_path = str(root(config["paths"]["feature_engineered_dir"]) / "train.parquet")

        self.data_path = data_path
        self.window_len = window_len if window_len is not None else env_cfg.get("window_len", 48)
        self.max_drawdown = max_drawdown if max_drawdown is not None else env_cfg.get("max_drawdown", 0.3)
        # Training wants a random episode start each reset (for
        # generalization across market regimes). Evaluation wants every
        # run to score the *same*, full held-out window so results are
        # reproducible and comparable across runs/models -- otherwise
        # total_return/Sharpe/trade counts differ purely because two runs
        # scored different, unequal-length slices of the test set.
        self.deterministic_start = deterministic_start

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
        # obs = single-timestep market features + portfolio vector.
        #
        # This branch trains a recurrent (LSTM) policy, which is
        # expected to carry temporal context in its own hidden state
        # across the episode rather than being handed a manually
        # flattened lookback window each step (the previous MLP-era
        # approach: `window_len * n_features` flattened into one
        # vector). Feeding a flattened window into an LSTM would waste
        # the point of the recurrence -- the network would be relearning
        # temporal structure that's already been hand-truncated to
        # `window_len` steps and pre-flattened. `window_len` is still
        # used below (see reset()) purely to pick a random episode start
        # far enough into the data for all rolling-window *features*
        # (ATR, z-scores, etc., computed upstream in the feature
        # pipeline) to already be warmed up -- it no longer shapes the
        # observation itself.
        #
        # portfolio vector: [cash_weight, *asset_weights, unrealized_pnl_pct, holding_time_norm]
        # = 1 (cash) + n_assets (weights) + 1 (unrealized pnl) + 1 (holding time)
        self.portfolio_vec_dim = 3 + self.n_assets
        # Market block is stacked per-asset: (n_assets, n_features), NOT
        # flattened. A Dict space is used (rather than SB3's default
        # flatten-everything Box) so a custom features extractor can
        # apply a shared per-asset encoder to the "market" block before
        # concatenating with the flat "portfolio" vector -- see the
        # multi-asset features extractor wired in via policy_kwargs.
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.n_assets, self.n_features), dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.portfolio_vec_dim,), dtype=np.float32,
                ),
            }
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
        # Read reward config live (like env_cfg above), rather than
        # relying on RewardCalculator's constructor defaults: those
        # defaults are module-level constants computed once when
        # rewards.py is first imported, before any per-run config
        # override is applied. In a long-lived process running many
        # trials/seeds (the multi-seed harness, the Optuna sweep), that
        # meant every reward.* override was silently ignored after the
        # first import -- every run used whatever was in config.yaml on
        # disk at process start, regardless of what was actually swept.
        reward_cfg = config.get("reward", {})
        self.reward_calc = RewardCalculator(
            window=reward_cfg.get("sharpe_window", 100),
            step_return_weight=reward_cfg.get("step_return_weight", 1.0),
            sharpe_weight=reward_cfg.get("sharpe_weight", 0.10),
            drawdown_scale=reward_cfg.get("drawdown_penalty_scale", 0.1),
            overtrade_scale=reward_cfg.get("overtrade_penalty_scale", 0.01),
            sharpe_aggregation_steps=reward_cfg.get("sharpe_aggregation_steps", 1),
        )
        #  Trade State -- one entry per asset so BTC and ETH can each hold
        # an independent position with its own stop-loss/take-profit,
        # rather than a single scalar that (pre-multiasset) only ever
        # tracked asset 0 and silently ignored every other asset.
        self.position_open = [False] * self.n_assets
        self.entry_price = [0.0] * self.n_assets
        self.entry_atr = [0.0] * self.n_assets
        self.entry_weight = [0.0] * self.n_assets
        self.stop_loss_multiple=config.get("risk", {}).get("stop_loss_atr_multiple", 1.5)
        self.take_profit_multiple= config.get("risk", {}).get("take_profit_atr_multiple", 3.0)
        self.target_atr_pct=config.get("risk", {}).get("target_atr_pct", 1.0)

        logger.info(
            f"GymBitcoinEnv initialized (single-timestep / recurrent-policy obs): "
            f"data={self.data_path}, episode_start_warmup={self.window_len}, "
            f"features={self.n_features}, market_shape=({self.n_assets},{self.n_features}), "
            f"max_steps={self.max_steps}, assets={self.portfolio.symbols}"
        )

    def load_data(self) -> None:
        df = pd.read_parquet(self.data_path)

        # Multi-asset feature files are asset-prefixed (e.g. "BTC_RSI",
        # "ETH_RSI") and share one "timestamp" + "{TAG}_close" per asset,
        # produced by src.features.pipeline.generate_multiasset_features.
        # Column order/asset order is driven by self.portfolio.symbols so
        # the env's asset axis always lines up with the portfolio's.
        selected = config["features"]["selected_features"]
        tags = [base_asset(s) for s in self.portfolio.symbols]

        self.feature_cols = [c for c in selected if f"{tags[0]}_{c}" in df.columns]
        logger.info("Observation Features: {}", self.feature_cols)
        if not self.feature_cols:
            raise ValueError(
                "No configured features found with expected asset prefix "
                f"'{tags[0]}_' in {self.data_path}. Was this file produced "
                "by generate_multiasset_features()?"
            )

        self.n_features = len(self.feature_cols)
        if "ATR_pct" not in self.feature_cols:
            raise ValueError("ATR_pct feature is required for risk management.")
        self.atr_feature_idx = self.feature_cols.index("ATR_pct")

        # Stack per-asset feature blocks -> shape (T, n_assets, n_features)
        # and per-asset close prices -> shape (T, n_assets). Assets were
        # already inner-joined on a shared timestamp axis upstream, so no
        # further alignment is needed here.
        per_asset_features = []
        per_asset_prices = []
        for tag in tags:
            cols = [f"{tag}_{c}" for c in self.feature_cols]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns for asset '{tag}': {missing}")
            per_asset_features.append(df[cols].values.astype(np.float32))
            close_col = f"{tag}_close"
            if close_col not in df.columns:
                raise ValueError(f"Missing close-price column '{close_col}' for asset '{tag}'.")
            per_asset_prices.append(df[close_col].values.astype(np.float32))

        # (T, n_assets, n_features)
        self.features = np.stack(per_asset_features, axis=1)
        # (T, n_assets)
        self.prices = np.stack(per_asset_prices, axis=1)
        self.timestamps = df["timestamp"].values

        #Max steps = data length - window_len - 1 (need at least one step ahead)
        self.max_steps = len(self.features) - self.window_len - 1

        if self.max_steps <= 0:
            raise ValueError(
                f"Not enough data: {len(self.features)} rows, "
                f"need at least {self.window_len + 2}"
            )

    def get_obs(self) -> dict[str, np.ndarray]:
        # Single timestep of market features, stacked per-asset:
        # shape (n_assets, n_features). Temporal context is the
        # recurrent policy's job (via its hidden state across steps),
        # not this method's -- see the observation_space comment in
        # __init__ for why this changed from a flattened window.
        market_step = self.features[self.current_step]

        prices = list(self.prices[self.current_step])
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

        return {
            "market": market_step.astype(np.float32),
            "portfolio": portfolio_vec.astype(np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.deterministic_start:
            # Evaluation: always score the full held-out window,
            # starting right after the first observable window.
            self.current_step = self.window_len
        else:
            # Training: random start for generalization.
            self.current_step = self.np_random.integers(
                self.window_len,
                max(self.window_len + 1, self.max_steps // 2)
            )
        self.portfolio.reset()
        self.current_weights = [0.0] * self.n_assets
        self.holding_time = 0
        self.peak_value = self.portfolio.cash
        self.reward_calc.reset()
        self.position_open = [False] * self.n_assets
        self.entry_price = [0.0] * self.n_assets
        self.entry_atr = [0.0] * self.n_assets
        self.entry_weight = [0.0] * self.n_assets

        obs = self.get_obs()
        prices = list(self.prices[self.current_step])
        info = {
            "step": self.current_step,
            "prices": prices,
            "weights": list(self.current_weights),
            "capital": self.portfolio.total_value(prices),
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

        prices_prev = list(self.prices[self.current_step])

        value_before = self.portfolio.total_value(prices_prev)

        self.current_step += 1

        prices = list(self.prices[self.current_step])

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
        "prices": prices,
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
        prices_prev = list(self.prices[self.current_step])

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
        # Dynamic ATR Position Sizing, computed independently per asset.
        # A short's equity risk for a given price move is `leverage`x a
        # long's, so its ATR budget must be divided by leverage before
        # solving for the multiplier -- otherwise the sizer treats a 3x
        # leveraged short as if it carried the same equity risk as an
        # unleveraged long of the same weight magnitude.
        current_atr_per_asset = [
            float(self.features[self.current_step, i, self.atr_feature_idx])
            for i in range(self.n_assets)
        ]
        leverage = self.portfolio.leverage
        risk_multiplier_long = [
            min(1.0, self.target_atr_pct / max(atr, 1e-6)) for atr in current_atr_per_asset
        ]
        risk_multiplier_short = [
            min(1.0, self.target_atr_pct / max(atr * leverage, 1e-6)) for atr in current_atr_per_asset
        ]
        # Per-asset multiplier actually applied, for logging/diagnostics
        # (previously only asset 0's value was captured, silently
        # dropping every other asset's risk-sizing from the logs).
        risk_multiplier_applied = [
            risk_multiplier_short[i] if new_weights[i] < 0 else risk_multiplier_long[i]
            for i in range(self.n_assets)
        ]
        new_weights = [
            weight * (risk_multiplier_short[i] if weight < 0 else risk_multiplier_long[i])
            for i, weight in enumerate(new_weights)
        ]

        # Capture portfolio value at the *old* prices, before this step's
        # trade and before advancing the market cursor. This is the
        # correct denominator for step_return below — valuing pre-trade
        # holdings at the pre-trade prices.
        value_before = self.portfolio.total_value(prices_prev)

        # 2. Advance market cursor to the next candle.
        self.current_step += 1
        prices = list(self.prices[self.current_step])
        # ATR Stop Loss / Take Profit -- evaluated independently per asset,
        # so e.g. a BTC stop-loss firing doesn't touch an open ETH
        # position, and vice versa.

        forced_exit = False
        exit_reason = None
        per_asset_exit_reason = [None] * self.n_assets
        for i in range(self.n_assets):
            if not self.position_open[i]:
                continue
            price = prices[i]
            entry_price = self.entry_price[i]
            unrealized_return = (price - entry_price) / entry_price
            if self.entry_weight[i] < 0:
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
            effective_leverage = self.portfolio.leverage if self.entry_weight[i] < 0 else 1.0
            equity_impact = unrealized_return * effective_leverage
            stop_loss_pct = (self.stop_loss_multiple * self.entry_atr[i] / 100.0)
            take_profit_pct = (self.take_profit_multiple * self.entry_atr[i] / 100.0)
            if equity_impact <= -stop_loss_pct:
                logger.info(f"ATR Stop Loss Triggered on asset {i} "
                            f"| Entry={entry_price:.2f} | Current={price:.2f}")
                new_weights[i] = 0.0  # Force exit this asset only
                forced_exit = True
                per_asset_exit_reason[i] = "stop_loss"
            elif equity_impact >= take_profit_pct:
                logger.info(f"ATR Take Profit Triggered on asset {i} "
                            f"| Entry={entry_price:.2f} | Current={price:.2f}")
                new_weights[i] = 0.0  # Force exit this asset only
                forced_exit = True
                per_asset_exit_reason[i] = "take_profit"
        if forced_exit:
            # Kept as a single scalar for back-compat with existing
            # consumers of info["exit_reason"]; per-asset detail is in
            # info["exit_reason_per_asset"].
            exit_reason = next(r for r in per_asset_exit_reason if r is not None)

        # 3. Execute the rebalance at the new candle's close.
        weight_delta_before = sum(abs(a - b) for a, b in zip(new_weights, self.current_weights))
        trades = self._rebalance_portfolio(
         new_weights,
         prices,
                         )

        # Detect new positions opening / existing ones fully closing,
        # independently per asset.
        for i in range(self.n_assets):
            weight_i = self.current_weights[i]
            if not self.position_open[i] and abs(weight_i) > 1e-6:
                self.position_open[i] = True
                self.entry_price[i] = prices[i]
                self.entry_atr[i] = float(self.features[self.current_step, i, self.atr_feature_idx])
                self.entry_weight[i] = weight_i
            elif self.position_open[i] and abs(weight_i) < 1e-6:
                self.position_open[i] = False
                self.entry_price[i] = 0.0
                self.entry_atr[i] = 0.0
                self.entry_weight[i] = 0.0

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
            "prices": prices,
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
            # Per-asset lists (index order == self.portfolio.symbols), so
            # nothing beyond asset 0 gets silently dropped from logs/CSVs.
            "risk_multiplier": [float(v) for v in risk_multiplier_applied],
            "risk_multiplier_long": [float(v) for v in risk_multiplier_long],
            "risk_multiplier_short": [float(v) for v in risk_multiplier_short],
            "forced_exit": forced_exit,
            "exit_reason": exit_reason,
            "exit_reason_per_asset": per_asset_exit_reason,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Optional render - not implemented."""
        pass

    def close(self):
        """Cleanup."""
        pass