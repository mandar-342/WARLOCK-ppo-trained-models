"""
Multi-asset features extractor for the recurrent PPO policy.

The env now emits a Dict observation:
    {"market": (n_assets, n_features), "portfolio": (portfolio_vec_dim,)}

Rather than flattening the market block into one long vector (which would
force the policy to relearn, separately for every asset slot, that e.g. an
RSI of 70 means the same thing for BTC and ETH), this extractor applies a
single shared-weight per-asset encoder to each asset's feature slice, then
concatenates the resulting per-asset embeddings (not pooled/averaged --
with only 2 assets, pooling would blur exactly the distinction the policy
needs to act on, since BTC and ETH can and should get different position
decisions) together with the flat portfolio vector. The concatenated
result feeds into RecurrentPPO's LSTM exactly as the old flat observation
did.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiAssetFeaturesExtractor(BaseFeaturesExtractor):
    """Shared per-asset MLP encoder + concatenation, for Dict obs spaces
    shaped like {"market": (n_assets, n_features), "portfolio": (k,)}."""

    def __init__(self, observation_space: gym.spaces.Dict, encoder_hidden_size: int = 64):
        market_space = observation_space["market"]
        portfolio_space = observation_space["portfolio"]
        self.n_assets, self.n_features = market_space.shape
        portfolio_dim = portfolio_space.shape[0]
        self.encoder_hidden_size = encoder_hidden_size

        features_dim = self.n_assets * encoder_hidden_size + portfolio_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Shared weights across assets: same Linear applied to every
        # asset's (n_features,) slice via a batched matmul over the
        # asset dimension, not a separate Linear per asset.
        self.asset_encoder = nn.Sequential(
            nn.Linear(self.n_features, encoder_hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        market = observations["market"]        # (batch, n_assets, n_features)
        portfolio = observations["portfolio"]  # (batch, portfolio_dim)

        batch = market.shape[0]
        flat = market.reshape(batch * self.n_assets, self.n_features)
        encoded = self.asset_encoder(flat)                       # (batch*n_assets, hidden)
        encoded = encoded.reshape(batch, self.n_assets * self.encoder_hidden_size)

        return torch.cat([encoded, portfolio], dim=1)