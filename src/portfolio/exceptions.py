"""
Exception hierarchy for the portfolio subsystem.
"""

from __future__ import annotations

class PortfolioError(Exception):
    """Base class for all portfolio subsystem errors."""

class InsufficientCashError(PortfolioError):
    """Raised when a buy order requires more cash than is available."""

class InsufficientPositionError(PortfolioError):
    """Raised when a sell order requires more of an asset than is held."""

class UnknownAssetError(PortfolioError):
    """Raised when an operation references a symbol not tracked by the portfolio."""

class InvalidActionError(PortfolioError):
    """
    Raised when an action/target-allocation vector is malformed.

    Examples: wrong length, negative weights, weights summing above 1.0
    beyond the configured tolerance.
    """

class InvalidConfigError(PortfolioError):
    """Raised when the `portfolio` section of config.yaml is missing or malformed."""