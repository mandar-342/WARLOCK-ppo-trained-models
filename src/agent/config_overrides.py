from __future__ import annotations

"""
Utilities for temporarily overriding values in the shared, module-level
`config` dict (src/utils/config.py loads config.yaml once at import time
and every module holds a reference to that *same* dict object).

Because every module (`PPOTrainer`, `GymBitcoinEnv`, `Evaluator`, ...)
reads config values live at construction time rather than caching them
at import time, mutating this dict in place is enough to change
behaviour for the next run -- no need to touch config.yaml on disk or
reload modules. This is what lets a single process run many seeds/trials
back-to-back with different hyperparameters.

This is NOT safe across multiple processes/threads mutating the same
dict concurrently. The multi-seed harness and Optuna sweep in this
package are intentionally sequential (one training run at a time) for
that reason -- see the `n_jobs` note in hpo.py.
"""

from typing import Any

from src.utils import config

_MISSING = object()


def _get_path(d: dict, path: list[str]) -> Any:
    node = d
    for key in path:
        node = node[key]
    return node


def _set_path(d: dict, path: list[str], value: Any) -> None:
    node = d
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def apply_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Applies dotted-key overrides to the global config, e.g.
    {"ppo.learning_rate": 1e-4, "env.max_drawdown": 0.25}.

    Returns a snapshot of the previous values (also dotted-key) so the
    caller can restore the config afterwards via `restore_overrides`.
    """

    previous: dict[str, Any] = {}

    for dotted_key, value in overrides.items():
        path = dotted_key.split(".")

        try:
            previous[dotted_key] = _get_path(config, path)
        except KeyError:
            previous[dotted_key] = _MISSING

        _set_path(config, path, value)

    return previous


def restore_overrides(previous: dict[str, Any]) -> None:
    """
    Restores config values captured by `apply_overrides`. Keys that did
    not exist before the override are removed again.
    """

    for dotted_key, value in previous.items():
        path = dotted_key.split(".")

        if value is _MISSING:
            node = config
            for key in path[:-1]:
                node = node.get(key, {})
            node.pop(path[-1], None)
            continue

        _set_path(config, path, value)


class override_config:
    """
    Context manager wrapper around apply_overrides/restore_overrides.

    Usage
    -----
    with override_config({"ppo.learning_rate": 1e-4}):
        ...  # build/train inside here
    """

    def __init__(self, overrides: dict[str, Any] | None) -> None:
        self._overrides = overrides or {}
        self._previous: dict[str, Any] = {}

    def __enter__(self) -> "override_config":
        self._previous = apply_overrides(self._overrides)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        restore_overrides(self._previous)