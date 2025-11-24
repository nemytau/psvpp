"""Registry utilities for pluggable RL modules.

This module provides decorators and lookup helpers for registering
action spaces, state encoders, and reward functions used by the
`ALNSEnvironment`. It allows new implementations to be added without
modifying the core training loop.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

ACTION_REGISTRY: Dict[str, Type[Any]] = {}
STATE_REGISTRY: Dict[str, Type[Any]] = {}
REWARD_REGISTRY: Dict[str, Type[Any]] = {}

DEFAULT_ACTION_KEY = "op_pair_v1"
DEFAULT_STATE_KEY = "features_v3"
DEFAULT_REWARD_KEY = "delta_cost_v4"

_ActionT = TypeVar("_ActionT")
_StateT = TypeVar("_StateT")
_RewardT = TypeVar("_RewardT")


def register_action(name: str) -> Callable[[Type[_ActionT]], Type[_ActionT]]:
    """Register an ActionSpace implementation.

    Parameters
    ----------
    name:
        Registry key used in configuration files.
    """

    key = name.strip()

    def decorator(cls: Type[_ActionT]) -> Type[_ActionT]:
        ACTION_REGISTRY[key] = cls
        return cls

    return decorator


def register_state(name: str) -> Callable[[Type[_StateT]], Type[_StateT]]:
    """Register a StateEncoder implementation."""

    key = name.strip()

    def decorator(cls: Type[_StateT]) -> Type[_StateT]:
        STATE_REGISTRY[key] = cls
        return cls

    return decorator


def register_reward(name: str) -> Callable[[Type[_RewardT]], Type[_RewardT]]:
    """Register a RewardFn implementation."""

    key = name.strip()

    def decorator(cls: Type[_RewardT]) -> Type[_RewardT]:
        REWARD_REGISTRY[key] = cls
        return cls

    return decorator


def _lookup(name: Optional[str], registry: Dict[str, Type[Any]], kind: str) -> Type[Any]:
    if not registry:
        raise KeyError(f"No implementations registered for {kind}")

    if name is None:
        # When no explicit key is supplied, return the first registered implementation.
        return next(iter(registry.values()))

    key = name.strip()
    try:
        return registry[key]
    except KeyError as exc:  # pragma: no cover - defensive branch
        available = ", ".join(sorted(registry)) or "<none>"
        raise KeyError(
            f"Unknown {kind} '{key}'. Available: {available}"
        ) from exc


def create_action_space(name: Optional[str], **kwargs: Any) -> Any:
    """Instantiate an ActionSpace by registry key."""

    cls = _lookup(name, ACTION_REGISTRY, "action space")
    return cls(**kwargs)


def create_state_encoder(name: Optional[str], **kwargs: Any) -> Any:
    """Instantiate a StateEncoder by registry key."""

    cls = _lookup(name, STATE_REGISTRY, "state encoder")
    return cls(**kwargs)


def create_reward_function(name: Optional[str], **kwargs: Any) -> Any:
    """Instantiate a RewardFn by registry key."""

    cls = _lookup(name, REWARD_REGISTRY, "reward function")
    return cls(**kwargs)


def list_registered_actions() -> Dict[str, Type[Any]]:
    """Return a shallow copy of registered action spaces."""

    return dict(ACTION_REGISTRY)


def list_registered_states() -> Dict[str, Type[Any]]:
    """Return a shallow copy of registered state encoders."""

    return dict(STATE_REGISTRY)


def list_registered_rewards() -> Dict[str, Type[Any]]:
    """Return a shallow copy of registered reward functions."""

    return dict(REWARD_REGISTRY)
