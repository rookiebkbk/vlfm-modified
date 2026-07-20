# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Hashable, Set, Tuple

import numpy as np


def _freeze(value: Any) -> Hashable:
    """Convert supported state values into immutable, hashable values."""
    if isinstance(value, np.ndarray):
        return ("array", value.shape, tuple(value.reshape(-1).tolist()))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class StateAction:
    def __init__(self, position: np.ndarray, action: Any, other: Any = None) -> None:
        self._key: Tuple[Hashable, Hashable, Hashable] = (
            _freeze(position),
            _freeze(action),
            _freeze(other),
        )

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        return self._key == other._key


class AcyclicEnforcer:
    def __init__(self) -> None:
        self.history: Set[StateAction] = set()

    def check_cyclic(self, position: np.ndarray, action: Any, other: Any = None) -> bool:
        state_action = StateAction(position, action, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: Any, other: Any = None) -> None:
        state_action = StateAction(position, action, other)
        self.history.add(state_action)
