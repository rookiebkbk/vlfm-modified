# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from myon.policy.utils.acyclic_enforcer import AcyclicEnforcer


def test_repeated_state_action_is_cyclic() -> None:
    position = np.array([1.0, 2.0])
    action = np.array([3.0, 4.0])
    values = (0.8, 0.7)
    enforcer = AcyclicEnforcer()

    assert not enforcer.check_cyclic(position, action, values)
    enforcer.add_state_action(position, action, values)
    assert enforcer.check_cyclic(position, action, values)

    position[:] = 0.0
    assert enforcer.check_cyclic(np.array([1.0, 2.0]), action, values)


def test_history_is_not_shared_between_instances() -> None:
    position = np.array([1.0, 2.0])
    action = np.array([3.0, 4.0])
    values = (0.8, 0.7)

    first_enforcer = AcyclicEnforcer()
    first_enforcer.add_state_action(position, action, values)

    second_enforcer = AcyclicEnforcer()
    assert not second_enforcer.check_cyclic(position, action, values)
