"""SSL "Crashing" foul (RoboCup SSL rules, section 8.4.2) for robots of different teams.

At the moment two robots of different teams collide, the difference of their
velocity vectors is projected onto the line through both robot centres:

* if the length of that projection is at most 1.5 m/s, nothing happens;
* if it is greater than 1.5 m/s, the faster robot commits a foul;
* if, in that case, their speeds differ by less than 0.3 m/s, both commit a foul.

Like all non-stopping fouls, the same crash cannot be counted again until the
robots separate, or 2 seconds after it was first counted.
"""

from math import hypot
from typing import Dict, FrozenSet, Optional, Tuple

from ssl_strategy_api.helpers.safety_checks import ROBOT_RADIUS_M, find_collision
from ssl_strategy_api.models import RobotState, WorldState

CRASH_PROJECTED_SPEED_MPS = 1.5
CRASH_EQUAL_SPEED_MPS = 0.3
REPEAT_FOUL_SECONDS = 2.0


def crash_fouls(first: RobotState, second: RobotState) -> Tuple[bool, bool]:
    """Return ``(first_at_fault, second_at_fault)`` for a collision at these velocities."""
    values = (first.vx_mps, first.vy_mps, second.vx_mps, second.vy_mps)
    if any(value is None for value in values):
        return False, False
    dx, dy = second.x_m - first.x_m, second.y_m - first.y_m
    distance = hypot(dx, dy)
    if distance < 1e-9:
        return False, False
    projected = abs((first.vx_mps - second.vx_mps) * dx / distance
                    + (first.vy_mps - second.vy_mps) * dy / distance)
    if projected <= CRASH_PROJECTED_SPEED_MPS:
        return False, False
    first_speed = hypot(first.vx_mps, first.vy_mps)
    second_speed = hypot(second.vx_mps, second.vy_mps)
    if abs(first_speed - second_speed) < CRASH_EQUAL_SPEED_MPS:
        return True, True
    return first_speed > second_speed, second_speed > first_speed


class CrashFoulTracker:
    """Count crash fouls per team from successive world states.

    Velocities from the previous step are used at the moment of contact,
    because the simulator has already applied the impact by the time the
    contact is seen.
    """

    def __init__(self, step_seconds: float, radius_m: float = ROBOT_RADIUS_M) -> None:
        self.radius_m = radius_m
        self.repeat_steps = round(REPEAT_FOUL_SECONDS / step_seconds)
        self.reset()

    def reset(self) -> None:
        self._previous: Dict[Tuple[str, int], RobotState] = {}
        self._touching = set()
        self._counted_at: Dict[FrozenSet, int] = {}
        self._step = 0

    def update(self, state: WorldState) -> Tuple[int, int]:
        """Return the (blue, yellow) fouls counted on this step."""
        self._step += 1
        blue = [(("blue", i), r) for i, r in state.robots_blue.items()]
        yellow = [(("yellow", i), r) for i, r in state.robots_yellow.items()]
        fouls = {"blue": 0, "yellow": 0}
        touching = set()
        for key_b, robot_b in blue:
            for key_y, robot_y in yellow:
                if find_collision(robot_b, [robot_y], self.radius_m) is None:
                    continue
                pair = frozenset((key_b, key_y))
                touching.add(pair)
                if pair not in self._touching:
                    # New contact: use the velocities from just before the impact.
                    before_b = self._previous.get(key_b, robot_b)
                    before_y = self._previous.get(key_y, robot_y)
                    self._counted_at.pop(pair, None)
                    self._count(pair, before_b, before_y, key_b, key_y, fouls)
                elif pair in self._counted_at and (
                        self._step - self._counted_at[pair] >= self.repeat_steps):
                    self._count(pair, robot_b, robot_y, key_b, key_y, fouls)
        for pair in set(self._counted_at) - touching:
            del self._counted_at[pair]
        self._touching = touching
        self._previous = dict(blue + yellow)
        return fouls["blue"], fouls["yellow"]

    def _count(self, pair, robot_b, robot_y, key_b, key_y, fouls) -> None:
        blue_at_fault, yellow_at_fault = crash_fouls(robot_b, robot_y)
        if blue_at_fault or yellow_at_fault:
            self._counted_at[pair] = self._step
        fouls["blue"] += blue_at_fault
        fouls["yellow"] += yellow_at_fault
