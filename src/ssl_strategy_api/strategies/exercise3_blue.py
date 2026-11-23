"""Blue team strategies for Exercise 3: the passer and the striker."""
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

from ssl_strategy_api import RobotIntent, TeamConfig, WorldState

# Put your team name here. It is shown at the top of the evaluation window.
TEAM_NAME = "Gradient Defenders"

@dataclass
class Coordinator:
    phase: str = "APPROACH"

    def reset(self) -> None:
        self.phase = "APPROACH"
coordinator = Coordinator()

class PasserStrategy:
    """
    Control blue robot 0 and pass to blue robot 1.

    The strategy is divided into 3 phases:
    - APPROACH:
        - rule-based
    - PASS:
        - one combined SAC-model for passer and striker
    - SHOOT:
    """

    def reset(self, seed=None) -> None:
        coordinator.phase = "APPROACH"

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if coordinator.phase == "APPROACH":
            return self.approach(state, config)
        else:
            return self.idle(config)

    def approach(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        passer_id = config.controlled_robot_ids[0]
        robots = state.robots_blue
        passer = robots.get(passer_id)

        dx = state.ball.x_m - passer.x_m
        dy = state.ball.y_m - passer.y_m
        distance = (dx * dx + dy * dy) ** 0.5  # calc the euclidian distance to the ball position
        target_angle = math.atan2(dy, dx)  # calc the direction vector to the ball
        angle_error = target_angle - passer.heading_rad  # calc the difference in the direction vector and robo's heading
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # choosing the smallest error
        speed_approach = min(2.0, max(0.3, 4.0 * distance))

        if passer.infrared:
            coordinator.phase = "PASS"

        return {passer_id: RobotIntent(
            vx_mps=speed_approach * dx / distance,
            vy_mps=speed_approach * dy / distance,
            angular_velocity_rad_s=max(-3, min(3, 4 * angle_error)),
            dribbler=distance < 0.3)}

    def idle(self, config: TeamConfig) -> Mapping[int, RobotIntent]:
        passer_id = config.controlled_robot_ids[0]
        return {passer_id: RobotIntent()}


class StrikerStrategy(PasserStrategy):
    """
    Control blue robot 1, receive the pass, and shoot.

    The strategy is divided into 3 phases:
    - APPROACH:
        - rule-based
    - PASS:
        - one combined SAC-model for passer and striker
    - SHOOT:
    """

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if coordinator.phase == 0:
            return self.approach(state, config)
        else:
            return self.idle(config)

    def approach(self, state, config: TeamConfig) -> Mapping[int, RobotIntent]:
        return self.idle(config) # TODO get into a good position

    def idle(self, config: TeamConfig) -> Mapping[int, RobotIntent]:
        striker_id = config.controlled_robot_ids[0]
        return {striker_id: RobotIntent()}
