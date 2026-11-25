"""Yellow team strategies for Exercise 3: the goalkeeper and the interceptor."""

from typing import Mapping

from ssl_strategy_api import RobotIntent, TeamConfig, WorldState

import math
import numpy as np

# Put your team name here. It is shown at the top of the evaluation window.
TEAM_NAME = "Yellow Team"

K_P = 3.0          # Verstärkung, ausprobieren
MAX_V = 1.5        # m/s
DEADBAND = 0.02    # 2 cm

class GoalkeeperStrategy:
    """Control yellow robot 0 and defend the goal."""

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        return {robot_id: RobotIntent() for robot_id in config.controlled_robot_ids}


class InterceptorStrategy(GoalkeeperStrategy):
    """Control yellow robot 1 and intercept the pass."""
    
    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids:
                return {}

        robot_id = config.controlled_robot_ids[0]
        robots = state.robots_yellow
        robot = robots.get(robot_id)
        ball = state.ball
        target_y = ball.y_m

        if robot is None:
            print("Robot not found")
            return {}

        # Simple motion example: move the robot in a small random-like direction.
        # Replace this with your own strategy logic.
        
        if ball.vx_mps is not None and abs(ball.vx_mps) > 0.1:
            t = (robot.x_m - ball.x_m) / ball.vx_mps
            if t > 0:
                target_y = ball.y_m + ball.vy_mps * t

        target_y = float(np.clip(target_y, -2.9, 2.9))

        dy = target_y - robot.y_m
        vy = 0.0 if abs(dy) < DEADBAND else float(np.clip(K_P * dy, -MAX_V, MAX_V))
        
        return {
            robot_id: RobotIntent(
                vx_mps=0,
                vy_mps = vy,
                angular_velocity_rad_s=0.0,
            )
        }
