"""Yellow team strategies for Exercise 3: the goalkeeper and the interceptor."""

from typing import Mapping

from ssl_strategy_api import RobotIntent, TeamConfig, WorldState

# Put your team name here. It is shown at the top of the evaluation window.
TEAM_NAME = "Yellow Team"


class GoalkeeperStrategy:
    """Control yellow robot 0 and defend the goal."""

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        return {robot_id: RobotIntent() for robot_id in config.controlled_robot_ids}


class InterceptorStrategy(GoalkeeperStrategy):
    """Control yellow robot 1 and intercept the pass."""
