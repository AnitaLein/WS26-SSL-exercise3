"""Blue team strategies for Exercise 3: the passer and the striker."""

from typing import Mapping

from ssl_strategy_api import RobotIntent, TeamConfig, WorldState

# Put your team name here. It is shown at the top of the evaluation window.
TEAM_NAME = "Blue Team"


class PasserStrategy:
    """Control blue robot 0 and pass to blue robot 1."""

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        return {robot_id: RobotIntent() for robot_id in config.controlled_robot_ids}


class StrikerStrategy(PasserStrategy):
    """Control blue robot 1, receive the pass, and shoot."""
