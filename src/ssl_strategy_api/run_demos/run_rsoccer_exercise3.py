"""Exercise 3: passing, shooting and defending in random episodes."""

import argparse
from itertools import count
from math import ceil, cos, hypot, isfinite, pi, sin
import random
import sys
import time

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands, world_state_from_rsoccer_frame
from ssl_strategy_api.helpers import is_in_opponent_goal_area
from ssl_strategy_api.run_demos.run_rsoccer_exercise1 import _load_strategy

STEP_SECONDS = 0.025
EPISODE_SECONDS = 60.0
STUCK_SECONDS = 2.0
STUCK_SPEED_MPS = 0.06
POSSESSION_DISTANCE_M = 0.15
POSSESSION_RELATIVE_SPEED_MPS = 0.4
POSSESSION_STEPS = 10
# The stationary-ball rule only applies when x < -3.5 or x > 3.5 (metres).
STATIONARY_X_LIMIT = 3.5
BALL_MAX_DISTANCE_M = 0.3

# These are centre-coordinate bounds, not physical walls.
PASSER_AREA = (-3.5, -1.5, -3.0, 3.0)
STRIKER_AREA = (1.5, 3.5, -3.0, 3.0)
INTERCEPTOR_AREA = (-1.5, 1.5, -3.0, 3.0)
GOALKEEPER_AREA = (3.5, 4.7, -3.0, 3.0)
PASSER_SPAWN = PASSER_AREA
STRIKER_SPAWN = STRIKER_AREA
RESTRICTED_AREAS = {(False, 0): PASSER_AREA, (False, 1): STRIKER_AREA,
                    (True, 0): GOALKEEPER_AREA, (True, 1): INTERCEPTOR_AREA}


def inside_area(x, y, area):
    return area[0] < x < area[1] and area[2] < y < area[3]


class Exercise3Env(SSLBaseEnv):
    """Random four-robot episodes; actions are rSoccer robot commands."""

    def __init__(self, seed=None, render_mode=None, stop_on_blue_goal_area=True,
                 episode_timeout=True):
        self.rng = random.Random(seed)
        self.stop_on_blue_goal_area = stop_on_blue_goal_area
        self.episode_timeout = episode_timeout
        self.passer_config = TeamConfig(TeamColor.BLUE, (0,), 1)
        self.striker_config = TeamConfig(TeamColor.BLUE, (1,), 1)
        self.goalkeeper_config = TeamConfig(TeamColor.YELLOW, (0,), -1)
        self.interceptor_config = TeamConfig(TeamColor.YELLOW, (1,), -1)
        self.state = None
        self.reason = None
        self.stuck_steps = 0
        self.episode_start_step = 0
        self.interception_steps = 0
        super().__init__(1, 2, 2, STEP_SECONDS, render_mode=render_mode)
        self.canonical_field = FieldGeometry(self.field.length, self.field.width, self.field.goal_width)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.reason = None
        self.stuck_steps = 0
        self.episode_start_step = 0
        self.interception_steps = 0
        return super().reset(seed=seed, options=options)

    def _sample_point(self, area):
        epsilon = 1e-6
        return (self.rng.uniform(area[0] + epsilon, area[1] - epsilon),
                self.rng.uniform(area[2] + epsilon, area[3] - epsilon))

    def _sample_ball_near_passer(self, passer, robots):
        """Random ball point within BALL_MAX_DISTANCE_M of the passer, inside the passer
        area and not overlapping any robot; None if no valid point is found."""
        clearance = self.field.rbt_radius + self.field.ball_radius
        for _ in range(500):
            angle = self.rng.uniform(-pi, pi)
            distance = self.rng.uniform(clearance + 0.01, BALL_MAX_DISTANCE_M)
            x, y = passer[0] + distance * cos(angle), passer[1] + distance * sin(angle)
            if (inside_area(x, y, PASSER_AREA)
                    and all(hypot(x - r.x, y - r.y) >= clearance for r in robots)):
                return x, y
        return None

    def _get_initial_positions_frame(self):
        for _ in range(1000):
            frame = Frame()
            passer = self._sample_point(PASSER_SPAWN)
            striker = self._sample_point(STRIKER_SPAWN)
            interceptor = self._sample_point(INTERCEPTOR_AREA)
            frame.robots_blue[0] = Robot(yellow=False, id=0, x=passer[0], y=passer[1], theta=0.0)
            frame.robots_blue[1] = Robot(yellow=False, id=1, x=striker[0], y=striker[1], theta=0.0)
            frame.robots_yellow[0] = Robot(yellow=True, id=0, x=4.3, y=0.0, theta=180.0)
            frame.robots_yellow[1] = Robot(yellow=True, id=1, x=interceptor[0], y=interceptor[1], theta=180.0)
            robots = list(frame.robots_blue.values()) + list(frame.robots_yellow.values())
            if any(hypot(a.x - b.x, a.y - b.y) <= 2 * self.field.rbt_radius
                   for i, a in enumerate(robots) for b in robots[i + 1:]):
                continue
            point = self._sample_ball_near_passer(passer, robots)
            if point is not None:
                frame.ball = Ball(x=point[0], y=point[1], v_x=0.0, v_y=0.0)
                return frame
        raise RuntimeError("could not sample a valid Exercise 3 layout")

    def _get_commands(self, commands):
        return commands

    def _frame_to_observations(self):
        self.state = world_state_from_rsoccer_frame(self.frame, self.canonical_field, frame_number=self.steps)
        return self.state

    def _calculate_reward_and_done(self):
        ball = self.state.ball
        if ball is None or ball.vx_mps is None or ball.vy_mps is None:
            raise RuntimeError("Exercise 3 requires the ball position and velocity")
        robot = self.state.robots_yellow.get(1)
        if robot is None or robot.vx_mps is None or robot.vy_mps is None:
            raise RuntimeError("Exercise 3 requires the interceptor position and velocity")
        has_ball = (hypot(ball.x_m - robot.x_m, ball.y_m - robot.y_m) < POSSESSION_DISTANCE_M
                    and hypot(ball.vx_mps - robot.vx_mps, ball.vy_mps - robot.vy_mps)
                    <= POSSESSION_RELATIVE_SPEED_MPS)
        self.interception_steps = self.interception_steps + 1 if has_ball else 0

        self.reason = None
        if abs(ball.x_m) >= self.field.length / 2 and abs(ball.y_m) <= self.field.goal_width / 2:
            self.reason = "goal" if ball.x_m > 0 else "own_goal"
        elif abs(ball.x_m) >= self.field.length / 2 or abs(ball.y_m) >= self.field.width / 2:
            self.reason = "out_of_bounds"
        elif self.stop_on_blue_goal_area and any(
            is_in_opponent_goal_area(robot.x_m, robot.y_m, self.field,
                                     self.passer_config.attack_direction_x)
            for robot in self.state.robots_blue.values()
        ):
            self.reason = "blue_in_opponent_goal_area"
        elif self.interception_steps >= POSSESSION_STEPS:
            self.reason = "interception"
        else:
            outside = ball.x_m < -STATIONARY_X_LIMIT or ball.x_m > STATIONARY_X_LIMIT
            stationary = hypot(ball.vx_mps, ball.vy_mps) < STUCK_SPEED_MPS
            self.stuck_steps = self.stuck_steps + 1 if outside and stationary else 0
            if self.stuck_steps >= ceil(STUCK_SECONDS / STEP_SECONDS):
                self.reason = "stationary"
        # Students implement their own reward function here.
        return 0.0, self.reason is not None

    def step(self, commands):
        state, reward, terminated, truncated, info = super().step(commands)
        if (self.episode_timeout and not terminated
                and self.steps - self.episode_start_step >= ceil(EPISODE_SECONDS / STEP_SECONDS)):
            truncated = True
            self.reason = "timeout"
        info["reason"] = self.reason
        return state, reward, terminated, truncated, info


def add_strategy_arguments(parser):
    for role in ("passer", "striker", "goalkeeper", "interceptor"):
        team = "blue" if role in ("passer", "striker") else "yellow"
        parser.add_argument("--{}-strategy".format(role),
                            default="ssl_strategy_api.strategies.exercise3_{}:{}Strategy".format(
                                team, role.capitalize()))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)


def load_strategies(args):
    return [_load_strategy(getattr(args, role + "_strategy"))
            for role in ("passer", "striker", "goalkeeper", "interceptor")]


def team_names(strategies, max_length=24):
    """Return (blue, yellow) team names from the ``TEAM_NAME`` of the strategies' modules.

    ``strategies`` is ordered passer, striker, goalkeeper, interceptor. The first
    strategy of each team whose module defines a non-empty ``TEAM_NAME`` wins.
    """
    names = []
    for default, candidates in (("Blue Team", strategies[0:2]), ("Yellow Team", strategies[2:4])):
        name = default
        for strategy in candidates:
            value = getattr(sys.modules.get(type(strategy).__module__), "TEAM_NAME", None)
            if isinstance(value, str) and value.strip():
                name = value.strip()[:max_length]
                break
        names.append(name)
    return tuple(names)


def reset_strategies(strategies, seed):
    for strategy in strategies:
        reset = getattr(strategy, "reset", None)
        if reset is not None:
            reset(seed=seed)


def strategy_commands(state, env, strategies):
    commands = []
    configs = (env.passer_config, env.striker_config, env.goalkeeper_config, env.interceptor_config)
    for strategy, config in zip(strategies, configs):
        commands.extend(to_rsoccer_commands(state, config, strategy.decide(state, config)))
    return commands


def main():
    parser = argparse.ArgumentParser(description="Exercise 3 training episodes")
    add_strategy_arguments(parser)
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="Run for this many simulated seconds; default: until interrupted")
    args = parser.parse_args()
    if args.duration_seconds is not None and (not isfinite(args.duration_seconds) or args.duration_seconds <= 0):
        parser.error("--duration-seconds must be positive and finite")
    strategies = load_strategies(args)
    env = Exercise3Env(seed=args.seed, render_mode="human" if args.render else None)
    try:
        state, _ = env.reset(seed=args.seed)
        reset_strategies(strategies, args.seed)
        start = time.monotonic()
        steps = count(1) if args.duration_seconds is None else range(1, ceil(args.duration_seconds / STEP_SECONDS) + 1)
        for step in steps:
            state, _, terminated, truncated, info = env.step(strategy_commands(state, env, strategies))
            if terminated or truncated:
                print("Episode ended: {}".format(info["reason"]), flush=True)
                state, _ = env.reset()
                reset_strategies(strategies, None)
            if args.render:
                delay = start + step * STEP_SECONDS - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print("Demo interrupted.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())