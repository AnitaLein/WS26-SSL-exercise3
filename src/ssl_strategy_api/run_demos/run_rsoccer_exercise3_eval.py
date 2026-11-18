"""Evaluate Exercise 3 over 60 simulated seconds."""

import argparse
from collections import Counter
from math import ceil, hypot
import time

import pygame
from rsoccer_gym.Entities import Ball, Frame, Robot

from ssl_strategy_api.helpers import CrashFoulTracker
from ssl_strategy_api.run_demos.run_rsoccer_exercise3 import (
    PASSER_AREA, RESTRICTED_AREAS, STEP_SECONDS, STUCK_SPEED_MPS, Exercise3Env, add_strategy_arguments,
    inside_area, load_strategies, reset_strategies, strategy_commands, team_names,
)

EVALUATION_SECONDS = 60.0
LONG_STATIONARY_SECONDS = 10.0  # ball stationary anywhere on the field
FOUL_INTERVAL_SECONDS = 2.0
FOUL_INTERVAL_STEPS = ceil(FOUL_INTERVAL_SECONDS / STEP_SECONDS)
AREA_COLOR = (220, 30, 30)
BAR_HEIGHT_PX = 30
HEADER_EXTRA_PX = 30  # extra window height added above the field for the team-names line
BLUE_TEXT_COLOR = (110, 170, 255)
YELLOW_TEXT_COLOR = (255, 215, 60)
DASH_LENGTH_PX = 8
DASH_GAP_PX = 12


def draw_area_boundaries(surface, renderer):
    """Draw the evaluation areas in field coordinates, beneath the robots."""
    for min_x, max_x, min_y, max_y in RESTRICTED_AREAS.values():
        color = AREA_COLOR
        x1 = round(min_x * renderer.scale + renderer.center_x)
        x2 = round(max_x * renderer.scale + renderer.center_x)
        y1 = round(min_y * renderer.scale + renderer.center_y)
        y2 = round(max_y * renderer.scale + renderer.center_y)
        for x in (x1, x2):
            for start in range(y1, y2, DASH_LENGTH_PX + DASH_GAP_PX):
                pygame.draw.line(surface, color, (x, start),
                                 (x, min(start + DASH_LENGTH_PX, y2)), 2)
        for y in (y1, y2):
            for start in range(x1, x2, DASH_LENGTH_PX + DASH_GAP_PX):
                pygame.draw.line(surface, color, (start, y),
                                 (min(start + DASH_LENGTH_PX, x2), y), 2)


def scoreboard_labels(stats):
    return ("Blue Team Goals: {}".format(stats["goal"]),
            "Blue Team Fouls: {}".format(stats["blue_team_fouls"]),
            "Yellow Team Fouls: {}".format(stats["yellow_team_fouls"]),
            "Blue Out of Area: {}".format(stats["blue_out_of_area"]),
            "Yellow Out of Area: {}".format(stats["yellow_out_of_area"]),
            "Ball Interceptions: {}".format(stats["interception"]))


class FoulTracker:
    """Count a foul on leaving an area and periodically while outside."""

    def __init__(self):
        self.outside_steps = {}

    def update(self, state, stats):
        for (yellow, robot_id), area in RESTRICTED_AREAS.items():
            robots = state.robots_yellow if yellow else state.robots_blue
            robot = robots.get(robot_id)
            if robot is None:
                raise RuntimeError("cannot count fouls without robot position")
            key = (yellow, robot_id)
            if inside_area(robot.x_m, robot.y_m, area):
                self.outside_steps.pop(key, None)
            else:
                steps = self.outside_steps.get(key)
                if steps is None or steps + 1 >= FOUL_INTERVAL_STEPS:
                    stats["yellow_out_of_area" if yellow else "blue_out_of_area"] += 1
                    steps = 0
                else:
                    steps += 1
                self.outside_steps[key] = steps

    def reset(self):
        self.outside_steps.clear()


class Exercise3EvalEnv(Exercise3Env):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("episode_timeout", False)  # evaluation has no 60 s timeout
        self.scoreboard_stats = Counter()
        self.team_names = ("Blue Team", "Yellow Team")
        self.scoreboard_font = None
        self.team_font = None
        self.long_stuck_steps = 0
        super().__init__(*args, **kwargs)
        # Make room above the field for the team-names line.
        self.window_size = (self.window_size[0], self.window_size[1] + HEADER_EXTRA_PX)

    def _render(self):
        renderer = self.field_renderer
        draw_field = renderer.draw

        def draw_field_and_boundaries(surface):
            draw_field(surface)
            draw_area_boundaries(surface, renderer)

        renderer.draw = draw_field_and_boundaries
        full_window = self.window_surface
        width, height = full_window.get_size()
        # rSoccer draws the field at the top-left of its surface, so draw it into a
        # sub-surface that starts below the header.
        self.window_surface = full_window.subsurface((0, HEADER_EXTRA_PX, width, height - HEADER_EXTRA_PX))
        try:
            super()._render()
        finally:
            self.window_surface = full_window
            del renderer.draw
        if self.scoreboard_font is None:
            self.scoreboard_font = pygame.font.Font(None, 20)
        width = self.window_surface.get_width()
        if self.team_font is None:
            self.team_font = pygame.font.Font(None, 26)
        pygame.draw.rect(self.window_surface, (25, 30, 35), (0, 0, width, BAR_HEIGHT_PX))
        parts = [
            self.team_font.render("Blue Team ({})".format(self.team_names[0]), True, BLUE_TEXT_COLOR),
            self.team_font.render("  -  ", True, (255, 255, 255)),
            self.team_font.render("Yellow Team ({})".format(self.team_names[1]), True, YELLOW_TEXT_COLOR),
        ]
        x = (width - sum(part.get_width() for part in parts)) // 2
        for part in parts:
            self.window_surface.blit(part, (x, (BAR_HEIGHT_PX - part.get_height()) // 2))
            x += part.get_width()
        pygame.draw.rect(self.window_surface, (25, 30, 35), (0, BAR_HEIGHT_PX, width, BAR_HEIGHT_PX))
        column_width = width // len(scoreboard_labels(self.scoreboard_stats))
        for index, label in enumerate(scoreboard_labels(self.scoreboard_stats)):
            text = self.scoreboard_font.render(label, True, (255, 255, 255))
            self.window_surface.blit(text, (index * column_width + 8,
                                            BAR_HEIGHT_PX + (BAR_HEIGHT_PX - text.get_height()) // 2))

    def reset(self, *, seed=None, options=None):
        self.long_stuck_steps = 0
        return super().reset(seed=seed, options=options)

    def _calculate_reward_and_done(self):
        reward, done = super()._calculate_reward_and_done()
        ball = self.state.ball
        if self.reason is None:
            stationary = hypot(ball.vx_mps, ball.vy_mps) < STUCK_SPEED_MPS
            self.long_stuck_steps = self.long_stuck_steps + 1 if stationary else 0
            if self.long_stuck_steps > ceil(LONG_STATIONARY_SECONDS / STEP_SECONDS):
                self.reason = "stationary"
        return reward, self.reason is not None

    def step(self, commands):
        # Render after the runner has counted the current step's events.
        render_mode = self.render_mode
        if render_mode == "human":
            self.render_mode = None
        try:
            return super().step(commands)
        finally:
            self.render_mode = render_mode

    def respawn_ball(self):
        """Replace the ball while preserving robot poses; rSoccer resets velocities."""
        frame = Frame()
        for yellow in (False, True):
            source = self.frame.robots_yellow if yellow else self.frame.robots_blue
            target = frame.robots_yellow if yellow else frame.robots_blue
            for robot_id in sorted(source):
                robot = source[robot_id]
                target[robot_id] = Robot(yellow=yellow, id=robot_id, x=robot.x, y=robot.y,
                                         theta=robot.theta)
        robots = list(frame.robots_blue.values()) + list(frame.robots_yellow.values())
        clearance = self.field.rbt_radius + self.field.ball_radius
        passer = frame.robots_blue[0]
        point = self._sample_ball_near_passer((passer.x, passer.y), robots)
        if point is None:
            # The passer is outside its area (or boxed in): fall back to a random point in the area.
            for _ in range(5000):
                x, y = self._sample_point(PASSER_AREA)
                if all(hypot(x - robot.x, y - robot.y) >= clearance for robot in robots):
                    point = (x, y)
                    break
            else:
                raise RuntimeError("could not place the evaluation ball safely")
        frame.ball = Ball(x=point[0], y=point[1], v_x=0.0, v_y=0.0)
        self.rsim.reset(frame)
        self.last_frame = None
        self.frame = self.rsim.get_frame()
        self.reason = None
        self.stuck_steps = 0
        self.long_stuck_steps = 0
        self.interception_steps = 0
        self.episode_start_step = self.steps
        return self._frame_to_observations()


def run_evaluation(strategies, seed=None, render=False):
    stats = Counter()
    fouls = FoulTracker()
    crashes = CrashFoulTracker(STEP_SECONDS)
    env = Exercise3EvalEnv(seed=seed, render_mode="human" if render else None,
                           stop_on_blue_goal_area=False)
    env.scoreboard_stats = stats
    env.team_names = team_names(strategies)
    try:
        state, _ = env.reset(seed=seed)
        reset_strategies(strategies, seed)
        started = time.monotonic()
        max_steps = ceil(EVALUATION_SECONDS / STEP_SECONDS)
        for step in range(1, max_steps + 1):
            state, _, terminated, truncated, info = env.step(strategy_commands(state, env, strategies))
            fouls.update(state, stats)
            blue_crashes, yellow_crashes = crashes.update(state)
            stats["blue_team_fouls"] += blue_crashes
            stats["yellow_team_fouls"] += yellow_crashes
            if terminated or truncated:
                stats[info["reason"]] += 1
                if step < max_steps:
                    state = env.respawn_ball()
                    fouls.reset()
            if render:
                env.render()
                delay = started + step * STEP_SECONDS - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        env.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Exercise 3 evaluation (60 simulated seconds)")
    add_strategy_arguments(parser)
    args = parser.parse_args()
    stats = run_evaluation(load_strategies(args), seed=args.seed, render=args.render)
    print("Goals: {}; blue team fouls: {}; yellow team fouls: {}; blue out of area: {}; "
          "yellow out of area: {}; interceptions: {}; own goals: {}; "
          "out of bounds: {}; stationary: {}".format(
              stats["goal"], stats["blue_team_fouls"], stats["yellow_team_fouls"],
              stats["blue_out_of_area"], stats["yellow_out_of_area"],
              stats["interception"], stats["own_goal"],
              stats["out_of_bounds"], stats["stationary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
