"""Automated benchmark suite for Yellow 1 (InterceptorStrategy) in Exercise 3.

Usage:
    python -m ssl_strategy_api.training.run_test_interceptor [OPTIONS]

Options:
    --scenario {all,straight,diagonal,fake_dribble,lead_pass,endurance_60s}
        Test scenario to run (default: all).
        - straight      : Passer and striker at similar y; 3.0-4.0 m/s pass after 2.5 s.
        - diagonal      : Opposite sides of the field; 4.5 m/s cross-field pass after 2.5 s.
        - fake_dribble  : Slow angled roll (vx=0.15, vy=+/-0.8) at 0.5 s before real pass at 2.5 s.
        - lead_pass     : Dummy striker moves vertically (1.5 m/s); pass aimed 1.5 m ahead.
        - endurance_60s : 60 s continuous run with ball respawns and no robot/strategy reset.
        - all           : Runs all 5 scenarios in sequence.

    --episodes INT
        Number of trials per scenario (default: 100).
        For endurance_60s, this is the number of 60 s matches (capped at 10).

    --seed INT
        Base random seed for reproducibility; episode i uses seed + i (default: 42).

    --render
        Open the rSoccer visualization window to watch trials live (default: off).

Output Metrics:
    Intrcpt : Official interceptions (<0.15 m & <=0.4 m/s rel. speed for 10 steps).
    Deflect : Touched the ball (<0.18 m) but failed to hold for 10 steps.
    Through : Ball crossed x = 1.5 m into the striker zone.
    OoA     : Out-of-area entries (|x| > 1.5 m or |y| > 3.0 m).
    Crash   : Collisions with Blue robots at >1.5 m/s relative speed.
    Avg T   : Mean time from pass release to official interception.

Examples:
    # Full headless benchmark (100 trials per scenario)
    python -m ssl_strategy_api.training.run_test_interceptor --scenario all --episodes 100

    # Watch 3 diagonal pass trials live
    python -m ssl_strategy_api.training.run_test_interceptor --scenario diagonal --episodes 3 --render

    # Run 1 endurance match with custom seed
    python -m ssl_strategy_api.training.run_test_interceptor --scenario endurance_60s --episodes 1 --seed 100 --render
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState
from ssl_strategy_api.run_demos.run_rsoccer_exercise3 import Exercise3Env
from ssl_strategy_api.strategies.exercise3_yellow import InterceptorStrategy

DT = 0.025  # 40 Hz
PREP_STEPS = int(2.5 / DT)  # 2.5 s positioning window before the pass
MAX_EPISODE_STEPS = int(8.0 / DT)  # 8.0 s per single-pass trial
ENDURANCE_STEPS = int(60.0 / DT)  # 60 s tournament-style endurance test

INTERCEPT_DIST_M = 0.15
INTERCEPT_REL_V_MPS = 0.4
INTERCEPT_HOLD_STEPS = 10
TOUCH_DIST_M = 0.18
CRASH_REL_V_MPS = 1.5
INTERCEPTOR_X_LIMIT = 1.5
FIELD_Y_LIMIT = 3.0

BLUE_CFG = TeamConfig(TeamColor.BLUE, (0, 1), 1)
GOALIE_CFG = TeamConfig(TeamColor.YELLOW, (0,), -1)
INTERCEPTOR_CFG = TeamConfig(TeamColor.YELLOW, (1,), -1)


@dataclass
class ScenarioStats:
    trials: int = 0
    official_interceptions: int = 0
    deflected_only: int = 0
    passed_through: int = 0
    out_of_bounds: int = 0
    out_of_area_events: int = 0
    crash_fouls: int = 0
    intercept_times_s: list[float] = field(default_factory=list)

    def summary_row(self, name: str) -> str:
        n = max(self.trials, 1)
        avg_t = (
            f"{np.mean(self.intercept_times_s):.2f}s"
            if self.intercept_times_s
            else "  -  "
        )
        return (
            f"{name:<15} | {self.trials:>4} | "
            f"{100 * self.official_interceptions / n:>6.1f}% | "
            f"{100 * self.deflected_only / n:>6.1f}% | "
            f"{100 * self.passed_through / n:>6.1f}% | "
            f"{self.out_of_area_events:>4} | "
            f"{self.crash_fouls:>5} | {avg_t:>6}"
        )


def _get_rsim(env: Exercise3Env):
    sim = getattr(env, "rsim", None) or getattr(env.unwrapped, "rsim", None)
    if sim is None:
        raise RuntimeError("Could not access rsim on Exercise3Env.")
    return sim


def _set_ball(env: Exercise3Env, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
    """Place the ball and set its velocity via RSimSSL frame reset."""
    sim = _get_rsim(env)
    frame = sim.get_frame()
    frame.ball.x = float(x)
    frame.ball.y = float(y)
    frame.ball.v_x = float(vx)
    frame.ball.v_y = float(vy)
    sim.reset(frame)
    if hasattr(env, "frame"):
        env.frame = sim.get_frame()


def _set_robot(
    env: Exercise3Env,
    is_blue: bool,
    robot_id: int,
    x: float,
    y: float,
    theta: float = 0.0,
) -> None:
    """Place a robot at a specific pose via RSimSSL frame reset."""
    sim = _get_rsim(env)
    frame = sim.get_frame()
    robots = frame.robots_blue if is_blue else frame.robots_yellow
    if robot_id in robots:
        r = robots[robot_id]
        r.x = float(x)
        r.y = float(y)
        r.theta = float(theta)
        r.v_x = 0.0
        r.v_y = 0.0
        r.v_theta = 0.0
        sim.reset(frame)
        if hasattr(env, "frame"):
            env.frame = sim.get_frame()


def _is_out_of_area(x: float, y: float) -> bool:
    return abs(x) > INTERCEPTOR_X_LIMIT or abs(y) > FIELD_Y_LIMIT


def _compute_pass_velocity(
    bx: float, by: float, tx: float, ty: float, speed: float
) -> tuple[float, float]:
    dx, dy = tx - bx, ty - by
    dist = max(math.hypot(dx, dy), 1e-6)
    return speed * dx / dist, speed * dy / dist


def _step_env(
    env: Exercise3Env,
    state: WorldState,
    interceptor: InterceptorStrategy,
    striker_vy: float = 0.0,
) -> tuple[WorldState, bool, dict]:
    """Run one 40 Hz step with only Yellow 1 active (plus optional dummy striker motion)."""
    commands = []
    commands.extend(
        to_rsoccer_commands(state, INTERCEPTOR_CFG, interceptor.decide(state, INTERCEPTOR_CFG))
    )
    commands.extend(to_rsoccer_commands(state, GOALIE_CFG, {0: RobotIntent()}))
    commands.extend(
        to_rsoccer_commands(
            state,
            BLUE_CFG,
            {
                0: RobotIntent(),
                1: RobotIntent(vx_mps=0.0, vy_mps=striker_vy),
            },
        )
    )

    next_state, _, terminated, truncated, info = env.step(commands)
    return next_state, (terminated or truncated), info


def run_single_pass_scenario(
    env: Exercise3Env,
    interceptor: InterceptorStrategy,
    scenario: str,
    episodes: int,
    base_seed: int,
) -> ScenarioStats:
    stats = ScenarioStats()

    for ep in range(episodes):
        rng = np.random.default_rng(base_seed + ep)
        state, _ = env.reset(seed=int(base_seed + ep))
        interceptor.reset(seed=int(base_seed + ep))

        passer_x = float(rng.uniform(-3.0, -1.8))
        striker_x = float(rng.uniform(1.8, 3.0))
        striker_vy = 0.0

        if scenario == "straight":
            y_base = float(rng.uniform(-2.0, 2.0))
            passer_y = y_base
            striker_y = float(np.clip(y_base + rng.uniform(-0.3, 0.3), -2.5, 2.5))
            pass_speed = float(rng.uniform(3.0, 4.0))
        elif scenario == "diagonal":
            side = 1.0 if rng.random() < 0.5 else -1.0
            passer_y = side * float(rng.uniform(1.5, 2.5))
            striker_y = -side * float(rng.uniform(1.5, 2.5))
            pass_speed = 4.5
        elif scenario == "fake_dribble":
            passer_y = float(rng.uniform(-1.2, 1.2))
            striker_y = float(rng.uniform(-2.2, 2.2))
            pass_speed = float(rng.uniform(3.5, 4.5))
        elif scenario == "lead_pass":
            passer_y = float(rng.uniform(-1.5, 1.5))
            striker_y = float(rng.uniform(-2.0, 0.0))
            striker_vy = 1.5
            pass_speed = float(rng.uniform(3.5, 4.2))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        ball_x, ball_y = passer_x + 0.14, passer_y
        _set_robot(env, True, 0, passer_x, passer_y, 0.0)
        _set_robot(env, True, 1, striker_x, striker_y, math.pi)
        _set_robot(
            env,
            False,
            1,
            float(rng.uniform(-0.8, 0.8)),
            float(rng.uniform(-1.5, 1.5)),
            math.pi,
        )
        _set_ball(env, ball_x, ball_y, 0.0, 0.0)

        touched = False
        hold_steps = 0
        was_out = False
        pass_fired = False
        pass_step = PREP_STEPS
        outcome = "timeout"

        for step in range(MAX_EPISODE_STEPS):
            if scenario == "fake_dribble" and step == int(0.5 / DT):
                fake_vy = 0.8 if striker_y < passer_y else -0.8
                _set_ball(env, state.ball.x_m, state.ball.y_m, 0.15, fake_vy)

            if step == pass_step:
                bx, by = state.ball.x_m, state.ball.y_m
                striker = state.robots_blue.get(1)
                tx = striker.x_m if striker else striker_x
                ty = striker.y_m if striker else striker_y
                if scenario == "lead_pass":
                    ty = float(np.clip(ty + 1.5, -2.6, 2.6))
                vx, vy = _compute_pass_velocity(bx, by, tx, ty, pass_speed)
                _set_ball(env, bx, by, vx, vy)
                pass_fired = True

            active_striker_vy = (
                striker_vy if (scenario == "lead_pass" and step >= pass_step - 20) else 0.0
            )
            state, _, _ = _step_env(env, state, interceptor, striker_vy=active_striker_vy)

            y1 = state.robots_yellow.get(1)
            ball = state.ball
            if y1 is None:
                continue

            out_now = _is_out_of_area(y1.x_m, y1.y_m)
            if out_now and not was_out:
                stats.out_of_area_events += 1
            was_out = out_now

            for b_robot in state.robots_blue.values():
                if math.hypot(y1.x_m - b_robot.x_m, y1.y_m - b_robot.y_m) <= 0.19:
                    rel_v = math.hypot(y1.vx_mps - b_robot.vx_mps, y1.vy_mps - b_robot.vy_mps)
                    if rel_v > CRASH_REL_V_MPS:
                        stats.crash_fouls += 1

            dist_b = math.hypot(ball.x_m - y1.x_m, ball.y_m - y1.y_m)
            rel_vb = math.hypot(ball.vx_mps - y1.vx_mps, ball.vy_mps - y1.vy_mps)

            if dist_b < TOUCH_DIST_M:
                touched = True
            if dist_b < INTERCEPT_DIST_M and rel_vb <= INTERCEPT_REL_V_MPS:
                hold_steps += 1
            else:
                hold_steps = 0

            if hold_steps >= INTERCEPT_HOLD_STEPS:
                outcome = "intercepted"
                stats.intercept_times_s.append((step - pass_step) * DT)
                break

            if pass_fired and ball.x_m >= INTERCEPTOR_X_LIMIT:
                outcome = "through"
                break

            if abs(ball.y_m) >= FIELD_Y_LIMIT or abs(ball.x_m) >= 4.5:
                outcome = "oob"
                break

        stats.trials += 1
        if outcome == "intercepted":
            stats.official_interceptions += 1
        elif outcome == "through":
            stats.passed_through += 1
        elif touched:
            stats.deflected_only += 1
        elif outcome == "oob":
            stats.out_of_bounds += 1

    return stats


def run_endurance_60s(
    env: Exercise3Env,
    interceptor: InterceptorStrategy,
    episodes: int,
    base_seed: int,
) -> ScenarioStats:
    """60s continuous test: ball respawns without resetting robot pose or strategy state."""
    stats = ScenarioStats()

    for ep in range(episodes):
        rng = np.random.default_rng(base_seed + ep)
        state, _ = env.reset(seed=int(base_seed + ep))
        interceptor.reset(seed=int(base_seed + ep))

        was_out = False
        hold_steps = 0
        touched = False
        steps_since_spawn = 0
        pass_fired = False

        def spawn_next_ball() -> None:
            nonlocal hold_steps, touched, steps_since_spawn, pass_fired
            bx = float(rng.uniform(-3.0, -1.8))
            by = float(rng.uniform(-2.2, 2.2))
            _set_ball(env, bx, by, 0.0, 0.0)
            hold_steps = 0
            touched = False
            steps_since_spawn = 0
            pass_fired = False

        spawn_next_ball()

        for _ in range(ENDURANCE_STEPS):
            if steps_since_spawn == int(1.5 / DT):
                bx, by = state.ball.x_m, state.ball.y_m
                ty = float(rng.uniform(-2.2, 2.2))
                vx, vy = _compute_pass_velocity(
                    bx, by, 2.5, ty, float(rng.uniform(3.2, 4.5))
                )
                _set_ball(env, bx, by, vx, vy)
                pass_fired = True

            state, _, _ = _step_env(env, state, interceptor)
            steps_since_spawn += 1

            y1 = state.robots_yellow.get(1)
            ball = state.ball
            if y1 is None:
                continue

            out_now = _is_out_of_area(y1.x_m, y1.y_m)
            if out_now and not was_out:
                stats.out_of_area_events += 1
            was_out = out_now

            dist_b = math.hypot(ball.x_m - y1.x_m, ball.y_m - y1.y_m)
            rel_vb = math.hypot(ball.vx_mps - y1.vx_mps, ball.vy_mps - y1.vy_mps)
            if dist_b < TOUCH_DIST_M:
                touched = True
            hold_steps = (
                hold_steps + 1
                if (dist_b < INTERCEPT_DIST_M and rel_vb <= INTERCEPT_REL_V_MPS)
                else 0
            )

            finished = False
            if hold_steps >= INTERCEPT_HOLD_STEPS:
                stats.official_interceptions += 1
                finished = True
            elif pass_fired and ball.x_m >= INTERCEPTOR_X_LIMIT:
                stats.passed_through += 1
                finished = True
            elif abs(ball.y_m) >= FIELD_Y_LIMIT or steps_since_spawn > int(5.0 / DT):
                if touched:
                    stats.deflected_only += 1
                else:
                    stats.out_of_bounds += 1
                finished = True

            if finished:
                stats.trials += 1
                spawn_next_ball()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Yellow 1 InterceptorStrategy.")
    parser.add_argument(
        "--scenario",
        choices=["all", "straight", "diagonal", "fake_dribble", "lead_pass", "endurance_60s"],
        default="all",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Trials per scenario (or 60s matches for endurance).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = Exercise3Env(render_mode="human" if args.render else None)
    interceptor = InterceptorStrategy()

    scenarios = (
        ["straight", "diagonal", "fake_dribble", "lead_pass", "endurance_60s"]
        if args.scenario == "all"
        else [args.scenario]
    )

    header = (
        f"{'Scenario':<15} | {'N':>4} | {'Intrcpt':>7} | {'Deflect':>7} | "
        f"{'Through':>7} | {'OoA':>4} | {'Crash':>5} | {'Avg T':>6}"
    )
    print(header)
    print("-" * len(header))

    for sc in scenarios:
        if sc == "endurance_60s":
            n_matches = max(1, min(args.episodes, 10))
            stats = run_endurance_60s(env, interceptor, n_matches, args.seed)
        else:
            stats = run_single_pass_scenario(env, interceptor, sc, args.episodes, args.seed)
        print(stats.summary_row(sc))

    env.close()


if __name__ == "__main__":
    main()