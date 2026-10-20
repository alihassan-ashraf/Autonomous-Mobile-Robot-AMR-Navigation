"""
navigation_sim.py
=================
Main entry point: the full perceive -> map -> plan -> act loop.

Run:
    python src/navigation_sim.py

Writes figures and a metrics report into ../output/.

The control loop below is the whole project in twelve lines of logic:

    every cycle:
        scan      <- LiDAR
        map       <- integrate(scan)                 # SLAM, mapping half
        if obstacle in the safety cone:              # reflex layer
            stop and replan
        elif the remaining path is now blocked:      # deliberative layer
            replan
        drive toward the next waypoint

The two re-plan triggers are separate on purpose. The reflex fires on raw
sensor data in a single cycle; the path-validity check fires on the accumulated
map and may lag by a scan or two. A system with only the second one hits things.

Author: Naveed Ali | DecodeLabs Robotics & Automation Internship, Project 3
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astar_planner import AStarPlanner, dijkstra          # noqa: E402
from environment import Environment                        # noqa: E402
from occupancy_grid import OccupancyGrid                    # noqa: E402
from robot import DifferentialDriveRobot                    # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

#: Planning resolution. 0.05 m gives a 400x400 grid and ~450 ms per A* call,
#: which is not a real-time planner. 0.10 m quarters the cell count and brings
#: a full re-plan under ~80 ms, while still resolving the 0.25 m racking walls
#: across 2-3 cells. Real stacks do exactly this: map finely, plan coarsely.
RESOLUTION = 0.10
DT = 0.1
MAX_STEPS = 4000

START_XY = (2.0, 2.0)
GOAL_XY = (18.0, 18.0)

#: Unexpected objects dropped onto the robot's current route as
#: (step, lead_distance_m, radius_m). Positions are resolved at runtime by
#: walking that far along the active path, so a test can never silently
#: degrade into "obstacle appeared somewhere the robot was never going".
#:
#: The two cases exercise the two different layers, which is the point:
#:   3.0 m lead -- far enough that mapping + re-planning absorbs it, the
#:                 robot never even slows down. The deliberative layer wins.
#:   1.1 m lead -- inside the sense->map->plan->actuate latency window. By
#:                 the time a new path exists the robot would already have
#:                 hit it, so only the reflex can save it.
DYNAMIC_OBSTACLES = [
    (300, 3.0, 0.55),
    (620, 1.1, 0.50),
]
DYNAMIC_OBSTACLE_RADIUS = 0.55

#: Minimum steps between deliberative re-plans. Without this the robot
#: re-plans on almost every map update as newly-seen walls shave cells off
#: the route -- correct, but it burns CPU recomputing a nearly identical path.
REPLAN_COOLDOWN_STEPS = 30


class NavigationMission:
    """One start-to-goal run with mapping, planning and dynamic re-routing."""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.verbose = verbose
        self.env = Environment(20.0, 20.0, RESOLUTION)
        self.grid = OccupancyGrid(20.0, 20.0, RESOLUTION)
        self.planner = AStarPlanner(resolution=RESOLUTION, robot_radius=0.35,
                                    inflation_radius=0.80, cost_scaling=3.0)
        self.robot = DifferentialDriveRobot(*START_XY, theta=np.pi / 2, seed=seed)
        self.rng = np.random.default_rng(seed)

        self.goal_cell = self.grid.world_to_grid(*GOAL_XY)
        self.path_cells = None
        self.waypoints = []
        self.wp_index = 0

        self.events = []
        self.snapshots = []       # (step, map, path, pose) for the GIF
        self.plan_log = []        # one record per planning call
        self.obstacles = []        # [{step, xy, radius, lead_m}]
        self.last_replan_step = -10 ** 6

    # ------------------------------------------------------------------
    def log(self, step: int, message: str) -> None:
        self.events.append({"step": step, "t": round(step * DT, 1), "msg": message})
        if self.verbose:
            print(f"[t={step * DT:6.1f}s] {message}")

    # ------------------------------------------------------------------
    def sense(self) -> None:
        """One LiDAR scan folded into the occupancy grid."""
        self.scan = self.env.lidar_scan(self.robot.true_pose, n_beams=360,
                                        max_range=6.0, rng=self.rng)
        # Mapping uses the robot's *estimated* pose, not ground truth -- that
        # is the honest version. A map built on ground truth would hide every
        # localisation error the EKF failed to catch.
        self.grid.integrate_scan(self.robot.ekf.state, self.scan)

    def plan(self, step: int, reason: str) -> bool:
        """Re-plan from the current estimated pose to the goal."""
        t0 = time.perf_counter()
        nav_grid = self.grid.to_nav_grid()
        start_cell = self.grid.world_to_grid(*self.robot.ekf.state[:2])
        path = self.planner.plan(start_cell, self.goal_cell, nav_grid)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if path is None:
            self.log(step, f"PLAN FAILED ({reason}) -- no route exists")
            return False

        self.path_cells = path
        simplified = self.planner.simplify(path)
        self.waypoints = [self.grid.grid_to_world(r, c) for r, c in simplified]
        self.wp_index = 1 if len(self.waypoints) > 1 else 0

        self.plan_log.append({
            "step": step,
            "reason": reason,
            "nodes_expanded": self.planner.nodes_expanded,
            "path_cells": len(path),
            "waypoints": len(simplified),
            "length_m": round(self.planner.path_length_m(path), 2),
            "plan_time_ms": round(elapsed_ms, 1),
        })
        self.log(step, f"Planned ({reason}): {self.planner.path_length_m(path):.1f} m, "
                       f"{len(simplified)} waypoints, "
                       f"{self.planner.nodes_expanded} nodes expanded, "
                       f"{elapsed_ms:.0f} ms")
        return True

    def path_is_blocked(self, horizon_m: float = 6.0) -> bool:
        """
        Does the freshly updated map invalidate the route we are driving?

        Two deliberate restrictions:

        * Only the path *ahead* of the robot is checked. Cells already driven
          through are irrelevant, and including them would trigger a re-plan
          every time the map behind the robot sharpened up.
        * Only the next `horizon_m` of it. A wall discovered 15 m away will
          very likely be re-planned around for free by the time the robot gets
          there, as the map keeps improving. Re-planning the instant any
          distant cell goes lethal produces constant churn for no change in
          the next few seconds of motion.
        """
        if not self.path_cells or self.planner.lethal is None:
            return False
        lethal = self.planner.lethal
        start = self._progress_index()
        horizon_cells = int(horizon_m / self.grid.resolution)
        ahead = self.path_cells[start:start + horizon_cells]
        for (r, c) in ahead:
            if 0 <= r < lethal.shape[0] and 0 <= c < lethal.shape[1] and lethal[r, c]:
                return True
        return False

    def _point_ahead_on_path(self, lead_m: float):
        """
        World point roughly `lead_m` along the current path from the robot.

        Used to place the dynamic obstacle somewhere that genuinely matters:
        far enough ahead that the robot is committed to that route, close
        enough that it is inside LiDAR range and will be met within seconds.
        """
        if not self.path_cells:
            return None
        start = self._progress_index()
        offset = int(lead_m / self.grid.resolution)
        idx = min(start + offset, len(self.path_cells) - 1)
        if idx <= start:
            return None
        return self.grid.grid_to_world(*self.path_cells[idx])

    def _progress_index(self) -> int:
        """Index of the path cell closest to the robot right now."""
        if not self.path_cells:
            return 0
        rc = np.array(self.grid.world_to_grid(*self.robot.ekf.state[:2]))
        arr = np.array(self.path_cells)
        return int(np.argmin(np.abs(arr - rc).sum(axis=1)))

    def refresh_costmap(self) -> None:
        self.planner.build_costmap(self.grid.to_nav_grid())

    # ------------------------------------------------------------------
    def run(self) -> dict:
        self.log(0, f"Mission start {START_XY} -> {GOAL_XY}")

        # Spin on the spot first: one scan from a standing start sees only
        # the aisle directly ahead, and planning on that produces a route
        # through what is really unexplored space.
        for _ in range(16):
            self.sense()
            self.robot.step(0.0, 0.6, DT, self.env)
        self.log(0, f"Initial 360-degree survey complete, "
                    f"map coverage {self.grid.coverage() * 100:.1f}%")

        if not self.plan(0, "initial"):
            return self._report(False, 0)

        reached = False
        stop_cooldown = 0

        for step in range(1, MAX_STEPS + 1):
            self.sense()

            # --- inject unexpected obstacles, directly onto the route ---
            for inject_step, lead_m, radius in DYNAMIC_OBSTACLES:
                if step != inject_step:
                    continue
                spot = self._point_ahead_on_path(lead_m)
                if spot is None:
                    continue
                self.env.add_dynamic_obstacle(*spot, radius=radius)
                self.obstacles.append({"step": step, "xy": [round(v, 2) for v in spot],
                                       "radius": radius, "lead_m": lead_m})
                self.log(step, f"!! Unexpected obstacle at "
                               f"({spot[0]:.1f}, {spot[1]:.1f}) -- {lead_m:.1f} m "
                               f"dead ahead on the planned route, in no prior map")

            # --- reflex layer: raw scan, no map, no plan ---
            blocked_now, nearest = self.robot.check_obstacle_ahead(
                self.scan, cone_deg=50.0, stop_distance=0.75)

            if blocked_now and stop_cooldown == 0:
                self.robot.emergency_stops += 1
                self.robot.step(0.0, 0.0, DT, self.env)   # hard stop
                self.log(step, f"EMERGENCY STOP -- obstacle at {nearest:.2f} m")
                self.refresh_costmap()
                if self.plan(step, "reflex stop"):
                    self.robot.replans += 1
                    self.last_replan_step = step
                stop_cooldown = 25
                self._snapshot(step)
                continue

            if stop_cooldown:
                stop_cooldown -= 1

            # --- deliberative layer: has the map invalidated the route? ---
            if step % 10 == 0 and (step - self.last_replan_step) >= REPLAN_COOLDOWN_STEPS:
                self.refresh_costmap()
                if self.path_is_blocked():
                    self.log(step, "Path invalidated by updated map -- re-routing")
                    if self.plan(step, "path blocked"):
                        self.robot.replans += 1
                        self.last_replan_step = step

            # --- goal check ---
            dist_to_goal = np.hypot(*(np.array(GOAL_XY) - self.robot.true_pose[:2]))
            if dist_to_goal < 0.4:
                reached = True
                self.log(step, f"GOAL REACHED, final error {dist_to_goal:.2f} m")
                self._snapshot(step)
                break

            # --- drive ---
            v, omega, new_index = self.robot.follow_path(
                self.waypoints, self.wp_index, DT, goal_tolerance=0.3)
            self.wp_index = new_index

            if self.wp_index >= len(self.waypoints):
                # Ran out of waypoints without arriving: the plan is stale.
                self.refresh_costmap()
                if not self.plan(step, "waypoints exhausted"):
                    break
                continue

            moved = self.robot.step(v, omega, DT, self.env)
            if not moved:
                # Blocked by geometry the controller did not expect. Back off
                # and re-plan rather than grinding against the wall.
                self.robot.step(-0.25, 0.5, DT, self.env)
                self.refresh_costmap()
                self.plan(step, "motion blocked")

            if step % 40 == 0:
                self._snapshot(step)

        return self._report(reached, step)

    # ------------------------------------------------------------------
    def _snapshot(self, step: int) -> None:
        self.snapshots.append({
            "step": step,
            "prob": self.grid.probability_map().copy(),
            "path": list(self.path_cells) if self.path_cells else [],
            "pose": self.robot.true_pose.copy(),
            "trail": np.array(self.robot.trajectory)[:, :2].copy(),
        })

    def _report(self, reached: bool, steps: int) -> dict:
        odom_err, ekf_err = self.robot.localisation_error()
        traj = np.array(self.robot.trajectory)[:, :2]
        travelled = float(np.sum(np.hypot(*np.diff(traj, axis=0).T)))

        plan_times = [p["plan_time_ms"] for p in self.plan_log]
        return {
            "goal_reached": reached,
            "sim_time_s": round(steps * DT, 1),
            "planning_resolution_m": RESOLUTION,
            "mean_plan_time_ms": round(float(np.mean(plan_times)), 1) if plan_times else None,
            "max_plan_time_ms": round(float(np.max(plan_times)), 1) if plan_times else None,
            "dynamic_obstacles": self.obstacles,
            "distance_travelled_m": round(travelled, 2),
            "map_coverage_pct": round(self.grid.coverage() * 100, 1),
            "emergency_stops": self.robot.emergency_stops,
            "replans": self.robot.replans,
            "planning_calls": len(self.plan_log),
            "final_odom_error_m": round(odom_err, 3),
            "final_ekf_error_m": round(ekf_err, 3),
            "ekf_improvement_pct": round(
                (1 - ekf_err / odom_err) * 100, 1) if odom_err > 1e-6 else 0.0,
            "plan_log": self.plan_log,
            "events": self.events,
        }


def benchmark_heuristics(mission: NavigationMission) -> dict:
    """
    A* vs Dijkstra on the final map, same start and goal.

    Both are optimal here, so the interesting number is not path length -- it
    is how many nodes each had to expand to prove that path optimal.
    """
    mission.refresh_costmap()
    start = mission.grid.world_to_grid(*START_XY)
    goal = mission.goal_cell

    t0 = time.perf_counter()
    a_path = mission.planner.plan(start, goal)
    a_time = (time.perf_counter() - t0) * 1000
    a_nodes = mission.planner.nodes_expanded

    t0 = time.perf_counter()
    d_path, d_nodes = dijkstra(mission.planner, start, goal)
    d_time = (time.perf_counter() - t0) * 1000

    return {
        "astar": {
            "nodes_expanded": a_nodes,
            "path_length_m": round(mission.planner.path_length_m(a_path), 2) if a_path else None,
            "time_ms": round(a_time, 1),
        },
        "dijkstra": {
            "nodes_expanded": d_nodes,
            "path_length_m": round(mission.planner.path_length_m(d_path), 2) if d_path else None,
            "time_ms": round(d_time, 1),
        },
        "speedup_nodes": round(d_nodes / a_nodes, 2) if a_nodes else None,
    }


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 68)
    print("  PROJECT 3 -- AUTONOMOUS MOBILE ROBOT NAVIGATION")
    print("  SLAM mapping | A* global planning | dynamic obstacle avoidance")
    print("=" * 68)

    mission = NavigationMission(seed=42)
    results = mission.run()

    print("\n" + "-" * 68)
    print("  MISSION REPORT")
    print("-" * 68)
    for key in ("goal_reached", "sim_time_s", "distance_travelled_m",
                "map_coverage_pct", "emergency_stops", "replans",
                "planning_calls", "mean_plan_time_ms", "max_plan_time_ms",
                "final_odom_error_m", "final_ekf_error_m",
                "ekf_improvement_pct"):
        print(f"  {key:<26} {results[key]}")

    print("\n  A* vs Dijkstra on the final map:")
    bench = benchmark_heuristics(mission)
    results["benchmark"] = bench
    print(f"    A*       {bench['astar']['nodes_expanded']:>7} nodes   "
          f"{bench['astar']['path_length_m']} m   {bench['astar']['time_ms']} ms")
    print(f"    Dijkstra {bench['dijkstra']['nodes_expanded']:>7} nodes   "
          f"{bench['dijkstra']['path_length_m']} m   {bench['dijkstra']['time_ms']} ms")
    print(f"    A* expanded {bench['speedup_nodes']}x fewer nodes for the same path")

    with open(os.path.join(OUTPUT_DIR, "mission_report.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Report written to output/mission_report.json")

    # Figures are generated by a separate module so that the control loop
    # stays free of plotting code.
    try:
        import visualise
        visualise.render_all(mission, results, OUTPUT_DIR)
        print("  Figures written to output/")
    except Exception as exc:                      # pragma: no cover
        print(f"  [warn] figure generation failed: {exc}")

    return 0 if results["goal_reached"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
