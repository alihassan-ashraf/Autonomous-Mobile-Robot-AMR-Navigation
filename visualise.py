"""
visualise.py
============
All figure generation for Project 3. Kept out of navigation_sim.py so the
control loop contains control logic only.

Produces, in ../output/:
    1_ground_truth_vs_slam.png   ground truth beside the map the robot built
    2_costmap_and_path.png       inflation layer + the A* route it produced
    3_dynamic_reroute.png        the re-route around the injected obstacle
    4_localisation_error.png     odometry drift vs EKF-fused estimate
    5_navigation.gif             the run, animated

Author: Naveed Ali | DecodeLabs Robotics & Automation Internship, Project 3
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.15,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

#: unknown -> grey, free -> white, occupied -> near-black
OCC_CMAP = LinearSegmentedColormap.from_list(
    "occ", ["#ffffff", "#c9ccd1", "#2b2f36"])


def _extent(grid):
    return [0, grid.width_m, 0, grid.height_m]


def _style(ax, title, grid):
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, grid.width_m)
    ax.set_ylim(0, grid.height_m)
    ax.set_aspect("equal")


# ----------------------------------------------------------------------
def fig_ground_truth_vs_slam(mission, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    axes[0].imshow(mission.env.grid, origin="lower", cmap="gray_r",
                   extent=_extent(mission.grid), interpolation="nearest")
    _style(axes[0], "Ground truth world\n(the robot never sees this)", mission.grid)

    prob = mission.grid.probability_map()
    axes[1].imshow(prob, origin="lower", cmap=OCC_CMAP, vmin=0, vmax=1,
                   extent=_extent(mission.grid), interpolation="nearest")

    traj = np.array(mission.robot.trajectory)
    axes[1].plot(traj[:, 0], traj[:, 1], color="#e8590c", lw=1.6,
                 label="path driven")
    axes[1].plot(*traj[0, :2], "o", color="#2f9e44", ms=9, label="start")
    axes[1].plot(*traj[-1, :2], "*", color="#c92a2a", ms=15, label="goal")
    axes[1].legend(loc="lower right", framealpha=0.92, fontsize=8)
    _style(axes[1],
           f"Occupancy grid built from LiDAR\n"
           f"{mission.grid.coverage() * 100:.0f}% of the map explored",
           mission.grid)

    fig.suptitle("SLAM: reconstructing the world from 360-beam LaserScans",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_ground_truth_vs_slam.png"), dpi=150)
    plt.close(fig)


def fig_costmap_and_path(mission, results, out_dir):
    mission.refresh_costmap()
    start = mission.grid.world_to_grid(2.0, 2.0)
    path = mission.planner.plan(start, mission.goal_cell)

    cost = np.array(mission.planner.costmap, dtype=float)
    display = np.where(np.isfinite(cost), cost, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    nav = mission.grid.to_nav_grid()
    axes[0].imshow(nav, origin="lower", cmap=OCC_CMAP, vmin=-1, vmax=100,
                   extent=_extent(mission.grid), interpolation="nearest")
    _style(axes[0], "Raw occupancy grid\n(what the sensors reported)", mission.grid)

    im = axes[1].imshow(display, origin="lower", cmap="turbo",
                        extent=_extent(mission.grid), interpolation="nearest",
                        vmin=1, vmax=np.nanpercentile(display, 99))
    lethal = np.ma.masked_where(~mission.planner.lethal, mission.planner.lethal)
    axes[1].imshow(lethal, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                   extent=_extent(mission.grid), interpolation="nearest")

    if path:
        pts = np.array([mission.grid.grid_to_world(r, c) for r, c in path])
        axes[1].plot(pts[:, 0], pts[:, 1], color="#ffffff", lw=3.4, alpha=0.9)
        axes[1].plot(pts[:, 0], pts[:, 1], color="#111111", lw=1.8,
                     label=f"A* route, {mission.planner.path_length_m(path):.1f} m")
        axes[1].plot(pts[0, 0], pts[0, 1], "o", color="#2f9e44", ms=9)
        axes[1].plot(pts[-1, 0], pts[-1, 1], "*", color="#c92a2a", ms=15)
        axes[1].legend(loc="lower right", framealpha=0.92, fontsize=8)

    bench = results.get("benchmark", {})
    nodes = bench.get("astar", {}).get("nodes_expanded", "-")
    _style(axes[1],
           f"Inflated costmap + A* solution\n"
           f"{nodes} nodes expanded (Manhattan heuristic)", mission.grid)
    fig.colorbar(im, ax=axes[1], fraction=0.045, label="traversal cost")

    fig.suptitle("Inflation keeps the path off the walls: "
                 "A* plans for a point, the robot is 0.7 m wide",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_costmap_and_path.png"), dpi=150)
    plt.close(fig)


def fig_dynamic_reroute(mission, out_dir):
    """Find the snapshots either side of the obstacle injection."""
    snaps = mission.snapshots
    if not snaps:
        return

    if not mission.obstacles:
        return
    first = mission.obstacles[0]
    inject_step = first["step"]

    before = None
    after = None
    for s in snaps:
        if s["step"] <= inject_step:
            before = s
        elif after is None:
            after = s
    before = before or snaps[0]
    after = after or snaps[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, snap, title in (
        (axes[0], before, f"BEFORE  (t = {before['step'] * 0.1:.0f} s)\n"
                          f"planned route runs straight through the aisle"),
        (axes[1], after, f"AFTER  (t = {after['step'] * 0.1:.0f} s)\n"
                         f"obstacle mapped, route recomputed around it"),
    ):
        ax.imshow(snap["prob"], origin="lower", cmap=OCC_CMAP, vmin=0, vmax=1,
                  extent=_extent(mission.grid), interpolation="nearest")
        if snap["path"]:
            pts = np.array([mission.grid.grid_to_world(r, c) for r, c in snap["path"]])
            ax.plot(pts[:, 0], pts[:, 1], color="#1971c2", lw=2.4, label="planned route")
        trail = snap["trail"]
        ax.plot(trail[:, 0], trail[:, 1], color="#e8590c", lw=1.5, label="driven so far")
        ax.plot(snap["pose"][0], snap["pose"][1], "o", color="#0b7285", ms=9,
                label="robot")
        for k, obs in enumerate(mission.obstacles):
            if obs["step"] > snap["step"]:
                continue          # not dropped yet at this point in the run
            ax.add_patch(plt.Circle(obs["xy"], obs["radius"], color="#c92a2a",
                                    alpha=0.45, zorder=5,
                                    label="unexpected obstacle" if k == 0 else None))
        ax.legend(loc="lower right", framealpha=0.92, fontsize=8)
        _style(ax, title, mission.grid)

    fig.suptitle("Dynamic obstacle avoidance: detect, map, re-plan "
                 f"({mission.robot.emergency_stops} emergency stops, "
                 f"{mission.robot.replans} re-routes)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_dynamic_reroute.png"), dpi=150)
    plt.close(fig)


def fig_localisation_error(mission, out_dir):
    true = np.array(mission.robot.trajectory)[:, :2]
    odom = np.array(mission.robot.odom_trajectory)[:, :2]
    ekf = np.array(mission.robot.ekf_trajectory)[:, :2]
    n = min(len(true), len(odom), len(ekf))
    t = np.arange(n) * 0.1

    odom_err = np.hypot(*(odom[:n] - true[:n]).T)
    ekf_err = np.hypot(*(ekf[:n] - true[:n]).T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(true[:n, 0], true[:n, 1], color="#2f9e44", lw=2.0, label="ground truth")
    axes[0].plot(odom[:n, 0], odom[:n, 1], color="#e8590c", lw=1.5, ls="--",
                 label="wheel odometry (drifting)")
    axes[0].plot(ekf[:n, 0], ekf[:n, 1], color="#1971c2", lw=1.5, ls=":",
                 label="EKF fused estimate")
    axes[0].legend(fontsize=8, framealpha=0.92)
    axes[0].set_title("Trajectory: three versions of 'where am I'")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")

    axes[1].fill_between(t, 0, odom_err, color="#e8590c", alpha=0.25)
    axes[1].plot(t, odom_err, color="#e8590c", lw=1.8,
                 label=f"odometry only (final {odom_err[-1]:.2f} m)")
    axes[1].plot(t, ekf_err, color="#1971c2", lw=1.8,
                 label=f"EKF fused (final {ekf_err[-1]:.2f} m)")
    axes[1].set_title("Position error against ground truth")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error (m)")
    axes[1].legend(fontsize=8, framealpha=0.92)

    fig.suptitle("Why sensor fusion: encoder error integrates without bound, "
                 "the IMU bounds the heading",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_localisation_error.png"), dpi=150)
    plt.close(fig)


def fig_animation(mission, out_dir):
    """Animated GIF of the run. Silently skipped if pillow is unavailable."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:                      # pragma: no cover
        return

    snaps = mission.snapshots
    if len(snaps) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    img = ax.imshow(snaps[0]["prob"], origin="lower", cmap=OCC_CMAP,
                    vmin=0, vmax=1, extent=_extent(mission.grid),
                    interpolation="nearest")
    (path_line,) = ax.plot([], [], color="#1971c2", lw=2.2, label="planned route")
    (trail_line,) = ax.plot([], [], color="#e8590c", lw=1.5, label="driven")
    (robot_dot,) = ax.plot([], [], "o", color="#0b7285", ms=10, label="robot")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    _style(ax, "", mission.grid)

    def update(i):
        snap = snaps[i]
        img.set_data(snap["prob"])
        if snap["path"]:
            pts = np.array([mission.grid.grid_to_world(r, c) for r, c in snap["path"]])
            path_line.set_data(pts[:, 0], pts[:, 1])
        trail = snap["trail"]
        trail_line.set_data(trail[:, 0], trail[:, 1])
        robot_dot.set_data([snap["pose"][0]], [snap["pose"][1]])
        ax.set_title(f"AMR navigation  |  t = {snap['step'] * 0.1:5.1f} s")
        return img, path_line, trail_line, robot_dot

    anim = FuncAnimation(fig, update, frames=len(snaps), blit=False)
    anim.save(os.path.join(out_dir, "5_navigation.gif"),
              writer=PillowWriter(fps=6), dpi=80)
    plt.close(fig)


# ----------------------------------------------------------------------
def render_all(mission, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig_ground_truth_vs_slam(mission, out_dir)
    fig_costmap_and_path(mission, results, out_dir)
    fig_dynamic_reroute(mission, out_dir)
    fig_localisation_error(mission, out_dir)
    fig_animation(mission, out_dir)
