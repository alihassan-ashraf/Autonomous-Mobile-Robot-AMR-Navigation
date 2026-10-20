"""
environment.py
==============
Ground-truth world + simulated 2D LiDAR sensor.

The robot NEVER reads this class's occupancy data directly. It may only call
`lidar_scan()`, which returns a LaserScan-style range array exactly as a real
RPLIDAR / Hokuyo driver would publish on /scan. Everything the robot knows
about the world has to be reconstructed from those range readings.

Author: Naveed Ali | DecodeLabs Robotics & Automation Internship, Project 3
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LaserScan:
    """Mirrors the sensor_msgs/LaserScan message used in ROS 2."""

    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray = field(repr=False)

    @property
    def angles(self) -> np.ndarray:
        """Beam bearings in the robot (base_link) frame."""
        return self.angle_min + np.arange(len(self.ranges)) * self.angle_increment


class Environment:
    """
    Ground-truth maze world.

    The world is stored as a boolean occupancy array at `resolution` metres per
    cell. World coordinates are metres with the origin at the array's [0, 0]
    corner, x to the right (columns) and y upward (rows).
    """

    def __init__(self, width_m: float = 20.0, height_m: float = 20.0,
                 resolution: float = 0.05):
        self.resolution = resolution
        self.width_m = width_m
        self.height_m = height_m
        self.n_cols = int(round(width_m / resolution))
        self.n_rows = int(round(height_m / resolution))
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=bool)

        # Dynamic obstacles are kept separately so they can be introduced
        # mid-run to test the re-routing logic.
        self._dynamic = []

        self._build_maze()

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def add_wall(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Fill an axis-aligned rectangle (metres) as solid wall."""
        c0, c1 = sorted((self.m2c(x0), self.m2c(x1)))
        r0, r1 = sorted((self.m2c(y0), self.m2c(y1)))
        r0 = max(r0, 0)
        c0 = max(c0, 0)
        r1 = min(r1, self.n_rows - 1)
        c1 = min(c1, self.n_cols - 1)
        self.grid[r0:r1 + 1, c0:c1 + 1] = True

    def _build_maze(self) -> None:
        """
        A warehouse-style maze: perimeter plus three long racking runs with
        staggered gaps, forcing a serpentine route.

        The aisles are deliberately 4-5 m wide. A corridor only slightly wider
        than the robot would make the run a test of the inflation radius rather
        than of the planner, and any tuning error would show up as "no path
        found" instead of as a visibly worse path.
        """
        t = 0.25  # wall thickness in metres
        w, h = self.width_m, self.height_m

        # Perimeter
        self.add_wall(0, 0, w, t)
        self.add_wall(0, h - t, w, h)
        self.add_wall(0, 0, t, h)
        self.add_wall(w - t, 0, w, h)

        # Three racking runs. Gaps alternate top / bottom / top, so the only
        # route from bottom-left to top-right is a serpentine -- a greedy
        # straight-line planner walks straight into the first rack.
        self.add_wall(5.0, 0.25, 5.0 + t, 14.0)      # gap at the TOP
        self.add_wall(10.0, 6.0, 10.0 + t, 19.75)    # gap at the BOTTOM
        self.add_wall(15.0, 0.25, 15.0 + t, 14.0)    # gap at the TOP

        # Partial shelves: they narrow an aisle without closing it.
        self.add_wall(5.25, 10.0, 8.0, 10.0 + t)
        self.add_wall(10.25, 17.0, 13.0, 17.0 + t)

        # Free-standing pillars
        self.add_wall(2.2, 8.0, 3.0, 8.8)
        self.add_wall(7.0, 3.0, 7.8, 3.8)
        self.add_wall(12.2, 9.0, 13.0, 9.8)
        self.add_wall(17.0, 10.0, 17.8, 10.8)

    def add_dynamic_obstacle(self, x: float, y: float, radius: float = 0.45) -> None:
        """
        Drop an unexpected object into the world (a pallet, a person, a
        stray AMR). It becomes physically solid immediately, so the LiDAR
        will start returning hits from it on the very next scan.
        """
        self._dynamic.append((x, y, radius))
        rr, cc = np.ogrid[:self.n_rows, :self.n_cols]
        xs = cc * self.resolution
        ys = rr * self.resolution
        mask = (xs - x) ** 2 + (ys - y) ** 2 <= radius ** 2
        self.grid |= mask

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def m2c(self, metres: float) -> int:
        """Metres -> cell index along one axis."""
        return int(np.floor(metres / self.resolution))

    def is_occupied(self, x: float, y: float) -> bool:
        """Ground-truth occupancy test at a world point."""
        c, r = self.m2c(x), self.m2c(y)
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return True  # outside the world counts as solid
        return bool(self.grid[r, c])

    # ------------------------------------------------------------------
    # Simulated LiDAR
    # ------------------------------------------------------------------
    def lidar_scan(self, pose, n_beams: int = 360, max_range: float = 6.0,
                   noise_std: float = 0.015, rng=None) -> LaserScan:
        """
        Cast `n_beams` rays from the robot pose and return a LaserScan.

        Ray-marching uses a fixed step of half a cell, which is the standard
        trade-off: fine enough that no wall thinner than a cell is missed,
        coarse enough to stay real-time.

        A beam that hits nothing within `max_range` returns `inf`, matching the
        ROS convention for "no return" rather than silently clamping to the max
        range (which would hallucinate a wall at exactly 6 m).
        """
        rng = rng or np.random.default_rng(0)
        x, y, theta = pose
        step = self.resolution * 0.5

        angle_min = -np.pi
        angle_increment = 2.0 * np.pi / n_beams
        bearings = angle_min + np.arange(n_beams) * angle_increment
        ranges = np.full(n_beams, np.inf)

        for i, bearing in enumerate(bearings):
            ang = theta + bearing
            dx, dy = np.cos(ang) * step, np.sin(ang) * step
            px, py = x, y
            dist = 0.0
            while dist < max_range:
                px += dx
                py += dy
                dist += step
                if self.is_occupied(px, py):
                    measured = dist + rng.normal(0.0, noise_std)
                    ranges[i] = max(measured, 0.0)
                    break

        return LaserScan(
            angle_min=angle_min,
            angle_max=angle_min + (n_beams - 1) * angle_increment,
            angle_increment=angle_increment,
            range_min=0.12,
            range_max=max_range,
            ranges=ranges,
        )
