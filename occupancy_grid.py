"""
occupancy_grid.py
=================
2D occupancy grid mapping from LaserScan data using a log-odds inverse
sensor model -- the "M" half of SLAM.

Why log-odds instead of just writing 0/100 into cells:

A single scan is noisy. If we overwrote each cell with the latest reading, one
spurious return would punch a permanent hole through a wall, and one missed
return would erase it. Log-odds lets evidence *accumulate*: every beam that
passes through a cell nudges it toward free, every beam that terminates in it
nudges it toward occupied, and the cell only flips once the evidence is strong
enough. Probabilities multiply, which in log space is just addition -- so the
update is a single `+=` and stays cheap at 360 beams x 10 Hz.

Cell convention matches nav_msgs/OccupancyGrid:
      -1  unknown        0  free        100  occupied

Author: Naveed Ali | DecodeLabs Robotics & Automation Internship, Project 3
"""

import numpy as np


def bresenham(r0: int, c0: int, r1: int, c1: int):
    """
    Integer line rasterisation between two cells (endpoint excluded).

    Used to find every cell a laser beam passes *through* before it hits
    something. Pure integer arithmetic, no floating point, no division.
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0

    while True:
        if (r, c) == (r1, c1):
            break
        cells.append((r, c))
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


class OccupancyGrid:
    """Log-odds occupancy grid built incrementally from LaserScans."""

    def __init__(self, width_m: float, height_m: float, resolution: float = 0.05,
                 p_hit: float = 0.75, p_miss: float = 0.35,
                 clamp: float = 6.0, occ_threshold: float = 0.65):
        self.resolution = resolution
        self.width_m = width_m
        self.height_m = height_m
        self.n_rows = int(round(height_m / resolution))
        self.n_cols = int(round(width_m / resolution))

        # Log-odds field. 0.0 == p(occupied) of 0.5 == totally unknown.
        self.log_odds = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)

        self.l_hit = np.log(p_hit / (1 - p_hit))
        self.l_miss = np.log(p_miss / (1 - p_miss))
        self.clamp = clamp
        self.l_occ_threshold = np.log(occ_threshold / (1 - occ_threshold))

    # ------------------------------------------------------------------
    # Coordinate transforms: continuous world <-> discrete grid
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, y: float):
        """(metres, metres) -> (row, col)."""
        return int(np.floor(y / self.resolution)), int(np.floor(x / self.resolution))

    def grid_to_world(self, row: int, col: int):
        """(row, col) -> centre of that cell in metres."""
        return ((col + 0.5) * self.resolution, (row + 0.5) * self.resolution)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    # ------------------------------------------------------------------
    # The mapping update
    # ------------------------------------------------------------------
    def integrate_scan(self, pose, scan) -> None:
        """
        Fold one LaserScan into the map.

        For each beam:
          * cells along the ray  -> free evidence  (l_miss)
          * the terminating cell -> occupied evidence (l_hit)
          * max-range returns    -> free evidence only, no endpoint hit,
            because `inf` means "nothing out there", not "wall at 6 m".
        """
        x, y, theta = pose
        r0, c0 = self.world_to_grid(x, y)

        for bearing, rng in zip(scan.angles, scan.ranges):
            no_return = not np.isfinite(rng)
            reach = scan.range_max if no_return else rng
            if reach < scan.range_min:
                continue

            ang = theta + bearing
            ex = x + reach * np.cos(ang)
            ey = y + reach * np.sin(ang)
            r1, c1 = self.world_to_grid(ex, ey)

            # Free space along the beam
            for (r, c) in bresenham(r0, c0, r1, c1):
                if self.in_bounds(r, c):
                    self.log_odds[r, c] = np.clip(
                        self.log_odds[r, c] + self.l_miss, -self.clamp, self.clamp)

            # Occupied endpoint (only for real returns)
            if not no_return and self.in_bounds(r1, c1):
                self.log_odds[r1, c1] = np.clip(
                    self.log_odds[r1, c1] + self.l_hit, -self.clamp, self.clamp)

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------
    def probability_map(self) -> np.ndarray:
        """Log-odds -> probability of occupancy in [0, 1]."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def to_nav_grid(self) -> np.ndarray:
        """
        Export in nav_msgs/OccupancyGrid encoding: -1 unknown, 0 free,
        100 occupied. This is the array the planner consumes.
        """
        out = np.full((self.n_rows, self.n_cols), -1, dtype=np.int8)
        out[self.log_odds < -0.2] = 0
        out[self.log_odds >= self.l_occ_threshold] = 100
        return out

    def coverage(self) -> float:
        """Fraction of the map that is no longer unknown -- an explore metric."""
        return float(np.mean(np.abs(self.log_odds) > 0.2))
