"""
astar_planner.py
================
Global path planning on an occupancy grid: costmap inflation + A* search.

Two separate ideas live in this file and it matters that they stay separate:

1. INFLATION turns a map of where the walls are into a map of where the robot
   is allowed to put its centre. A* is a point-search algorithm -- it has no
   concept of the robot's 0.35 m radius. If you plan on the raw grid you get a
   mathematically perfect path that scrapes the chassis along every wall.
   Inflating obstacles by the robot radius, then adding a soft decaying cost
   beyond that, moves the solution into the middle of the aisle where a real
   AMR would actually drive.

2. A* finds the cheapest route through that inflated costmap, with
   f(n) = g(n) + h(n).

Heuristic choice: Manhattan, not Euclidean. The robot expands 4-connected
neighbours, so it physically cannot travel a diagonal. Euclidean distance would
report a shorter distance than any achievable route -- it stays admissible, so
A* would still return the optimal path, but it under-informs the search and
expands far more nodes than necessary. Manhattan is exactly the true remaining
cost on an obstacle-free 4-grid, which makes it both admissible and tight.

Author: Naveed Ali | DecodeLabs Robotics & Automation Internship, Project 3
"""

import heapq

import numpy as np


class AStarPlanner:
    """A* over an inflated occupancy costmap."""

    #: 4-connected motion. A diagonal move would clip wall corners on a grid
    #: map, so the AMR is restricted to axis-aligned steps.
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, resolution: float = 0.05, robot_radius: float = 0.35,
                 inflation_radius: float = 0.75, cost_scaling: float = 3.0,
                 unknown_is_free: bool = True, unknown_penalty: float = 2.5):
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.inflation_radius = inflation_radius
        self.cost_scaling = cost_scaling
        self.unknown_is_free = unknown_is_free
        self.unknown_penalty = unknown_penalty

        self.costmap = None   # populated by build_costmap()
        self.lethal = None    # boolean mask of cells the centre may not enter
        self.nodes_expanded = 0

    # ------------------------------------------------------------------
    # Costmap construction
    # ------------------------------------------------------------------
    def build_costmap(self, nav_grid: np.ndarray) -> np.ndarray:
        """
        Turn a -1/0/100 occupancy grid into a float costmap.

        Layers, in the order Nav2 applies them:
          * lethal         -- obstacle cells themselves
          * inscribed      -- within one robot radius of an obstacle; the robot
                              centre here means a guaranteed collision
          * inflated       -- beyond the radius, cost decays exponentially so
                              the planner prefers the centre of free space
          * unknown        -- a flat penalty: traversable, but only if there is
                              no known-free alternative
        """
        occupied = (nav_grid == 100)
        unknown = (nav_grid == -1)

        dist_cells = self._distance_transform(occupied)
        dist_m = dist_cells * self.resolution

        cost = np.ones_like(dist_m, dtype=np.float32)

        # Exponential decay away from obstacles (Nav2's inflation formula).
        inflate_band = (dist_m > self.robot_radius) & (dist_m <= self.inflation_radius)
        cost[inflate_band] += 50.0 * np.exp(
            -self.cost_scaling * (dist_m[inflate_band] - self.robot_radius))

        if self.unknown_is_free:
            cost[unknown] += self.unknown_penalty

        # Anything within the robot radius of an obstacle is off-limits.
        self.lethal = (dist_m <= self.robot_radius) | occupied
        cost[self.lethal] = np.inf

        self.costmap = cost
        return cost

    @staticmethod
    def _distance_transform(mask: np.ndarray) -> np.ndarray:
        """
        Exact Euclidean distance (in cells) from every cell to the nearest
        True cell in `mask`.

        Uses scipy when available; otherwise falls back to the two-pass
        Felzenszwalb 1D squared-distance transform, so the project runs with
        numpy alone and has no hard scipy dependency.
        """
        try:
            from scipy.ndimage import distance_transform_edt
            return distance_transform_edt(~mask)
        except ImportError:
            pass

        INF = 1e12
        f = np.where(mask, 0.0, INF)

        def edt_1d(f_row):
            n = len(f_row)
            d = np.empty(n)
            v = np.zeros(n, dtype=int)
            z = np.empty(n + 1)
            k = 0
            v[0] = 0
            z[0], z[1] = -INF, INF
            for q in range(1, n):
                s = ((f_row[q] + q * q) - (f_row[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
                while s <= z[k]:
                    k -= 1
                    s = ((f_row[q] + q * q) - (f_row[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = INF
            k = 0
            for q in range(n):
                while z[k + 1] < q:
                    k += 1
                d[q] = (q - v[k]) ** 2 + f_row[v[k]]
            return d

        for r in range(f.shape[0]):
            f[r, :] = edt_1d(f[r, :])
        for c in range(f.shape[1]):
            f[:, c] = edt_1d(f[:, c])
        return np.sqrt(f)

    # ------------------------------------------------------------------
    # The search
    # ------------------------------------------------------------------
    @staticmethod
    def heuristic(a, b) -> float:
        """Manhattan distance -- exact on a 4-connected grid, so admissible."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start, goal, nav_grid=None):
        """
        Find the cheapest route from `start` to `goal`, both (row, col).

        Returns a list of cells from start to goal inclusive, or None if the
        goal is unreachable. `self.nodes_expanded` reports the search effort,
        which is what makes the heuristic comparison in the README measurable.
        """
        if nav_grid is not None:
            self.build_costmap(nav_grid)
        if self.costmap is None:
            raise RuntimeError("build_costmap() must run before plan()")

        n_rows, n_cols = self.costmap.shape
        start = tuple(int(v) for v in start)
        goal = tuple(int(v) for v in goal)

        # A goal buried inside the inflation band is common -- the operator
        # clicks slightly too close to a shelf. Nudge it to the nearest cell
        # the robot centre can legally occupy rather than failing outright.
        if self.lethal[goal]:
            goal = self._nearest_free(goal)
            if goal is None:
                return None
        if self.lethal[start]:
            start = self._nearest_free(start)
            if start is None:
                return None

        open_heap = [(self.heuristic(start, goal), 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        self.nodes_expanded = 0

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            self.nodes_expanded += 1

            if current == goal:
                return self._reconstruct(came_from, current)

            cr, cc = current
            for dr, dc in self.MOVES:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                neighbour = (nr, nc)
                if neighbour in closed:
                    continue
                step_cost = self.costmap[nr, nc]
                if not np.isfinite(step_cost):
                    continue

                tentative_g = g + step_cost
                if tentative_g < g_score.get(neighbour, np.inf):
                    g_score[neighbour] = tentative_g
                    came_from[neighbour] = current
                    f = tentative_g + self.heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbour))

        return None  # open list exhausted -> no route exists

    def _nearest_free(self, cell, max_radius: int = 40):
        """Spiral outward from `cell` to the closest non-lethal cell."""
        n_rows, n_cols = self.lethal.shape
        r0, c0 = cell
        for radius in range(1, max_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < n_rows and 0 <= c < n_cols and not self.lethal[r, c]:
                        return (r, c)
        return None

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def simplify(self, path):
        """
        Collapse runs of collinear cells into waypoints.

        A* on a grid returns one cell per step -- hundreds of points for a
        20 m route. The controller only needs the corners, and a shorter
        waypoint list means fewer heading corrections and smoother motion.
        """
        if not path or len(path) < 3:
            return path
        out = [path[0]]
        for prev, cur, nxt in zip(path, path[1:], path[2:]):
            d1 = (cur[0] - prev[0], cur[1] - prev[1])
            d2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            if d1 != d2:
                out.append(cur)
        out.append(path[-1])
        return out

    def path_length_m(self, path) -> float:
        """Path length in metres (4-connected, so steps are axis-aligned)."""
        if not path or len(path) < 2:
            return 0.0
        return (len(path) - 1) * self.resolution


# ----------------------------------------------------------------------
# Dijkstra, kept for the benchmark in the README
# ----------------------------------------------------------------------
def dijkstra(planner: AStarPlanner, start, goal):
    """
    Identical search with h(n) = 0.

    Included purely so the README can quote a real node-expansion count for
    A* vs Dijkstra on this exact map rather than asserting that A* is faster.
    """
    costmap = planner.costmap
    n_rows, n_cols = costmap.shape
    start, goal = tuple(start), tuple(goal)

    open_heap = [(0.0, start)]
    came_from, g_score, closed = {}, {start: 0.0}, set()
    expanded = 0

    while open_heap:
        g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1
        if current == goal:
            return AStarPlanner._reconstruct(came_from, current), expanded
        cr, cc = current
        for dr, dc in AStarPlanner.MOVES:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            step = costmap[nr, nc]
            if not np.isfinite(step):
                continue
            tentative = g + step
            if tentative < g_score.get((nr, nc), np.inf):
                g_score[(nr, nc)] = tentative
                came_from[(nr, nc)] = current
                heapq.heappush(open_heap, (tentative, (nr, nc)))
    return None, expanded
