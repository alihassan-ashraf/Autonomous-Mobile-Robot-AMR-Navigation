"""
robot.py
========
Differential-drive AMR: motion model, noisy sensors, EKF state estimation,
path-following controller and the reactive safety layer.

Three distinct notions of "where the robot is" live here, and conflating them
is the classic beginner mistake:

  * `true_pose`      -- ground truth. The simulator knows it; the robot does not.
  * `odom_pose`      -- dead reckoning from wheel encoders. Smooth and
                        continuous, but drifts without bound because every
                        slip and every calibration error integrates forever.
  * `ekf.state`      -- odometry fused with an IMU heading. Still drifts, but
                        far more slowly, because the two sensors fail in
                        uncorrelated ways and the filter weights each by how
                        much it currently trusts them.

Author: Naveed Ali | DecodeLabs Robotics & Automation Internship, Project 3
"""

import numpy as np


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    """
    Extended Kalman Filter over the state [x, y, theta].

    The "extended" part is needed because the differential-drive motion model
    is nonlinear -- heading enters through sin/cos -- so the covariance has to
    be propagated through the Jacobian of that model rather than through a
    constant matrix as in a plain Kalman filter.

    Predict  : push the state forward using the commanded velocities and grow
               the covariance by the process noise Q.
    Update   : correct using the IMU's absolute heading, shrinking covariance
               by the Kalman gain.
    """

    def __init__(self, x: float, y: float, theta: float):
        self.state = np.array([x, y, theta], dtype=float)
        self.P = np.diag([0.01, 0.01, 0.01])
        # Process noise: how much we distrust the motion model per step.
        self.Q = np.diag([0.002, 0.002, 0.004])
        # Measurement noise: how much we distrust the IMU heading.
        self.R = np.array([[0.02]])

    def predict(self, v: float, omega: float, dt: float) -> None:
        x, y, theta = self.state
        self.state = np.array([
            x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            wrap_angle(theta + omega * dt),
        ])
        # Jacobian of the motion model w.r.t. the state
        F = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt],
            [0.0, 1.0,  v * np.cos(theta) * dt],
            [0.0, 0.0,  1.0],
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update_heading(self, measured_theta: float) -> None:
        """Correct the state with an absolute heading measurement."""
        H = np.array([[0.0, 0.0, 1.0]])
        innovation = wrap_angle(measured_theta - self.state[2])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + (K.flatten() * innovation)
        self.state[2] = wrap_angle(self.state[2])
        self.P = (np.eye(3) - K @ H) @ self.P


class DifferentialDriveRobot:
    """A wheeled AMR with encoders, an IMU and a 2D LiDAR."""

    def __init__(self, x: float, y: float, theta: float,
                 radius: float = 0.35,
                 max_linear: float = 0.6, max_angular: float = 1.4,
                 seed: int = 42):
        self.true_pose = np.array([x, y, theta], dtype=float)
        self.odom_pose = np.array([x, y, theta], dtype=float)
        self.ekf = EKF(x, y, theta)

        self.radius = radius
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.rng = np.random.default_rng(seed)

        self.trajectory = [self.true_pose.copy()]
        self.odom_trajectory = [self.odom_pose.copy()]
        self.ekf_trajectory = [self.ekf.state.copy()]

        self.emergency_stops = 0
        self.replans = 0

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def step(self, v: float, omega: float, dt: float, env=None) -> bool:
        """
        Advance the robot by one control cycle.

        Returns False if the commanded motion would drive the chassis into a
        wall -- the simulator refuses the move rather than letting the robot
        clip through geometry, which is what a physics engine would do.
        """
        v = float(np.clip(v, -self.max_linear, self.max_linear))
        omega = float(np.clip(omega, -self.max_angular, self.max_angular))

        # --- Ground truth, with wheel slip ---
        slip = self.rng.normal(1.0, 0.012)          # multiplicative wheel slip
        drift = self.rng.normal(0.0, 0.004)         # small heading bias
        x, y, theta = self.true_pose
        nx = x + v * slip * np.cos(theta) * dt
        ny = y + v * slip * np.sin(theta) * dt
        ntheta = wrap_angle(theta + (omega + drift) * dt)

        if env is not None and self._would_collide(nx, ny, env):
            return False

        self.true_pose = np.array([nx, ny, ntheta])

        # --- Wheel odometry: integrates the *commanded* motion, so it never
        #     sees the slip. This is precisely why it drifts. ---
        ox, oy, otheta = self.odom_pose
        enc_noise = self.rng.normal(1.0, 0.02)
        self.odom_pose = np.array([
            ox + v * enc_noise * np.cos(otheta) * dt,
            oy + v * enc_noise * np.sin(otheta) * dt,
            wrap_angle(otheta + omega * enc_noise * dt),
        ])

        # --- EKF: predict from commands, correct with the IMU ---
        self.ekf.predict(v, omega, dt)
        imu_heading = wrap_angle(self.true_pose[2] + self.rng.normal(0.0, 0.03))
        self.ekf.update_heading(imu_heading)

        self.trajectory.append(self.true_pose.copy())
        self.odom_trajectory.append(self.odom_pose.copy())
        self.ekf_trajectory.append(self.ekf.state.copy())
        return True

    def _would_collide(self, x: float, y: float, env) -> bool:
        """Sample the chassis footprint against the ground-truth world."""
        for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            px = x + self.radius * 0.95 * np.cos(ang)
            py = y + self.radius * 0.95 * np.sin(ang)
            if env.is_occupied(px, py):
                return True
        return env.is_occupied(x, y)

    # ------------------------------------------------------------------
    # Reactive safety layer
    # ------------------------------------------------------------------
    def check_obstacle_ahead(self, scan, cone_deg: float = 50.0,
                             stop_distance: float = 0.75):
        """
        Reflex-level collision check, run every cycle before the path
        controller gets a vote.

        This deliberately does not consult the map or the plan. The global
        planner works on a map that is already a few hundred milliseconds old;
        if a pallet falls off a forklift right now, the only thing that can
        save the robot is the raw scan. Planning is advice, this is a reflex.

        Returns (blocked, nearest_range_in_cone).
        """
        half = np.radians(cone_deg) / 2.0
        angles = scan.angles
        in_cone = np.abs(wrap_angle_array(angles)) <= half
        cone_ranges = scan.ranges[in_cone]
        finite = cone_ranges[np.isfinite(cone_ranges)]
        if finite.size == 0:
            return False, np.inf
        nearest = float(np.min(finite))
        return nearest < stop_distance, nearest

    # ------------------------------------------------------------------
    # Path following
    # ------------------------------------------------------------------
    def follow_path(self, waypoints, lookahead_index: int, dt: float,
                    goal_tolerance: float = 0.25):
        """
        Proportional heading controller toward the current waypoint.

        Turn-in-place dominates when the heading error is large: driving
        forward while badly misaligned carves a wide arc that can clip the
        inside of a corner, so linear speed is scaled down by the cosine of
        the heading error and zeroed past 60 degrees.
        """
        if lookahead_index >= len(waypoints):
            return 0.0, 0.0, lookahead_index

        x, y, theta = self.ekf.state
        tx, ty = waypoints[lookahead_index]

        dx, dy = tx - x, ty - y
        distance = np.hypot(dx, dy)

        if distance < goal_tolerance:
            return 0.0, 0.0, lookahead_index + 1

        heading_error = wrap_angle(np.arctan2(dy, dx) - theta)
        omega = 2.2 * heading_error

        if abs(heading_error) > np.radians(60):
            v = 0.0                       # rotate in place first
        else:
            v = self.max_linear * np.cos(heading_error)
            v = min(v, 0.9 * distance + 0.12)   # ease in near the waypoint

        return v, omega, lookahead_index

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def localisation_error(self):
        """(odometry error, EKF error) against ground truth, in metres."""
        odom_err = float(np.hypot(*(self.odom_pose[:2] - self.true_pose[:2])))
        ekf_err = float(np.hypot(*(self.ekf.state[:2] - self.true_pose[:2])))
        return odom_err, ekf_err


def wrap_angle_array(a: np.ndarray) -> np.ndarray:
    """Vectorised angle wrap into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
