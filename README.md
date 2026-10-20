# Project 3 — Autonomous Mobile Robot (AMR) Navigation

**DecodeLabs Robotics & Automation Internship, Batch 2026**
A wheeled robot that maps an unknown warehouse from simulated LiDAR, plans an
optimal route with A*, and re-routes around obstacles that appear while it is
driving.

---

## What this is

A complete navigation stack written from first principles — no ROS, no Nav2,
no `move_base`. Every component the brief lists is implemented directly:
occupancy grid mapping, costmap inflation, A* search, EKF sensor fusion and a
two-layer obstacle-avoidance scheme.

| Component | File |
|---|---|
| Ground-truth world + simulated 360-beam LiDAR | [`src/environment.py`](src/environment.py) |
| Occupancy grid mapping (log-odds inverse sensor model) | [`src/occupancy_grid.py`](src/occupancy_grid.py) |
| A* planner, costmap inflation, Dijkstra benchmark | [`src/astar_planner.py`](src/astar_planner.py) |
| Differential-drive robot, EKF, path controller, safety reflex | [`src/robot.py`](src/robot.py) |
| Main perceive → map → plan → act loop | [`src/navigation_sim.py`](src/navigation_sim.py) |
| Figure generation | [`src/visualise.py`](src/visualise.py) |

The robot never reads the ground-truth world. It may only call `lidar_scan()`,
which returns a `LaserScan` exactly as an RPLIDAR driver would publish on
`/scan`. Everything it knows is reconstructed from those range readings.

---

## Run it

```bash
pip install -r ../requirements.txt
python src/navigation_sim.py
```

Runs the full mission and writes figures plus `output/mission_report.json`.
Takes about a minute. `scipy` is optional — the distance transform falls back
to a pure-numpy Felzenszwalb implementation if it is absent.

---

## Results

```
  goal_reached               True
  sim_time_s                 100.2
  distance_travelled_m       48.58
  map_coverage_pct           84.7
  emergency_stops            1
  replans                    13
  mean_plan_time_ms          148.2
  final_odom_error_m         0.436
  final_ekf_error_m          0.047      <- 89% better than odometry alone
```

The robot starts at (2, 2) with **no map at all** and reaches (18, 18) through
a serpentine warehouse it has never seen, while two unexpected obstacles are
dropped onto its route mid-run.

---

## 1. Where am I? — SLAM mapping

![Ground truth vs the map the robot built](output/1_ground_truth_vs_slam.png)

Left is the real world; the robot never sees it. Right is what it reconstructed
from LiDAR alone, with its driven path overlaid.

**Cells are accumulated in log-odds, not overwritten.** A single scan is noisy.
If each cell took the latest reading, one spurious return would punch a
permanent hole through a wall and one missed return would erase it. In log
space, probabilities multiply as additions — every beam passing *through* a
cell nudges it toward free, every beam terminating *in* one nudges it toward
occupied, and a cell only flips once the evidence accumulates. The update is a
single `+=`, which keeps it cheap at 360 beams × 10 Hz.

Two details that matter:

- **Max-range returns write free space only.** `inf` means "nothing out
  there", not "a wall at exactly 6 m". Treating a no-return as an obstacle
  hallucinates a wall ring at the sensor's range limit.
- **Mapping uses the robot's *estimated* pose, not ground truth.** That is the
  honest version — a map built on ground truth would hide every localisation
  error the EKF failed to catch.

---

## 2. Sensor fusion — why odometry alone is not enough

![Odometry drift vs EKF](output/4_localisation_error.png)

Wheel odometry integrates the *commanded* motion, so it never sees wheel slip.
Every slip and every calibration error accumulates and **never comes back** —
the error curve only climbs, reaching **0.44 m** by the end of a 48 m run.

An Extended Kalman Filter fuses that drifting odometry with an IMU heading.
"Extended" because the differential-drive motion model is nonlinear — heading
enters through `sin`/`cos` — so covariance is propagated through the model's
Jacobian rather than a constant matrix. The two sensors fail in uncorrelated
ways, so the filter weights each by how much it currently trusts it.

Result: **0.047 m final error, an 89% reduction.** At 0.05 m grid resolution,
0.44 m of drift is nearly nine cells — enough to write walls into the wrong
place and plan a route through solid racking.

---

## 3. Where am I going? — costmap inflation and A*

![Inflated costmap and the A* route](output/2_costmap_and_path.png)

**A* is a point-search algorithm and the robot is 0.7 m wide.** Planning on
the raw grid (left) yields a mathematically perfect path that scrapes the
chassis along every wall. The costmap (right) fixes this in layers, the way
Nav2 does:

| Layer | Effect |
|---|---|
| **Lethal** | obstacle cells themselves |
| **Inscribed** | within one robot radius — centre here is a guaranteed collision |
| **Inflated** | beyond the radius, cost decays exponentially, pulling the path to the centre of the aisle |
| **Unknown** | flat penalty — traversable, but only if nothing known-free exists |

The exponential decay is what produces the rainbow bands. The planner is free
to cut a corner tightly when it has to, but pays for it, so in open aisles it
drives down the middle like a competent forklift driver.

### Heuristic choice: Manhattan, not Euclidean

The robot expands 4-connected neighbours, so it physically cannot travel a
diagonal. Euclidean distance reports a shorter distance than any achievable
route — it stays *admissible*, so A* still returns the optimal path, but it
under-informs the search and expands far more nodes. Manhattan is exactly the
true remaining cost on an obstacle-free 4-grid: admissible **and** tight.

### A* vs Dijkstra, measured on the final map

| | Nodes expanded | Path length | Time |
|---|---|---|---|
| **A\*** (Manhattan) | **16,224** | 51.6 m | 78 ms |
| Dijkstra (`h=0`) | 24,337 | 51.6 m | 169 ms |

Identical paths — both are optimal — but A* proves it with **1.5× fewer
expansions**.

That ratio is lower than the textbook figure, and the reason is worth stating:
this maze is a *serpentine*. The heuristic points straight at the goal while
the only real route doubles back twice, so its guidance is actively misleading
for much of the search. A* degrades gracefully toward Dijkstra exactly when the
heuristic stops being informative — it never does worse, which is the property
that matters. In an open warehouse the same code shows a far wider margin.

### Planning resolution

Mapping runs at a fine resolution; planning does not need to. At 0.05 m the
grid is 400×400 and a full re-plan takes ~450 ms — not a real-time planner. At
**0.10 m** the cell count quarters, a re-plan drops to ~80 ms, and the 0.25 m
racking walls still resolve across 2–3 cells. Real stacks do exactly this: map
finely, plan coarsely.

---

## 4. How do I avoid that wall? — two-layer obstacle avoidance

![Dynamic re-route](output/3_dynamic_reroute.png)

The brief asks the robot to "stop **or** re-route". It needs both, at
different latencies, and they are separate code paths on purpose:

| Layer | Input | Latency | Action |
|---|---|---|---|
| **Reflex** | raw LaserScan, 50° cone | 1 cycle (100 ms) | hard stop |
| **Deliberative** | accumulated map | ~1 s | re-plan around it |

**The reflex deliberately consults neither the map nor the plan.** The global
planner works on a map that is already a few hundred milliseconds old. If a
pallet falls off a forklift *right now*, the only thing that can save the
robot is the raw scan. Planning is advice; this is a reflex.

Both layers are exercised by dropping obstacles onto the robot's live route at
two different lead distances:

```
t = 30.0s   obstacle 3.0 m dead ahead, in no prior map
t = 31.0s   path invalidated -> re-routed, 17.1 m -> 25.9 m
            (never slowed down; the deliberative layer absorbed it)

t = 62.0s   obstacle 1.1 m dead ahead  <- inside the sense->plan->act window
t = 62.1s   EMERGENCY STOP, obstacle at 0.58 m
t = 62.1s   re-planned from a standstill, 19.0 m
```

At 3 m there is time to map and re-plan and the robot never breaks stride. At
1.1 m a new path would not exist until after impact — only the reflex can
save it. A system with just the deliberative layer hits things.

Obstacle positions are resolved **at runtime** by walking a fixed distance
along the active path, so the test cannot silently degrade into "obstacle
appeared somewhere the robot was never going".

### Re-plan damping

Two restrictions stop the robot re-planning itself into uselessness:

- **Only the path ahead is checked**, within a 6 m horizon. A wall discovered
  15 m away will very likely be planned around for free by the time the robot
  arrives, as the map keeps improving.
- **A 3 s cooldown** between deliberative re-plans, so a burst of map updates
  cannot trigger several near-identical re-plans in a row.

Measured on the same seed, holding everything else fixed:

| Horizon check | Cooldown | Re-plans | Goal reached |
|---|---|---|---|
| whole remaining path | off | 28 | yes |
| 6 m ahead | off | 14 | yes |
| 6 m ahead | 3 s | **13** | yes |

The horizon restriction does nearly all the work — halving re-plans on its
own — because most invalidations were distant cells that resolved themselves
before the robot got near them. The cooldown trims one more. Neither changed
the route driven or the distance travelled by more than a metre, so this is
CPU saved rather than path quality traded away.

---

## 5. The run, animated

![Navigation animation](output/5_navigation.gif)

Grey fades to white as the map fills in; blue is the current plan, orange the
path actually driven.

---

## Control loop

The whole project, in the order it executes:

```python
every cycle (10 Hz):
    scan  = lidar()                       # perceive
    map.integrate(scan, ekf_pose)         # SLAM, mapping half

    if obstacle in safety cone:           # reflex layer
        stop(); replan()
    elif remaining path now blocked:      # deliberative layer
        replan()

    v, w = follow_path(waypoints)         # act
    robot.step(v, w)                      # EKF predict + IMU correct
```

Before any of that, the robot spins 360° on the spot. One scan from a standing
start sees only the aisle directly ahead, and planning on that produces a
route confidently drawn through unexplored space.

---

## Design decisions worth defending

**Unknown space is traversable but penalised.** A planner that refuses to
enter unknown cells cannot move at all on the first scan — everything is
unknown. A planner that treats unknown as free drives into walls. A flat
penalty means "go there only if nothing known-free works", which is what
frontier exploration does.

**Goals inside the inflation band are nudged, not rejected.** An operator
clicking slightly too close to a shelf is routine. The planner spirals outward
to the nearest cell the robot's centre can legally occupy rather than
returning "unreachable".

**Paths are simplified to waypoints before driving.** A* returns one cell per
step — hundreds of points for a 20 m route. The controller only needs the
corners, and a shorter list means fewer heading corrections and smoother
motion.

**Turn-in-place dominates when heading error is large.** Driving forward while
badly misaligned carves a wide arc that can clip the inside of a corner, so
linear speed scales with the cosine of heading error and is zeroed past 60°.

---

## Known limitations

Stated plainly, because they bound what this code can claim:

- **The map→odom correction is not closed.** The EKF fuses odometry with an
  IMU, which bounds heading drift but not position drift — there is no scan
  matching, so the `map → odom` transform is never corrected. A full SLAM
  stack (GMapping, Cartographer) closes that loop by aligning each scan
  against the accumulated map. Over a much longer run, position error would
  grow without bound even with the EKF.
- **No loop closure.** Revisiting a previously mapped area will not retroactively
  correct the trajectory.
- **4-connected motion** produces axis-aligned paths. 8-connected movement
  with a diagonal-aware heuristic would be shorter but needs corner-cutting
  checks to avoid clipping wall corners diagonally.
- **The dynamic obstacle is static once placed.** Genuinely moving obstacles
  (people, other AMRs) need velocity estimation and a time-indexed costmap —
  the robot would have to plan around where a pedestrian *will be*.

---

*Naveed Ali — DecodeLabs Robotics & Automation Internship, Batch 2026*
