# Agents

A simulation has **UEs** (devices that receive) and **pedestrians** (people who block). Each is driven by an agent.
At every step the simulator asks each agent where its entity is, inserts the pedestrians, renders every UE's view
and spectrum, and works out the beams.

```python
class UEAgent:
    def reset(self, sim, name): ...          # called by sim.reset()
    def act(self, t, obs) -> Pose: ...       # device pose at time t; obs = previous StepResult (or None)

class PedestrianAgent:
    def reset(self, sim, name): ...
    def act(self, t, obs) -> PedestrianState: ...   # PedestrianState(x, y, heading, phase)
```

Positions can be `(x, y)`, `(x, y, z)`, or a landmark name: a base station (`"bs1"`), a labelled object
(`"tripod"`, `"cabinet"`), or `"ue_area"` (a recorded device position in the middle of the recorded walk).

## Built-in agents

| Agent | What it does |
|---|---|
| `StaticUE(position=None, look_at="bs1")` | stands still; default position `ue_area` |
| `ReplayUE(trajectory, loop=True, start=0, rate=1)` | replays a trajectory (the measurement walk, a recorded activity, your own) |
| `WaypointUE(points, speed=1.0, look_at=None)` | carried along a planned path at device height; faces the walking direction or `look_at` |
| `FunctionUE(fn)` | `fn(t, obs, sim) -> Pose`, any policy |
| `ExternalUE()` | pose given each step: `sim.step({"ue0": pose})` |
| `StandingPedestrian(position, heading_deg)` | stands still |
| `WaypointPedestrian(points, speed=1.2, pause=1.0, standoff=0.8)` | walks between points on walkable floor (A* around furniture); stops `standoff` m in front of base stations and objects |
| `ReplayPedestrian(trajectory)` | follows the ground track of a trajectory |
| `FunctionPedestrian(fn)` | `fn(t, obs, sim) -> PedestrianState` |
| `ExternalPedestrian(initial_state)` | state given each step: `sim.step({"p0": state})` |

## People

`Simulator(twin, people="auto")` loads the animated people from a `people/` folder next to the twin
(`radiosight-sim download people`). `add_pedestrian(name, agent, avatar="kate")` picks who walks; without `avatar`
the cast is used in turn. `add_ue(name, agent, body="remy")` draws a person wearing the UE (with an XR headset) in the
other UEs' views; that person also blocks their signals. For `StaticUE` and `WaypointUE` the device then sits at the
person's eye height. A UE never sees its own carrier.

A walking person's gait follows the distance they cover (`PedestrianState.phase`, in walk cycles; `None` = standing).

## What an agent can read

`obs` is the previous `StepResult`:

```python
obs.t
obs.pedestrians["p0"]            # PedestrianState
o = obs.ues["ue0"]               # UEObservation
o.pose, o.rx_position
o.render.rf_db, o.render.rgb, o.render.depth, o.render.camera   # spectrum (dB), camera, depth
o.paths                          # strongest arrival directions (power_db, az_deg, el_deg, dir_world)
o.beams.codebook_db, o.beams.best_beam, o.beams.best_db
o.beams.peak_pixel, o.beams.peak_db, o.beams.peak_dir_world
o.beams.los["bs1"]               # {"visible": bool, "blocked_by": [names]}
o.beams.bs_beam["bs1"]           # {"azimuth_deg", "elevation_deg", "via", "point"}
o.sees                           # [{"name", "kind": "pedestrian" | "ue", "bbox": [u0, v0, u1, v1], "visible_fraction",
                                 #   "distance_m", "x", "y", "heading_deg", "height_m"}, ...]
obs.carriers                     # where the people carrying the UEs stand (PedestrianState)
```

The twin itself is available as `sim.twin`: `landmark(name)`, `walkable.is_free(xy)`, `walkable.plan(a, b)`,
`render(pose, ...)`, `move_object(name, translation, yaw_deg)`.

## Replaying a recorded activity

```python
traj = Trajectory.load("twins/motions/gaming.csv")
moved = traj.retarget(twin)                      # placed for you
moved = traj.retarget(twin, anchor_xy=(-1.0, 0.5), yaw=0.0)   # or fix them
print(moved.info)   # where it was placed
```

The headset keeps its height above the floor. Without `anchor_xy`/`yaw`, a place and heading in the room are chosen
for you.

## Moving objects

```python
twin.move_object("cabinet", translation=(0.6, 0.0, 0.0), yaw_deg=30)
twin.reset_objects()
```

The object moves with its radio values.
