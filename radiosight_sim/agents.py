"""Agents decide where each UE (receiver) and each pedestrian is at every simulation step.

Every agent has two methods:
    reset(sim, name)     called by Simulator.reset()
    act(t, obs) -> state called once per step; `obs` is the previous step's observations (or None)

A UE agent returns a :class:`~radiosight_sim.geometry.Pose` (the device pose; the receive array sits at a
fixed offset from the camera). A pedestrian agent returns a :class:`PedestrianState`.

Points may be given as (x, y) / (x, y, z) or as landmark names of the twin: a base station name
(e.g. "bs1"), an object name (e.g. "tripod") or "ue_area" (centre of the recorded walk).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import Pose
from .people import STRIDE_M
from .trajectory import Trajectory


@dataclass
class PedestrianState:
    x: float
    y: float
    heading: float                  # rad, direction the person faces
    phase: float | None = None      # gait phase in walk cycles (any real number); None = standing still


# ---------------------------------------------------------------------------- helpers
def _xy(sim, p) -> np.ndarray:
    return np.asarray(sim.twin.landmark(p) if isinstance(p, str) else p, np.float64)[:2]


def _goal_xy(sim, p, standoff: float) -> np.ndarray:
    """Where a person walking to `p` stops: for a base station or an object, `standoff` metres in front of it
    (towards the UE area), so they stand in the room rather than against the furniture."""
    xy = _xy(sim, p)
    if isinstance(p, str) and p != "ue_area" and standoff > 0:
        d = _xy(sim, "ue_area") - xy
        xy = xy + standoff * d / max(np.linalg.norm(d), 1e-6)
    return xy


class _Route:
    """Walk through goals on walkable floor (A* between goals), pausing at each goal."""

    def __init__(self, sim, points, speed: float, pause: float, loop: bool, plan: bool, standoff: float = 0.8):
        goals = [sim.twin.walkable.nearest_free(_goal_xy(sim, p, standoff)) for p in points]
        if loop and len(goals) > 1:
            goals = goals + [goals[0]]
        legs = []
        for a, b in zip(goals[:-1], goals[1:]):
            legs.append(sim.twin.walkable.plan(a, b) if plan else np.array([a, b]))
        self.legs, self.speed, self.pause, self.loop = legs, speed, pause, loop
        self.leg_len = [float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in legs]
        self.leg_time = [L / speed + pause for L in self.leg_len]
        self.cycle = sum(self.leg_time)
        self.start = goals[0]

    def at(self, t: float) -> tuple[np.ndarray, float, bool, float]:
        """position, heading, moving?, distance walked so far"""
        if not self.legs:
            return self.start, 0.0, False, 0.0
        t = t % self.cycle if self.loop else min(t, self.cycle - 1e-9)
        walked = 0.0
        for leg, L, T in zip(self.legs, self.leg_len, self.leg_time):
            if t < T:
                s = min(t * self.speed, L)
                seg = np.diff(leg, axis=0)
                cum = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
                i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
                d = seg[i] / max(np.linalg.norm(seg[i]), 1e-9)
                pos = leg[i] + d * (s - cum[i])
                return pos, float(np.arctan2(d[1], d[0])), s < L, walked + s
            t -= T
            walked += L
        last = self.legs[-1]
        d = last[-1] - last[-2]
        return last[-1], float(np.arctan2(d[1], d[0])), False, walked


# ---------------------------------------------------------------------------- UE agents
class UEAgent:
    def reset(self, sim, name: str) -> None:
        self.sim, self.name = sim, name

    def act(self, t: float, obs) -> Pose:
        raise NotImplementedError


class StaticUE(UEAgent):
    """Stand still at `position` (default: centre of the recorded walk) looking at `look_at`."""

    def __init__(self, position=None, look_at="bs1", pose: Pose | None = None):
        self.position, self.look_at, self.pose = position, look_at, pose

    def reset(self, sim, name):
        super().reset(sim, name)
        if self.pose is None:
            p = np.asarray(sim.twin.landmark("ue_area") if self.position is None else _point3(sim, self.position), np.float64)
            if self.position is None or len(np.atleast_1d(self.position)) == 2 or isinstance(self.position, str):
                p[2] = sim.twin.floor_z + sim.device_height(name)
            tgt = sim.twin.landmark(self.look_at) if isinstance(self.look_at, str) else np.asarray(self.look_at, np.float64)
            self.pose = Pose.look_at(p, tgt)

    def act(self, t, obs):
        return self.pose


class ReplayUE(UEAgent):
    """Replay a trajectory (e.g. ``twin.trajectory("recorded_walk")`` or a recorded activity moved into the twin)."""

    def __init__(self, trajectory: Trajectory, loop: bool = True, start: float = 0.0, rate: float = 1.0):
        self.traj, self.loop, self.start, self.rate = trajectory, loop, start, rate

    def act(self, t, obs):
        return self.traj.pose_at(self.traj.t[0] + self.start + self.rate * t, loop=self.loop)


class WaypointUE(UEAgent):
    """Carry the device through waypoints at walking speed, at the twin's device height.
    The device faces the walking direction, or keeps looking at `look_at` if given."""

    def __init__(self, points, speed: float = 1.0, pause: float = 0.0, loop: bool = True, look_at=None, plan: bool = True):
        self.points, self.speed, self.pause, self.loop, self.look_at, self.plan = points, speed, pause, loop, look_at, plan

    def reset(self, sim, name):
        super().reset(sim, name)
        self.route = _Route(sim, self.points, self.speed, self.pause, self.loop, self.plan)
        self.z = sim.twin.floor_z + sim.device_height(name)

    def act(self, t, obs):
        xy, heading, _, _ = self.route.at(t)
        p = np.r_[xy, self.z]
        if self.look_at is not None:
            tgt = self.sim.twin.landmark(self.look_at) if isinstance(self.look_at, str) else np.asarray(self.look_at)
            return Pose.look_at(p, tgt)
        return Pose.from_heading(p, heading)


class FunctionUE(UEAgent):
    """Any policy: ``fn(t, obs, sim) -> Pose``. `obs` holds the previous step (rf spectrum, beams, ...)."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def act(self, t, obs):
        return self.fn(t, obs, self.sim)


class ExternalUE(UEAgent):
    """Pose supplied from outside every step: ``sim.step({"ue0": pose})``. Keeps the last pose otherwise."""

    def __init__(self, initial: Pose | None = None):
        self.pose = initial

    def reset(self, sim, name):
        super().reset(sim, name)
        if self.pose is None:
            self.pose = Pose.look_at(sim.twin.landmark("ue_area"), sim.twin.landmark(next(iter(sim.twin.base_stations))))

    def act(self, t, obs):
        return self.pose


# ---------------------------------------------------------------------------- pedestrian agents
class PedestrianAgent:
    def reset(self, sim, name: str) -> None:
        self.sim, self.name = sim, name

    def act(self, t: float, obs) -> PedestrianState:
        raise NotImplementedError


class StandingPedestrian(PedestrianAgent):
    def __init__(self, position, heading_deg: float = 0.0):
        self.position, self.heading = position, np.radians(heading_deg)

    def act(self, t, obs):
        x, y = _xy(self.sim, self.position)
        return PedestrianState(x, y, self.heading)


class WaypointPedestrian(PedestrianAgent):
    """Walk between points (planned around furniture), pausing `pause` seconds at each one."""

    def __init__(self, points, speed: float = 1.2, pause: float = 1.0, loop: bool = True, plan: bool = True, start: float = 0.0,
                 standoff: float = 0.8):
        self.points, self.speed, self.pause, self.loop, self.plan, self.start = points, speed, pause, loop, plan, start
        self.standoff = standoff

    def reset(self, sim, name):
        super().reset(sim, name)
        self.route = _Route(sim, self.points, self.speed, self.pause, self.loop, self.plan, self.standoff)

    def act(self, t, obs):
        xy, heading, moving, walked = self.route.at(t + self.start)
        return PedestrianState(float(xy[0]), float(xy[1]), heading, walked / STRIDE_M if moving else None)


class ReplayPedestrian(PedestrianAgent):
    """Follow the ground track (and heading) of a trajectory, e.g. a recorded walk."""

    def __init__(self, trajectory: Trajectory, loop: bool = True, start: float = 0.0):
        self.traj, self.loop, self.start = trajectory, loop, start
        step = np.linalg.norm(np.diff(trajectory.positions[:, :2], axis=0), axis=1)
        self._dist = np.r_[0.0, np.cumsum(step)]

    def act(self, t, obs):
        tt = self.traj.t[0] + self.start + t
        p = self.traj.pose_at(tt, loop=self.loop)
        tq = self.traj.t[0] + ((tt - self.traj.t[0]) % max(self.traj.duration, 1e-9) if self.loop else tt - self.traj.t[0])
        walked = float(np.interp(tq, self.traj.t, self._dist))
        speed = float(np.interp(tq, self.traj.t[1:], np.diff(self._dist) / np.maximum(np.diff(self.traj.t), 1e-6)))
        return PedestrianState(float(p.position[0]), float(p.position[1]), p.yaw, walked / STRIDE_M if speed > 0.25 else None)


class FunctionPedestrian(PedestrianAgent):
    """Any policy: ``fn(t, obs, sim) -> PedestrianState``."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def act(self, t, obs):
        return self.fn(t, obs, self.sim)


class ExternalPedestrian(PedestrianAgent):
    """State supplied from outside: ``sim.step({"p0": PedestrianState(...)})``."""

    def __init__(self, initial: PedestrianState):
        self.state = initial

    def act(self, t, obs):
        return self.state


def _point3(sim, p) -> np.ndarray:
    if isinstance(p, str):
        return sim.twin.landmark(p)
    p = np.asarray(p, np.float64)
    return p if len(p) == 3 else np.r_[p, sim.twin.floor_z + sim.twin.device_height]
