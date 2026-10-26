"""The simulation loop: move agents, insert people, render each UE's view and spectrum, decide beams,
and report which people each UE can see (multi-UE observation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .agents import PedestrianAgent, PedestrianState, UEAgent
from .beams import BeamReport, Codebook, analyze
from .bodies import merge
from .geometry import Pose
from .people import STRIDE_M, Avatar, Cast
from .spectrum import arrivals
from .twin import RadioTwin, RenderResult


@dataclass
class UEObservation:
    t: float
    pose: Pose
    rx_position: np.ndarray
    render: RenderResult          # .rf_db, .rgb, .depth, .alpha
    beams: BeamReport
    sees: list = field(default_factory=list)   # people in view: dicts with name, kind, bbox, visible_fraction, ...
    paths: list = field(default_factory=list)  # strongest arrival directions (spectrum.Arrival)


@dataclass
class StepResult:
    t: float
    ues: dict[str, UEObservation]
    pedestrians: dict[str, PedestrianState]
    carriers: dict[str, PedestrianState] = field(default_factory=dict)   # people carrying the UEs


@dataclass
class _Person:
    avatar: Avatar | None
    headset: bool


class Simulator:
    """Example::

        twin = RadioTwin.load("twins/nlos_tx1")
        sim = Simulator(twin, dt=0.1)                       # people from twins/people
        sim.add_ue("ue0", ReplayUE(twin.trajectory("recorded_walk")), body="remy")
        sim.add_pedestrian("p0", WaypointPedestrian(["ue_area", "bs1", "tripod"]), avatar="kate")
        for step in sim.run(30.0):
            print(step.t, step.ues["ue0"].beams.best_beam, [s["name"] for s in step.ues["ue0"].sees])
    """

    def __init__(self, twin: RadioTwin, dt: float = 0.1, width: int = 480, rgb: bool = True,
                 people: str | Path | Cast = "auto", pedestrian_height: float = 1.75):
        self.twin, self.dt, self.width, self.rgb = twin, dt, width, rgb
        self.ped_height = pedestrian_height
        self.codebook = Codebook.from_twin(twin)
        if people == "auto":
            people = Cast(twin.root.parent / "people")
        elif not isinstance(people, Cast):
            people = Cast(people)
        self.cast = people
        self.ue_agents: dict[str, UEAgent] = {}
        self.ped_agents: dict[str, PedestrianAgent] = {}
        self._ped_person: dict[str, _Person] = {}
        self._ue_person: dict[str, _Person] = {}
        self.t = 0.0
        self.last: StepResult | None = None

    def _avatar(self, name, index):
        return self.cast[name] if name else self.cast.pick(index)

    def add_ue(self, name: str, agent: UEAgent, body: str | None = None) -> None:
        """`body`: an avatar name to draw a person carrying this UE (with a headset) in the other UEs' views."""
        self.ue_agents[name] = agent
        if body:
            self._ue_person[name] = _Person(self._avatar(body, 0), headset=True)

    def add_pedestrian(self, name: str, agent: PedestrianAgent, avatar: str | None = None) -> None:
        """`avatar`: which person to draw (default: the next one in the cast)."""
        self.ped_agents[name] = agent
        self._ped_person[name] = _Person(self._avatar(avatar, len(self._ped_person) + 1), headset=False)

    def device_height(self, ue_name: str) -> float:
        """Height of a UE above the floor for agents that place it themselves: the carrier's eye height if the UE
        has a body, otherwise the height of the recorded walk."""
        p = self._ue_person.get(ue_name)
        if p is not None and p.avatar is not None:
            return p.avatar.eye_height
        return self.twin.device_height

    def reset(self) -> None:
        self.t, self.last = 0.0, None
        self._carrier_walked = {n: 0.0 for n in self._ue_person}
        self._carrier_prev = {}
        for name, a in {**self.ue_agents, **self.ped_agents}.items():
            a.reset(self, name)

    # ------------------------------------------------------------------ people
    def _body(self, person: _Person, s: PedestrianState) -> dict:
        return person.avatar.gaussians(s.x, s.y, s.heading, self.twin.floor_z, s.phase, headset=person.headset)

    def _carrier_state(self, name: str, pose: Pose) -> PedestrianState:
        """Where the person carrying a UE stands: their eyes at the device, facing where it looks."""
        av = self._ue_person[name].avatar
        heading = pose.yaw
        xy = pose.position[:2].copy()
        if av is not None:
            off = av.head[av.rest_frame][:2, 3]                 # eyes relative to the body centre
            c, s = np.cos(heading), np.sin(heading)
            xy -= [c * off[0] - s * off[1], s * off[0] + c * off[1]]
        prev = self._carrier_prev.get(name)
        moved = 0.0 if prev is None else float(np.linalg.norm(xy - prev))
        self._carrier_prev[name] = xy
        self._carrier_walked[name] += moved
        walking = moved / self.dt > 0.25
        return PedestrianState(float(xy[0]), float(xy[1]), heading, self._carrier_walked[name] / STRIDE_M if walking else None)

    # ------------------------------------------------------------------ step
    def step(self, commands: dict | None = None) -> StepResult:
        """Advance one step. `commands` overrides agents for this step: {ue_name: Pose, ped_name: PedestrianState};
        external agents keep the last command they were given."""
        commands = commands or {}
        peds = {}
        for name, a in self.ped_agents.items():
            if name in commands:
                a.state = commands[name]
            peds[name] = commands.get(name) or a.act(self.t, self.last)
        poses = {}
        for name, a in self.ue_agents.items():
            if name in commands:
                a.pose = commands[name]
            poses[name] = commands.get(name) or a.act(self.t, self.last)
        carriers = {n: self._carrier_state(n, poses[n]) for n in self._ue_person}

        people = {n: (self._ped_person[n], s, "pedestrian") for n, s in peds.items()}
        people.update({n: (self._ue_person[n], s, "ue") for n, s in carriers.items()})
        bodies = {n: self._body(p, s) for n, (p, s, _) in people.items()}

        ues = {}
        for name, pose in poses.items():
            others = {n: b for n, b in bodies.items() if n != name}        # a UE does not see its own carrier
            r = self.twin.render(pose, width=self.width, rgb=self.rgb, depth=True, extra=merge(list(others.values())))
            paths = arrivals(r.rf_db, r.camera, pose)
            blockers = {n: people[n][1] for n in others}
            rep = analyze(self.twin, r, self.codebook, blockers, r.depth, self.ped_height)
            sees = self._observe(r, {n: (people[n], bodies[n]) for n in others})
            ues[name] = UEObservation(self.t, pose, self.twin.rx_position(pose), r, rep, sees, paths)
        self.last = StepResult(self.t, ues, peds, carriers)
        self.t += self.dt
        return self.last

    def _observe(self, r: RenderResult, persons: dict) -> list:
        """2D box and visible share of each person in a UE's view (from its own rendered depth)."""
        cam, pose = r.camera, r.pose
        out = []
        for name, ((person, s, kind), body) in persons.items():
            pts = body["means"][:: max(1, len(body["means"]) // 400)].astype(np.float64)
            pc = (pts - pose.position) @ pose.R
            front = pc[:, 2] > 0.1
            if not front.any():
                continue
            u = cam.fx * pc[front, 0] / pc[front, 2] + cam.cx
            v = cam.fy * pc[front, 1] / pc[front, 2] + cam.cy
            inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            if not inside.any():
                continue
            ui, vi = u[inside].astype(int), v[inside].astype(int)
            vis = pc[front][inside, 2] <= r.depth[vi, ui] + 0.05
            frac = float(vis.sum()) / len(pts)
            if frac < 0.03:
                continue
            h = person.avatar.height if person.avatar is not None else self.ped_height
            out.append({"name": name, "kind": kind,
                        "bbox": [float(u[inside].min()), float(v[inside].min()), float(u[inside].max()), float(v[inside].max())],
                        "visible_fraction": round(frac, 3),
                        "distance_m": round(float(np.hypot(s.x - pose.position[0], s.y - pose.position[1])), 3),
                        "x": round(s.x, 3), "y": round(s.y, 3), "heading_deg": round(float(np.degrees(s.heading)), 1),
                        "height_m": round(h, 2)})
        return out

    def run(self, duration: float, reset: bool = True) -> Iterator[StepResult]:
        if reset:
            self.reset()
        for _ in range(int(round(duration / self.dt))):
            yield self.step()
