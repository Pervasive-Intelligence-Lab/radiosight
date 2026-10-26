"""Build a simulation from a YAML scenario file (no Python needed).

    twin: twins/nlos_tx1            # folder of the twin (relative to the scenario file or the cwd)
    dt: 0.1                         # seconds per step
    duration: 30
    width: 480                      # image width of the rendered views
    rgb: true
    people: twins/people            # animated people (default: a `people` folder next to the twin)
    ues:
      - name: ue0
        body: remy                      # optional: draw a person wearing this UE (seen and blocking in other UEs' views)
        agent: {type: replay, trajectory: recorded_walk}                 # a trajectory stored in the twin
      # {type: replay, file: twins/motions/gaming.csv, retarget: true}   # a recorded activity, replayed in this twin
      # {type: static, position: ue_area, look_at: bs1}
      # {type: waypoints, points: [ue_area, tripod], speed: 1.0, look_at: bs1}
    pedestrians:
      - name: p0
        avatar: kate                    # kate | martha | remy (default: next in the cast)
        agent: {type: waypoints, points: [ue_area, bs1, tripod], speed: 1.2, pause: 1.0, standoff: 0.8}
      # {type: standing, position: [-1.8, 0.6], heading_deg: 90}
    record: {rgb: true, rf: true, depth: false, video: true}
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import agents as A
from .sim import Simulator
from .trajectory import Trajectory
from .twin import RadioTwin


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    return base / q


def ue_agent(spec: dict, twin: RadioTwin, base: Path) -> A.UEAgent:
    kind = spec["type"]
    if kind == "replay":
        traj = twin.trajectory(spec["trajectory"]) if "trajectory" in spec else Trajectory.load(_resolve(base, spec["file"]))
        if spec.get("retarget", "file" in spec):
            traj = traj.retarget(twin, anchor_xy=spec.get("anchor"), yaw=spec.get("yaw"))
        return A.ReplayUE(traj, loop=spec.get("loop", True), start=spec.get("start", 0.0), rate=spec.get("rate", 1.0))
    if kind == "static":
        return A.StaticUE(spec.get("position"), spec.get("look_at", "bs1"))
    if kind == "waypoints":
        return A.WaypointUE(spec["points"], spec.get("speed", 1.0), spec.get("pause", 0.0), spec.get("loop", True), spec.get("look_at"))
    raise ValueError(f"unknown UE agent type {kind!r} (replay | static | waypoints)")


def pedestrian_agent(spec: dict, twin: RadioTwin, base: Path) -> A.PedestrianAgent:
    kind = spec["type"]
    if kind == "waypoints":
        return A.WaypointPedestrian(spec["points"], spec.get("speed", 1.2), spec.get("pause", 1.0), spec.get("loop", True),
                                    start=spec.get("start", 0.0), standoff=spec.get("standoff", 0.8))
    if kind == "standing":
        return A.StandingPedestrian(spec["position"], spec.get("heading_deg", 0.0))
    if kind == "replay":
        traj = twin.trajectory(spec["trajectory"]) if "trajectory" in spec else Trajectory.load(_resolve(base, spec["file"]))
        if spec.get("retarget", "file" in spec):
            traj = traj.retarget(twin, anchor_xy=spec.get("anchor"), yaw=spec.get("yaw"))
        return A.ReplayPedestrian(traj, start=spec.get("start", 0.0))
    raise ValueError(f"unknown pedestrian agent type {kind!r} (waypoints | standing | replay)")


def load_scenario(path: str | Path, twin_dir: str | Path | None = None, device: str = "cuda") -> tuple[Simulator, dict]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    twin = RadioTwin.load(twin_dir or _resolve(path.parent, cfg["twin"]), device=device)
    people = cfg.get("people", "auto")
    if people != "auto" and people is not None:
        people = _resolve(path.parent, people)
    sim = Simulator(twin, dt=cfg.get("dt", 0.1), width=cfg.get("width", 480), rgb=cfg.get("rgb", True), people=people)
    for u in cfg.get("ues", []):
        sim.add_ue(u["name"], ue_agent(u["agent"], twin, path.parent), body=u.get("body"))
    for p in cfg.get("pedestrians", []) or []:
        sim.add_pedestrian(p["name"], pedestrian_agent(p["agent"], twin, path.parent), avatar=p.get("avatar"))
    return sim, cfg
