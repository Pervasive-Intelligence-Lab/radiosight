"""Generate a small dataset: every recorded activity, with and without a person walking by.

python examples/03_generate_dataset.py --twins twins --out data_generated --seconds 20
"""
import argparse
from pathlib import Path

from radiosight_sim import RadioTwin, Recorder, ReplayUE, Simulator, Trajectory, WaypointPedestrian

ap = argparse.ArgumentParser()
ap.add_argument("--twins", default="twins")
ap.add_argument("--out", default="data_generated")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--width", type=int, default=320)
a = ap.parse_args()

root = Path(a.twins)
motions = {p.stem: Trajectory.load(p) for p in sorted((root / "motions").glob("*.csv"))}
for twin_dir in sorted(p for p in root.iterdir() if (p / "twin.json").exists()):
    twin = RadioTwin.load(twin_dir)
    runs = {"recorded_walk": twin.trajectory("recorded_walk")}
    runs.update({m: t.retarget(twin) for m, t in motions.items()})
    for motion, traj in runs.items():
        for with_ped in (False, True):
            sim = Simulator(twin, dt=0.1, width=a.width)
            sim.add_ue("ue0", ReplayUE(traj))
            if with_ped:
                pts = ["ue_area", "bs1"] + (["tripod"] if any(o["name"] == "tripod" for o in twin.meta["objects"]) else [])
                sim.add_pedestrian("p0", WaypointPedestrian(pts, speed=1.2))
            out = Path(a.out) / twin.name / f"{motion}{'_pedestrian' if with_ped else ''}"
            rec = Recorder(out, sim, save_rgb=False)
            for step in sim.run(a.seconds):
                rec.write(step)
            rec.close()
            print(f"{out}: {int(a.seconds / sim.dt)} steps")
    del twin
