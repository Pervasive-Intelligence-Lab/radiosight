"""Drive the agents yourself, step by step.

- ue0 walks between two points carried at device height and always faces the base station.
- p0 is controlled from this script: it walks back and forth across the line of sight.
- ue1 follows its own spectrum: every step it turns a little towards the strongest direction.

python examples/02_custom_agents.py twins/nlos_tx1
"""
import sys

import numpy as np

from radiosight_sim import (ExternalPedestrian, FunctionUE, PedestrianState, Pose, RadioTwin, Simulator,
                            WaypointUE)

twin = RadioTwin.load(sys.argv[1] if len(sys.argv) > 1 else "twins/nlos_tx1")
sim = Simulator(twin, dt=0.1, width=320)
ue_area = twin.landmark("ue_area")
bs = twin.landmark("bs1")

sim.add_ue("ue0", WaypointUE([ue_area[:2] + [-0.8, 0.0], ue_area[:2] + [0.8, 0.0]], speed=0.6, look_at="bs1"))


def follow_peak(t, obs, sim):
    """Turn 20 % of the way towards last step's strongest direction (a simple beam-tracking head)."""
    if obs is None or "ue1" not in obs.ues:
        return Pose.look_at(ue_area + [0.0, 0.6, 0.0], bs)
    o = obs.ues["ue1"]
    target = o.pose.forward + 0.2 * (o.beams.peak_dir_world - o.pose.forward)
    return Pose.look_at(o.pose.position, o.pose.position + target)


sim.add_ue("ue1", FunctionUE(follow_peak))

# the pedestrian walks on the segment between the UE area and the base station
mid = 0.5 * (ue_area[:2] + bs[:2])
across = np.array([-(bs - ue_area)[1], (bs - ue_area)[0]])
across /= np.linalg.norm(across)
sim.add_pedestrian("p0", ExternalPedestrian(PedestrianState(*mid, 0.0)))

sim.reset()
for k in range(150):
    s = 1.2 * np.sin(2 * np.pi * k / 75)                       # 1.2 m either side of the line of sight
    p = mid + s * across
    step = sim.step({"p0": PedestrianState(p[0], p[1], np.arctan2(across[1], across[0]))})
    if k % 10 == 0:
        for name, o in step.ues.items():
            los = o.beams.los["bs1"]
            print(f"t={step.t:4.1f}s {name}: best beam {o.beams.best_beam:2d} {o.beams.best_db:5.1f} dB | "
                  f"bs1 visible={los['visible']!s:5} blocked_by={los['blocked_by']} | bs1 beam via {o.beams.bs_beam['bs1']['via']}")
