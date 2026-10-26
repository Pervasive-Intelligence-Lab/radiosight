<a href="https://radioverum.com/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/radioverum-horizontal-dark.svg">
    <img alt="RadioVerum" src="assets/radioverum-horizontal.svg" width="200">
  </picture>
</a>

# RadioSight

**Powered by [RadioVerum](https://radioverum.com/)**

**Predictive mmWave XR Network Optimization from Dynamic Neural Radio Fields**

Lihao Zhang, Paul Kudyba, Zhenlin An, Haijian Sun · University of Georgia · ACM MobiCom 2026

[Project page](https://pervasive-intelligence-lab.github.io/radiosight/) · [Paper (ACM)](https://doi.org/10.1145/3795866.3844465) · [arXiv](https://arxiv.org/abs/2608.29504) · [Cite](#cite)

![RadioSight overview. Left: the physical world, with an access point, an XR user whose predicted path is drawn as a dashed arrow, and a person walking past. Right: the neural radio field, where the system finds a reflected path to the user and switches to it.](assets/fig1-overview.jpg)

> **This release: one measured radio twin, ready to query.**
> RadioSight is part of [RadioVerum](https://radioverum.com/). More rooms, hardware and tools join the RadioVerum
> system in upcoming versions.

The twin is a lab room we measured at 28 GHz with Project Aria glasses and a phased-array receiver. With
`radiosight-sim` you put XR headsets and people into it and get what each headset would see:

- the camera view, the **28 GHz spatial spectrum** and the power of all **55 receive beams**, from any head pose;
- **recorded head motion**: the walk we took while measuring the room, five recorded XR activities (walking, gaming,
  interacting, navigating), or your own trajectories;
- **people who block the signal**: animated people who walk planned routes, or go where you send them;
- **several XR users** at once; each headset reports **who it sees** (where, how far, how much of them is visible);
- control of every headset and person from Python, or a short YAML file;
- the data in one folder per run, with a preview video.

![Six generated clips: the measurement walk; a recorded walking activity; a recorded gaming activity while someone walks by; someone walking past; a person crossing the line of sight of a standing headset; two XR users and a pedestrian. Each shows the floor map, the headset's camera view and the 28 GHz spectrum it receives.](assets/previews.jpg)

*Generated, not filmed. Videos on the [project page](https://pervasive-intelligence-lab.github.io/radiosight/#previews).*

## 1. Install

Linux with an NVIDIA GPU (about 2 GB of GPU memory is enough). Tested with Python 3.10, PyTorch 2.4.1 and gsplat 1.5.3.

```bash
conda create -n radiosight python=3.10 -y && conda activate radiosight
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install gsplat==1.5.3 --extra-index-url https://docs.gsplat.studio/whl/pt24cu124   # prebuilt CUDA kernels
git clone https://github.com/Pervasive-Intelligence-Lab/radiosight.git
cd radiosight && pip install -e .
```

With another PyTorch version, `pip install gsplat` also works; it compiles its CUDA kernels the first time it
runs, which needs the CUDA toolkit.

## 2. Download

```bash
radiosight-sim download all --dest twins        # about 240 MB
```

| Folder | What it is |
|---|---|
| `twins/nlos_tx1` | The twin: a lab room with the transmitter on a desk behind a drawer cabinet (the paper's NLoS Tx1 scene). Labelled objects: tripod, cabinet, chairs. Includes the 14-minute measurement walk. |
| `twins/motions` | Five XR activities recorded with the same glasses, 40–54 s each: `walking_1`, `walking_2`, `gaming` (standing, quick head turns), `interacting`, `navigating`. |
| `twins/people` | Three animated people: `kate`, `martha`, `remy`. |

## 3. Run

```bash
radiosight-sim info twins/nlos_tx1                       # what is in the twin
radiosight-sim render twins/nlos_tx1 --out view.png      # one view: camera image | spectrum
radiosight-sim run scenarios/nlos_tx1_recorded_walk_pedestrian.yaml --out runs/demo
```

`run` writes the data, a `preview.mp4`, and a `report.html` with the video and how to load the data. About 20 steps
per second at 480 × 480 on an RTX 4090.

| Scenario | Headset | People |
|---|---|---|
| `nlos_tx1_recorded_walk.yaml` | the measurement walk | none |
| `nlos_tx1_recorded_walk_pedestrian.yaml` | the measurement walk | one person walking between the headset, the base station and the tripod |
| `nlos_tx1_static_pedestrian.yaml` | standing, facing the base station | one person crossing its line of sight |
| `nlos_tx1_motion_walk_around.yaml` | the `walking_1` activity | none |
| `nlos_tx1_motion_gaming_pedestrian.yaml` | the `gaming` activity | one person walking by |
| `nlos_tx1_multi_ue.yaml` | two XR users: the measurement walk and the `walking_1` activity | one more person walking |

A scenario is a few lines of YAML; the top of [`radiosight_sim/scenario.py`](radiosight_sim/scenario.py) lists every
option.

```yaml
twin: twins/nlos_tx1
dt: 0.1
duration: 60
ues:
  - name: ue0
    body: remy                                  # optional: draw a person wearing this headset
    agent: {type: replay, file: twins/motions/gaming.csv, retarget: true}
pedestrians:
  - name: p0
    avatar: kate
    agent: {type: waypoints, points: [ue_area, bs1, tripod], speed: 1.2, pause: 1.0}
```

## 4. Control everything from Python

```python
from radiosight_sim import (RadioTwin, Simulator, Trajectory, ReplayUE, WaypointPedestrian,
                            ExternalPedestrian, PedestrianState)

twin = RadioTwin.load("twins/nlos_tx1")
sim = Simulator(twin, dt=0.1)
sim.add_ue("ue0", ReplayUE(twin.trajectory("recorded_walk")), body="remy")            # the measurement walk, worn by remy
# sim.add_ue("ue0", ReplayUE(Trajectory.load("twins/motions/walking_1.csv").retarget(twin)))
sim.add_pedestrian("p0", WaypointPedestrian(["ue_area", "bs1", "tripod"], speed=1.2), avatar="kate")   # walks around furniture
sim.add_pedestrian("p1", ExternalPedestrian(PedestrianState(-1.5, 0.5, heading=0.0)), avatar="martha")  # you move this one

sim.reset()
for k in range(300):
    step = sim.step({"p1": PedestrianState(-1.5 + 0.01 * k, 0.5, heading=0.0)})
    o = step.ues["ue0"]
    o.render.rf_db            # 28 GHz spatial spectrum (dB)
    o.beams.codebook_db       # power of each of the 55 receive beams
    o.beams.los["bs1"]        # {'visible': ..., 'blocked_by': [...]}
    o.beams.bs_beam["bs1"]    # base-station beam: direct, or via the reflection found by back-tracing
    o.sees                    # who ue0 sees: name, 2D box, visible share, distance, position
```

| Headset agents | People agents |
|---|---|
| `StaticUE`, `ReplayUE`, `WaypointUE`, `FunctionUE(fn)`, `ExternalUE` | `StandingPedestrian`, `WaypointPedestrian`, `ReplayPedestrian`, `FunctionPedestrian(fn)`, `ExternalPedestrian` |

`FunctionUE` and `FunctionPedestrian` take any policy `fn(t, previous_step, sim)`, so an agent can react to what it
measured in the previous step. [`examples/02_custom_agents.py`](examples/02_custom_agents.py) has a headset that turns
towards its strongest direction and a person walked across its line of sight. More in
[docs/AGENT_API.md](docs/AGENT_API.md).

## 5. What a run writes

```
runs/demo/
  ue0/spectrum/000123.npy  28 GHz spatial spectrum (dB), same view as the camera
  ue0/rgb/000123.jpg       camera view
  ue0/beams.csv            power of the 55 receive beams per step (dB)
  ue0/frames.csv           head pose, best beam, line of sight and base-station beam per step
  ue0/arrivals.csv         strongest arrival directions per step
  ue0/observations.csv     the people ue0 sees per step
  pedestrians.csv          where every person is
  meta.json                twin, camera, receive beams, the scenario
  preview.mp4, report.html
```

Column by column: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).
[`examples/03_generate_dataset.py`](examples/03_generate_dataset.py) runs every recorded activity, with and without
a person walking by.

## Cite

```bibtex
@inproceedings{zhang2026radiosight,
  title     = {RadioSight: Predictive mmWave XR Network Optimization from Dynamic Neural Radio Fields},
  author    = {Zhang, Lihao and Kudyba, Paul and An, Zhenlin and Sun, Haijian},
  booktitle = {Proceedings of the 32nd Annual International Conference on Mobile Computing and Networking (MobiCom '26)},
  year      = {2026},
  address   = {Austin, TX, USA},
  publisher = {ACM},
  doi       = {10.1145/3795866.3844465}
}
```

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Twin and recorded activities: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The animated people are built from Mixamo characters and animations (Adobe) and are provided for use with this
simulator under Adobe's terms. The RadioVerum name and logo are not covered by these licences.

---

<sub>Measurement platform: Project Aria glasses with a TMYTEK 28 GHz phased-array receiver and USRP software radios
running GNU Radio. The hardware details and the platform software are available on request:
contact@radioverum.com.</sub>
