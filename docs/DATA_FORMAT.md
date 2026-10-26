# Data formats

## Conventions

- **World frame**: the twin's frame, metres, +z up (gravity along −z). It is the gravity-aligned world of the
  capture's Aria SLAM map. `twin.json → frame.floor_z` is the floor height.
- **Poses** are camera-to-world with OpenCV camera axes: +x right, +y down, +z forward (viewing direction).
  Quaternions are (w, x, y, z).
- **Receive array**: its phase centre sits at `radio.rx_offset_m` from the camera, in camera axes. Spectra are
  rendered from there.
- **Receive beams**: beam *(θ, φ)* points along (−sin θ cos φ, sin θ sin φ, cos θ) in camera axes. The codebook has
  55 beams, θ ∈ {0, 15, 30, 45}°, φ in 20° steps.

## The twin

The twin folder (`twins/nlos_tx1`) is read by `RadioTwin.load`. `twin.json` lists what you can query by name:

| Key | Content |
|---|---|
| `frame` | units, up axis, `floor_z` |
| `provenance` | capture, date, hardware, paper |
| `radio` | frequency, what the spectrum means, `rx_offset_m` |
| `camera` | the capture camera (pinhole, undistorted, upright) |
| `codebook` | the 55 receive beams |
| `base_stations` | name and position of each transmitter |
| `objects` | name, anchor position and bounding box of each labelled object (tripod, cabinet, chairs) |
| `navigation` | where people can walk, the device height, the outline of the recorded walk |
| `trajectories` | recorded trajectories in this twin |

## People

`people/` holds the animated people (`kate`, `martha`, `remy`), read by `Simulator`. Pick them by name.

## Trajectories (`*.csv` + `*.json`)

```
t_s, x, y, z, qw, qx, qy, qz        # camera-to-world pose, 30 Hz
```

The sidecar `.json` gives `floor_z` (the floor height in the trajectory's own frame) and where the motion comes
from. Trajectories in `motions/` are in the frame of their own recording; `Trajectory.retarget(twin)` moves them
into a twin at the same height above the floor.

## Generated data (a `run` folder)

| File | Content |
|---|---|
| `meta.json` | twin, `dt_s`, camera intrinsics, `rx_offset_m`, receive beams, base stations, the scenario |
| `pedestrians.csv` | `t_s, name, kind` (`pedestrian` or `ue_carrier`), `x, y, heading_deg, walking` |
| `<ue>/observations.csv` | the people this UE sees: `t_s, name, kind, u0, v0, u1, v1` (2D box, pixels), `visible_fraction` (share of the person not hidden), `distance_m`, ground position `x, y`, `heading_deg`, `height_m` |
| `<ue>/frames.csv` | `t_s`, pose (`x y z qw qx qy qz`), receive-array position (`rx_*`), `best_beam`, `best_theta_deg`, `best_phi_deg`, `best_db`, strongest pixel (`peak_u`, `peak_v`, `peak_db`) and its world direction (`peak_dir_*`); per base station: `visible`, `blocked_by`, `beam_az_deg`, `beam_el_deg`, `via` (`direct` or `reflection`) |
| `<ue>/beams.csv` | `t_s, beam_00 … beam_54`: power of each receive beam (dB); empty if outside the image |
| `<ue>/spectrum/NNNNNN.npy` | 28 GHz spatial spectrum, float16 dB, 160 × 160, same field of view as the camera |
| `<ue>/arrivals.csv` | `t_s, k, power_db, az_deg, el_deg, dir_x, dir_y, dir_z, u, v`: the strongest arrival directions (k = 0 strongest) |
| `report.html` | an overview of the run and how to load it |
| `<ue>/rgb/NNNNNN.jpg` | camera view |
| `<ue>/depth/NNNNNN.npy` | expected depth (if enabled) |

`<bs>_via` says whether the base station reaches the headset directly or via a reflection. Azimuth is measured in
the world x-y plane from +x, elevation from the horizontal.
