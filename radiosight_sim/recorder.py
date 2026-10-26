"""Write a simulation run to disk in the same shape as our measured scans.

    <run>/
      meta.json                  twin, step, camera, radio window, codebook, scenario
      pedestrians.csv            t_s, name, kind (pedestrian / ue_carrier), x, y, heading_deg, walking
      <ue>/frames.csv            one row per step: pose, receiver position, beam decisions, line of sight
      <ue>/beams.csv             one row per step: power of each codebook beam (dB) -- like a measured beam scan
      <ue>/observations.csv      the people this UE sees: 2D box, visible share, distance, ground position, height
      <ue>/spectrum/000000.npy   spatial spectrum, float16 dB, spectrum_size x spectrum_size (same field of view as rgb)
      <ue>/arrivals.csv          per step: the strongest arrival directions (power, azimuth, elevation, pixel)
      <ue>/rgb/000000.jpg        camera view (if enabled)
      <ue>/depth/000000.npy      expected depth, float16 metres (if enabled)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .sim import Simulator, StepResult


class Recorder:
    def __init__(self, out: str | Path, sim: Simulator, save_rgb: bool = True, save_rf: bool = True,
                 save_depth: bool = False, scenario: dict | None = None, spectrum_size: int = 160):
        self.out, self.sim = Path(out), sim
        self.save_rgb, self.save_rf, self.save_depth = save_rgb, save_rf, save_depth
        self.spectrum_size = spectrum_size
        self.out.mkdir(parents=True, exist_ok=True)
        tw = sim.twin
        cam = tw.camera.scaled(sim.width)
        (self.out / "meta.json").write_text(json.dumps({
            "twin": tw.name, "dt_s": sim.dt, "window_db": tw.window_db,
            "camera": {"width": cam.width, "height": cam.height, "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                       "convention": "pinhole, OpenCV axes (x right, y down, z forward)"},
            "rx_offset_m": tw.rx_offset.tolist(),
            "spectrum_size": self.spectrum_size,
            "codebook": {"name": sim.codebook.name, "theta_phi_deg": sim.codebook.theta_phi_deg.tolist()},
            "base_stations": {k: v.tolist() for k, v in tw.base_stations.items()},
            "ues": list(sim.ue_agents), "pedestrians": list(sim.ped_agents),
            "scenario": scenario or {},
        }, indent=1))
        self._files, self._writers, self.k = {}, {}, 0
        self._writer("pedestrians", self.out / "pedestrians.csv", ["t_s", "name", "kind", "x", "y", "heading_deg", "walking"])
        bs_names = list(tw.base_stations)
        for ue in sim.ue_agents:
            d = self.out / ue
            for sub, on in (("spectrum", save_rf), ("rgb", save_rgb), ("depth", save_depth)):
                if on:
                    (d / sub).mkdir(parents=True, exist_ok=True)
            cols = ["t_s", "x", "y", "z", "qw", "qx", "qy", "qz", "rx_x", "rx_y", "rx_z",
                    "best_beam", "best_theta_deg", "best_phi_deg", "best_db", "peak_u", "peak_v", "peak_db",
                    "peak_dir_x", "peak_dir_y", "peak_dir_z"]
            for b in bs_names:
                cols += [f"{b}_visible", f"{b}_blocked_by", f"{b}_beam_az_deg", f"{b}_beam_el_deg", f"{b}_via"]
            self._writer(f"{ue}/frames", d / "frames.csv", cols)
            self._writer(f"{ue}/beams", d / "beams.csv", ["t_s"] + [f"beam_{i:02d}" for i in range(len(sim.codebook))])
            self._writer(f"{ue}/arrivals", d / "arrivals.csv", ["t_s", "k", "power_db", "az_deg", "el_deg", "dir_x", "dir_y", "dir_z", "u", "v"])
            self._writer(f"{ue}/observations", d / "observations.csv",
                         ["t_s", "name", "kind", "u0", "v0", "u1", "v1", "visible_fraction", "distance_m", "x", "y", "heading_deg", "height_m"])

    def _writer(self, key, path, header):
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "w", newline="")
        self._files[key], self._writers[key] = f, csv.writer(f)
        self._writers[key].writerow(header)

    def write(self, step: StepResult) -> None:
        k, t = self.k, f"{step.t:.3f}"
        for kind, group in (("pedestrian", step.pedestrians), ("ue_carrier", step.carriers)):
            for name, s in group.items():
                self._writers["pedestrians"].writerow([t, name, kind, f"{s.x:.3f}", f"{s.y:.3f}", f"{np.degrees(s.heading):.1f}",
                                                       int(s.phase is not None)])
        for ue, o in step.ues.items():
            b = o.beams
            tp = self.sim.codebook.theta_phi_deg[b.best_beam] if b.best_beam >= 0 else (np.nan, np.nan)
            row = [t, *np.round(o.pose.position, 4), *np.round(o.pose.quat_wxyz, 5), *np.round(o.rx_position, 4),
                   b.best_beam, tp[0], tp[1], f"{b.best_db:.2f}", f"{b.peak_pixel[0]:.0f}", f"{b.peak_pixel[1]:.0f}",
                   f"{b.peak_db:.2f}", *np.round(b.peak_dir_world, 4)]
            for bs in self.sim.twin.base_stations:
                L, B = b.los[bs], b.bs_beam[bs]
                row += [int(L["visible"]), "|".join(L["blocked_by"]), f"{B['azimuth_deg']:.1f}", f"{B['elevation_deg']:.1f}", B["via"]]
            self._writers[f"{ue}/frames"].writerow(row)
            self._writers[f"{ue}/beams"].writerow([t] + [f"{v:.2f}" for v in b.codebook_db])
            for k_, pth in enumerate(o.paths):
                self._writers[f"{ue}/arrivals"].writerow([t, k_, f"{pth.power_db:.2f}", f"{pth.az_deg:.1f}", f"{pth.el_deg:.1f}",
                                                       *np.round(pth.dir_world, 4), f"{pth.u:.0f}", f"{pth.v:.0f}"])
            for o_ in o.sees:
                self._writers[f"{ue}/observations"].writerow([t, o_["name"], o_["kind"], *[f"{x:.1f}" for x in o_["bbox"]], o_["visible_fraction"],
                                                              o_["distance_m"], o_["x"], o_["y"], o_["heading_deg"], o_["height_m"]])
            d = self.out / ue
            if self.save_rf:
                n = self.spectrum_size
                spec = np.asarray(Image.fromarray(o.render.rf_db.astype(np.float32)).resize((n, n), Image.BILINEAR))
                np.save(d / "spectrum" / f"{k:06d}.npy", spec.astype(np.float16))
            if self.save_rgb and o.render.rgb is not None:
                Image.fromarray(o.render.rgb).save(d / "rgb" / f"{k:06d}.jpg", quality=90)
            if self.save_depth and o.render.depth is not None:
                np.save(d / "depth" / f"{k:06d}.npy", o.render.depth.astype(np.float16))
        self.k += 1

    def close(self) -> None:
        for f in self._files.values():
            f.close()
