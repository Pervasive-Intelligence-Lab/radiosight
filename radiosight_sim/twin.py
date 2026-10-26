"""A measured radio twin: 3D Gaussians that carry both colour and 28 GHz receive power, plus scene metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import Camera, Pose, yaw_matrix, matrix_to_quat
from .io_ply import read_gaussians
from .navigation import WalkableMap
from .sh import eval_sh
from .spectrum import through_beam


@dataclass
class RenderResult:
    rf_db: np.ndarray          # [H, W] spatial spectrum: power through a receive beam pointed at each pixel (dB)
    rgb: np.ndarray | None     # [H, W, 3] uint8
    depth: np.ndarray | None   # [H, W] metres along the optical axis (expected depth)
    alpha: np.ndarray          # [H, W] accumulated opacity
    pose: Pose
    camera: Camera


class RadioTwin:
    """Load with :meth:`RadioTwin.load` and render camera views and radio spectra from any 6-DoF pose."""

    def __init__(self, root: Path, meta: dict, g: dict, device: str = "cuda"):
        self.root, self.meta, self.device = root, meta, torch.device(device)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        self._means0 = t(g["means"])
        self._quats0 = torch.nn.functional.normalize(t(g["quats_wxyz"]), dim=-1)
        self.means, self.quats = self._means0.clone(), self._quats0.clone()
        self.scales = torch.exp(t(g["log_scales"]))
        self.opacities = torch.sigmoid(t(g["opacities"]))
        self.rgb_sh = t(g["rgb_sh"])
        self.rf_sh = t(g["rf_sh"])
        self.object_id = torch.as_tensor(g["object_id"], device=self.device)
        radio = meta["radio"]
        self.window_db = tuple(radio["window_db"])
        self.rx_offset = np.asarray(radio["rx_offset_m"], np.float64)
        self.rx_beam_fwhm = float(radio.get("rx_beam_fwhm_deg", 18.0))
        c = meta["camera"]
        self.camera = Camera(c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"])
        self.floor_z = float(meta["frame"]["floor_z"])
        nav = meta["navigation"]
        self.device_height = float(nav["device_height_m"])
        self.walkable = WalkableMap.load(root / nav["walkable"], nav["origin_xy"], nav["resolution_m"])
        self.ue_area = np.asarray(nav["ue_area"], np.float64)
        self._traj_cache = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, root: str | Path, device: str = "cuda") -> "RadioTwin":
        root = Path(root)
        meta = json.loads((root / "twin.json").read_text())
        g = read_gaussians(root / meta["gaussians"])
        if g["rf_sh"] is None:
            raise ValueError(f"{root} has no radio fields; is it a RadioSight twin?")
        return cls(root, meta, g, device)

    @property
    def name(self) -> str:
        return self.meta["name"]

    def __repr__(self) -> str:
        return (f"RadioTwin({self.name!r}, {len(self.means):,} Gaussians, window {self.window_db} dB, "
                f"base stations {list(self.base_stations)}, objects {[o['name'] for o in self.meta['objects']]})")

    # ------------------------------------------------------------------ scene facts
    @property
    def base_stations(self) -> dict[str, np.ndarray]:
        return {b["name"]: np.asarray(b["position"], np.float64) for b in self.meta["base_stations"]}

    def bs_facing(self, name: str) -> np.ndarray:
        """Unit vector the base station's array faces: `facing` in twin.json, else towards the room."""
        b = next(b for b in self.meta["base_stations"] if b["name"] == name)
        if b.get("facing") is not None:
            f = np.asarray(b["facing"], np.float64)
        else:
            f = np.r_[self.ue_area.mean(0) - np.asarray(b["position"])[:2], 0.0]
        return f / np.linalg.norm(f)

    def object(self, name: str) -> dict:
        for o in self.meta["objects"]:
            if o["name"] == name:
                return o
        raise KeyError(f"no object {name!r}; objects are {[o['name'] for o in self.meta['objects']]}")

    def landmark(self, name: str) -> np.ndarray:
        """3D position of a base station, an object anchor, or 'ue_area' (centre of the recorded walk)."""
        if name in self.base_stations:
            return self.base_stations[name]
        if name == "ue_area":
            # the recorded device position closest to the middle of the recorded walk (a place the device really was)
            walk = self.trajectory("recorded_walk").positions
            return walk[np.argmin(np.linalg.norm(walk[:, :2] - self.ue_area.mean(0), axis=1))].copy()
        return np.asarray(self.object(name)["anchor"], np.float64)

    def trajectory(self, name: str):
        from .trajectory import Trajectory
        if name not in self._traj_cache:
            self._traj_cache[name] = Trajectory.load(self.root / self.meta["trajectories"][name])
        return self._traj_cache[name]

    # ------------------------------------------------------------------ editing
    def move_object(self, name: str, translation=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> None:
        """Move a labelled object (about its anchor); its radio values move with it."""
        o = self.object(name)
        m = self.object_id == o["id"]
        a = torch.as_tensor(o["anchor"], dtype=torch.float32, device=self.device)
        R = torch.as_tensor(yaw_matrix(np.radians(yaw_deg)), dtype=torch.float32, device=self.device)
        self.means[m] = (self._means0[m] - a) @ R.T + a + torch.as_tensor(translation, dtype=torch.float32, device=self.device)
        qR = torch.as_tensor(matrix_to_quat(yaw_matrix(np.radians(yaw_deg))), dtype=torch.float32, device=self.device)
        self.quats[m] = _quat_mul(qR.expand_as(self._quats0[m]), self._quats0[m])

    def reset_objects(self) -> None:
        self.means, self.quats = self._means0.clone(), self._quats0.clone()

    # ------------------------------------------------------------------ rendering
    def rx_position(self, pose: Pose) -> np.ndarray:
        """Phase centre of the receive array for a device pose (fixed offset in the camera frame)."""
        return pose.position + pose.R @ self.rx_offset

    @torch.no_grad()
    def render(self, pose: Pose, camera: Camera | None = None, width: int | None = 480, rgb: bool = True,
               depth: bool = False, extra=None) -> RenderResult:
        """Render the radio spectrum (and optionally colour and depth) seen from `pose`.

        camera: intrinsics; default = the capture's pinhole camera resized to `width`.
        extra:  optional dict of additional Gaussians (e.g. pedestrians from :mod:`bodies`) with
                activated values: means, scales, quats, opacities, rgb [N,3] in 0..1, rf [N,1] in 0..1.
        """
        from gsplat import rasterization

        cam = camera or (self.camera.scaled(width) if width else self.camera)
        eye = torch.as_tensor(pose.position, dtype=torch.float32, device=self.device)
        rx = torch.as_tensor(self.rx_position(pose), dtype=torch.float32, device=self.device)
        colors = [torch.clamp_min(eval_sh(3, self.rgb_sh, self.means - eye) + 0.5, 0.0)] if rgb else []
        colors.append(torch.clamp(eval_sh(3, self.rf_sh, self.means - rx) + 0.5, 0.0, 1.0))
        means, quats, scales, opac, col = self.means, self.quats, self.scales, self.opacities, torch.cat(colors, -1)
        if extra is not None and len(extra["means"]):
            e = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in extra.items()}
            ecol = torch.cat(([e["rgb"]] if rgb else []) + [e["rf"]], -1)
            means, quats = torch.cat([means, e["means"]]), torch.cat([quats, e["quats"]])
            scales, opac, col = torch.cat([scales, e["scales"]]), torch.cat([opac, e["opacities"]]), torch.cat([col, ecol])
        w2c = np.linalg.inv(pose.matrix)
        out, alpha, _ = rasterization(
            means, quats, scales, opac, col,
            torch.as_tensor(w2c, dtype=torch.float32, device=self.device)[None],
            torch.as_tensor(cam.K, dtype=torch.float32, device=self.device)[None],
            cam.width, cam.height, render_mode="RGB+ED", packed=False, near_plane=0.02, far_plane=1e3)
        out = out[0]
        lo, hi = self.window_db
        rf = out[..., 3 if rgb else 0].clamp(0.0, 1.0)
        return RenderResult(
            rf_db=through_beam((lo + rf * (hi - lo)).cpu().numpy(), cam, self.rx_beam_fwhm, lo),
            rgb=(out[..., :3].clamp(0, 1) * 255).byte().cpu().numpy() if rgb else None,
            depth=out[..., -1].cpu().numpy() if depth else None,
            alpha=alpha[0, ..., 0].cpu().numpy(),
            pose=pose, camera=cam)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)
