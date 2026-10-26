"""Animated people: skinned characters baked into surface samples, rendered as Gaussians.

An avatar file (``people/<name>.npz``) holds one in-place walk
loop: per-frame sample positions and normals, sample colours, the head pose per frame and, optionally,
an XR headset in the head frame. Avatar frame: +z up, facing +x, feet at z = 0.

People are opaque to 28 GHz in the radio channel (they render at the bottom of the radio window).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STRIDE_M = 1.25          # distance covered by one walk loop at ~1.2 m/s


class Avatar:
    def __init__(self, name: str, d: dict):
        self.name = name
        self.pos = d["pos"].astype(np.float32)           # [T, N, 3]
        self.nrm = d["nrm"].astype(np.float32) / 127.0   # [T, N, 3]
        self.rgb = d["rgb"].astype(np.float32) / 255.0   # [N, 3]
        self.head = d["head"].astype(np.float64)         # [T, 4, 4]
        self.loop_s = float(d["loop_s"])
        self.height = float(self.pos[..., 2].max())
        n = self.pos.shape[1]
        self.splat = 0.6 * float(np.sqrt(float(d["area"]) / n))
        self.hs = None
        if "hs_pos" in d:
            self.hs = (d["hs_pos"].astype(np.float32), d["hs_nrm"].astype(np.float32) / 127.0, d["hs_rgb"].astype(np.float32) / 255.0)
        # standing pose: the frame with the feet closest together
        low = self.pos[..., 2] < 0.15
        spread = [np.ptp(self.pos[f][low[f], 0]) if low[f].any() else 9 for f in range(len(self.pos))]
        self.rest_frame = int(np.argmin(spread))

    @classmethod
    def load(cls, path: str | Path) -> "Avatar":
        path = Path(path)
        return cls(path.stem, dict(np.load(path)))

    @property
    def eye_height(self) -> float:
        return float(self.head[self.rest_frame][2, 3])

    def _frame(self, phase):
        T = len(self.pos)
        if phase is None:
            return self.pos[self.rest_frame], self.nrm[self.rest_frame], self.head[self.rest_frame]
        f = (phase % 1.0) * T
        i, a = int(f) % T, f - int(f)
        j = (i + 1) % T
        p = (1 - a) * self.pos[i] + a * self.pos[j]
        n = (1 - a) * self.nrm[i] + a * self.nrm[j]
        h = self.head[i] if a < 0.5 else self.head[j]
        return p, n, h

    def placement(self, x: float, y: float, heading: float, floor_z: float) -> np.ndarray:
        c, s = np.cos(heading), np.sin(heading)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = [x, y, floor_z]
        return T

    def head_pose(self, x, y, heading, floor_z, phase=None) -> np.ndarray:
        """World pose of the point between the eyes (x forward, z up)."""
        return self.placement(x, y, heading, floor_z) @ self._frame(phase)[2]

    def gaussians(self, x: float, y: float, heading: float, floor_z: float, phase: float | None = None,
                  headset: bool = False) -> dict:
        P, Nn, H = self._frame(phase)
        C = self.rgb
        if headset and self.hs is not None:
            hp, hn, hc = self.hs
            P = np.r_[P, hp @ H[:3, :3].T.astype(np.float32) + H[:3, 3].astype(np.float32)]
            Nn = np.r_[Nn, hn @ H[:3, :3].T.astype(np.float32)]
            C = np.r_[C, hc]
        T = self.placement(x, y, heading, floor_z)
        R = T[:3, :3].astype(np.float32)
        Pw = P @ R.T + T[:3, 3].astype(np.float32)
        Nw = Nn @ R.T
        Nw /= np.maximum(np.linalg.norm(Nw, axis=1, keepdims=True), 1e-6)
        light = np.array([0.4, -0.3, 0.85], np.float32)
        light /= np.linalg.norm(light)
        shade = 0.62 + 0.38 * np.abs(Nw @ light)
        n = len(Pw)
        return {
            "means": Pw.astype(np.float32),
            "scales": np.tile(np.array([self.splat, self.splat, self.splat * 0.2], np.float32), (n, 1)),
            "quats": _quats_from_normals(Nw),
            "opacities": np.full(n, 0.99, np.float32),
            "rgb": (C * shade[:, None]).astype(np.float32),
            "rf": np.zeros((n, 1), np.float32),
        }


def _quats_from_normals(n: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0], np.float32)
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    s = np.linalg.norm(axis, axis=1)
    ang = np.arctan2(s, n @ z)
    axis = np.where(s[:, None] > 1e-8, axis / np.maximum(s, 1e-8)[:, None], np.array([1.0, 0.0, 0.0], np.float32))
    return np.concatenate([np.cos(ang / 2)[:, None], axis * np.sin(ang / 2)[:, None]], 1).astype(np.float32)


class Cast:
    """The avatars available to a simulation (all ``*.npz`` in a folder)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.avatars = {p.stem: Avatar.load(p) for p in sorted(self.folder.glob("*.npz"))}
        if not self.avatars:
            raise FileNotFoundError(f"no avatars (*.npz) in {self.folder}; run `radiosight-sim download people`")

    def __getitem__(self, name: str) -> Avatar:
        return self.avatars[name]

    def pick(self, i: int) -> Avatar:
        return list(self.avatars.values())[i % len(self.avatars)]
