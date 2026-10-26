"""Timed 6-DoF trajectories: loading, sampling and moving a recorded motion into a twin."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .geometry import Pose, matrix_to_quat, quat_to_matrix, slerp, yaw_matrix

_COLS = ["t_s", "x", "y", "z", "qw", "qx", "qy", "qz"]


class Trajectory:
    """Device poses (camera-to-world, OpenCV axes) at increasing times.

    `floor_z` is the floor height in the trajectory's own frame, so a motion can be replayed in
    another room at the same height above the floor."""

    def __init__(self, t, positions, quats_wxyz, floor_z: float | None = None, info: dict | None = None):
        self.t = np.asarray(t, np.float64)
        self.positions = np.asarray(positions, np.float64)
        self.quats = np.asarray(quats_wxyz, np.float64)
        self.floor_z = floor_z
        self.info = dict(info or {})

    def __len__(self) -> int:
        return len(self.t)

    def __repr__(self) -> str:
        return f"Trajectory({len(self)} poses, {self.duration:.1f} s, path {self.path_length:.1f} m)"

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        path = Path(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        a = np.array([[float(r[c]) for c in _COLS] for r in rows])
        side = path.with_suffix(".json")
        info = json.loads(side.read_text()) if side.exists() else {}
        return cls(a[:, 0], a[:, 1:4], a[:, 4:8], info.get("floor_z"), info)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_COLS)
            for t, p, q in zip(self.t, self.positions, self.quats):
                w.writerow([f"{t:.4f}", *(f"{v:.5f}" for v in p), *(f"{v:.6f}" for v in q)])
        path.with_suffix(".json").write_text(json.dumps({**self.info, "floor_z": self.floor_z}, indent=1))

    # ------------------------------------------------------------------ sampling
    def pose_at(self, t: float, loop: bool = True) -> Pose:
        t = float(t)
        if loop and self.duration > 0:
            t = self.t[0] + (t - self.t[0]) % self.duration
        t = min(max(t, self.t[0]), self.t[-1])
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a = (t - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)
        return Pose((1 - a) * self.positions[i] + a * self.positions[i + 1], slerp(self.quats[i], self.quats[i + 1], a))

    def resampled(self, rate_hz: float) -> "Trajectory":
        ts = np.arange(self.t[0], self.t[-1], 1.0 / rate_hz)
        poses = [self.pose_at(t, loop=False) for t in ts]
        return Trajectory(ts - ts[0], [p.position for p in poses], [p.quat_wxyz for p in poses], self.floor_z, self.info)

    def segment(self, t0: float, t1: float) -> "Trajectory":
        m = (self.t >= t0) & (self.t <= t1)
        return Trajectory(self.t[m] - self.t[m][0], self.positions[m], self.quats[m], self.floor_z, self.info)

    # ------------------------------------------------------------------ moving into a twin
    def transformed(self, yaw: float, translation, pivot_xy=(0.0, 0.0), dz: float = 0.0) -> "Trajectory":
        """Rotate about the vertical axis through `pivot_xy`, then translate; heights shift by `dz`."""
        R = yaw_matrix(yaw)
        piv = np.array([pivot_xy[0], pivot_xy[1], 0.0])
        p = (self.positions - piv) @ R.T + piv + np.asarray(translation, np.float64) + np.array([0, 0, dz])
        q = np.array([matrix_to_quat(R @ quat_to_matrix(qi)) for qi in self.quats])
        return Trajectory(self.t, p, q, None if self.floor_z is None else self.floor_z + dz, self.info)

    def retarget(self, twin, anchor_xy=None, yaw: float | None = None, yaw_step_deg: float = 10.0,
                 view_weight: float = 0.5) -> "Trajectory":
        """Place this motion in `twin`, keeping the head height above the floor. `anchor_xy` (centre of the
        ground track) and `yaw` (turn, radians) are chosen for you unless given; `info` records the placement."""
        from scipy.spatial import cKDTree
        if self.floor_z is None:
            raise ValueError("trajectory has no floor_z; cannot keep the height above the floor")
        ref = twin.trajectory("recorded_walk")
        ref_fwd = np.array([quat_to_matrix(q)[:, 2] for q in ref.quats[::5]])
        tree = cKDTree(ref.positions[::5, :2])
        centre = self.positions[:, :2].mean(0)
        dz = twin.floor_z - self.floor_z
        sub = slice(None, None, max(1, len(self) // 200))
        fwd = np.array([quat_to_matrix(q)[:, 2] for q in self.quats[sub]])
        if anchor_xy is not None:
            anchors = [np.asarray(anchor_xy, np.float64)[:2]]
        else:
            pts = ref.positions[:, :2]
            keep = [pts[0]]
            for q in pts:  # recorded-walk points at least 0.4 m apart
                if np.min(np.linalg.norm(np.array(keep) - q, axis=1)) > 0.4:
                    keep.append(q)
            anchors = keep
        yaws = [yaw] if yaw is not None else np.radians(np.arange(0.0, 360.0, yaw_step_deg))
        best, best_score, best_terms = None, -np.inf, None
        for a in anchors:
            for y in yaws:
                R = yaw_matrix(y)
                xy = (self.positions[sub, :2] - centre) @ R[:2, :2].T + a
                frac = float(twin.walkable.is_free(xy).mean())
                if frac + view_weight < best_score:
                    continue
                d, i = tree.query(xy, distance_upper_bound=1.0)
                ok = np.isfinite(d)
                view = float(np.mean(np.where(ok, np.einsum("ij,ij->i", fwd @ R.T, ref_fwd[np.minimum(i, len(ref_fwd) - 1)]), -1.0).clip(0, 1)))
                score = frac + view_weight * view
                if score > best_score:
                    best_score, best, best_terms = score, (a, y), (frac, view)
        a, y = best
        out = self.transformed(y, np.r_[a - centre, 0.0], pivot_xy=centre, dz=dz)
        out.info = {**self.info, "retargeted_to": twin.name, "anchor_xy": np.round(a, 3).tolist(),
                    "yaw_deg": round(float(np.degrees(y)), 1), "walkable_fraction": round(best_terms[0], 3),
                    "view_match": round(best_terms[1], 3)}
        return out
