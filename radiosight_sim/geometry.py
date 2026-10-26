"""Poses, cameras and small rotation helpers.

Conventions used everywhere in this package:
  * World frame = the twin's frame: metres, +z up (gravity along -z).
  * A pose is camera-to-world with OpenCV camera axes: +x right, +y down, +z forward (the viewing direction).
  * Quaternions are (w, x, y, z).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def slerp(q0, q1, a: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Pose:
    """Camera-to-world pose (OpenCV camera axes) in the twin frame."""

    position: np.ndarray            # (3,)
    quat_wxyz: np.ndarray           # (4,) camera-to-world rotation

    def __post_init__(self):
        self.position = np.asarray(self.position, np.float64).reshape(3)
        q = np.asarray(self.quat_wxyz, np.float64).reshape(4)
        self.quat_wxyz = q / np.linalg.norm(q)

    @property
    def R(self) -> np.ndarray:
        return quat_to_matrix(self.quat_wxyz)

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.position
        return T

    @property
    def forward(self) -> np.ndarray:
        return self.R[:, 2]

    @property
    def yaw(self) -> float:
        """Heading of the viewing direction in the ground plane (rad, from +x towards +y)."""
        f = self.forward
        return float(np.arctan2(f[1], f[0]))

    @classmethod
    def from_matrix(cls, T) -> "Pose":
        T = np.asarray(T, np.float64)
        return cls(T[:3, 3], matrix_to_quat(T[:3, :3]))

    @classmethod
    def look_at(cls, eye, target, up=(0.0, 0.0, 1.0)) -> "Pose":
        eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-6:  # looking straight up or down
            x = np.cross(z, [1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return cls(eye, matrix_to_quat(np.stack([x, y, z], 1)))

    @classmethod
    def from_heading(cls, position, yaw: float, pitch: float = 0.0) -> "Pose":
        """Level camera at `position` looking along `yaw` (rad), tilted by `pitch` (rad, + = up)."""
        d = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
        return cls.look_at(position, np.asarray(position, np.float64) + d)

    def interpolate(self, other: "Pose", a: float) -> "Pose":
        return Pose((1 - a) * self.position + a * other.position, slerp(self.quat_wxyz, other.quat_wxyz, a))


@dataclass
class Camera:
    """Pinhole intrinsics."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Camera":
        f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    def scaled(self, width: int, height: int | None = None) -> "Camera":
        height = height or round(self.height * width / self.width)
        sx, sy = width / self.width, height / self.height
        return Camera(width, height, self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def pixel_to_ray(self, u, v) -> np.ndarray:
        """Unit ray(s) in the camera frame for pixel coordinates."""
        d = np.stack([(np.asarray(u, np.float64) - self.cx) / self.fx, (np.asarray(v, np.float64) - self.cy) / self.fy,
                      np.ones_like(np.asarray(u, np.float64))], -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, d_cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project camera-frame directions; returns u, v and a mask of directions in front of the camera."""
        d = np.atleast_2d(np.asarray(d_cam, np.float64))
        front = d[:, 2] > 1e-6
        z = np.where(front, d[:, 2], 1.0)
        return self.fx * d[:, 0] / z + self.cx, self.fy * d[:, 1] / z + self.cy, front
