"""Helpers for people in the radio channel: merging their Gaussians and testing line-of-sight blockage."""
from __future__ import annotations

import numpy as np


def merge(bodies: list[dict]) -> dict | None:
    if not bodies:
        return None
    return {k: np.concatenate([b[k] for b in bodies]) for k in bodies[0]}


def segment_hits_body(p0, p1, x: float, y: float, floor_z: float, height: float = 1.75, radius: float = 0.32) -> bool:
    """Does the straight segment p0 -> p1 pass through a person standing at (x, y)? (vertical cylinder test)"""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    d = p1 - p0
    rel = p0[:2] - np.array([x, y])
    a, b, c = d[:2] @ d[:2], rel @ d[:2], rel @ rel - radius ** 2
    if a < 1e-12:
        return bool(c < 0 and floor_z <= p0[2] <= floor_z + height)
    disc = b * b - a * c
    if disc < 0:
        return False
    t0, t1 = max((-b - disc ** 0.5) / a, 0.0), min((-b + disc ** 0.5) / a, 1.0)
    if t0 > t1:
        return False
    z0, z1 = sorted((p0[2] + t0 * d[2], p0[2] + t1 * d[2]))
    return bool(z1 >= floor_z and z0 <= floor_z + height)
