"""The spatial spectrum a receiver sees, and its strongest arrival directions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .geometry import Camera, Pose


def through_beam(rf_db: np.ndarray, cam: Camera, fwhm_deg: float, floor_db: float) -> np.ndarray:
    """Power a receive beam of `fwhm_deg` pointed at each pixel would measure (dB)."""
    sigma_px = np.radians(fwhm_deg) / 2.355 * cam.fx
    lin = gaussian_filter(10.0 ** (rf_db.astype(np.float64) / 10.0), sigma_px, mode="nearest")
    return np.maximum(10.0 * np.log10(np.maximum(lin, 1e-30)), floor_db).astype(np.float32)


@dataclass
class Arrival:
    """One arrival direction in a spectrum."""
    u: float                     # pixel
    v: float
    power_db: float
    dir_world: np.ndarray        # unit vector from the receiver
    az_deg: float                # in the world x-y plane, from +x
    el_deg: float                # from the horizontal


def arrivals(spec_db: np.ndarray, cam: Camera, pose: Pose, max_n: int = 6, min_sep_deg: float = 12.0,
             rel_db: float = 15.0) -> list[Arrival]:
    """Local maxima of the spectrum, strongest first, at least `min_sep_deg` apart, within `rel_db` of the strongest."""
    r = max(1, int(np.radians(min_sep_deg) / 2 * cam.fx))
    peak = (spec_db >= maximum_filter(spec_db, size=2 * r + 1, mode="nearest")) & (spec_db > spec_db.min() + 1e-3)
    m = max(2, r // 2)
    peak[:m], peak[-m:], peak[:, :m], peak[:, -m:] = False, False, False, False
    vs, us = np.nonzero(peak)
    order = np.argsort(-spec_db[vs, us])
    top = float(spec_db.max())
    cos_sep = np.cos(np.radians(min_sep_deg))
    out: list[Arrival] = []
    for v, u in zip(vs[order], us[order]):
        p = float(spec_db[v, u])
        if p < top - rel_db or len(out) >= max_n:
            break
        d_w = pose.R @ cam.pixel_to_ray(u, v)
        if any(float(d_w @ q.dir_world) > cos_sep for q in out):
            continue
        out.append(Arrival(float(u), float(v), p, d_w, float(np.degrees(np.arctan2(d_w[1], d_w[0]))),
                           float(np.degrees(np.arcsin(np.clip(d_w[2], -1, 1))))))
    return out
