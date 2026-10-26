"""From a rendered spectrum to beam decisions: codebook scan, strongest direction, line of sight,
and the base-station beam found by tracing the strongest direction back to the surface it came from
(single bounce, as in the RadioSight paper)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bodies import segment_hits_body


@dataclass
class Codebook:
    """Receive beams of the measuring array. Directions are in the camera frame (OpenCV axes)."""

    theta_phi_deg: np.ndarray
    dirs_cam: np.ndarray
    name: str = ""

    @classmethod
    def from_twin(cls, twin) -> "Codebook":
        cb = twin.meta["codebook"]
        tp = np.asarray(cb["theta_phi_deg"], np.float64)
        t, p = np.radians(tp[:, 0]), np.radians(tp[:, 1])
        d = np.stack([-np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
        return cls(tp, d, cb.get("name", ""))

    def __len__(self) -> int:
        return len(self.theta_phi_deg)


@dataclass
class BeamReport:
    codebook_db: np.ndarray                 # [B] power of each receive beam (dB, twin window); NaN if outside the image
    best_beam: int                          # index into the codebook
    best_db: float
    peak_pixel: tuple[float, float]         # (u, v) of the strongest pixel
    peak_db: float
    peak_dir_world: np.ndarray              # unit vector from the receiver
    los: dict = field(default_factory=dict)        # base station -> {"visible": bool, "blocked_by": [names]}
    bs_beam: dict = field(default_factory=dict)    # base station -> {"azimuth_deg", "elevation_deg", "via": "direct"|"reflection", "point": [x,y,z]}


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = img.shape
    out = np.full(len(u), np.nan)
    ok = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    u0, v0 = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
    u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
    a, b = u[ok] - u0, v[ok] - v0
    out[ok] = ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u1] + (1 - a) * b * img[v1, u0] + a * b * img[v1, u1])
    return out


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return img
    p = np.pad(img, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _az_el(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


def analyze(twin, render, codebook: Codebook, pedestrians: dict, depth: np.ndarray, ped_height: float = 1.75,
            direct_cone_deg: float = 12.0, direct_margin_db: float = 3.0, bs_zone_m: float = 0.8,
            max_steer_deg: float = 80.0) -> BeamReport:
    """`render` is a RenderResult; `pedestrians` maps name -> PedestrianState; `depth` is the expected depth image.
    The direct path is used when the base station is visible and unblocked and either the strongest direction is within
    `direct_cone_deg` of it or the spectrum towards it is within `direct_margin_db` of the strongest direction.
    `bs_zone_m`: surfaces this close to a base station are its own (blurred) radiation, not reflectors.
    `max_steer_deg`: how far from its facing direction a base station can steer."""
    cam, pose = render.camera, render.pose
    S = render.rf_db
    direct_cone_deg = max(direct_cone_deg, 0.75 * twin.rx_beam_fwhm)
    rx = twin.rx_position(pose)
    u, v, front = cam.project(codebook.dirs_cam)
    cb_db = _bilinear(S, np.where(front, u, -1), v)
    best = int(np.nanargmax(cb_db)) if np.isfinite(cb_db).any() else -1

    sm = _box_blur(S, max(1, cam.width // 160))
    pv, pu = np.unravel_index(int(np.argmax(sm)), sm.shape)
    ray_cam = cam.pixel_to_ray(pu, pv)
    peak_dir = pose.R @ ray_cam

    # 3D point seen by every other pixel (from the expected depth); back-tracing does not need full resolution
    vv, uu = np.mgrid[0:cam.height:2, 0:cam.width:2]
    d_sub, sm_sub = depth[::2, ::2], sm[::2, ::2]
    rays = np.stack([(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones(uu.shape)], -1)
    pts = pose.position + (rays * d_sub[..., None]) @ pose.R.T
    solid = render.alpha[::2, ::2] > 0.5

    los, bs_beam = {}, {}
    for name, bs in twin.base_stations.items():
        # geometric visibility of the base station from the receiver, from the rendered depth
        d_cam = pose.R.T @ (bs - pose.position)
        bu, bv, f = cam.project(d_cam)
        visible, bs_db = False, -np.inf
        if f[0] and 0 <= bu[0] < cam.width and 0 <= bv[0] < cam.height:
            visible = depth[int(bv[0]), int(bu[0])] >= d_cam[2] - max(0.25, 0.15 * d_cam[2])   # twin depth is less certain far away
            bs_db = float(sm[int(bv[0]), int(bu[0])])
        blocked = [n for n, st in pedestrians.items() if segment_hits_body(rx, bs, st.x, st.y, twin.floor_z, ped_height)]
        los[name] = {"visible": bool(visible), "blocked_by": blocked}
        to_bs = (bs - rx) / np.linalg.norm(bs - rx)
        ang = np.degrees(np.arccos(np.clip(peak_dir @ to_bs, -1, 1)))
        # direct path: geometrically visible, not blocked, and the twin puts it within `direct_margin_db` of the best
        if visible and not blocked and (ang < direct_cone_deg or bs_db >= float(sm[pv, pu]) - direct_margin_db):
            az, el = _az_el(rx - bs)
            bs_beam[name] = {"azimuth_deg": az, "elevation_deg": el, "via": "direct", "point": rx.round(3).tolist()}
            continue
        # back-trace: the strongest direction whose surface is a real reflector, i.e. not the base station's own
        # blurred hot spot (within `bs_zone_m` of it, or behind it as seen from the receiver)
        rel = pts - rx
        along = rel @ to_bs
        cosang = along / np.maximum(np.linalg.norm(rel, axis=-1), 1e-6)
        bs_zone = (np.linalg.norm(pts - bs, axis=-1) < bs_zone_m) | ((along > np.linalg.norm(bs - rx) - 0.3) & (cosang > np.cos(np.radians(25))))
        # the base station can only aim into its front half-space; its array is taken to face the room (see twin.json)
        facing = twin.bs_facing(name)
        rel_bs = pts - bs
        in_front = rel_bs @ facing > np.cos(np.radians(max_steer_deg)) * np.linalg.norm(rel_bs, axis=-1)
        cand = np.where(solid & ~bs_zone & in_front, sm_sub, -np.inf)
        if not np.isfinite(cand).any():
            bs_beam[name] = {"azimuth_deg": float("nan"), "elevation_deg": float("nan"), "via": "none", "point": None}
            continue
        rv, ru = np.unravel_index(int(np.argmax(cand)), cand.shape)
        hit = pts[rv, ru]
        az, el = _az_el(hit - bs)
        bs_beam[name] = {"azimuth_deg": az, "elevation_deg": el, "via": "reflection", "point": hit.round(3).tolist(),
                         "reflection_db": float(S[2 * rv, 2 * ru])}
    return BeamReport(cb_db, best, float(cb_db[best]) if best >= 0 else float("nan"), (float(pu), float(pv)),
                      float(S[pv, pu]), peak_dir, los, bs_beam)
