"""Preview frames: top-down map | camera view | radio spectrum."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import StepResult

UE_COLOURS = [(37, 99, 181), (199, 106, 0), (14, 143, 129), (150, 60, 160)]


def _inferno(x: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps
    return (colormaps["inferno"](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def spectrum_rgb(S: np.ndarray, floor_db: float, dyn_db: float = 15.0, top_db: float | None = None) -> np.ndarray:
    """Colour a spectrum over `dyn_db` below `top_db` (default: its own peak). Nothing above the floor stays dark."""
    top = float(S.max()) if top_db is None else float(top_db)
    if float(S.max()) < floor_db + 3.0:
        return _inferno(np.zeros_like(S))
    vmin = max(top - dyn_db, floor_db)
    return _inferno((S - vmin) / max(top - vmin, 1e-6))


class Previewer:
    def __init__(self, twin, panel: int = 480):
        self.twin, self.panel = twin, panel
        m = twin.meta["map"]
        self.map = Image.open(twin.root / m["image"]).convert("RGB")
        self.map_origin, self.map_res = np.asarray(m["origin_xy"]), float(m["resolution_m"])
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except OSError:
            self.font = ImageFont.load_default()
        self.trail: dict[str, list] = {}
        # one colour scale for the whole video, so a blocked signal shows up as a dimmer spectrum
        walk = twin.trajectory("recorded_walk")
        peaks = [float(twin.render(walk.pose_at(t, loop=False), width=96, rgb=False).rf_db.max())
                 for t in np.linspace(walk.t[0], walk.t[-1], 24)]
        self.top_db = float(np.percentile(peaks, 90))

    def _px(self, xy):
        return ((xy[0] - self.map_origin[0]) / self.map_res, self.map.height - 1 - (xy[1] - self.map_origin[1]) / self.map_res)

    def frame(self, step: StepResult, ue: str | None = None) -> np.ndarray:
        ue = ue or next(iter(step.ues))
        o = step.ues[ue]
        # --- map
        im = self.map.copy()
        d = ImageDraw.Draw(im)
        for name, bs in self.twin.base_stations.items():
            x, y = self._px(bs)
            d.rectangle([x - 7, y - 7, x + 7, y + 7], fill=(220, 40, 40))
            d.text((x + 9, y - 8), name, fill=(160, 20, 20), font=self.font)
        for obj in self.twin.meta["objects"]:
            if obj.get("show_on_map"):
                x, y = self._px(obj["anchor"])
                d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(90, 90, 90), width=2)
                d.text((x + 8, y - 8), obj["name"], fill=(70, 70, 70), font=self.font)
        for i, (name, s) in enumerate(step.pedestrians.items()):
            x, y = self._px((s.x, s.y))
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(60, 60, 60))
            d.line([x, y, x + 16 * np.cos(s.heading), y - 16 * np.sin(s.heading)], fill=(60, 60, 60), width=3)
            d.text((x + 10, y + 4), name, fill=(60, 60, 60), font=self.font)
        hf = np.arctan(0.5 * o.render.camera.width / o.render.camera.fx)
        for i, (name, uo) in enumerate(step.ues.items()):
            c = UE_COLOURS[i % len(UE_COLOURS)]
            tr = self.trail.setdefault(name, [])
            tr.append(self._px(uo.pose.position))
            tr[:] = tr[-200:]
            if len(tr) > 1:
                d.line(tr, fill=c, width=2)
            x, y = tr[-1]
            f = uo.pose.forward
            a0 = np.arctan2(f[1], f[0])
            L = 1.2 / self.map_res
            d.line([(x + L * np.cos(a0 + hf), y - L * np.sin(a0 + hf)), (x, y), (x + L * np.cos(a0 - hf), y - L * np.sin(a0 - hf))],
                   fill=c, width=2)                                  # field of view
            d.polygon([(x + 14 * np.cos(a), y - 14 * np.sin(a)) for a in (a0, a0 + 2.5, a0 - 2.5)], fill=c)
            d.text((x + 10, y + 4), name, fill=c, font=self.font)
            for bs, B in uo.beams.bs_beam.items():
                if B["point"] is None:
                    continue
                p = self._px(B["point"]); q = self._px(self.twin.base_stations[bs])
                col = (40, 160, 70) if B["via"] == "direct" else (230, 150, 20)
                d.line([q, p], fill=col, width=3)
                d.line([p, (x, y)], fill=col, width=3)
        side = max(im.width, im.height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        mp = canvas.resize((self.panel, self.panel))
        # --- camera view and spectrum
        P = self.panel
        rgb = Image.fromarray(o.render.rgb).resize((P, P)) if o.render.rgb is not None else Image.new("RGB", (P, P))
        dr = ImageDraw.Draw(rgb)
        k = P / o.render.camera.width
        names = list(step.ues)
        for seen in o.sees:                                          # what this UE observes: people boxes
            u0, v0, u1, v1 = (np.asarray(seen["bbox"]) * k).tolist()
            col = UE_COLOURS[names.index(seen["name"]) % len(UE_COLOURS)] if seen["kind"] == "ue" else (40, 200, 90)
            dr.rectangle([u0, v0, u1, v1], outline=col, width=3)
            dr.text((u0 + 3, max(0, v0 - 17)), f"{seen['name']} {seen['distance_m']:.1f} m", fill=col, font=self.font)
        sp = o.render.rf_db
        spec = Image.fromarray(spectrum_rgb(sp, self.twin.window_db[0], 20.0, self.top_db)).resize((P, P))
        ds = ImageDraw.Draw(spec)
        sc = P / o.render.camera.width
        cam = o.render.camera
        u, v, front = cam.project(self.twin_codebook_dirs())
        for i, (uu, vv, ff) in enumerate(zip(u, v, front)):
            if ff and 0 <= uu < cam.width and 0 <= vv < cam.height:
                r = 7 if i == o.beams.best_beam else 2
                ds.ellipse([uu * sc - r, vv * sc - r, uu * sc + r, vv * sc + r],
                           outline=(255, 255, 255), width=2 if i == o.beams.best_beam else 1)
        out = Image.new("RGB", (3 * P, P + 32), (250, 249, 246))
        out.paste(mp, (0, 32)); out.paste(rgb, (P, 32)); out.paste(spec, (2 * P, 32))
        dt = ImageDraw.Draw(out)
        L = o.beams.los.get(next(iter(self.twin.base_stations)), {})
        state = "blocked by " + ",".join(L["blocked_by"]) if L.get("blocked_by") else ("line of sight" if L.get("visible") else "no line of sight")
        dt.text((8, 8), f"{self.twin.name}   t = {step.t:5.1f} s", fill=(24, 40, 62), font=self.font)
        dt.text((P + 8, 8), f"{ue} camera view", fill=(24, 40, 62), font=self.font)
        sc = P / o.render.camera.width
        for pth in o.paths:
            ds.ellipse([pth.u * sc - 7, pth.v * sc - 7, pth.u * sc + 7, pth.v * sc + 7], outline=(120, 220, 255), width=2)
        dt.text((2 * P + 8, 8), f"28 GHz spectrum | best beam {o.beams.best_db:.1f} dB | {state}",
                fill=(24, 40, 62), font=self.font)
        return np.asarray(out)

    def twin_codebook_dirs(self):
        cb = np.radians(np.asarray(self.twin.meta["codebook"]["theta_phi_deg"], np.float64))
        t, p = cb[:, 0], cb[:, 1]
        return np.stack([-np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
