"""Where people can stand: a floor occupancy grid with A* path planning."""
from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
from PIL import Image


class WalkableMap:
    """Boolean grid over the floor. Row 0 is the far +y edge; `origin_xy` is the world (x, y) of the
    lower-left corner of the grid; cells are `res` metres square."""

    def __init__(self, free: np.ndarray, origin_xy, res: float):
        self.free = free.astype(bool)
        self.origin = np.asarray(origin_xy, np.float64)
        self.res = float(res)

    @classmethod
    def load(cls, png: Path, origin_xy, res: float) -> "WalkableMap":
        return cls(np.array(Image.open(png).convert("L")) > 127, origin_xy, res)

    def save(self, png: Path) -> None:
        Image.fromarray((self.free * 255).astype(np.uint8)).save(png)

    def to_cell(self, xy) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        col = np.floor((xy[:, 0] - self.origin[0]) / self.res).astype(int)
        row = self.free.shape[0] - 1 - np.floor((xy[:, 1] - self.origin[1]) / self.res).astype(int)
        return row, col

    def to_xy(self, row, col) -> np.ndarray:
        x = self.origin[0] + (np.asarray(col) + 0.5) * self.res
        y = self.origin[1] + (self.free.shape[0] - 1 - np.asarray(row) + 0.5) * self.res
        return np.stack([x, y], -1)

    def is_free(self, xy) -> np.ndarray:
        r, c = self.to_cell(xy)
        inside = (r >= 0) & (r < self.free.shape[0]) & (c >= 0) & (c < self.free.shape[1])
        out = np.zeros(len(r), bool)
        out[inside] = self.free[r[inside], c[inside]]
        return out

    @property
    def components(self) -> np.ndarray:
        if not hasattr(self, "_labels"):
            from scipy import ndimage
            self._labels, _ = ndimage.label(self.free, structure=np.ones((3, 3)))
        return self._labels

    def nearest_free(self, xy, within=None) -> np.ndarray:
        """The free cell centre closest to `xy` (to put a goal next to, not inside, an obstacle).
        `within`: only consider cells reachable from this point (default: the largest walkable area)."""
        lab = self.components
        if within is not None:
            r, c = self.to_cell(self._snap(within))
            comp = lab[r[0], c[0]]
        else:
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            comp = ids[np.argmax(counts)]
        rows, cols = np.nonzero(lab == comp)
        p = self.to_xy(rows, cols)
        return p[np.argmin(np.linalg.norm(p - np.asarray(xy, np.float64)[:2], axis=1))]

    def _snap(self, xy) -> np.ndarray:
        rows, cols = np.nonzero(self.free)
        p = self.to_xy(rows, cols)
        return p[np.argmin(np.linalg.norm(p - np.asarray(xy, np.float64)[:2], axis=1))]

    def plan(self, start_xy, goal_xy) -> np.ndarray:
        """Shortest 8-connected path through free cells, returned as world (x, y) waypoints."""
        H, W = self.free.shape
        start = self.nearest_free(start_xy)
        (sr,), (sc,) = self.to_cell(start)
        (gr,), (gc,) = self.to_cell(self.nearest_free(goal_xy, within=start))
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        g = {(sr, sc): 0.0}
        parent = {}
        heap = [(0.0, sr, sc)]
        while heap:
            _, r, c = heapq.heappop(heap)
            if (r, c) == (gr, gc):
                break
            for dr, dc, w in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and self.free[nr, nc]:
                    ng = g[(r, c)] + w
                    if ng < g.get((nr, nc), np.inf):
                        g[(nr, nc)] = ng
                        parent[(nr, nc)] = (r, c)
                        heapq.heappush(heap, (ng + np.hypot(nr - gr, nc - gc), nr, nc))
        if (gr, gc) not in g:
            raise ValueError(f"no walkable path from {start_xy} to {goal_xy}")
        cells = [(gr, gc)]
        while cells[-1] != (sr, sc):
            cells.append(parent[cells[-1]])
        cells = np.array(cells[::-1])
        return _shortcut(self, self.to_xy(cells[:, 0], cells[:, 1]))


def _shortcut(m: WalkableMap, path: np.ndarray) -> np.ndarray:
    """Drop waypoints that a straight walkable segment can skip."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            seg = np.linspace(path[i], path[j], max(2, int(np.linalg.norm(path[j] - path[i]) / (m.res / 2))))
            if m.is_free(seg).all():
                break
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)
