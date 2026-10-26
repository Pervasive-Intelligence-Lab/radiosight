"""Read and write radio twin Gaussians as a binary PLY.

The vertex layout is the standard 3D Gaussian Splatting one, so any 3DGS viewer can open the file
and show the visual scene, plus RadioSight's extra fields:

    x y z                       position (m)
    f_dc_0..2, f_rest_0..44     RGB spherical harmonics, degree 3 (f_rest is channel-major, as in 3DGS)
    opacity                     logit
    scale_0..2                  log of the three axis lengths (m)
    rot_0..3                    quaternion w, x, y, z
    rf_dc_0, rf_rest_0..14      radio spherical harmonics, degree 3, one channel
    object_id                   0 = background, otherwise an id from twin.json "objects"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_FLOAT_FIELDS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
    + ["rf_dc_0"]
    + [f"rf_rest_{i}" for i in range(15)]
)


def write_gaussians(path: str | Path, g: dict[str, np.ndarray]) -> None:
    """Write arrays (means [N,3], rgb_sh [N,16,3], opacities [N], log_scales [N,3], quats_wxyz [N,4],
    rf_sh [N,16,1], object_id [N]) to a binary little-endian PLY."""
    n = len(g["means"])
    dtype = [(name, "<f4") for name in _FLOAT_FIELDS] + [("object_id", "<u2")]
    v = np.empty(n, dtype=dtype)
    for i, a in enumerate("xyz"):
        v[a] = g["means"][:, i]
    for i in range(3):
        v[f"f_dc_{i}"] = g["rgb_sh"][:, 0, i]
    rest = np.transpose(g["rgb_sh"][:, 1:, :], (0, 2, 1)).reshape(n, 45)  # channel-major
    for i in range(45):
        v[f"f_rest_{i}"] = rest[:, i]
    v["opacity"] = g["opacities"]
    for i in range(3):
        v[f"scale_{i}"] = g["log_scales"][:, i]
    for i in range(4):
        v[f"rot_{i}"] = g["quats_wxyz"][:, i]
    v["rf_dc_0"] = g["rf_sh"][:, 0, 0]
    for i in range(15):
        v[f"rf_rest_{i}"] = g["rf_sh"][:, 1 + i, 0]
    v["object_id"] = g["object_id"]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name in _FLOAT_FIELDS] + ["property ushort object_id", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(v.tobytes())


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
              "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int": "<i4", "uint": "<u4"}


def read_gaussians(path: str | Path) -> dict[str, np.ndarray]:
    """Read a PLY written by :func:`write_gaussians` (or any 3DGS PLY; radio fields then come back as None)."""
    with open(path, "rb") as f:
        names, types, n = [], [], 0
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        while True:
            line = f.readline().strip().decode("ascii")
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError("only binary_little_endian PLY is supported")
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                names.append(name)
                types.append(_PLY_TYPES[t])
            elif line == "end_header":
                break
        v = np.frombuffer(f.read(), dtype=list(zip(names, types)), count=n)

    def stack(fields):
        return np.stack([v[k].astype(np.float32) for k in fields], -1)

    rgb_dc = stack([f"f_dc_{i}" for i in range(3)])
    if "f_rest_0" in v.dtype.names:
        k = sum(1 for nm in v.dtype.names if nm.startswith("f_rest_")) // 3
        rest = stack([f"f_rest_{i}" for i in range(3 * k)]).reshape(n, 3, k).transpose(0, 2, 1)
        rgb_sh = np.concatenate([rgb_dc[:, None, :], rest], 1)
    else:
        rgb_sh = rgb_dc[:, None, :]
    out = {
        "means": stack(["x", "y", "z"]),
        "rgb_sh": rgb_sh,
        "opacities": v["opacity"].astype(np.float32),
        "log_scales": stack([f"scale_{i}" for i in range(3)]),
        "quats_wxyz": stack([f"rot_{i}" for i in range(4)]),
        "rf_sh": None,
        "object_id": np.zeros(n, np.int64),
    }
    if "rf_dc_0" in v.dtype.names:
        out["rf_sh"] = stack(["rf_dc_0"] + [f"rf_rest_{i}" for i in range(15)])[:, :, None]
    if "object_id" in v.dtype.names:
        out["object_id"] = v["object_id"].astype(np.int64)
    return out
