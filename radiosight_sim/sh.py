"""Real spherical harmonics up to degree 3 for any number of channels (same basis as 3DGS / gsplat)."""
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: [N, K, C] coefficients (K >= (degree+1)^2); dirs: [N, 3] directions from the viewer to each Gaussian.
    Returns [N, C] before any activation."""
    out = C0 * sh[:, 0]
    if degree < 1:
        return out
    d = F.normalize(dirs, dim=-1)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    out = out - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if degree < 2:
        return out
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    out = (out + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5] + C2[2] * (2 * zz - xx - yy) * sh[:, 6]
           + C2[3] * xz * sh[:, 7] + C2[4] * (xx - yy) * sh[:, 8])
    if degree < 3:
        return out
    return (out + C3[0] * y * (3 * xx - yy) * sh[:, 9] + C3[1] * xy * z * sh[:, 10]
            + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
            + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + C3[5] * z * (xx - yy) * sh[:, 14]
            + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
