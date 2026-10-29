"""
uromt_core.py - Core Numerical Algorithms for Unbalanced Regularized Optimal Mass Transport (urOMT)

High-performance implementation featuring:
  1. Numba JIT-compiled parallel Particle-in-Cell (PIC) advection, adjoint pull-back, and fused velocity gradient kernels.
  2. Multi-threaded 3D spectral diffusion solver (I - sigma*dt*Laplacian)^(-1) via Discrete Cosine Transform (DCT-II)
     with precomputed inverse filters and zero-flux Neumann boundary conditions.
  3. Tight bounding-box ROI cropping to eliminate non-tissue background air voxels.
  4. Operator-splitting forward simulation (Reaction -> Advection -> Diffusion).
  5. Exact adjoint sensitivity gradient computation w.r.t velocity and source.
  6. High-performance L-BFGS-B optimization with callback loss caching.
  7. Sequential dynamic 4D time series solver for multi-frame MRI datasets (optional temporal
     warm-starting, disabled by default).
  8. Comprehensive Eulerian transport metrics (speed, flux, kinetic energy, influx, efflux, divergence).
"""

from __future__ import annotations
import os
import time
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
from scipy.fft import dctn, idctn
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

from uromt_io import compute_mask_bbox

# =============================================================================
# Optional Numba JIT Acceleration Detection
# =============================================================================

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _resolve_worker_count(workers: int) -> int:
    """
    Resolve the --workers CLI value (-1 = use all available cores) to a concrete,
    positive thread count, clamped to the number of CPUs actually available.
    """
    cpu_total = os.cpu_count() or 4
    if workers is None or workers <= 0:
        return cpu_total
    return max(1, min(int(workers), cpu_total))


# =============================================================================
# Numba JIT Fused Kernels (High-Performance Parallel CPU Implementations)
# =============================================================================

if HAS_NUMBA:
    @nb.njit(parallel=True, fastmath=True)
    def _advect_pic_3d_numba_parallel(
        rho_src: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        n_threads: int
    ) -> np.ndarray:
        """
        Thread-safe parallel forward 3D Particle-in-Cell scatter accumulation using chunked thread-local buffers.
        """
        Nx, Ny, Nz = rho_src.shape
        thread_bufs = np.zeros((n_threads, Nx, Ny, Nz), dtype=np.float32)
        chunk_size = (Nx + n_threads - 1) // n_threads

        for t in nb.prange(n_threads):
            i_start = t * chunk_size
            i_end = min(i_start + chunk_size, Nx)
            buf = thread_bufs[t]

            for i in range(i_start, i_end):
                for j in range(Ny):
                    for k in range(Nz):
                        val = rho_src[i, j, k]
                        if val <= 1e-12:
                            continue

                        xp = i + vx[i, j, k] * (dt / dx)
                        yp = j + vy[i, j, k] * (dt / dy)
                        zp = k + vz[i, j, k] * (dt / dz)

                        if xp < 0.0:
                            xp = 0.0
                        elif xp > Nx - 1.0001:
                            xp = Nx - 1.0001

                        if yp < 0.0:
                            yp = 0.0
                        elif yp > Ny - 1.0001:
                            yp = Ny - 1.0001

                        if zp < 0.0:
                            zp = 0.0
                        elif zp > Nz - 1.0001:
                            zp = Nz - 1.0001

                        x0 = int(xp)
                        y0 = int(yp)
                        z0 = int(zp)

                        x1 = x0 + 1 if x0 < Nx - 1 else x0
                        y1 = y0 + 1 if y0 < Ny - 1 else y0
                        z1 = z0 + 1 if z0 < Nz - 1 else z0

                        fx = xp - x0
                        fy = yp - y0
                        fz = zp - z0

                        cx = 1.0 - fx
                        cy = 1.0 - fy
                        cz = 1.0 - fz

                        buf[x0, y0, z0] += val * (cx * cy * cz)
                        buf[x1, y0, z0] += val * (fx * cy * cz)
                        buf[x0, y1, z0] += val * (cx * fy * cz)
                        buf[x0, y0, z1] += val * (cx * cy * fz)
                        buf[x1, y1, z0] += val * (fx * fy * cz)
                        buf[x1, y0, z1] += val * (fx * cy * fz)
                        buf[x0, y1, z1] += val * (cx * fy * fz)
                        buf[x1, y1, z1] += val * (fx * fy * fz)

        # Sum reduction across thread buffers
        out = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        for t in range(n_threads):
            out += thread_bufs[t]
        return out


    @nb.njit(parallel=True, fastmath=True)
    def _advect_backward_fused_numba(
        rho_src: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        lambda_adv: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        lambda_src: np.ndarray,
        grad_vx: np.ndarray,
        grad_vy: np.ndarray,
        grad_vz: np.ndarray
    ) -> None:
        """
        Fused adjoint pull-back and velocity gradient computation in a single OpenMP parallel pass.
        Eliminates intermediate array allocations and redundant coordinate calculations.
        """
        Nx, Ny, Nz = rho_src.shape

        for i in nb.prange(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    xp_raw = i + vx[i, j, k] * (dt / dx)
                    yp_raw = j + vy[i, j, k] * (dt / dy)
                    zp_raw = k + vz[i, j, k] * (dt / dz)

                    xp = xp_raw
                    if xp < 0.0:
                        xp = 0.0
                    elif xp > Nx - 1.0001:
                        xp = Nx - 1.0001

                    yp = yp_raw
                    if yp < 0.0:
                        yp = 0.0
                    elif yp > Ny - 1.0001:
                        yp = Ny - 1.0001

                    zp = zp_raw
                    if zp < 0.0:
                        zp = 0.0
                    elif zp > Nz - 1.0001:
                        zp = Nz - 1.0001

                    x0 = int(xp)
                    y0 = int(yp)
                    z0 = int(zp)

                    x1 = x0 + 1 if x0 < Nx - 1 else x0
                    y1 = y0 + 1 if y0 < Ny - 1 else y0
                    z1 = z0 + 1 if z0 < Nz - 1 else z0

                    fx = xp - x0
                    fy = yp - y0
                    fz = zp - z0

                    cx = 1.0 - fx
                    cy = 1.0 - fy
                    cz = 1.0 - fz

                    # 8 vertex adjoint values
                    L000 = lambda_adv[x0, y0, z0]
                    L100 = lambda_adv[x1, y0, z0]
                    L010 = lambda_adv[x0, y1, z0]
                    L001 = lambda_adv[x0, y0, z1]
                    L110 = lambda_adv[x1, y1, z0]
                    L101 = lambda_adv[x1, y0, z1]
                    L011 = lambda_adv[x0, y1, z1]
                    L111 = lambda_adv[x1, y1, z1]

                    # 1. Adjoint Gather: lambda_src = S(v)^T lambda_adv
                    lambda_src[i, j, k] = (
                        (cx * cy * cz) * L000 +
                        (fx * cy * cz) * L100 +
                        (cx * fy * cz) * L010 +
                        (cx * cy * fz) * L001 +
                        (fx * fy * cz) * L110 +
                        (fx * cy * fz) * L101 +
                        (cx * fy * fz) * L011 +
                        (fx * fy * fz) * L111
                    )

                    # 2. Advection Velocity Gradient: [d(S(v) rho_src)/dv]^T lambda_adv
                    rho_val = rho_src[i, j, k]
                    dL_dfx = (
                        cy * cz * (L100 - L000) +
                        fy * cz * (L110 - L010) +
                        cy * fz * (L101 - L001) +
                        fy * fz * (L111 - L011)
                    )
                    dL_dfy = (
                        cx * cz * (L010 - L000) +
                        fx * cz * (L110 - L100) +
                        cx * fz * (L011 - L001) +
                        fx * fz * (L111 - L101)
                    )
                    dL_dfz = (
                        cx * cy * (L001 - L000) +
                        fx * cy * (L101 - L100) +
                        cx * fy * (L011 - L010) +
                        fx * fy * (L111 - L110)
                    )

                    gvx = rho_val * dL_dfx * (dt / dx)
                    gvy = rho_val * dL_dfy * (dt / dy)
                    gvz = rho_val * dL_dfz * (dt / dz)

                    # The trilinear formulas above implicitly assume the destination
                    # coordinate depends smoothly on v. Where the raw (pre-clamp)
                    # position fell outside the domain, the clamped coordinate is
                    # actually constant w.r.t. v along that axis, so its true
                    # derivative is zero -- zero the corresponding component instead
                    # of returning the (incorrect) unclamped trilinear derivative.
                    if xp_raw < 0.0 or xp_raw > Nx - 1.0001:
                        gvx = 0.0
                    if yp_raw < 0.0 or yp_raw > Ny - 1.0001:
                        gvy = 0.0
                    if zp_raw < 0.0 or zp_raw > Nz - 1.0001:
                        gvz = 0.0

                    grad_vx[i, j, k] = gvx
                    grad_vy[i, j, k] = gvy
                    grad_vz[i, j, k] = gvz


# =============================================================================
# Helper & Boundary Utilities
# =============================================================================

def _compute_laplacian_eigenvalues(
    shape: Tuple[int, int, int],
    voxel_size: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute 3D Laplacian eigenvalues for DCT-II with Neumann boundary conditions.
    """
    Nx, Ny, Nz = shape
    dx, dy, dz = voxel_size

    kx = np.arange(Nx, dtype=np.float32)
    ky = np.arange(Ny, dtype=np.float32)
    kz = np.arange(Nz, dtype=np.float32)

    eig_x = (2.0 / (dx ** 2)) * (np.cos(np.pi * kx / Nx) - 1.0)
    eig_y = (2.0 / (dy ** 2)) * (np.cos(np.pi * ky / Ny) - 1.0)
    eig_z = (2.0 / (dz ** 2)) * (np.cos(np.pi * kz / Nz) - 1.0)

    # 3D grid of eigenvalues: lambda(kx, ky, kz) <= 0
    eig_3d = (
        eig_x[:, np.newaxis, np.newaxis] +
        eig_y[np.newaxis, :, np.newaxis] +
        eig_z[np.newaxis, np.newaxis, :]
    )
    return eig_3d


# =============================================================================
# Tight ROI Bounding-Box Masking Utilities
# =============================================================================
#
# compute_mask_bbox is imported from uromt_io (see import at top of file) rather
# than redefined here, so both modules always agree on ROI bounding-box geometry.
# uromt_io.compute_mask_bbox returns (bbox_slices, cropped_shape, bbox_origin);
# call sites in this module only need the first two and ignore bbox_origin.


def crop_volume_3d(vol: np.ndarray, bbox_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    """Crop 3D volume using bounding-box slices."""
    return np.ascontiguousarray(vol[bbox_slices])


def crop_volume_4d(vol: np.ndarray, bbox_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    """Crop 4D dynamic volume (Nx, Ny, Nz, Nt) using spatial bounding-box slices."""
    return np.ascontiguousarray(vol[bbox_slices + (slice(None),)])


def uncrop_volume(
    cropped: np.ndarray,
    full_shape: Tuple[int, ...],
    bbox_slices: Tuple[slice, slice, slice],
    is_velocity: bool = False
) -> np.ndarray:
    """
    Embed cropped output array back into the full-resolution spatial canvas.
    """
    out = np.zeros(full_shape, dtype=cropped.dtype)
    if is_velocity:
        if cropped.ndim == 5:
            out[:, :, bbox_slices[0], bbox_slices[1], bbox_slices[2]] = cropped
        elif cropped.ndim == 4:
            out[:, bbox_slices[0], bbox_slices[1], bbox_slices[2]] = cropped
    else:
        if cropped.ndim == 4:
            out[:, bbox_slices[0], bbox_slices[1], bbox_slices[2]] = cropped
        elif cropped.ndim == 3:
            out[bbox_slices[0], bbox_slices[1], bbox_slices[2]] = cropped
    return out


# =============================================================================
# Spectral Diffusion Solver (Multi-Threaded & Precomputed Filters)
# =============================================================================

class SpectralDiffusionSolver:
    """
    Solves (I - sigma * dt * Laplacian) u = f on a 3D grid with Neumann boundary conditions
    using exact 3D Discrete Cosine Transforms (DCT-II) with multi-threaded execution.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sigma: float = 0.1,
        dt: float = 1.0,
        workers: int = -1
    ):
        self.shape = shape
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.dt = dt
        self.workers = workers
        self.eigenvalues = _compute_laplacian_eigenvalues(shape, voxel_size)
        denom = 1.0 - (sigma * dt) * self.eigenvalues
        self.inv_denom = (1.0 / denom).astype(np.float32)

    def solve(self, f: np.ndarray, sigma: Optional[float] = None, dt: Optional[float] = None) -> np.ndarray:
        """
        Solve (I - sigma * dt * Delta) u = f
        """
        if (sigma is not None and sigma <= 0.0) or (dt is not None and dt <= 0.0):
            return f.copy()

        if (sigma is not None and sigma != self.sigma) or (dt is not None and dt != self.dt):
            denom = 1.0 - (sigma * dt) * self.eigenvalues
            inv_denom = (1.0 / denom).astype(np.float32)
        else:
            inv_denom = self.inv_denom

        f_hat = dctn(f.astype(np.float32), type=2, norm='ortho', workers=self.workers)
        u = idctn(f_hat * inv_denom, type=2, norm='ortho', workers=self.workers)
        return np.maximum(u, 0.0, out=u).astype(np.float32)

    def solve_adjoint(self, adj_in: np.ndarray, sigma: Optional[float] = None, dt: Optional[float] = None) -> np.ndarray:
        """
        Solve L^(-T) adj_in = L^(-1) adj_in since L is symmetric positive-definite.
        """
        if (sigma is not None and sigma <= 0.0) or (dt is not None and dt <= 0.0):
            return adj_in.copy()

        if (sigma is not None and sigma != self.sigma) or (dt is not None and dt != self.dt):
            denom = 1.0 - (sigma * dt) * self.eigenvalues
            inv_denom = (1.0 / denom).astype(np.float32)
        else:
            inv_denom = self.inv_denom

        hat = dctn(adj_in.astype(np.float32), type=2, norm='ortho', workers=self.workers)
        sol = idctn(hat * inv_denom, type=2, norm='ortho', workers=self.workers)
        return sol.astype(np.float32)


# =============================================================================
# 3D Particle-In-Cell Advection & Adjoint Operators
# =============================================================================

def advect_pic_3d(
    rho_src: np.ndarray,
    v: np.ndarray,
    dt: float,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_numba: bool = True,
    workers: int = -1
) -> np.ndarray:
    """
    Forward 3D Particle-in-Cell trilinear mass transport: rho_adv = S(v) rho_src.

    ``workers`` controls how many per-thread scatter buffers are allocated for the
    Numba-parallel path (-1 uses all available cores, matching the --workers CLI
    default). Each buffer is a full copy of the volume, so on many-core machines
    honoring a smaller worker count noticeably reduces peak memory use.
    """
    Nx, Ny, Nz = rho_src.shape
    dx, dy, dz = voxel_size

    if HAS_NUMBA and use_numba:
        n_threads = _resolve_worker_count(workers)
        vx = np.ascontiguousarray(v[0], dtype=np.float32)
        vy = np.ascontiguousarray(v[1], dtype=np.float32)
        vz = np.ascontiguousarray(v[2], dtype=np.float32)
        src = np.ascontiguousarray(rho_src, dtype=np.float32)
        return _advect_pic_3d_numba_parallel(src, vx, vy, vz, float(dt), float(dx), float(dy), float(dz), n_threads)

    # Pure NumPy Fallback
    ix, iy, iz = np.meshgrid(
        np.arange(Nx, dtype=np.float32),
        np.arange(Ny, dtype=np.float32),
        np.arange(Nz, dtype=np.float32),
        indexing='ij'
    )

    xp = np.clip(ix + (v[0] * dt / dx), 0.0, Nx - 1.0001)
    yp = np.clip(iy + (v[1] * dt / dy), 0.0, Ny - 1.0001)
    zp = np.clip(iz + (v[2] * dt / dz), 0.0, Nz - 1.0001)

    x0 = xp.astype(np.int32)
    y0 = yp.astype(np.int32)
    z0 = zp.astype(np.int32)

    x1 = np.minimum(x0 + 1, Nx - 1)
    y1 = np.minimum(y0 + 1, Ny - 1)
    z1 = np.minimum(z0 + 1, Nz - 1)

    fx = xp - x0
    fy = yp - y0
    fz = zp - z0

    w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
    w100 = fx * (1.0 - fy) * (1.0 - fz)
    w010 = (1.0 - fx) * fy * (1.0 - fz)
    w001 = (1.0 - fx) * (1.0 - fy) * fz
    w110 = fx * fy * (1.0 - fz)
    w101 = fx * (1.0 - fy) * fz
    w011 = (1.0 - fx) * fy * fz
    w111 = fx * fy * fz

    rho_adv = np.zeros_like(rho_src, dtype=np.float32)

    def scatter_add(target, x_idx, y_idx, z_idx, weights, source):
        lin_idx = (x_idx * Ny + y_idx) * Nz + z_idx
        contrib = weights * source
        np.add.at(target.ravel(), lin_idx.ravel(), contrib.ravel())

    scatter_add(rho_adv, x0, y0, z0, w000, rho_src)
    scatter_add(rho_adv, x1, y0, z0, w100, rho_src)
    scatter_add(rho_adv, x0, y1, z0, w010, rho_src)
    scatter_add(rho_adv, x0, y0, z1, w001, rho_src)
    scatter_add(rho_adv, x1, y1, z0, w110, rho_src)
    scatter_add(rho_adv, x1, y0, z1, w101, rho_src)
    scatter_add(rho_adv, x0, y1, z1, w011, rho_src)
    scatter_add(rho_adv, x1, y1, z1, w111, rho_src)

    return rho_adv


def advect_pic_3d_adjoint(
    lambda_adv: np.ndarray,
    v: np.ndarray,
    dt: float,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Adjoint pull-back of advection operator: lambda_src = S(v)^T lambda_adv.
    """
    Nx, Ny, Nz = lambda_adv.shape
    dx, dy, dz = voxel_size

    ix, iy, iz = np.meshgrid(
        np.arange(Nx, dtype=np.float32),
        np.arange(Ny, dtype=np.float32),
        np.arange(Nz, dtype=np.float32),
        indexing='ij'
    )

    xp = np.clip(ix + (v[0] * dt / dx), 0.0, Nx - 1.0001)
    yp = np.clip(iy + (v[1] * dt / dy), 0.0, Ny - 1.0001)
    zp = np.clip(iz + (v[2] * dt / dz), 0.0, Nz - 1.0001)

    x0 = xp.astype(np.int32)
    y0 = yp.astype(np.int32)
    z0 = zp.astype(np.int32)

    x1 = np.minimum(x0 + 1, Nx - 1)
    y1 = np.minimum(y0 + 1, Ny - 1)
    z1 = np.minimum(z0 + 1, Nz - 1)

    fx = xp - x0
    fy = yp - y0
    fz = zp - z0

    w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
    w100 = fx * (1.0 - fy) * (1.0 - fz)
    w010 = (1.0 - fx) * fy * (1.0 - fz)
    w001 = (1.0 - fx) * (1.0 - fy) * fz
    w110 = fx * fy * (1.0 - fz)
    w101 = fx * (1.0 - fy) * fz
    w011 = (1.0 - fx) * fy * fz
    w111 = fx * fy * fz

    lambda_src = (
        w000 * lambda_adv[x0, y0, z0] +
        w100 * lambda_adv[x1, y0, z0] +
        w010 * lambda_adv[x0, y1, z0] +
        w001 * lambda_adv[x0, y0, z1] +
        w110 * lambda_adv[x1, y1, z0] +
        w101 * lambda_adv[x1, y0, z1] +
        w011 * lambda_adv[x0, y1, z1] +
        w111 * lambda_adv[x1, y1, z1]
    )
    return lambda_src.astype(np.float32)


def compute_advection_velocity_gradient(
    rho_src: np.ndarray,
    v: np.ndarray,
    lambda_adv: np.ndarray,
    dt: float,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Compute gradient of the advection step w.r.t velocity field v:
    grad_v = [d(S(v) rho_src) / dv]^T lambda_adv.
    """
    Nx, Ny, Nz = rho_src.shape
    dx, dy, dz = voxel_size

    ix, iy, iz = np.meshgrid(
        np.arange(Nx, dtype=np.float32),
        np.arange(Ny, dtype=np.float32),
        np.arange(Nz, dtype=np.float32),
        indexing='ij'
    )

    xp_raw = ix + (v[0] * dt / dx)
    yp_raw = iy + (v[1] * dt / dy)
    zp_raw = iz + (v[2] * dt / dz)

    xp = np.clip(xp_raw, 0.0, Nx - 1.0001)
    yp = np.clip(yp_raw, 0.0, Ny - 1.0001)
    zp = np.clip(zp_raw, 0.0, Nz - 1.0001)

    x0 = xp.astype(np.int32)
    y0 = yp.astype(np.int32)
    z0 = zp.astype(np.int32)

    x1 = np.minimum(x0 + 1, Nx - 1)
    y1 = np.minimum(y0 + 1, Ny - 1)
    z1 = np.minimum(z0 + 1, Nz - 1)

    fx = xp - x0
    fy = yp - y0
    fz = zp - z0

    L000 = lambda_adv[x0, y0, z0]
    L100 = lambda_adv[x1, y0, z0]
    L010 = lambda_adv[x0, y1, z0]
    L001 = lambda_adv[x0, y0, z1]
    L110 = lambda_adv[x1, y1, z0]
    L101 = lambda_adv[x1, y0, z1]
    L011 = lambda_adv[x0, y1, z1]
    L111 = lambda_adv[x1, y1, z1]

    dL_dfx = (
        (1.0 - fy) * (1.0 - fz) * (L100 - L000) +
        fy * (1.0 - fz) * (L110 - L010) +
        (1.0 - fy) * fz * (L101 - L001) +
        fy * fz * (L111 - L011)
    )
    dL_dfy = (
        (1.0 - fx) * (1.0 - fz) * (L010 - L000) +
        fx * (1.0 - fz) * (L110 - L100) +
        (1.0 - fx) * fz * (L011 - L001) +
        fx * fz * (L111 - L101)
    )
    dL_dfz = (
        (1.0 - fx) * (1.0 - fy) * (L001 - L000) +
        fx * (1.0 - fy) * (L101 - L100) +
        (1.0 - fx) * fy * (L011 - L010) +
        fx * fy * (L111 - L110)
    )

    grad_vx = rho_src * dL_dfx * (dt / dx)
    grad_vy = rho_src * dL_dfy * (dt / dy)
    grad_vz = rho_src * dL_dfz * (dt / dz)

    # Zero the gradient along any axis where the advected position saturated at
    # the domain boundary: the clamped coordinate no longer depends on v there,
    # so its true derivative is zero, unlike the raw trilinear formula above.
    boundary_x = (xp_raw < 0.0) | (xp_raw > Nx - 1.0001)
    boundary_y = (yp_raw < 0.0) | (yp_raw > Ny - 1.0001)
    boundary_z = (zp_raw < 0.0) | (zp_raw > Nz - 1.0001)
    grad_vx = np.where(boundary_x, 0.0, grad_vx)
    grad_vy = np.where(boundary_y, 0.0, grad_vy)
    grad_vz = np.where(boundary_z, 0.0, grad_vz)

    grad_v = np.stack([grad_vx, grad_vy, grad_vz], axis=0).astype(np.float32)
    return grad_v


# =============================================================================
# Forward Simulation & Loss Evaluation
# =============================================================================

def forward_simulation(
    rho_0: np.ndarray,
    v_seq: np.ndarray,
    r_seq: np.ndarray,
    chi_seq: np.ndarray,
    sigma: float,
    dt: float,
    diffusion_solver: SpectralDiffusionSolver,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_numba: bool = True,
    workers: int = -1
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run forward operator-splitting simulation across m sub-steps:
      rho_i -> rho_i_src -> rho_i_adv -> rho_{i+1}
    """
    m = v_seq.shape[0]
    rho_history = [rho_0.copy()]
    rho_src_history = []
    rho_adv_history = []

    curr_rho = rho_0.copy()
    for i in range(m):
        # 1. Source / Reaction Step: allows mass influx into empty or low-density voxels
        rho_eff = np.maximum(curr_rho, 1.0e-3)
        rho_src = np.maximum(curr_rho + dt * (chi_seq[i] * rho_eff * r_seq[i]), 0.0)
        rho_src_history.append(rho_src)

        # 2. Advection Step (Particle-in-Cell)
        rho_adv = advect_pic_3d(rho_src, v_seq[i], dt, voxel_size, use_numba=use_numba, workers=workers)
        rho_adv_history.append(rho_adv)

        # 3. Diffusion Step (DCT solve)
        rho_next = diffusion_solver.solve(rho_adv, sigma, dt)
        rho_history.append(rho_next)
        curr_rho = rho_next

    return rho_history, rho_src_history, rho_adv_history


def compute_uromt_cost_and_gradient(
    controls_flat: np.ndarray,
    rho_0: np.ndarray,
    rho_target: np.ndarray,
    chi_seq: np.ndarray,
    sigma: float,
    alpha: float,
    beta: float,
    dt: float,
    m_substeps: int,
    shape: Tuple[int, int, int],
    diffusion_solver: SpectralDiffusionSolver,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_numba: bool = True,
    workers: int = -1
) -> Tuple[float, np.ndarray]:
    """
    Compute total urOMT cost functional and exact analytic gradients via the adjoint sensitivity method.
    """
    Nx, Ny, Nz = shape
    n_voxels = Nx * Ny * Nz
    dx, dy, dz = voxel_size
    dV = dx * dy * dz

    # Unpack controls
    v_size = m_substeps * 3 * n_voxels
    v_flat = controls_flat[:v_size]
    r_flat = controls_flat[v_size:]

    v_seq = v_flat.reshape((m_substeps, 3, Nx, Ny, Nz))
    r_seq = r_flat.reshape((m_substeps, Nx, Ny, Nz))

    # 1. Forward Pass
    rho_history, rho_src_history, rho_adv_history = forward_simulation(
        rho_0, v_seq, r_seq, chi_seq, sigma, dt, diffusion_solver, voxel_size,
        use_numba=use_numba, workers=workers
    )

    # 2. Compute Objective Value
    kinetic_loss = 0.0
    source_loss = 0.0

    for i in range(m_substeps):
        v_sq = np.sum(v_seq[i] ** 2, axis=0)
        kinetic_loss += dt * dV * np.sum(rho_history[i] * v_sq)

        r_sq = r_seq[i] ** 2
        rho_eff = np.maximum(rho_history[i], 1.0e-3)
        source_loss += alpha * dt * dV * np.sum(chi_seq[i] * rho_eff * r_sq)

    # Endpoint mismatch fitting term: beta * int (rho_m - rho_target)^2 dx
    rho_m = rho_history[-1]
    mismatch = rho_m - rho_target
    fitting_loss = beta * dV * np.sum(mismatch ** 2)

    total_loss = float(kinetic_loss + source_loss + fitting_loss)

    # 3. Backward Pass (Adjoint Sensitivity Propagation)
    grad_v_seq = np.zeros_like(v_seq, dtype=np.float32)
    grad_r_seq = np.zeros_like(r_seq, dtype=np.float32)

    # Terminal adjoint state at t = T (step m)
    lambda_curr = 2.0 * beta * dV * mismatch  # shape (Nx, Ny, Nz)

    for i in range(m_substeps - 1, -1, -1):
        # A. Backprop through Diffusion step: lambda_adv = L^(-T) lambda_curr
        lambda_adv = diffusion_solver.solve_adjoint(lambda_curr, sigma, dt)

        # A'. The forward source/reaction step clamps its output to be non-negative:
        #   rho_src = max(rho_i + dt*chi*rho_eff*r_i, 0)
        # max(x, 0) has zero subgradient wherever the clamp was active, so the
        # adjoint must not backprop through those voxels. Recompute the pre-clamp
        # value here (cheap elementwise op) to build that mask.
        rho_eff_i = np.maximum(rho_history[i], 1.0e-3)
        rho_src_pre_clamp = rho_history[i] + dt * (chi_seq[i] * rho_eff_i * r_seq[i])
        clip_mask = (rho_src_pre_clamp >= 0.0).astype(np.float32)

        # B & C. Backprop through Advection step and compute velocity gradient
        if HAS_NUMBA and use_numba:
            lambda_src = np.empty((Nx, Ny, Nz), dtype=np.float32)
            grad_vx = np.empty((Nx, Ny, Nz), dtype=np.float32)
            grad_vy = np.empty((Nx, Ny, Nz), dtype=np.float32)
            grad_vz = np.empty((Nx, Ny, Nz), dtype=np.float32)

            _advect_backward_fused_numba(
                np.ascontiguousarray(rho_src_history[i], dtype=np.float32),
                np.ascontiguousarray(v_seq[i, 0], dtype=np.float32),
                np.ascontiguousarray(v_seq[i, 1], dtype=np.float32),
                np.ascontiguousarray(v_seq[i, 2], dtype=np.float32),
                np.ascontiguousarray(lambda_adv, dtype=np.float32),
                float(dt), float(dx), float(dy), float(dz),
                lambda_src, grad_vx, grad_vy, grad_vz
            )
            # lambda_src is d(loss)/d(rho_src) (post-clamp); v only enters through
            # the advection step (which sees the already-clamped rho_src), so the
            # clamp mask does not apply to the velocity gradient, only to lambda_src
            # itself before it is propagated further upstream (source term & rho_i).
            lambda_src *= clip_mask

            # Direct kinetic energy gradient: 2 * dt * dV * rho_i * v_i
            grad_v_direct = 2.0 * dt * dV * (rho_history[i][np.newaxis, ...] * v_seq[i])
            grad_v_seq[i, 0] = grad_v_direct[0] + grad_vx
            grad_v_seq[i, 1] = grad_v_direct[1] + grad_vy
            grad_v_seq[i, 2] = grad_v_direct[2] + grad_vz
        else:
            grad_v_direct = 2.0 * dt * dV * (rho_history[i][np.newaxis, ...] * v_seq[i])
            grad_v_advect = compute_advection_velocity_gradient(
                rho_src_history[i], v_seq[i], lambda_adv, dt, voxel_size
            )
            grad_v_seq[i] = grad_v_direct + grad_v_advect
            lambda_src = advect_pic_3d_adjoint(lambda_adv, v_seq[i], dt, voxel_size)
            lambda_src = lambda_src * clip_mask

        # D. Compute gradient w.r.t relative source r_i
        rho_eff = np.maximum(rho_history[i], 1.0e-3)
        grad_r_direct = 2.0 * alpha * dt * dV * (chi_seq[i] * rho_eff * r_seq[i])
        grad_r_src = dt * (chi_seq[i] * rho_eff * lambda_src)
        grad_r_seq[i] = grad_r_direct + grad_r_src

        # E. Backprop to lambda for previous step (rho_i)
        v_sq = np.sum(v_seq[i] ** 2, axis=0)
        r_sq = r_seq[i] ** 2
        direct_rho_cost = dt * dV * (v_sq + alpha * chi_seq[i] * r_sq * (rho_history[i] > 1.0e-3))
        d_src_d_rho = np.where(rho_history[i] > 1.0e-3, 1.0 + dt * (chi_seq[i] * r_seq[i]), 1.0)
        lambda_curr = lambda_src * d_src_d_rho + direct_rho_cost

    grad_flat = np.concatenate([grad_v_seq.ravel(), grad_r_seq.ravel()]).astype(np.float64)
    return total_loss, grad_flat


# =============================================================================
# Optimization Solver (Single-Scale L-BFGS-B with Cached Callbacks)
# =============================================================================

def solve_uromt_pair(
    rho_0: np.ndarray,
    rho_1: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: float = 0.01,
    alpha: float = 100.0,
    beta: float = 100.0,
    n_substeps: int = 4,
    max_iter: int = 50,
    tol: float = 1e-8,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    v_init: Optional[np.ndarray] = None,
    r_init: Optional[np.ndarray] = None,
    use_numba: bool = True,
    workers: int = -1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solve the urOMT optimization problem between two consecutive 3D density distributions.

    Parameters
    ----------
    rho_0 : np.ndarray
        Initial density distribution (Nx, Ny, Nz).
    rho_1 : np.ndarray
        Target density distribution (Nx, Ny, Nz).
    mask : np.ndarray, optional
        Spatial indicator mask restricting source and domain (Nx, Ny, Nz).
    sigma : float, default 0.01
        Diffusion regularization parameter.
    alpha : float, default 100.0
        Relative weighting of source term vs transport kinetic energy.
    beta : float, default 100.0
        Endpoint matching penalty weight.
    n_substeps : int, default 4
        Number of time sub-steps m.
    max_iter : int, default 50
        Maximum L-BFGS-B iterations.
    tol : float, default 1e-8
        Convergence gradient tolerance (gtol for L-BFGS-B on [0, 1] scale).
    voxel_size : tuple of float
        Grid resolution (dx, dy, dz).
    v_init : np.ndarray, optional
        Initial velocity guess (n_substeps, 3, Nx, Ny, Nz).
    r_init : np.ndarray, optional
        Initial source guess (n_substeps, Nx, Ny, Nz).
    use_numba : bool, default True
        Use Numba JIT compiled kernels if available.
    workers : int, default -1
        Number of parallel CPU threads for FFT/DCT (-1 for all cores).
    verbose : bool, default True
        Print optimization progress.

    Returns
    -------
    result : dict
        Solved velocity, source, interpolated density sequence, loss history, and metrics.
    """
    shape = rho_0.shape
    Nx, Ny, Nz = shape
    dt = 1.0 / n_substeps
    n_voxels = Nx * Ny * Nz

    # Make the Numba-parallel kernels (which use nb.prange internally rather than
    # accepting an explicit thread count) honor --workers as well, so it isn't
    # silently limited to just the FFT/DCT diffusion solver.
    if HAS_NUMBA and use_numba:
        nb.set_num_threads(_resolve_worker_count(workers))

    # Check for near-zero baseline frame (pre-contrast state)
    if np.sum(rho_0) < 1e-6 and np.sum(rho_1) < 1e-6:
        if verbose:
            print("      [Baseline Frame] Pre-contrast density is near-zero; setting velocity to zero.")
        v_zeros = np.zeros((n_substeps, 3, Nx, Ny, Nz), dtype=np.float32)
        r_zeros = np.zeros((n_substeps, Nx, Ny, Nz), dtype=np.float32)
        return {
            'velocity_substeps': v_zeros,
            'source_substeps': r_zeros,
            'density_substeps': np.zeros((n_substeps + 1, Nx, Ny, Nz), dtype=np.float32),
            'velocity': np.zeros((3, Nx, Ny, Nz), dtype=np.float32),
            'source': np.zeros((Nx, Ny, Nz), dtype=np.float32),
            'speed': np.zeros((Nx, Ny, Nz), dtype=np.float32),
            'flux': np.zeros((Nx, Ny, Nz), dtype=np.float32),
            'influx': 0.0,
            'efflux': 0.0,
            'kinetic_energy': 0.0,
            'loss': 0.0,
            'loss_history': [0.0],
            'num_iterations': 0,
            'status': 0,
            'status_message': 'baseline frame (optimization skipped)',
            'elapsed_time': 0.0
        }

    # Setup spatial indicator mask chi
    if mask is not None:
        chi = (mask > 0).astype(np.float32)
    else:
        chi = np.ones(shape, dtype=np.float32)
    chi_seq = np.tile(chi[np.newaxis, ...], (n_substeps, 1, 1, 1))

    # Initialize spectral solver with precomputed filters
    diffusion_solver = SpectralDiffusionSolver(shape, voxel_size, sigma=sigma, dt=dt, workers=workers)

    # Initialize controls: seed with small perturbation if no warm-start is available
    if v_init is not None and v_init.shape == (n_substeps, 3, Nx, Ny, Nz):
        v0 = v_init.astype(np.float64)
    else:
        rng = np.random.default_rng(0)
        v0 = (1.0e-5 * rng.standard_normal((n_substeps, 3, Nx, Ny, Nz))).astype(np.float64)

    if r_init is not None and r_init.shape == (n_substeps, Nx, Ny, Nz):
        r0 = r_init.astype(np.float64)
    else:
        r0 = np.zeros((n_substeps, Nx, Ny, Nz), dtype=np.float64)

    x0_flat = np.concatenate([v0.ravel(), r0.ravel()])

    loss_history = []
    iter_count = [0]
    last_evaluated_loss = [0.0]
    _cache_x = [None]
    _cache_out = [None]

    def objective_fn(x_flat):
        # Memoize the most recent evaluation so the verbose initial-cost probe below
        # is not recomputed by the optimizer's identical first function evaluation.
        if _cache_x[0] is not None and np.array_equal(_cache_x[0], x_flat):
            last_evaluated_loss[0] = _cache_out[0][0]
            return _cache_out[0]
        loss, grad = compute_uromt_cost_and_gradient(
            x_flat, rho_0, rho_1, chi_seq, sigma, alpha, beta, dt,
            n_substeps, shape, diffusion_solver, voxel_size, use_numba=use_numba,
            workers=workers
        )
        _cache_x[0] = x_flat.copy()
        _cache_out[0] = (loss, grad)
        last_evaluated_loss[0] = loss
        return loss, grad

    def callback_fn(xk):
        iter_count[0] += 1
        loss_val = last_evaluated_loss[0]
        loss_history.append(loss_val)
        if verbose and (iter_count[0] % 5 == 0 or iter_count[0] == 1):
            print(f"      [Iter {iter_count[0]:3d}/{max_iter}] Cost: {loss_val:12.4e}")

    init_loss, _ = objective_fn(x0_flat)
    loss_history.append(init_loss)
    if verbose:
        print(f"      [Iter   0/{max_iter}] Initial Cost: {init_loss:12.4e}")

    t_start = time.time()
    res = minimize(
        objective_fn,
        x0_flat,
        method='L-BFGS-B',
        jac=True,
        callback=callback_fn,
        options={
            'maxiter': max_iter,
            'ftol': 1e-7,
            'gtol': float(tol),
            'maxcor': 10,
            'disp': False
        }
    )
    t_elapsed = time.time() - t_start

    # Extract optimal solutions
    v_size = n_substeps * 3 * n_voxels
    opt_v = res.x[:v_size].reshape((n_substeps, 3, Nx, Ny, Nz)).astype(np.float32)
    opt_r = res.x[v_size:].reshape((n_substeps, Nx, Ny, Nz)).astype(np.float32)

    # Compute final forward density trajectory
    rho_history, _, _ = forward_simulation(
        rho_0, opt_v, opt_r, chi_seq, sigma, dt, diffusion_solver, voxel_size,
        use_numba=use_numba, workers=workers
    )

    # Compute summary transport speed and flux
    avg_v = np.mean(opt_v, axis=0)  # (3, Nx, Ny, Nz)
    speed = np.sqrt(np.sum(avg_v ** 2, axis=0))  # (Nx, Ny, Nz)
    avg_r = np.mean(opt_r, axis=0)  # (Nx, Ny, Nz)

    # Representative transition density: if rho_0 is near-zero (e.g. after baseline subtraction),
    # use 0.5 * (rho_0 + rho_1) so transport kinetics are not zeroed out at contrast arrival
    if np.sum(rho_0) < 1e-6 and np.sum(rho_1) >= 1e-6:
        rho_trans = 0.5 * (rho_0 + rho_1)
    else:
        rho_trans = rho_0

    dx, dy, dz = voxel_size
    dV = dx * dy * dz

    flux = rho_trans * speed

    # Compute influx, efflux, and kinetic energy scaled by physical voxel volume dV
    influx_vol = dV * np.sum(rho_trans * np.maximum(avg_r * chi, 0.0))
    efflux_vol = dV * np.sum(rho_trans * np.maximum(-avg_r * chi, 0.0))
    kinetic_energy = 0.5 * dV * np.sum(rho_trans * (speed ** 2))

    # Report the actual L-BFGS-B termination reason (status 2 is a line-search
    # failure, NOT convergence - previously all exits were logged as "converged").
    status_labels = {
        0: "converged",
        1: "stopped at iteration/eval limit",
        2: "stopped: line-search failure (no further progress)",
    }
    status_message = status_labels.get(int(res.status), f"stopped (status {int(res.status)})")
    n_iters = int(getattr(res, 'nit', iter_count[0]))

    if verbose:
        print(f"      urOMT {status_message} after {n_iters} iters "
              f"({t_elapsed:.2f}s) | Final Cost: {res.fun:12.4e}")

    return {
        'velocity_substeps': opt_v,
        'source_substeps': opt_r,
        'density_substeps': np.array(rho_history, dtype=np.float32),
        'velocity': avg_v,
        'source': avg_r,
        'speed': speed,
        'flux': flux,
        'influx': float(influx_vol),
        'efflux': float(efflux_vol),
        'kinetic_energy': float(kinetic_energy),
        'loss': float(res.fun),
        'loss_history': loss_history,
        'num_iterations': n_iters,
        'status': int(res.status),
        'status_message': status_message,
        'elapsed_time': t_elapsed
    }


# =============================================================================
# Full Dynamic 4D Time Series Sequence Runner
# =============================================================================

def run_uromt_sequence(
    signal_4d: np.ndarray,
    mask_3d: Optional[np.ndarray] = None,
    sigma: float = 0.01,
    alpha: float = 100.0,
    beta: float = 100.0,
    n_substeps: int = 4,
    max_iter: int = 40,
    tol: float = 1e-8,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth_sigma: float = 0.5,
    baseline_frames: int = 1,
    subtract_baseline: bool = True,
    normalize_max: bool = True,
    downsample_factor: Optional[Union[float, Tuple[float, float, float]]] = None,
    crop_roi: bool = True,
    warm_start: bool = False,
    use_numba: bool = True,
    workers: int = -1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute urOMT across an entire 4D MRI time series (Nx, Ny, Nz, Nt) with ROI cropping,
    baseline subtraction, peak normalization, and optional temporal warm-starting.

    Parameters
    ----------
    signal_4d : np.ndarray
        4D dynamic MRI series, shape (Nx, Ny, Nz, Nt).
    mask_3d : np.ndarray, optional
        3D spatial mask, shape (Nx, Ny, Nz).
    sigma : float, default 0.01
        Diffusion coefficient.
    alpha : float, default 100.0
        Source term weight.
    beta : float, default 100.0
        Endpoint fitting weight.
    n_substeps : int, default 4
        Number of time sub-steps between consecutive frames.
    max_iter : int, default 40
        Max optimizer iterations per frame transition.
    voxel_size : tuple of float
        Voxel dimensions (dx, dy, dz).
    smooth_sigma : float, default 0.5
        Gaussian smoothing kernel standard deviation to regularize image noise.
    baseline_frames : int, default 1
        Number of leading pre-contrast frames to average as baseline.
    subtract_baseline : bool, default True
        Whether to subtract pre-contrast baseline to isolate contrast agent concentration.
    normalize_max : bool, default True
        Whether to divide by global peak so concentration is normalized to [0, 1].
    downsample_factor : float or tuple, optional
        Spatial downsampling factor for rapid prototyping on large volumes.
    crop_roi : bool, default True
        Automatically crop to tight bounding box of mask_3d to eliminate non-tissue air voxels.
    warm_start : bool, default False
        Seed each frame transition's optimizer with the previous transition's velocity and
        source solution. Disabled by default: because the L-BFGS-B objective is only
        piecewise-differentiable (PIC advection) and evaluated in float32, warm-starting from
        an already-fitted point causes the line search to fail on the first step, so the
        optimizer returns the seed unchanged and the whole velocity/source sequence collapses
        to the first transition's solution. Cold-starting every transition avoids this at the
        cost of more iterations per frame.
    use_numba : bool, default True
        Use Numba JIT parallel acceleration.
    workers : int, default -1
        Number of threads for FFT/DCT (-1 for all cores).
    verbose : bool, default True
        Print sequence progress logs.

    Returns
    -------
    results : dict
        Contains 'density' (Nt, Nx, Ny, Nz), 'velocity' (Nt-1, 3, Nx, Ny, Nz),
        'source' (Nt-1, Nx, Ny, Nz), 'metrics' dictionary, and execution stats.
    """
    # Optional spatial downsampling if requested
    if downsample_factor is not None:
        if isinstance(downsample_factor, (int, float)) and downsample_factor > 1:
            ds = int(round(downsample_factor))
            signal_4d = signal_4d[::ds, ::ds, ::ds, :]
            if mask_3d is not None:
                mask_3d = mask_3d[::ds, ::ds, ::ds]
            voxel_size = tuple(v * ds for v in voxel_size)
            Nx, Ny, Nz, Nt = signal_4d.shape
            if verbose:
                print(f"   [Downsampling] Downsampled input volume by {ds}x to ({Nx}, {Ny}, {Nz}).")

    Nx, Ny, Nz, Nt = signal_4d.shape
    if verbose:
        print(f"\n=======================================================")
        print(f"   Starting Accelerated urOMT 4D Sequence Solver")
        print(f"   Original Volume: ({Nx}, {Ny}, {Nz}), Time Steps: {Nt}")
        print(f"   Acceleration: Numba={HAS_NUMBA and use_numba}, Workers={workers}")
        print(f"   Parameters: sigma={sigma}, alpha={alpha}, beta={beta}, sub-steps={n_substeps}")
        print(f"   Preprocessing: Baseline Subtraction={subtract_baseline} (frames={baseline_frames}), Max Normalize={normalize_max}")
        print(f"=======================================================\n")

    # Preprocessing: Masking and noise smoothing
    processed_signal = signal_4d.copy().astype(np.float32)
    processed_mask = (mask_3d > 0).astype(np.float32) if mask_3d is not None else np.ones((Nx, Ny, Nz), dtype=np.float32)

    for t in range(Nt):
        frame = processed_signal[..., t]
        if smooth_sigma > 0:
            frame = gaussian_filter(frame, sigma=smooth_sigma)
        frame = frame * processed_mask
        processed_signal[..., t] = np.maximum(frame, 0.0)

    # Baseline Subtraction: Convert raw MRI signal to relative tracer concentration ΔS
    if subtract_baseline and Nt > baseline_frames:
        baseline_img = np.mean(processed_signal[..., :baseline_frames], axis=-1, keepdims=True)
        processed_signal = np.maximum(processed_signal - baseline_img, 0.0)
        if verbose:
            print(f"   [Preprocessing] Subtracted baseline (mean of first {baseline_frames} frames).")

    # Global Max Normalization: Scale tracer concentration to [0, 1]
    if normalize_max:
        peak_val = float(np.max(processed_signal))
        if peak_val > 0:
            processed_signal = processed_signal / peak_val
            if verbose:
                print(f"   [Preprocessing] Normalized peak density to 1.0 (original peak was {peak_val:.2f}).")

    # Tight ROI Bounding-Box Cropping
    is_cropped = False
    bbox_slices = (slice(0, Nx), slice(0, Ny), slice(0, Nz))
    if crop_roi and mask_3d is not None and np.sum(mask_3d > 0) > 0:
        bbox_slices, cropped_shape, _bbox_origin = compute_mask_bbox(processed_mask, pad=4)
        orig_vol_size = Nx * Ny * Nz
        crop_vol_size = cropped_shape[0] * cropped_shape[1] * cropped_shape[2]

        if crop_vol_size < 0.9 * orig_vol_size:
            is_cropped = True
            cNx, cNy, cNz = cropped_shape
            if verbose:
                print(f"   [ROI Cropping] Mask ROI cropped from {Nx}x{Ny}x{Nz} to {cNx}x{cNy}x{cNz} ({crop_vol_size/orig_vol_size*100:.1f}% volume, saving {(1 - crop_vol_size/orig_vol_size)*100:.1f}% computations)\n")
            sub_signal = crop_volume_4d(processed_signal, bbox_slices)
            sub_mask = crop_volume_3d(processed_mask, bbox_slices)
        else:
            sub_signal = processed_signal
            sub_mask = processed_mask
            cNx, cNy, cNz = Nx, Ny, Nz
    else:
        sub_signal = processed_signal
        sub_mask = processed_mask
        cNx, cNy, cNz = Nx, Ny, Nz

    # Storage arrays for working region
    sub_velocities = np.zeros((Nt - 1, 3, cNx, cNy, cNz), dtype=np.float32)
    sub_sources = np.zeros((Nt - 1, cNx, cNy, cNz), dtype=np.float32)
    sub_densities = np.zeros((Nt, cNx, cNy, cNz), dtype=np.float32)
    sub_densities[0] = sub_signal[..., 0]

    # Physical voxel volume for metric scaling
    dx, dy, dz = voxel_size
    dV = dx * dy * dz

    # Metrics trackers
    total_mass = [float(dV * np.sum(processed_signal[..., t])) for t in range(Nt)]
    mean_speed = []
    kinetic_energies = []
    influx_rates = []
    efflux_rates = []
    loss_values = []

    prev_v = None
    prev_r = None
    t_start_seq = time.time()

    if verbose and not warm_start:
        print("   [Solver] Temporal warm-starting disabled: each transition is optimized "
              "from a cold start (see run_uromt_sequence docstring).\n")

    # Loop over frame transitions: t -> t+1
    for t in range(Nt - 1):
        if verbose:
            print(f"--- Processing Transition Frame [{t+1:2d}/{Nt-1}] (t={t} -> t={t+1}) ---")

        # Frame-to-frame OMT: start each transition from observed frame t
        rho_0 = sub_signal[..., t]
        rho_1 = sub_signal[..., t + 1]

        step_res = solve_uromt_pair(
            rho_0=rho_0,
            rho_1=rho_1,
            mask=sub_mask,
            sigma=sigma,
            alpha=alpha,
            beta=beta,
            n_substeps=n_substeps,
            max_iter=max_iter,
            tol=tol,
            voxel_size=voxel_size,
            v_init=prev_v if warm_start else None,
            r_init=prev_r if warm_start else None,
            use_numba=use_numba,
            workers=workers,
            verbose=verbose
        )

        sub_velocities[t] = step_res['velocity']
        sub_sources[t] = step_res['source']
        sub_densities[t + 1] = step_res['density_substeps'][-1]

        # Retain current solution as the seed for the next transition (only used
        # when warm_start=True).
        prev_v = step_res['velocity_substeps']
        prev_r = step_res['source_substeps']

        # Collect metrics
        if np.any(sub_mask > 0):
            mean_speed.append(float(np.mean(step_res['speed'][sub_mask > 0])))
        else:
            mean_speed.append(0.0)
        kinetic_energies.append(step_res['kinetic_energy'])
        influx_rates.append(step_res['influx'])
        efflux_rates.append(step_res['efflux'])
        loss_values.append(step_res['loss'])

    t_total = time.time() - t_start_seq

    # Uncrop back to full resolution canvas if cropped
    if is_cropped:
        all_densities = uncrop_volume(sub_densities, (Nt, Nx, Ny, Nz), bbox_slices, is_velocity=False)
        all_velocities = uncrop_volume(sub_velocities, (Nt - 1, 3, Nx, Ny, Nz), bbox_slices, is_velocity=True)
        all_sources = uncrop_volume(sub_sources, (Nt - 1, Nx, Ny, Nz), bbox_slices, is_velocity=False)
    else:
        all_densities = sub_densities
        all_velocities = sub_velocities
        all_sources = sub_sources

    if verbose:
        print(f"\n=======================================================")
        print(f"   urOMT Sequence Successfully Completed in {t_total:.2f}s")
        print(f"   Processed {Nt-1} time transitions.")
        print(f"=======================================================\n")

    metrics = {
        'total_mass': total_mass,
        'mean_speed': mean_speed,
        'kinetic_energy': kinetic_energies,
        'influx': influx_rates,
        'efflux': efflux_rates,
        'loss': loss_values,
        'total_runtime_seconds': t_total
    }

    return {
        'density': all_densities,
        'velocity': all_velocities,
        'source': all_sources,
        'mask': processed_mask,
        'metrics': metrics
    }
