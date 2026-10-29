"""
uromt_io.py - Input/Output and Visualization for urOMT

Provides functions to:
  1. Load 3D/4D NIfTI files (.nii, .nii.gz) with metadata (nibabel with pure NumPy fallback).
  2. Save density, velocity, and source fields as NumPy (.npy, .npz) and NIfTI (.nii) files.
  3. Generate publication-quality 2D slice visualizations of density, velocity fields (quiver/streamlines),
     source distributions, and summary transport time courses.
"""

from __future__ import annotations
import os
import struct
import gzip
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt

# Optional import of nibabel; fallback to pure NumPy parser if unavailable
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# =============================================================================
# Pure NumPy NIfTI-1 Binary Parser (Fallback when nibabel is not installed)
# =============================================================================

def _qform_to_affine(
    quatern_b: float,
    quatern_c: float,
    quatern_d: float,
    qoffset_x: float,
    qoffset_y: float,
    qoffset_z: float,
    pixdim: Tuple[float, ...],
    qfac: float
) -> np.ndarray:
    """
    Build a 4x4 affine from a NIfTI-1 qform quaternion (method 2), per the NIfTI-1
    spec (nifti1.h). Used as a fallback when a file carries a qform but no sform,
    which the previous pure-NumPy parser silently ignored (falling back to an
    unrotated diagonal affine and discarding orientation).
    """
    b, c, d = float(quatern_b), float(quatern_c), float(quatern_d)
    a_sq = 1.0 - (b * b + c * c + d * d)
    a = np.sqrt(a_sq) if a_sq > 0.0 else 0.0

    r = np.array([
        [a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
        [2.0 * (b * c + a * d), a * a + c * c - b * b - d * d, 2.0 * (c * d - a * b)],
        [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a + d * d - b * b - c * c],
    ], dtype=np.float64)

    dx = pixdim[1] if len(pixdim) > 1 else 1.0
    dy = pixdim[2] if len(pixdim) > 2 else 1.0
    dz = pixdim[3] if len(pixdim) > 3 else 1.0
    qfac = 1.0 if qfac == 0.0 else float(qfac)

    affine = np.eye(4, dtype=np.float32)
    affine[0:3, 0] = r[:, 0] * dx
    affine[0:3, 1] = r[:, 1] * dy
    affine[0:3, 2] = r[:, 2] * dz * qfac
    affine[0, 3] = qoffset_x
    affine[1, 3] = qoffset_y
    affine[2, 3] = qoffset_z
    return affine


def _read_nifti_binary(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a NIfTI-1 file (.nii or .nii.gz) directly from raw bytes using NumPy and struct.
    Supports 3D and 4D datasets with standard datatypes.
    """
    is_gz = file_path.endswith('.gz')
    open_fn = gzip.open if is_gz else open

    with open_fn(file_path, 'rb') as f:
        hdr_bytes = f.read(348)
        if len(hdr_bytes) < 348:
            raise ValueError(f"Invalid NIfTI file (header too short): {file_path}")

        # Check endianness via sizeof_hdr (offset 0)
        sizeof_hdr = struct.unpack('<i', hdr_bytes[0:4])[0]
        if sizeof_hdr == 348:
            endian = '<'
        elif struct.unpack('>i', hdr_bytes[0:4])[0] == 348:
            endian = '>'
        else:
            raise ValueError(f"Unsupported NIfTI header size in {file_path}")

        # Dimensions: offset 40, array of 8 int16
        dim = struct.unpack(f'{endian}8h', hdr_bytes[40:56])
        ndim = dim[0]
        shape = tuple(dim[1:ndim + 1])

        # Datatype: offset 70 (int16), bitpix: offset 72 (int16)
        datatype = struct.unpack(f'{endian}h', hdr_bytes[70:72])[0]
        bitpix = struct.unpack(f'{endian}h', hdr_bytes[72:74])[0]

        # Voxel dimensions (pixdim): offset 76, array of 8 float32
        pixdim = struct.unpack(f'{endian}8f', hdr_bytes[76:108])
        voxel_size = tuple(pixdim[1:ndim + 1])

        # Vox offset: offset 108 (float32)
        vox_offset = struct.unpack(f'{endian}f', hdr_bytes[108:112])[0]
        vox_offset = int(vox_offset)

        # Scale slope and intercept
        scl_slope = struct.unpack(f'{endian}f', hdr_bytes[112:116])[0]
        scl_inter = struct.unpack(f'{endian}f', hdr_bytes[116:120])[0]
        if scl_slope == 0.0:
            scl_slope = 1.0

        # Mapping NIfTI datatype codes to numpy dtypes
        dtype_map = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
            256: np.int8,
            512: np.uint16,
            768: np.uint32,
        }

        if datatype not in dtype_map:
            raise ValueError(f"Unsupported NIfTI datatype code: {datatype}")

        np_dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)

        # Affine matrix: prefer sform, then qform, then a default diagonal built
        # from voxel spacing. (Previously only sform was read; a qform-only file --
        # common output from many scanners -- silently lost its orientation.)
        qform_code = struct.unpack(f'{endian}h', hdr_bytes[252:254])[0]
        sform_code = struct.unpack(f'{endian}h', hdr_bytes[254:256])[0]
        affine = np.eye(4, dtype=np.float32)
        if sform_code > 0:
            srow_x = struct.unpack(f'{endian}4f', hdr_bytes[280:296])
            srow_y = struct.unpack(f'{endian}4f', hdr_bytes[296:312])
            srow_z = struct.unpack(f'{endian}4f', hdr_bytes[312:328])
            affine[0, :] = srow_x
            affine[1, :] = srow_y
            affine[2, :] = srow_z
        elif qform_code > 0:
            quatern_b = struct.unpack(f'{endian}f', hdr_bytes[256:260])[0]
            quatern_c = struct.unpack(f'{endian}f', hdr_bytes[260:264])[0]
            quatern_d = struct.unpack(f'{endian}f', hdr_bytes[264:268])[0]
            qoffset_x = struct.unpack(f'{endian}f', hdr_bytes[268:272])[0]
            qoffset_y = struct.unpack(f'{endian}f', hdr_bytes[272:276])[0]
            qoffset_z = struct.unpack(f'{endian}f', hdr_bytes[276:280])[0]
            affine = _qform_to_affine(
                quatern_b, quatern_c, quatern_d,
                qoffset_x, qoffset_y, qoffset_z,
                pixdim, qfac=pixdim[0]
            )
        else:
            # Construct default diagonal affine using voxel dimensions
            for i in range(min(3, len(voxel_size))):
                affine[i, i] = voxel_size[i]

        # Read data payload
        f.seek(vox_offset)
        total_elements = np.prod(shape)
        data = np.frombuffer(f.read(int(total_elements * (bitpix // 8))), dtype=np_dtype)
        data = data.reshape(shape, order='F').astype(np.float32)

        # Apply scaling if specified
        if scl_slope != 1.0 or scl_inter != 0.0:
            data = data * scl_slope + scl_inter

        meta = {
            'shape': shape,
            'voxel_size': voxel_size,
            'affine': affine,
            'datatype': datatype,
            'bitpix': bitpix,
            'scl_slope': scl_slope,
            'scl_inter': scl_inter,
            'file_path': file_path
        }
        return data, meta


def _write_nifti_binary(data: np.ndarray, file_path: str, affine: Optional[np.ndarray] = None) -> None:
    """
    Write a 3D or 4D numpy array to an uncompressed NIfTI-1 (.nii) file using pure binary layout.
    """
    data = np.asfortranarray(data, dtype=np.float32)
    shape = data.shape
    ndim = len(shape)

    hdr = bytearray(348)
    struct.pack_into('<i', hdr, 0, 348)  # sizeof_hdr

    # dim
    dim = [ndim] + list(shape) + [1] * (7 - ndim)
    struct.pack_into('<8h', hdr, 40, *dim)

    # datatype = 16 (FLOAT32), bitpix = 32
    struct.pack_into('<h', hdr, 70, 16)
    struct.pack_into('<h', hdr, 72, 32)

    # pixdim
    pixdim = [1.0] * 8
    if affine is not None:
        for i in range(min(3, ndim)):
            pixdim[i + 1] = float(np.linalg.norm(affine[:3, i]))
    struct.pack_into('<8f', hdr, 76, *pixdim)

    # vox_offset = 352.0
    struct.pack_into('<f', hdr, 108, 352.0)
    struct.pack_into('<f', hdr, 112, 1.0)  # scl_slope
    struct.pack_into('<f', hdr, 116, 0.0)  # scl_inter

    # sform
    if affine is not None:
        struct.pack_into('<h', hdr, 254, 1)  # sform_code = 1 (NIFTI_XFORM_SCANNER_ANAT)
        struct.pack_into('<4f', hdr, 280, *affine[0, :].tolist())
        struct.pack_into('<4f', hdr, 296, *affine[1, :].tolist())
        struct.pack_into('<4f', hdr, 312, *affine[2, :].tolist())

    # magic = b"n+1\0"
    hdr[344:348] = b'n+1\0'

    # 4 bytes padding after 348 to reach vox_offset 352
    pad = b'\0\0\0\0'

    is_gz = file_path.endswith('.gz')
    open_fn = gzip.open if is_gz else open
    with open_fn(file_path, 'wb') as f:
        f.write(hdr)
        f.write(pad)
        f.write(data.tobytes(order='F'))


# =============================================================================
# Primary High-Level I/O Functions
# =============================================================================

def load_nifti(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a 3D or 4D NIfTI file (.nii or .nii.gz) and return data with header metadata.

    Parameters
    ----------
    file_path : str
        Path to NIfTI file.

    Returns
    -------
    data : np.ndarray
        Array containing the image volume(s), dtype float32.
        For 4D time series, shape is (Nx, Ny, Nz, Nt).
        For 3D volumes/masks, shape is (Nx, Ny, Nz).
    meta : dict
        Metadata dictionary including 'affine', 'voxel_size', 'shape', 'header'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    if HAS_NIBABEL:
        try:
            img = nib.load(file_path)
            data = img.get_fdata(dtype=np.float32)
            affine = img.affine
            hdr = img.header
            zooms = hdr.get_zooms()
            meta = {
                'shape': data.shape,
                'voxel_size': zooms,
                'affine': affine,
                'header': hdr,
                'file_path': file_path
            }
            return data, meta
        except Exception:
            # Fall back to pure NumPy parser if nibabel encounters format issues
            pass

    return _read_nifti_binary(file_path)


def normalise_intensities(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    baseline_frames: int = 1,
    subtract_baseline: bool = True,
    normalize_max: bool = True,
    time_axis: Optional[int] = None
) -> np.ndarray:
    """
    Convert raw DCE-MRI signal intensities to non-negative tracer concentration
    by subtracting the pre-contrast baseline (mean of first baseline_frames)
    and normalizing the global peak to 1.0.

    This is the standard DCE-MRI preprocessing used in the urOMT literature
    for dynamic glymphatic and perfusion imaging studies.

    Parameters
    ----------
    data : np.ndarray
        4-D dynamic contrast series. Supports either (Nx, Ny, Nz, Nt) or (Nt, Nx, Ny, Nz).
    mask : np.ndarray, optional
        3-D binary brain mask.
    baseline_frames : int, default 1
        Number of leading pre-contrast frames to average as baseline.
    subtract_baseline : bool, default True
        Whether to subtract pre-contrast baseline.
    normalize_max : bool, default True
        Whether to divide by global maximum so density is in [0, 1].
    time_axis : int, optional
        Index of time dimension (0 or -1 / 3). If None, inferred from data and mask shapes.

    Returns
    -------
    rho : np.ndarray
        Preprocessed non-negative tracer concentration array (dtype float32).
    """
    rho = data.astype(np.float32).copy()
    if rho.ndim != 4:
        return rho

    # Robust time-axis detection
    if time_axis is not None:
        is_t_last = (time_axis in (-1, 3))
    elif mask is not None and mask.ndim >= 3:
        if mask.shape[:3] == rho.shape[:3]:
            is_t_last = True
        elif mask.shape[:3] == rho.shape[1:]:
            is_t_last = False
        else:
            is_t_last = (rho.shape[-1] < rho.shape[0])
    else:
        is_t_last = (rho.shape[-1] < rho.shape[0])

    if is_t_last:
        # Array shape is (Nx, Ny, Nz, Nt)
        if subtract_baseline and rho.shape[-1] > baseline_frames:
            base = np.mean(rho[..., :baseline_frames], axis=-1, keepdims=True)
            rho = np.maximum(rho - base, 0.0)
        if mask is not None:
            mask_3d = mask if mask.ndim == 3 else (mask[..., 0] if mask.shape[-1] < mask.shape[0] else mask[0])
            rho = rho * (mask_3d[..., np.newaxis] > 0)
        if normalize_max:
            m = np.max(rho)
            if m > 0:
                rho = rho / m
    else:
        # Array shape is (Nt, Nx, Ny, Nz)
        if subtract_baseline and rho.shape[0] > baseline_frames:
            base = np.mean(rho[:baseline_frames], axis=0, keepdims=True)
            rho = np.maximum(rho - base, 0.0)
        if mask is not None:
            mask_3d = mask if mask.ndim == 3 else (mask[0] if mask.shape[0] < mask.shape[-1] else mask[..., 0])
            rho = rho * (mask_3d[np.newaxis, ...] > 0)
        if normalize_max:
            m = np.max(rho)
            if m > 0:
                rho = rho / m

    return rho


# =============================================================================
# Bounding Box Cropping & Affine Geometry Utilities
# =============================================================================

def compute_mask_bbox(
    mask: np.ndarray,
    pad: int = 2,
    time_axis: Optional[int] = None
) -> Tuple[Tuple[slice, slice, slice], Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Compute tight bounding box slices around the non-zero voxels of a mask.

    Parameters
    ----------
    mask : np.ndarray
        3D (Nx, Ny, Nz) or 4D mask array.
    pad : int, default 2
        Padding border voxels added around the tight mask boundary.
    time_axis : int, optional
        Axis of time if mask is 4D (0 or -1 / 3). If None, inferred from shape.

    Returns
    -------
    bbox_slices : tuple of slice
        (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
    cropped_shape : tuple of int
        (cNx, cNy, cNz)
    bbox_origin : tuple of int
        (x_min, y_min, z_min) voxel coordinates of the cropped origin in the original canvas.
    """
    if mask is None:
        return (slice(None), slice(None), slice(None)), (0, 0, 0), (0, 0, 0)

    # If mask is 4D, project across time
    if mask.ndim == 4:
        if time_axis is not None:
            mask_3d = np.any(mask > 0, axis=time_axis)
        elif mask.shape[0] in (1, 2, 3, 4) and mask.shape[-1] > 10:
            mask_3d = np.any(mask > 0, axis=0)
        elif mask.shape[-1] in (1, 2, 3, 4) and mask.shape[0] > 10:
            mask_3d = np.any(mask > 0, axis=-1)
        else:
            mask_3d = np.any(mask > 0, axis=-1) if mask.shape[-1] < mask.shape[0] else np.any(mask > 0, axis=0)
    else:
        mask_3d = (mask > 0)

    if np.sum(mask_3d) == 0:
        Nx, Ny, Nz = mask_3d.shape
        return (slice(0, Nx), slice(0, Ny), slice(0, Nz)), (Nx, Ny, Nz), (0, 0, 0)

    coords = np.argwhere(mask_3d)
    Nx, Ny, Nz = mask_3d.shape

    x_min = max(0, int(np.min(coords[:, 0])) - pad)
    x_max = min(Nx, int(np.max(coords[:, 0])) + 1 + pad)
    y_min = max(0, int(np.min(coords[:, 1])) - pad)
    y_max = min(Ny, int(np.max(coords[:, 1])) + 1 + pad)
    z_min = max(0, int(np.min(coords[:, 2])) - pad)
    z_max = min(Nz, int(np.max(coords[:, 2])) + 1 + pad)

    bbox_slices = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
    cropped_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
    bbox_origin = (x_min, y_min, z_min)
    return bbox_slices, cropped_shape, bbox_origin


def crop_array_to_bbox(
    arr: Optional[np.ndarray],
    bbox_slices: Tuple[slice, slice, slice],
    time_axis: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Crop array to 3D spatial bounding box slices.
    Handles 3D (Nx, Ny, Nz), 4D (Nt, Nx, Ny, Nz) or (Nx, Ny, Nz, Nt), and 5D (Nt, 3, Nx, Ny, Nz).
    """
    if arr is None:
        return None
    sx, sy, sz = bbox_slices

    # 3D: (Nx, Ny, Nz)
    if arr.ndim == 3:
        return arr[sx, sy, sz]

    # 4D: (Nt, Nx, Ny, Nz), (Nx, Ny, Nz, Nt), or (3, Nx, Ny, Nz)
    elif arr.ndim == 4:
        if arr.shape[0] == 3:
            # Single-transition velocity vector: (3, Nx, Ny, Nz)
            return arr[:, sx, sy, sz]
        elif time_axis is not None:
            if time_axis in (-1, 3):
                return arr[sx, sy, sz, :]
            else:
                return arr[:, sx, sy, sz]
        else:
            # Check bounding box limits against array dimensions to determine spatial axes
            x_stop = sx.stop if sx.stop is not None else 0
            z_stop = sz.stop if sz.stop is not None else 0

            # If z_stop exceeds the last dimension, last dimension cannot be Z (spatial)
            if z_stop > arr.shape[-1]:
                return arr[:, sx, sy, sz]
            # If x_stop exceeds the first dimension, first dimension cannot be X (spatial)
            elif x_stop > arr.shape[0]:
                return arr[:, sx, sy, sz]
            # If arr is standard NIfTI 4D (Nx, Ny, Nz, Nt) where time is last
            elif arr.shape[-1] <= 200 and arr.shape[0] > arr.shape[-1]:
                return arr[sx, sy, sz, :]
            elif arr.shape[0] <= 200 and arr.shape[-1] > arr.shape[0]:
                return arr[:, sx, sy, sz]
            else:
                return arr[sx, sy, sz, :]

    # 5D: (Nt-1, 3, Nx, Ny, Nz) or (Nt-1, Nx, Ny, Nz, 3)
    elif arr.ndim == 5:
        if arr.shape[1] == 3:
            # (Nt-1, 3, Nx, Ny, Nz)
            return arr[:, :, sx, sy, sz]
        elif arr.shape[-1] == 3:
            # (Nt-1, Nx, Ny, Nz, 3)
            return arr[:, sx, sy, sz, :]
        else:
            # Default to spatial dimensions at the end
            return arr[:, :, sx, sy, sz]

    return arr


def adjust_affine_for_crop(
    affine: Optional[np.ndarray],
    bbox_origin: Tuple[int, int, int]
) -> np.ndarray:
    """
    Update 4x4 NIfTI affine matrix to account for the shifted origin (x_min, y_min, z_min)
    so physical millimeter space is accurately preserved in 3D viewers.
    """
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    new_affine = np.array(affine, dtype=np.float32).copy()
    x_min, y_min, z_min = bbox_origin
    shift_vector = np.array([x_min, y_min, z_min], dtype=np.float32)
    offset = new_affine[:3, :3] @ shift_vector
    new_affine[:3, 3] += offset
    return new_affine


def save_nifti(
    data: np.ndarray,
    file_path: str,
    affine: Optional[np.ndarray] = None,
    header: Any = None,
    mask: Optional[np.ndarray] = None,
    crop_to_mask: bool = True,
    bbox_pad: int = 2
) -> None:
    """
    Save a numpy array as a NIfTI file, optionally cropped to the tight bounding box of the mask ROI.

    Parameters
    ----------
    data : np.ndarray
        3D or 4D numpy array to save.
    file_path : str
        Destination path (.nii or .nii.gz).
    affine : np.ndarray, optional
        4x4 affine coordinate transformation matrix.
    header : Any, optional
        NIfTI header template.
    mask : np.ndarray, optional
        3D binary brain mask. If provided and crop_to_mask is True, crops data to mask ROI.
    crop_to_mask : bool, default True
        Whether to crop volume to the tight mask bounding box.
    bbox_pad : int, default 2
        Padding border around mask bounding box.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    save_data = data
    save_affine = affine

    # Apply tight ROI bounding box cropping if mask is provided
    if mask is not None and crop_to_mask and np.sum(mask > 0) > 0:
        bbox_slices, cropped_shape, bbox_origin = compute_mask_bbox(mask, pad=bbox_pad)
        # This function writes NIfTI files, so 4D input is always in NIfTI's
        # spatial-first / time-last (Nx, Ny, Nz, T) convention -- pass that
        # explicitly rather than relying on crop_array_to_bbox's shape-guessing
        # heuristic, which can misclassify unusual (e.g. long time-series) shapes.
        save_data = crop_array_to_bbox(data, bbox_slices, time_axis=-1)
        save_affine = adjust_affine_for_crop(affine, bbox_origin)

    if HAS_NIBABEL:
        try:
            nii_img = nib.Nifti1Image(save_data.astype(np.float32), affine=save_affine, header=header)
            nib.save(nii_img, file_path)
            return
        except Exception:
            pass

    _write_nifti_binary(save_data, file_path, affine=save_affine)


def save_numpy_results(
    output_dir: str,
    prefix: str,
    density: np.ndarray,
    velocity: np.ndarray,
    source: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, Any]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_size: Optional[Tuple[float, ...]] = None,
    crop_to_mask: bool = True,
    bbox_pad: int = 2
) -> Dict[str, str]:
    """
    Save density and velocity data (and optional source/metrics) as .npy files and a compressed .npz archive,
    cropped to the tight bounding box of the mask ROI.

    Parameters
    ----------
    output_dir : str
        Destination directory.
    prefix : str
        Filename prefix (e.g. 'sub32521_s1').
    density : np.ndarray
        Dynamic density field array, shape (T, Nx, Ny, Nz) or (Nx, Ny, Nz, T).
    velocity : np.ndarray
        Dynamic velocity field array, shape (T-1, 3, Nx, Ny, Nz) or (T-1, Nx, Ny, Nz, 3).
    source : np.ndarray, optional
        Dynamic relative source field array, shape (T-1, Nx, Ny, Nz).
    mask : np.ndarray, optional
        3D binary brain mask. If provided and crop_to_mask is True, crops results to tight ROI.
    metrics : dict, optional
        Eulerian/Lagrangian metrics dictionary.
    affine : np.ndarray, optional
        4x4 affine matrix.
    voxel_size : tuple, optional
        Voxel dimensions.
    crop_to_mask : bool, default True
        Whether to crop results to the tight bounding box of the mask.
    bbox_pad : int, default 2
        Padding border voxels around mask bounding box.

    Returns
    -------
    saved_files : dict
        Mapping of data name to saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    save_dens = density
    save_vel = velocity
    save_src = source
    save_mask = mask
    save_aff = affine
    bbox_origin = (0, 0, 0)
    cropped_shape = None

    # Apply tight ROI bounding box cropping if mask is provided
    if mask is not None and crop_to_mask and np.sum(mask > 0) > 0:
        bbox_slices, cropped_shape, bbox_origin = compute_mask_bbox(mask, pad=bbox_pad)
        # density/source arrays from run_uromt_sequence are always time-first
        # (T, Nx, Ny, Nz); pass that explicitly instead of relying on
        # crop_array_to_bbox's shape-guessing heuristic (velocity is 5D and
        # already unambiguous; mask is 3D and unaffected by time_axis).
        save_dens = crop_array_to_bbox(density, bbox_slices, time_axis=0)
        save_vel = crop_array_to_bbox(velocity, bbox_slices)
        save_src = crop_array_to_bbox(source, bbox_slices, time_axis=0)
        save_mask = crop_array_to_bbox(mask, bbox_slices)
        save_aff = adjust_affine_for_crop(affine, bbox_origin)

    density_path = os.path.join(output_dir, f"{prefix}_density.npy")
    np.save(density_path, save_dens.astype(np.float32))
    saved['density'] = density_path

    velocity_path = os.path.join(output_dir, f"{prefix}_velocity.npy")
    np.save(velocity_path, save_vel.astype(np.float32))
    saved['velocity'] = velocity_path

    # Save flow speed field if velocity is available
    if save_vel is not None:
        if save_vel.ndim == 5 and save_vel.shape[1] == 3:
            save_spd = np.sqrt(np.sum(save_vel ** 2, axis=1))
        elif save_vel.ndim == 5 and save_vel.shape[-1] == 3:
            save_spd = np.sqrt(np.sum(save_vel ** 2, axis=-1))
        elif save_vel.ndim == 4 and save_vel.shape[0] == 3:
            save_spd = np.sqrt(np.sum(save_vel ** 2, axis=0))
        elif save_vel.ndim == 4 and save_vel.shape[-1] == 3:
            save_spd = np.sqrt(np.sum(save_vel ** 2, axis=-1))
        else:
            # None of the recognized (vector-component-axis) shapes matched --
            # fail loudly rather than silently guessing an axis, which could
            # silently produce a wrong "speed" array.
            raise ValueError(
                f"Cannot infer the vector-component axis of velocity array with "
                f"shape {save_vel.shape}; expected a size-3 axis at position 0, 1, "
                f"or -1."
            )
        speed_path = os.path.join(output_dir, f"{prefix}_speed.npy")
        np.save(speed_path, save_spd.astype(np.float32))
        saved['speed'] = speed_path
    else:
        save_spd = None

    if save_src is not None:
        source_path = os.path.join(output_dir, f"{prefix}_source.npy")
        np.save(source_path, save_src.astype(np.float32))
        saved['source'] = source_path

    if save_mask is not None:
        mask_path = os.path.join(output_dir, f"{prefix}_mask.npy")
        np.save(mask_path, save_mask.astype(np.float32))
        saved['mask'] = mask_path

    # Save comprehensive compressed npz archive
    npz_data = {
        'density': save_dens.astype(np.float32),
        'velocity': save_vel.astype(np.float32),
    }
    if save_spd is not None:
        npz_data['speed'] = save_spd.astype(np.float32)
    if save_src is not None:
        npz_data['source'] = save_src.astype(np.float32)
    if save_mask is not None:
        npz_data['mask'] = save_mask.astype(np.float32)
    if save_aff is not None:
        npz_data['affine'] = save_aff
    if voxel_size is not None:
        npz_data['voxel_size'] = np.array(voxel_size)
    if bbox_origin != (0, 0, 0):
        npz_data['bbox_origin'] = np.array(bbox_origin)
        if cropped_shape:
            npz_data['bbox_shape'] = np.array(cropped_shape)
    if metrics is not None:
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list, float, int)):
                npz_data[f"metric_{k}"] = np.array(v)

    npz_path = os.path.join(output_dir, f"{prefix}_uromt_results.npz")
    np.savez_compressed(npz_path, **npz_data)
    saved['npz_archive'] = npz_path

    return saved


def load_numpy_results(npz_or_dir_path: str, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Load urOMT results from a .npz file or output directory.
    """
    if os.path.isfile(npz_or_dir_path) and npz_or_dir_path.endswith('.npz'):
        archive = np.load(npz_or_dir_path, allow_pickle=True)
        return {k: archive[k] for k in archive.files}

    if os.path.isdir(npz_or_dir_path):
        pref = f"{prefix}_" if prefix else ""
        results = {}
        for name in ['density', 'velocity', 'source', 'mask', 'speed']:
            fp = os.path.join(npz_or_dir_path, f"{pref}{name}.npy")
            if os.path.exists(fp):
                results[name] = np.load(fp)
        return results

    raise FileNotFoundError(f"Results path not found: {npz_or_dir_path}")


# =============================================================================
# Plotting and 2D Slice Visualization Functions
# =============================================================================

def get_orthogonal_slice_indices(
    volume_shape: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
) -> Tuple[int, int, int]:
    """
    Compute (sagittal_idx, coronal_idx, axial_idx) for orthogonal slicing.
    Defaults to the center of mass of the mask ROI (or midpoint of volume).
    """
    Nx, Ny, Nz = volume_shape
    if mask is not None and np.sum(mask > 0) > 0:
        coords = np.argwhere(mask > 0)
        cx = int(np.round(np.mean(coords[:, 0])))
        cy = int(np.round(np.mean(coords[:, 1])))
        cz = int(np.round(np.mean(coords[:, 2])))
    else:
        cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

    sx = slice_indices[0] if (slice_indices and slice_indices[0] is not None) else cx
    sy = slice_indices[1] if (slice_indices and len(slice_indices) > 1 and slice_indices[1] is not None) else cy
    sz = slice_indices[2] if (slice_indices and len(slice_indices) > 2 and slice_indices[2] is not None) else cz

    sx = max(0, min(int(sx), Nx - 1))
    sy = max(0, min(int(sy), Ny - 1))
    sz = max(0, min(int(sz), Nz - 1))
    return sx, sy, sz


def plot_density_orthogonal_slices(
    density: np.ndarray,
    time_indices: Optional[List[int]] = None,
    mask: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    crop_to_mask: bool = True,
    bbox_pad: int = 2,
    save_path: Optional[str] = None,
    title_prefix: str = "Tracer Concentration Density (ρ)",
    dpi: int = 150
) -> plt.Figure:
    """
    Plot 3-row orthogonal views (Sagittal, Coronal, Axial) of Density ρ across time points,
    cropped to the tight bounding box of the mask ROI.
    """
    plot_dens = density
    plot_mask = mask

    # Crop to tight bounding box of the mask ROI
    if mask is not None and crop_to_mask and np.sum(mask > 0) > 0:
        bbox_slices, cropped_shape, _ = compute_mask_bbox(mask, pad=bbox_pad)
        plot_dens = crop_array_to_bbox(density, bbox_slices)
        plot_mask = crop_array_to_bbox(mask, bbox_slices)

    Nt = plot_dens.shape[0]
    spatial_shape = plot_dens.shape[1:]
    sx, sy, sz = get_orthogonal_slice_indices(spatial_shape, plot_mask, slice_indices)

    if time_indices is None:
        if Nt <= 4:
            time_indices = list(range(Nt))
        else:
            time_indices = [0, Nt // 3, (2 * Nt) // 3, Nt - 1]
    time_indices = [t for t in time_indices if 0 <= t < Nt]
    n_cols = len(time_indices)

    fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 11.5), constrained_layout=True)
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    rho_vmax = float(np.percentile(plot_dens, 99.5)) if np.max(plot_dens) > 0 else 1.0

    for col, t in enumerate(time_indices):
        d_t = plot_dens[t]
        t_str = f"t = {t}" if Nt > 1 else "Time-Averaged"

        # Row 0: Sagittal (x = sx) -> Y vs Z
        s_sag = d_t[sx, :, :].T
        im_sag = axes[0, col].imshow(s_sag, origin='lower', cmap='magma', vmin=0.0, vmax=rho_vmax)
        if plot_mask is not None and np.any(plot_mask[sx, :, :] > 0):
            axes[0, col].contour(plot_mask[sx, :, :].T > 0, levels=[0.5], colors='cyan', linewidths=0.9, alpha=0.8)
        axes[0, col].set_title(f"Sagittal (x={sx})\n{t_str}", fontsize=11, fontweight='bold')
        axes[0, col].set_xlabel("Voxel Y")
        axes[0, col].set_ylabel("Voxel Z")

        # Row 1: Coronal (y = sy) -> X vs Z
        s_cor = d_t[:, sy, :].T
        im_cor = axes[1, col].imshow(s_cor, origin='lower', cmap='magma', vmin=0.0, vmax=rho_vmax)
        if plot_mask is not None and np.any(plot_mask[:, sy, :] > 0):
            axes[1, col].contour(plot_mask[:, sy, :].T > 0, levels=[0.5], colors='cyan', linewidths=0.9, alpha=0.8)
        axes[1, col].set_title(f"Coronal (y={sy})\n{t_str}", fontsize=11, fontweight='bold')
        axes[1, col].set_xlabel("Voxel X")
        axes[1, col].set_ylabel("Voxel Z")

        # Row 2: Axial (z = sz) -> X vs Y
        s_ax = d_t[:, :, sz].T
        im_ax = axes[2, col].imshow(s_ax, origin='lower', cmap='magma', vmin=0.0, vmax=rho_vmax)
        if plot_mask is not None and np.any(plot_mask[:, :, sz] > 0):
            axes[2, col].contour(plot_mask[:, :, sz].T > 0, levels=[0.5], colors='cyan', linewidths=0.9, alpha=0.8)
        axes[2, col].set_title(f"Axial (z={sz})\n{t_str}", fontsize=11, fontweight='bold')
        axes[2, col].set_xlabel("Voxel X")
        axes[2, col].set_ylabel("Voxel Y")

        if col == n_cols - 1:
            row_ims = [im_sag, im_cor, im_ax]
            for r in range(3):
                cbar = fig.colorbar(row_ims[r], ax=axes[r, col], fraction=0.046, pad=0.04)
                cbar.set_label("Density ρ", fontsize=9)

    fig.suptitle(f"{title_prefix} — Orthogonal Planes (ROI Bounding Box)", fontsize=14, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Plotter] Saved density orthogonal visualization: {save_path}")

    return fig


def plot_velocity_orthogonal_slices(
    velocity: np.ndarray,
    speed: Optional[np.ndarray] = None,
    time_indices: Optional[List[int]] = None,
    mask: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    scale_quiver: float = 10.0,
    quiver_subsample: int = 4,
    crop_to_mask: bool = True,
    bbox_pad: int = 2,
    save_path: Optional[str] = None,
    title_prefix: str = "Fluid Flow Velocity & Speed (||v||)",
    dpi: int = 150
) -> plt.Figure:
    """
    Plot 3-row orthogonal views (Sagittal, Coronal, Axial) of Speed ||v|| with Quiver overlays,
    cropped to the tight bounding box of the mask ROI.
    """
    plot_vel = velocity
    plot_spd = speed if speed is not None else np.sqrt(np.sum(velocity ** 2, axis=1))
    plot_mask = mask

    # Crop to tight bounding box of the mask ROI
    if mask is not None and crop_to_mask and np.sum(mask > 0) > 0:
        bbox_slices, cropped_shape, _ = compute_mask_bbox(mask, pad=bbox_pad)
        plot_vel = crop_array_to_bbox(velocity, bbox_slices)
        plot_spd = crop_array_to_bbox(plot_spd, bbox_slices)
        plot_mask = crop_array_to_bbox(mask, bbox_slices)

    Nt_vel = plot_vel.shape[0]
    spatial_shape = plot_vel.shape[2:]  # (Nx, Ny, Nz)
    sx, sy, sz = get_orthogonal_slice_indices(spatial_shape, plot_mask, slice_indices)

    if time_indices is None:
        if Nt_vel <= 4:
            time_indices = list(range(Nt_vel))
        else:
            time_indices = [0, Nt_vel // 3, (2 * Nt_vel) // 3, Nt_vel - 1]
    time_indices = [t for t in time_indices if 0 <= t < Nt_vel]
    n_cols = len(time_indices)

    fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 11.5), constrained_layout=True)
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    v_vmax = float(np.percentile(plot_spd, 99.0)) if np.max(plot_spd) > 0 else 1.0
    if v_vmax == 0:
        v_vmax = 1e-4

    eff_scale = 30.0 / max(1e-4, float(scale_quiver))
    step = max(1, quiver_subsample)

    for col, t in enumerate(time_indices):
        v_t = plot_vel[t]    # (3, Nx, Ny, Nz)
        sp_t = plot_spd[t]   # (Nx, Ny, Nz)
        t_str = f"t = {t}" if Nt_vel > 1 else "Time-Averaged"

        # -------------------------------------------------------------
        # Row 0: Sagittal (x = sx) -> In-plane components: vy, vz
        # -------------------------------------------------------------
        sp_sag = sp_t[sx, :, :].T
        u_sag = v_t[1, sx, :, :].T
        v_sag = v_t[2, sx, :, :].T

        im_sag = axes[0, col].imshow(sp_sag, origin='lower', cmap='viridis', vmin=0.0, vmax=v_vmax)
        if plot_mask is not None and np.any(plot_mask[sx, :, :] > 0):
            axes[0, col].contour(plot_mask[sx, :, :].T > 0, levels=[0.5], colors='red', linewidths=0.9, alpha=0.7)

        H, W = sp_sag.shape
        Y_g, X_g = np.mgrid[0:H:step, 0:W:step]
        U_sub = u_sag[0:H:step, 0:W:step]
        V_sub = v_sag[0:H:step, 0:W:step]
        q_mask = np.sqrt(U_sub ** 2 + V_sub ** 2) > (0.01 * v_vmax)
        if np.any(q_mask):
            axes[0, col].quiver(
                X_g[q_mask], Y_g[q_mask], U_sub[q_mask], V_sub[q_mask],
                color='white', alpha=0.88, scale=eff_scale,
                scale_units='width', width=0.005, headwidth=3.5, headlength=4.5
            )
        axes[0, col].set_title(f"Sagittal (x={sx})\n{t_str}", fontsize=11, fontweight='bold')
        axes[0, col].set_xlabel("Voxel Y")
        axes[0, col].set_ylabel("Voxel Z")

        # -------------------------------------------------------------
        # Row 1: Coronal (y = sy) -> In-plane components: vx, vz
        # -------------------------------------------------------------
        sp_cor = sp_t[:, sy, :].T
        u_cor = v_t[0, :, sy, :].T
        v_cor = v_t[2, :, sy, :].T

        im_cor = axes[1, col].imshow(sp_cor, origin='lower', cmap='viridis', vmin=0.0, vmax=v_vmax)
        if plot_mask is not None and np.any(plot_mask[:, sy, :] > 0):
            axes[1, col].contour(plot_mask[:, sy, :].T > 0, levels=[0.5], colors='red', linewidths=0.9, alpha=0.7)

        H, W = sp_cor.shape
        Y_g, X_g = np.mgrid[0:H:step, 0:W:step]
        U_sub = u_cor[0:H:step, 0:W:step]
        V_sub = v_cor[0:H:step, 0:W:step]
        q_mask = np.sqrt(U_sub ** 2 + V_sub ** 2) > (0.01 * v_vmax)
        if np.any(q_mask):
            axes[1, col].quiver(
                X_g[q_mask], Y_g[q_mask], U_sub[q_mask], V_sub[q_mask],
                color='white', alpha=0.88, scale=eff_scale,
                scale_units='width', width=0.005, headwidth=3.5, headlength=4.5
            )
        axes[1, col].set_title(f"Coronal (y={sy})\n{t_str}", fontsize=11, fontweight='bold')
        axes[1, col].set_xlabel("Voxel X")
        axes[1, col].set_ylabel("Voxel Z")

        # -------------------------------------------------------------
        # Row 2: Axial (z = sz) -> In-plane components: vx, vy
        # -------------------------------------------------------------
        sp_ax = sp_t[:, :, sz].T
        u_ax = v_t[0, :, :, sz].T
        v_ax = v_t[1, :, :, sz].T

        im_ax = axes[2, col].imshow(sp_ax, origin='lower', cmap='viridis', vmin=0.0, vmax=v_vmax)
        if plot_mask is not None and np.any(plot_mask[:, :, sz] > 0):
            axes[2, col].contour(plot_mask[:, :, sz].T > 0, levels=[0.5], colors='red', linewidths=0.9, alpha=0.7)

        H, W = sp_ax.shape
        Y_g, X_g = np.mgrid[0:H:step, 0:W:step]
        U_sub = u_ax[0:H:step, 0:W:step]
        V_sub = v_ax[0:H:step, 0:W:step]
        q_mask = np.sqrt(U_sub ** 2 + V_sub ** 2) > (0.01 * v_vmax)
        if np.any(q_mask):
            axes[2, col].quiver(
                X_g[q_mask], Y_g[q_mask], U_sub[q_mask], V_sub[q_mask],
                color='white', alpha=0.88, scale=eff_scale,
                scale_units='width', width=0.005, headwidth=3.5, headlength=4.5
            )
        axes[2, col].set_title(f"Axial (z={sz})\n{t_str}", fontsize=11, fontweight='bold')
        axes[2, col].set_xlabel("Voxel X")
        axes[2, col].set_ylabel("Voxel Y")

        if col == n_cols - 1:
            row_ims = [im_sag, im_cor, im_ax]
            for r in range(3):
                cbar = fig.colorbar(row_ims[r], ax=axes[r, col], fraction=0.046, pad=0.04)
                cbar.set_label("Speed ||v|| (vox/frame)", fontsize=9)

    fig.suptitle(f"{title_prefix} — Orthogonal Planes with Quivers (Scale: {scale_quiver}x, ROI Bounding Box)", fontsize=14, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Plotter] Saved velocity orthogonal visualization: {save_path}")

    return fig


def plot_source_orthogonal_slices(
    source: np.ndarray,
    time_indices: Optional[List[int]] = None,
    mask: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    crop_to_mask: bool = True,
    bbox_pad: int = 2,
    save_path: Optional[str] = None,
    title_prefix: str = "Relative Source / Sink Rate (r)",
    dpi: int = 150
) -> plt.Figure:
    """
    Plot 3-row orthogonal views (Sagittal, Coronal, Axial) of Source Rate r (Red=Influx, Blue=Efflux),
    cropped to the tight bounding box of the mask ROI.
    """
    plot_src = source
    plot_mask = mask

    # Crop to tight bounding box of the mask ROI
    if mask is not None and crop_to_mask and np.sum(mask > 0) > 0:
        bbox_slices, cropped_shape, _ = compute_mask_bbox(mask, pad=bbox_pad)
        plot_src = crop_array_to_bbox(source, bbox_slices)
        plot_mask = crop_array_to_bbox(mask, bbox_slices)

    Nt_src = plot_src.shape[0]
    spatial_shape = plot_src.shape[1:]
    sx, sy, sz = get_orthogonal_slice_indices(spatial_shape, plot_mask, slice_indices)

    if time_indices is None:
        if Nt_src <= 4:
            time_indices = list(range(Nt_src))
        else:
            time_indices = [0, Nt_src // 3, (2 * Nt_src) // 3, Nt_src - 1]
    time_indices = [t for t in time_indices if 0 <= t < Nt_src]
    n_cols = len(time_indices)

    fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 11.5), constrained_layout=True)
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    src_abs_max = max(1e-4, float(np.percentile(np.abs(plot_src), 99.0))) if np.max(np.abs(plot_src)) > 0 else 0.5

    for col, t in enumerate(time_indices):
        s_t = plot_src[t]
        t_str = f"t = {t}" if Nt_src > 1 else "Time-Averaged"

        # Row 0: Sagittal (x = sx) -> Y vs Z
        r_sag = s_t[sx, :, :].T
        im_sag = axes[0, col].imshow(r_sag, origin='lower', cmap='bwr', vmin=-src_abs_max, vmax=src_abs_max)
        if plot_mask is not None and np.any(plot_mask[sx, :, :] > 0):
            axes[0, col].contour(plot_mask[sx, :, :].T > 0, levels=[0.5], colors='black', linewidths=0.8, alpha=0.6)
        axes[0, col].set_title(f"Sagittal (x={sx})\n{t_str}", fontsize=11, fontweight='bold')
        axes[0, col].set_xlabel("Voxel Y")
        axes[0, col].set_ylabel("Voxel Z")

        # Row 1: Coronal (y = sy) -> X vs Z
        r_cor = s_t[:, sy, :].T
        im_cor = axes[1, col].imshow(r_cor, origin='lower', cmap='bwr', vmin=-src_abs_max, vmax=src_abs_max)
        if plot_mask is not None and np.any(plot_mask[:, sy, :] > 0):
            axes[1, col].contour(plot_mask[:, sy, :].T > 0, levels=[0.5], colors='black', linewidths=0.8, alpha=0.6)
        axes[1, col].set_title(f"Coronal (y={sy})\n{t_str}", fontsize=11, fontweight='bold')
        axes[1, col].set_xlabel("Voxel X")
        axes[1, col].set_ylabel("Voxel Z")

        # Row 2: Axial (z = sz) -> X vs Y
        r_ax = s_t[:, :, sz].T
        im_ax = axes[2, col].imshow(r_ax, origin='lower', cmap='bwr', vmin=-src_abs_max, vmax=src_abs_max)
        if plot_mask is not None and np.any(plot_mask[:, :, sz] > 0):
            axes[2, col].contour(plot_mask[:, :, sz].T > 0, levels=[0.5], colors='black', linewidths=0.8, alpha=0.6)
        axes[2, col].set_title(f"Axial (z={sz})\n{t_str}", fontsize=11, fontweight='bold')
        axes[2, col].set_xlabel("Voxel X")
        axes[2, col].set_ylabel("Voxel Y")

        if col == n_cols - 1:
            row_ims = [im_sag, im_cor, im_ax]
            for r in range(3):
                cbar = fig.colorbar(row_ims[r], ax=axes[r, col], fraction=0.046, pad=0.04)
                cbar.set_label("Source Rate r (1/frame)", fontsize=9)

    fig.suptitle(f"{title_prefix} — Orthogonal Planes (Red=Influx, Blue=Efflux, ROI Bounding Box)", fontsize=14, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Plotter] Saved source orthogonal visualization: {save_path}")

    return fig


def plot_density_and_velocity_slices(
    density: np.ndarray,
    velocity: np.ndarray,
    time_indices: Optional[List[int]] = None,
    slice_axis: Union[str, int] = 'axial',
    slice_index: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    source: Optional[np.ndarray] = None,
    quiver_subsample: int = 4,
    scale_quiver: float = 10.0,
    crop_to_mask: bool = True,
    bbox_pad: int = 2,
    save_path: Optional[str] = None,
    title_prefix: str = "urOMT Dynamic Transport",
    voxel_size: Optional[Tuple[float, ...]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Unified entry point to generate the 3 separate orthogonal slice figures cropped to the mask ROI.
    """
    # Map slice_axis and slice_index to orthogonal slice_indices tuple (sx, sy, sz)
    slice_indices = None
    if slice_index is not None:
        axis_str = str(slice_axis).lower()
        if axis_str in ('axial', 'z', '2'):
            slice_indices = (None, None, int(slice_index))
        elif axis_str in ('coronal', 'y', '1'):
            slice_indices = (None, int(slice_index), None)
        elif axis_str in ('sagittal', 'x', '0'):
            slice_indices = (int(slice_index), None, None)

    out_dir = os.path.dirname(os.path.abspath(save_path)) if save_path else "results"
    base_prefix = os.path.splitext(os.path.basename(save_path))[0] if save_path else "uromt"
    base_prefix = base_prefix.replace("_slices", "").replace("_density_velocity", "")

    path_dens = os.path.join(out_dir, f"{base_prefix}_density_slices.png")
    path_vel = os.path.join(out_dir, f"{base_prefix}_velocity_slices.png")
    path_src = os.path.join(out_dir, f"{base_prefix}_source_slices.png")

    fig_dens = plot_density_orthogonal_slices(
        density=density,
        time_indices=time_indices,
        mask=mask,
        slice_indices=slice_indices,
        crop_to_mask=crop_to_mask,
        bbox_pad=bbox_pad,
        save_path=path_dens,
        title_prefix=f"{title_prefix}: Density (ρ)",
        dpi=dpi
    )
    if save_path:
        plt.close(fig_dens)

    fig_vel = plot_velocity_orthogonal_slices(
        velocity=velocity,
        time_indices=time_indices,
        mask=mask,
        slice_indices=slice_indices,
        scale_quiver=scale_quiver,
        quiver_subsample=quiver_subsample,
        crop_to_mask=crop_to_mask,
        bbox_pad=bbox_pad,
        save_path=path_vel,
        title_prefix=f"{title_prefix}: Velocity & Speed",
        dpi=dpi
    )

    if source is not None:
        fig_src = plot_source_orthogonal_slices(
            source=source,
            time_indices=time_indices,
            mask=mask,
            slice_indices=slice_indices,
            crop_to_mask=crop_to_mask,
            bbox_pad=bbox_pad,
            save_path=path_src,
            title_prefix=f"{title_prefix}: Source / Sink Rate (r)",
            dpi=dpi
        )
        if save_path:
            plt.close(fig_src)

    # If an explicit save_path was provided and differs from path_vel, also save fig_vel there
    if save_path and os.path.abspath(save_path) != os.path.abspath(path_vel):
        fig_vel.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig_vel


def plot_summary_metrics(metrics: Dict[str, Any], save_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    """
    Plot time series curves of transport kinetics: mass, mean velocity, kinetic energy, influx/efflux.

    Parameters
    ----------
    metrics : dict
        Dictionary containing metric time-courses (e.g. 'total_mass', 'mean_speed', 'kinetic_energy', 'influx', 'efflux').
    save_path : str, optional
        Destination path to save the plot figure.
    dpi : int, default 150
        Resolution for saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # 1. Total Mass Over Time
    if 'total_mass' in metrics:
        mass = metrics['total_mass']
        t_arr = np.arange(len(mass))
        axes[0, 0].plot(t_arr, mass, 'o-', color='#1f77b4', linewidth=2, markersize=5)
        axes[0, 0].set_title("Total Mass $\\int \\rho(t, x) dx$", fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel("Time Frame Index")
        axes[0, 0].set_ylabel("Total Density")
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Mean Transport Speed
    if 'mean_speed' in metrics:
        speed = metrics['mean_speed']
        t_arr = np.arange(len(speed))
        axes[0, 1].plot(t_arr, speed, 's-', color='#2ca02c', linewidth=2, markersize=5)
        axes[0, 1].set_title("Mean Transport Speed $\\langle \\|v\\| \\rangle$", fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel("Time Frame Transition")
        axes[0, 1].set_ylabel("Speed (voxel/frame)")
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Kinetic Energy
    if 'kinetic_energy' in metrics:
        ke = metrics['kinetic_energy']
        t_arr = np.arange(len(ke))
        axes[1, 0].plot(t_arr, ke, '^-', color='#d62728', linewidth=2, markersize=5)
        axes[1, 0].set_title("Transport Kinetic Energy $\\int \\rho \\|v\\|^2 dx$", fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel("Time Frame Transition")
        axes[1, 0].set_ylabel("Kinetic Energy")
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Influx vs. Efflux Rates
    ax_rates = axes[1, 1]
    has_rates = False
    if 'influx' in metrics and 'efflux' in metrics:
        influx = metrics['influx']
        efflux = metrics['efflux']
        t_arr = np.arange(len(influx))
        ax_rates.plot(t_arr, influx, 'o-', color='#ff7f0e', label="Influx (+)", linewidth=2, markersize=5)
        ax_rates.plot(t_arr, efflux, 'x-', color='#9467bd', label="Efflux (-)", linewidth=2, markersize=5)
        ax_rates.legend(loc='best', frameon=True)
        has_rates = True
    elif 'loss' in metrics:
        loss = metrics['loss']
        t_arr = np.arange(len(loss))
        ax_rates.plot(t_arr, loss, 'd-', color='#8c564b', linewidth=2, markersize=5)
        ax_rates.set_ylabel("urOMT Cost Objective")
        has_rates = True

    ax_rates.set_title("Source Influx & Efflux Rates", fontsize=11, fontweight='bold')
    ax_rates.set_xlabel("Time Frame Transition")
    ax_rates.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle("urOMT Transport Kinetics Summary", fontsize=14, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
