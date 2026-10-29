"""
main.py - Driver script for Unbalanced Regularized Optimal Mass Transport (urOMT)

Orchestrates:
  1. Command-line argument parsing and configuration.
  2. Importing 4D dynamic volumetric MRI data and optional 3D/4D masks from NIfTI files.
  3. Running the accelerated urOMT solver across time frames with tight ROI cropping (optional temporal warm-starting via --warm_start).
  4. Exporting density, velocity, source, and metric arrays as NumPy (.npy, .npz) and NIfTI (.nii) files.
  5. Generating publication-quality 2D slice visualizations and flow vector field quiver plots.
  6. Synthetic test dataset generation for verification and rapid demonstration.

Usage Examples:
  # Run on sample MRI dataset with mask:
  python main.py --signal data/sub32521_s1_signal.nii --mask data/sub32521_s1_mask.nii --output_dir results/sub32521_s1

  # Run on a subset of time frames with custom regularization:
  python main.py --signal data/sub32521_s1_signal.nii --mask data/sub32521_s1_mask.nii --time_start 0 --time_end 5 --sigma 0.1 --alpha 1.0

  # Run self-contained synthetic benchmark test:
  python main.py --synthetic --output_dir results/synthetic_demo
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import traceback
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# Import project modules
from uromt_io import (
    load_nifti,
    save_nifti,
    save_numpy_results,
    plot_density_and_velocity_slices,
    plot_summary_metrics
)
from uromt_core import run_uromt_sequence


def create_synthetic_data(
    grid_size: Tuple[int, int, int] = (40, 40, 30),
    n_frames: int = 5,
    include_source: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 4D dynamic density series (moving and growing 3D Gaussian sphere)
    and corresponding 3D mask for validation and benchmarking.
    """
    Nx, Ny, Nz = grid_size
    signal_4d = np.zeros((Nx, Ny, Nz, n_frames), dtype=np.float32)

    # Coordinate grids
    x, y, z = np.ogrid[:Nx, :Ny, :Nz]

    # Motion trajectory from (12, 12, 10) towards (28, 28, 20)
    for t in range(n_frames):
        alpha = t / max(1, n_frames - 1)
        cx = 12.0 + 16.0 * alpha
        cy = 12.0 + 16.0 * alpha
        cz = 10.0 + 10.0 * alpha

        # Intensity grows over time (unbalanced source influx)
        amplitude = 100.0 * (1.0 + 0.4 * alpha) if include_source else 100.0
        width = 4.5

        r_sq = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        frame = amplitude * np.exp(-r_sq / (2.0 * width ** 2))
        signal_4d[..., t] = frame.astype(np.float32)

    # Spatial mask encompassing the domain
    mask_3d = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    mask_3d[3:Nx - 3, 3:Ny - 3, 3:Nz - 3] = 1.0

    return signal_4d, mask_3d


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unbalanced Regularized Optimal Mass Transport (urOMT) for Dynamic 4D MRI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input & Output options
    parser.add_argument("--signal", "--input", "-i", type=str, default=None,
                        help="Path to input 4D NIfTI signal file (.nii / .nii.gz)")
    parser.add_argument("--mask", "-m", type=str, default=None,
                        help="Optional path to 3D NIfTI binary mask file (.nii / .nii.gz)")
    parser.add_argument("--output_dir", "-o", type=str, default="results",
                        help="Directory to save output numpy files, nifti files, and plots")
    parser.add_argument("--prefix", "-p", type=str, default=None,
                        help="Prefix for saved output files (defaults to input filename stem)")

    # urOMT Physical and Regularization Hyperparameters
    parser.add_argument("--sigma", type=float, default=2.0e-3,
                        help="Diffusion regularization coefficient (sigma > 0)")
    parser.add_argument("--alpha", type=float, default=100.0,
                        help="Weighting parameter for relative source term in cost functional (alpha > 0)")
    parser.add_argument("--beta", type=float, default=20.0,
                        help="Weighting parameter for endpoint fitting term (beta > 0)")
    parser.add_argument("--n_substeps", type=int, default=3,
                        help="Number of temporal sub-steps between consecutive MRI frames (m >= 1)")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum optimizer iterations per frame transition")
    parser.add_argument("--tol", type=float, default=1.0e-8,
                        help="Convergence gradient tolerance for L-BFGS-B (gtol)")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Standard deviation of Gaussian smoothing kernel for noise reduction")
    parser.add_argument("--warm_start", action="store_true", default=False,
                        help="Seed each frame transition's optimizer with the previous transition's "
                             "solution. Disabled by default: warm-starting traps L-BFGS-B at the "
                             "seed (line-search failure), collapsing the velocity/source output to "
                             "the first transition's solution (time-invariant fields).")

    # Time frame filtering and spatial downsampling
    parser.add_argument("--time_start", type=int, default=0,
                        help="Start time frame index (default: 0)")
    parser.add_argument("--time_end", type=int, default=None,
                        help="End time frame index (inclusive, None for all frames)")
    parser.add_argument("--time_stride", type=int, default=1,
                        help="Stride between time frames to skip intermediate acquisitions")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Spatial downsampling factor (e.g. 2 for 2x downsampling on large grids)")

    # Visualization settings
    parser.add_argument("--slice_axis", type=str, default="axial",
                        choices=["axial", "coronal", "sagittal", "x", "y", "z"],
                        help="2D slice visualization orientation plane")
    parser.add_argument("--slice_index", type=int, default=None,
                        help="Specific slice index along slice axis (auto-selects ROI center if None)")
    parser.add_argument("--quiver_subsample", type=int, default=4,
                        help="Subsampling interval for 2D velocity quiver arrows")
    parser.add_argument("--scale_quiver", type=float, default=1.0,
                        help="Scaling factor for quiver arrow lengths")

    # Preprocessing: Baseline Subtraction & Normalization
    parser.add_argument("--baseline_frames", type=int, default=1,
                        help="Number of leading pre-contrast frames to average as baseline (default: 1)")
    parser.add_argument("--no_baseline_subtraction", action="store_false", dest="subtract_baseline", default=True,
                        help="Disable pre-contrast baseline subtraction")
    parser.add_argument("--no_normalize", action="store_false", dest="normalize_max", default=True,
                        help="Disable global peak normalization to [0, 1]")

    # Performance & Numerical Acceleration Options
    parser.add_argument("--crop_roi", action="store_true", default=True,
                        help="Automatically crop volume to tight bounding box of mask ROI (default: True)")
    parser.add_argument("--no_crop_roi", action="store_false", dest="crop_roi",
                        help="Disable automatic tight bounding-box ROI cropping")
    parser.add_argument("--no_numba", action="store_false", dest="use_numba", default=True,
                        help="Disable Numba JIT parallel acceleration and use pure NumPy")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Number of parallel CPU worker threads for FFT/DCT (-1 for all cores)")

    # Execution modes
    parser.add_argument("--save_nifti", action="store_true", default=True,
                        help="Export computed density and velocity fields as NIfTI (.nii) files")
    parser.add_argument("--no_save_nifti", action="store_false", dest="save_nifti",
                        help="Disable NIfTI file export")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Run self-contained synthetic benchmark test")

    return parser


def main() -> int:
    """Main execution driver function."""
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        return _run(args)
    except Exception as exc:
        print(f"\nError: urOMT processing failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


def _run(args: argparse.Namespace) -> int:
    """Core processing pipeline, isolated from main() so it can be wrapped in a
    single top-level try/except that reports a clean error instead of a raw
    traceback dumped mid-run (e.g. on a corrupt NIfTI file or an OOM condition)."""
    # 1. Determine Data Source (Synthetic vs. NIfTI input)
    if args.synthetic or (args.signal is None):
        if args.signal is None and not args.synthetic:
            print("No input NIfTI file specified. Running synthetic demonstration benchmark...")
        else:
            print("Running synthetic benchmark mode...")

        prefix = "synthetic_demo" if args.prefix is None else args.prefix
        signal_4d, mask_3d = create_synthetic_data(grid_size=(40, 40, 30), n_frames=5, include_source=True)
        voxel_size = (1.0, 1.0, 1.0)
        affine = np.eye(4, dtype=np.float32)
        meta = {'voxel_size': voxel_size, 'affine': affine}
    else:
        # Load real NIfTI files
        signal_path = args.signal
        if not os.path.exists(signal_path):
            print(f"Error: Input signal file not found: {signal_path}", file=sys.stderr)
            return 1

        print(f"Loading input NIfTI signal: {signal_path}")
        signal_raw, meta = load_nifti(signal_path)
        voxel_size = meta.get('voxel_size', (1.0, 1.0, 1.0))[:3]
        affine = meta.get('affine', np.eye(4, dtype=np.float32))

        if signal_raw.ndim == 3:
            # Expand single 3D frame to 4D with 1 time step
            signal_raw = signal_raw[..., np.newaxis]

        if signal_raw.ndim != 4:
            print(f"Error: Expected 4D dynamic NIfTI (Nx, Ny, Nz, Nt), got shape: {signal_raw.shape}", file=sys.stderr)
            return 1

        # Load optional mask
        mask_3d = None
        if args.mask:
            if os.path.exists(args.mask):
                print(f"Loading mask NIfTI: {args.mask}")
                mask_raw, _ = load_nifti(args.mask)
                if mask_raw.ndim == 4:
                    mask_raw = mask_raw[..., 0]
                mask_3d = (mask_raw > 0).astype(np.float32)
                if mask_3d.shape != signal_raw.shape[:3]:
                    print(f"Error: Mask spatial shape {mask_3d.shape} does not match signal volume shape {signal_raw.shape[:3]}.", file=sys.stderr)
                    return 1
            else:
                print(f"Warning: Mask file not found: {args.mask}. Proceeding without mask.")

        # Slice time range (named frame_start/frame_end to avoid shadowing the
        # t_start = time.time() timer variable used later in this function)
        frame_start = max(0, args.time_start)
        frame_end = signal_raw.shape[-1] if args.time_end is None else min(args.time_end + 1, signal_raw.shape[-1])
        frame_stride = max(1, args.time_stride)
        time_slice = slice(frame_start, frame_end, frame_stride)
        signal_4d = signal_raw[..., time_slice]

        # Spatial downsampling if requested
        if args.downsample > 1:
            ds = args.downsample
            print(f"Applying {ds}x spatial downsampling (with anti-aliasing pre-filter) for accelerated processing...")
            # Smooth before decimating so high-frequency spatial structure isn't
            # aliased into the coarser grid (plain strided slicing would just pick
            # every ds-th voxel with no low-pass filtering first).
            signal_4d = uniform_filter(signal_4d, size=(ds, ds, ds, 1))
            signal_4d = signal_4d[::ds, ::ds, ::ds, :]
            if mask_3d is not None:
                mask_3d = uniform_filter(mask_3d.astype(np.float32), size=(ds, ds, ds))
                mask_3d = (mask_3d[::ds, ::ds, ::ds] > 0.5).astype(np.float32)
            voxel_size = tuple(v * ds for v in voxel_size)
            affine_ds = affine.copy()
            affine_ds[:3, :3] *= ds
            affine = affine_ds

        # Output prefix
        if args.prefix:
            prefix = args.prefix
        else:
            base_name = os.path.basename(signal_path)
            prefix = os.path.splitext(base_name.replace('.nii.gz', '').replace('.nii', ''))[0]

    Nx, Ny, Nz, Nt = signal_4d.shape
    print(f"\n--- urOMT Processing Configuration ---")
    print(f"  Volume Dimensions  : {Nx} x {Ny} x {Nz}")
    print(f"  Time Frames        : {Nt}")
    print(f"  Voxel Resolution   : ({voxel_size[0]:.2f}, {voxel_size[1]:.2f}, {voxel_size[2]:.2f}) mm")
    print(f"  Diffusion Coeff    : sigma = {args.sigma}")
    print(f"  Source Weight      : alpha = {args.alpha}")
    print(f"  Endpoint Penalty   : beta  = {args.beta}")
    print(f"  Temporal Sub-steps : m = {args.n_substeps}")
    print(f"  Max Iterations     : {args.max_iter}")
    print(f"  Temporal Warm-Start: {args.warm_start}")
    print(f"  Crop Mask ROI      : {args.crop_roi}")
    print(f"  Numba Acceleration : {args.use_numba}")
    print(f"  Worker Threads     : {args.workers}")
    print(f"  Output Directory   : {args.output_dir}\n")

    if Nt < 2:
        print("Error: Need at least 2 time steps to compute transport velocity fields.", file=sys.stderr)
        return 1

    # 2. Run urOMT Solver
    t_start = time.time()
    results = run_uromt_sequence(
        signal_4d=signal_4d,
        mask_3d=mask_3d,
        sigma=args.sigma,
        alpha=args.alpha,
        beta=args.beta,
        n_substeps=args.n_substeps,
        max_iter=args.max_iter,
        tol=args.tol,
        voxel_size=voxel_size,
        smooth_sigma=args.smooth_sigma,
        baseline_frames=args.baseline_frames,
        subtract_baseline=args.subtract_baseline,
        normalize_max=args.normalize_max,
        crop_roi=args.crop_roi,
        warm_start=args.warm_start,
        use_numba=args.use_numba,
        workers=args.workers,
        verbose=True
    )
    runtime = time.time() - t_start

    density = results['density']      # (Nt, Nx, Ny, Nz)
    velocity = results['velocity']    # (Nt-1, 3, Nx, Ny, Nz)
    source = results['source']        # (Nt-1, Nx, Ny, Nz)
    metrics = results['metrics']

    # 3. Save NumPy Results (.npy and .npz)
    print("\n--- Saving Results to Disk (Mask ROI Bounding Box) ---")
    saved_npy = save_numpy_results(
        output_dir=args.output_dir,
        prefix=prefix,
        density=density,
        velocity=velocity,
        source=source,
        mask=mask_3d,
        metrics=metrics,
        affine=affine,
        voxel_size=voxel_size,
        crop_to_mask=True,
        bbox_pad=4
    )
    for name, path in saved_npy.items():
        print(f"  Saved {name:12s}: {path}")

    # 4. Save NIfTI Files (.nii)
    if args.save_nifti:
        # Save 4D dynamic density volume: transpose from (Nt, Nx, Ny, Nz) to (Nx, Ny, Nz, Nt)
        nii_density_path = os.path.join(args.output_dir, f"{prefix}_density.nii")
        density_4d_nii = np.transpose(density, (1, 2, 3, 0))
        save_nifti(density_4d_nii, nii_density_path, affine=affine, mask=mask_3d, crop_to_mask=True, bbox_pad=4)
        print(f"  Saved NIfTI density: {nii_density_path}")

        # Save 4D dynamic source rate volume: (Nx, Ny, Nz, Nt-1)
        nii_source_path = os.path.join(args.output_dir, f"{prefix}_source.nii")
        source_4d_nii = np.transpose(source, (1, 2, 3, 0))
        save_nifti(source_4d_nii, nii_source_path, affine=affine, mask=mask_3d, crop_to_mask=True, bbox_pad=4)
        print(f"  Saved NIfTI source : {nii_source_path}")

        # Save 4D flow speed volume: (Nx, Ny, Nz, Nt-1)
        speed_4d = np.sqrt(np.sum(velocity ** 2, axis=1))  # (Nt-1, Nx, Ny, Nz)
        speed_4d_nii = np.transpose(speed_4d, (1, 2, 3, 0))
        nii_speed_path = os.path.join(args.output_dir, f"{prefix}_speed.nii")
        save_nifti(speed_4d_nii, nii_speed_path, affine=affine, mask=mask_3d, crop_to_mask=True, bbox_pad=4)
        print(f"  Saved NIfTI speed  : {nii_speed_path}")

    # 5. Generate 2D Multi-Slice & Quiver Vector Field Visualizations
    print("\n--- Generating Publication Visualizations ---")
    slice_plot_path = os.path.join(args.output_dir, f"{prefix}_density_velocity_slices.png")
    fig_slices = plot_density_and_velocity_slices(
        density=density,
        velocity=velocity,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        mask=mask_3d,
        source=source,
        quiver_subsample=args.quiver_subsample,
        scale_quiver=args.scale_quiver,
        crop_to_mask=True,
        bbox_pad=4,
        save_path=slice_plot_path,
        title_prefix=f"urOMT Dynamic Transport ({prefix})",
        voxel_size=voxel_size,
        dpi=150
    )
    plt.close(fig_slices)
    print(f"  Saved 2D slice & quiver plots in {args.output_dir}:")
    print(f"    - {prefix}_density_slices.png")
    print(f"    - {prefix}_velocity_slices.png")
    print(f"    - {prefix}_source_slices.png")

    # 6. Generate Transport Kinetics Time-Course Plots
    metrics_plot_path = os.path.join(args.output_dir, f"{prefix}_kinetics_summary.png")
    fig_metrics = plot_summary_metrics(metrics, save_path=metrics_plot_path, dpi=150)
    plt.close(fig_metrics)
    print(f"  Saved transport kinetics summary: {metrics_plot_path}")

    print(f"\n=======================================================")
    print(f"   urOMT Processing Finished Successfully!")
    print(f"   Total Execution Time: {runtime:.2f} seconds")
    print(f"   All outputs stored in: {os.path.abspath(args.output_dir)}")
    print(f"=======================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
