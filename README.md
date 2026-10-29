# Unbalanced Regularized Optimal Mass Transport (urOMT) in Python

A high-performance Python implementation of the **Unbalanced Regularized Optimal Mass Transport (urOMT)** framework for 4D dynamic volumetric medical imaging (e.g., Dynamic Contrast-Enhanced MRI, dynamic PET/CT).

---

## 1. Overview & Mathematical Model

Optimal Mass Transport (OMT) in fluid dynamics seeks a velocity field $v(t, x)$ that minimizes the transport kinetic energy between mass distributions. Conventional OMT enforces strict mass conservation. In dynamic biological systems (such as DCE-MRI tracer delivery, tumor microvascular perfusion, or glymphatic clearance), mass enters and leaves the tissue over time.

The **urOMT** method introduces a relative source variable $r(t, x)$ into the Benamou–Brenier formulation with diffusion regularization:

$$\min_{v, r} \quad \int_0^T \int_{\Omega} \rho(t, x) \left( \|v(t, x)\|^2 + \alpha \, \chi(t, x) r(t, x)^2 \right) dx \, dt + \beta \int_{\Omega} \left( \rho(T, x) - \rho_1^{\text{img}}(x) \right)^2 dx$$

subject to the advection-diffusion-reaction PDE constraint:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho v) = \sigma \Delta \rho + \chi \rho r$$

$$\rho(0, x) = \rho_0^{\text{img}}(x)$$

### Variable Definitions
- $\rho(t, x) \ge 0$: Dynamic mass density (tracer concentration).
- $v(t, x) = (v_x, v_y, v_z)$: 3D velocity vector field.
- $r(t, x)$: Relative source rate ($r > 0$ represents influx, $r < 0$ represents clearance/efflux).
- $\chi(t, x) \in \{0, 1\}$: Spatial indicator mask restricting source/sink locations.
- $\sigma > 0$: Diffusion regularization coefficient.
- $\alpha > 0$: Source penalty weighting factor.
- $\beta > 0$: End-point fitting penalty factor.

---

## 2. Project Architecture

The codebase is organized into three clean, modular components:

```
AG-urOMT/
├── uromt_io.py          # Input/Output, NIfTI loading/saving, NumPy exporting, 2D slice & quiver visualization
├── uromt_core.py        # Core numerical solvers, operator splitting, adjoint gradients, sequence runner
├── main.py              # CLI driver orchestrating the workflow, preprocessing, execution, and reporting
├── data/                # Input 4D dynamic NIfTI MRI signal and mask files
├── requirements.txt     # Python dependencies
└── README.md            # Documentation and user guide
```

---

## 3. Installation & Dependencies

Ensure Python 3.8+ is installed with the following packages:

```bash
pip install -r requirements.txt
```

*(Note: `uromt_io.py` includes a standalone pure-NumPy binary NIfTI parser, so the codebase can function even in environments where `nibabel` is not installed!)*

---

## 4. Usage Guide

### 4.1 Running on Example 4D MRI NIfTI Data
Run urOMT on a 4D MRI signal file with an optional 3D mask:

```bash
python main.py --signal data/sub32521_s1_signal.nii --mask data/sub32521_s1_mask.nii --output_dir results/sub32521_s1
```

### 4.2 Running on a Time-Frame Subset
For rapid testing or targeted window analysis (e.g. frames 0 to 6):

```bash
python main.py --signal data/sub32521_s1_signal.nii --mask data/sub32521_s1_mask.nii --time_start 0 --time_end 6 --output_dir results/sub32521_s1_sub
```

### 4.3 Accelerated Prototyping with Spatial Downsampling
For rapid testing on large 3D grids:

```bash
python main.py --signal data/sub32521_s1_signal.nii --mask data/sub32521_s1_mask.nii --downsample 2 --max_iter 20 --output_dir results/sub32521_fast
```

### 4.4 Running the Built-in Synthetic Benchmark
To test and verify the solver without external data:

```bash
python main.py --synthetic --output_dir results/synthetic_demo
```

---

## 5. Command-Line Arguments Reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--signal` / `--input` | `None` | Path to 4D dynamic NIfTI MRI file (`.nii` / `.nii.gz`) |
| `--mask` | `None` | Optional path to 3D/4D NIfTI mask file |
| `--output_dir` | `results` | Output directory for `.npy`, `.npz`, `.nii`, and figures |
| `--prefix` | `auto` | Prefix for saved file names |
| `--sigma` | `0.01` | Diffusion coefficient ($\sigma > 0$) |
| `--alpha` | `100.0` | Relative source weighting ($\alpha > 0$) |
| `--beta` | `100.0` | End-point mismatch penalty ($\beta > 0$) |
| `--n_substeps` | `4` | Intermediate time sub-steps between consecutive frames |
| `--max_iter` | `40` | Maximum L-BFGS-B optimizer iterations per frame |
| `--tol` | `1e-8` | Convergence gradient tolerance for L-BFGS-B (gtol) |
| `--baseline_frames` | `1` | Number of leading pre-contrast frames to average as baseline |
| `--no_baseline_subtraction` | `False` | Disable pre-contrast baseline subtraction ($\Delta S = S - S_0$) |
| `--no_normalize` | `False` | Disable global peak normalization to $[0, 1]$ |
| `--crop_roi` / `--no_crop_roi` | `True` | Automatically crop volume to tight bounding box of mask ROI |
| `--no_numba` | `False` | Disable Numba JIT parallel acceleration (uses pure NumPy) |
| `--workers` | `-1` | Parallel CPU worker threads for FFT/DCT (-1 for all cores) |
| `--smooth_sigma` | `0.5` | Gaussian smoothing standard deviation for noise reduction |
| `--time_start` | `0` | Starting time frame index |
| `--time_end` | `None` | Ending time frame index |
| `--time_stride` | `1` | Time frame stride step |
| `--downsample` | `1` | Spatial downsampling factor |
| `--slice_axis` | `axial` | 2D visualization plane (`axial`, `sagittal`, `coronal`) |
| `--slice_index` | `None` | Specific slice index (auto-detects center of mass if None) |
| `--quiver_subsample` | `4` | Step interval for 2D velocity vector quiver arrows |
| `--save_nifti` | `True` | Export density and velocity fields as NIfTI volumes |
| `--synthetic` | `False` | Run self-contained synthetic 3D test simulation |


---

## 6. Output Files

For an input dataset with prefix `sub32521_s1`, the output directory contains:

- **NumPy Output Arrays**:
  - `sub32521_s1_density.npy`: Dynamic density volume sequence, shape $(N_t, N_x, N_y, N_z)$.
  - `sub32521_s1_velocity.npy`: 3D velocity vector fields, shape $(N_t-1, 3, N_x, N_y, N_z)$.
  - `sub32521_s1_source.npy`: Relative source rates, shape $(N_t-1, N_x, N_y, N_z)$.
  - `sub32521_s1_uromt_results.npz`: Compressed archive containing density, velocity, source, affine, voxel size, and all metrics.
- **NIfTI Volumes (Preserving Coordinate Affine for Medical Viewers)**:
  - `sub32521_s1_density.nii`: 4D density sequence $(N_x, N_y, N_z, N_t)$.
  - `sub32521_s1_speed.nii`: 4D flow speed volume $(N_x, N_y, N_z, N_t-1)$.
  - `sub32521_s1_source.nii`: 4D relative source rate $(N_x, N_y, N_z, N_t-1)$.
- **Diagnostic Visualizations**:
  - `sub32521_s1_density_velocity_slices.png`: 2D multi-slice panels showing density evolution, speed heatmaps with overlaid velocity vector quiver arrows, and source influx/efflux maps.
  - `sub32521_s1_kinetics_summary.png`: Time-series curves of total mass, mean speed, kinetic energy, and influx/efflux rates.

---

## 7. Python API Example

```python
import numpy as np
from uromt_io import load_nifti, save_numpy_results, plot_density_and_velocity_slices
from uromt_core import run_uromt_sequence

# 1. Load data
signal_4d, meta = load_nifti("data/sub32521_s1_signal.nii")
mask_3d, _ = load_nifti("data/sub32521_s1_mask.nii")

# 2. Run urOMT solver
results = run_uromt_sequence(
    signal_4d=signal_4d[..., :6],  # First 6 frames
    mask_3d=mask_3d,
    sigma=0.1,
    alpha=1.0,
    beta=100.0,
    n_substeps=3,
    max_iter=30
)

# 3. Save results
save_numpy_results(
    output_dir="results/api_example",
    prefix="sub32521_s1",
    density=results['density'],
    velocity=results['velocity'],
    source=results['source'],
    metrics=results['metrics']
)

# 4. Generate 2D slice visualization
fig = plot_density_and_velocity_slices(
    density=results['density'],
    velocity=results['velocity'],
    source=results['source'],
    mask=mask_3d,
    slice_axis="axial",
    save_path="results/api_example/slices.png"
)
```

---

## 8. Replotting & Post-Processing (`replot_uromt.py`)

You can replot, rescale velocity vectors, and perform spatiotemporal averaging on existing results:

```bash
# 1. Replot with 15x velocity vector scaling:
python replot_uromt.py results/sub33394_s1 --scale_quiver 15.0

# 2. Time-averaged velocity field (all frames):
python replot_uromt.py results/sub33394_s1 --time_avg --scale_quiver 20.0 --output_dir replots/time_avg

# 3. Time-averaged over specific window (frames 15-45) with spatial Gaussian smoothing:
python replot_uromt.py results/sub33394_s1 --time_avg --time_start 15 --time_end 45 --spatial_smooth_sigma 1.5 --scale_quiver 20.0

# 4. Spatial smoothing on dynamic sequence (along coronal axis):
python replot_uromt.py results/sub33394_s1 --spatial_smooth_sigma 1.0 --scale_quiver 10.0 --slice_axis coronal
```

---

## 9. Multi-Session Comparison & Hypothesis Testing (`compare_session_speeds.py`)

Compare fluid transport speeds between baseline Session 1 (`s1`) and both Session 2 (`s2`) and Session 3 (`s3`) across subjects with 3-session speed histograms, longitudinal trajectories, and paired $t$-tests:

```bash
# 1. Run multi-session comparison with 3D Brain Tissue Masking (S1 vs. S2 and S1 vs. S3):
python compare_session_speeds.py --use_mask

# 2. Run unmasked across all voxels (uniform x-axis upper bound: 0.5):
python compare_session_speeds.py

# 3. Specify sessions explicitly and save to custom folder:
python compare_session_speeds.py --use_mask --sessions s1 s2 s3 --ref_session s1 --output_dir results/session_comparison_3sessions
```

---

## 10. Spatial Statistical Parametric Mapping & Hotspots (`spatial_speed_change_analysis.py`)

Identify 3D anatomical regions and hotspots with statistically significant fluid transport speed changes between Session 1 and Session 2 across the cohort:

```bash
# Run voxelwise paired t-test across all subjects and generate 3D statistical maps:
python spatial_speed_change_analysis.py

# Specify significance threshold and minimum cluster size:
python spatial_speed_change_analysis.py --p_threshold 0.05 --min_cluster_size 10 --output_dir results/spatial_hotspots
```

---

## 11. Dynamic 3D Continuous Density & Velocity Movie Generator (`animate_uromt.py`)

Generate continuous, single-perspective 3D volumetric dynamic movies (`.mp4` / `.gif`) and static publication snapshots (`.png`) displaying tracer density $\rho(t)$ evolving over time as smooth nested 3D isosurface shells with progressive transparency, multi-level color-coding, and optional **dynamic, spatially smoothed 3D velocity vectors**:

```bash
# 1. Generate Continuous 3D Density MP4 movie (Default, fast):
python animate_uromt.py --subject 33394 --session s1

# 2. Add dynamic, spatially smoothed 3D velocity vectors (sigma = 1.5 voxels, arrow scale 15.0):
python animate_uromt.py --subject 33394 --session s1 --show_velocity --spatial_smooth_sigma 1.5 --scale_quiver 15.0

# 3. Add 3D velocity vectors with custom quiver subsampling and color:
python animate_uromt.py --subject 33394 --session s1 --show_velocity --scale_quiver 12.0 --quiver_subsample 3 --quiver_color cyan

# 4. Display velocity vectors across the entire brain volume (disable density wavefront masking):
python animate_uromt.py --subject 33394 --session s1 --show_velocity --no_quiver_density_mask --scale_quiver 12.0

# 5. Generate with Plasma colormap and save static 3D PNG snapshot:
python animate_uromt.py --subject 33394 --session s1 --cmap plasma --save_static_image

# 6. Customize 3D camera perspective (e.g. elevation 30 deg, azimuth -45 deg):
python animate_uromt.py --subject 33394 --session s1 --elevation 30 --azimuth -45

# 7. Specify direct results folder, frame rate, and custom output path:
python animate_uromt.py --results_dir results/sub32521_s2 -o movies/sub32521_s2_density_continuous_3d.mp4 --fps 8

# 8. Generate high-FPS animated GIF:
python animate_uromt.py --subject 33633 --session s3 --format gif --fps 8
```

---

## 12. References
- Chen, X., Benveniste, H., & Tannenbaum, A. R. (2024). *Unbalanced regularized optimal mass transport with applications to fluid flows in the brain*. **Scientific Reports**, 14(1), 1111. [DOI: 10.1038/s41598-023-50874-y](https://doi.org/10.1038/s41598-023-50874-y).
- Benamou, J. D., & Brenier, Y. (2000). *A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem*. **Numerische Mathematik**, 84(3), 375-393.
