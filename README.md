# Intrinsic Perception Demo

A hands-on tutorial exploring the geometry behind **6-DoF object pose estimation** from RGB-D data. 

Five progressive demos walk through SE(3) transformations, depth back-projection, mesh reconstruction, CAD reprojection, and common spatial failure modes, all grounded in a single plant scene.

## Overview

| Demo | Script | What it covers |
|------|--------|----------------|
| **1. Frame Transforms** | `frames.py` | SE(3) homogeneous matrices, world ↔ camera frame conversions, 3D frame visualisation |
| **2. RGB-D Geometry** | `rgbd_geometry.py` | Depth back-projection, point cloud generation, 3D centroid computation, relative → metric scaling |
| **3. Plant Pose Pipeline** | `plant_pose_pipeline.py` | End-to-end: RGB + depth + mask → metric point cloud → surface mesh (Ball Pivoting) → reprojection overlay |
| **4. Pose & CAD Reprojection** | `pose_cad_reprojection_diagnostics.py` | Pose representation (SE(3), R+t, JSON), CAD mesh loading & reprojection, look-at camera, full pipeline rebuild with diagnostics |
| **5. Failure Analysis** | `failure_analysis.py` | Five spatial failure modes: symmetry flips, depth ambiguity, occlusion drift, scale inconsistency, multi-view noise, with quantitative error analysis and diagnostic figures |

Each script can be run as a standalone `.py` file. Corresponding Jupyter notebooks are also provided in `notebooks/`.

## Project Structure

```
intrinsic-perception-demo/
├── data/                       # Raw input data
│   ├── plant.jpeg              # RGB image of a potted plant
│   ├── depth.npy               # Monocular relative depth map (NumPy)
│   ├── depth.png               # Depth map visualisation
│   ├── mask.npy                # Segmentation mask (NumPy)
│   └── mask.png                # Segmentation mask (PNG)
├── meshes/                     # 3D mesh files (input & generated)
│   ├── plant.obj               # Original mesh
│   ├── plant_metric_mesh.obj   # Metric-scale mesh (from Demo 3)
│   ├── plant_cam.obj           # Mesh in camera frame (from Demo 4)
│   └── plant_centered.obj      # Object-centred mesh (from Demo 4)
├── scripts/                    # Python demo scripts
│   ├── utils.py                # Shared utilities (SE(3), projection, I/O, visualisation)
│   ├── frames.py               # Demo 1 — Coordinate frame transforms
│   ├── rgbd_geometry.py        # Demo 2 — Depth back-projection & point clouds
│   ├── plant_pose_pipeline.py  # Demo 3 — RGB → metric mesh → reprojection
│   ├── pose_cad_reprojection_diagnostics.py  # Demo 4 — Pose representation & CAD
│   └── failure_analysis.py     # Demo 5 — Failure modes & diagnosis
├── notebooks/                  # Jupyter notebook versions of each demo
│   ├── frames.ipynb
│   ├── rgb-d_geometry.ipynb
│   ├── plant_pose_pipeline.ipynb
│   └── pose_representation_and_failure.ipynb
├── output/                     # Generated outputs
│   ├── failure_figs/           # Diagnostic figures from Demo 5 (17 PNGs)
│   ├── pose_cam_obj.json       # Exported SE(3) pose
│   ├── depth_plant.npy         # Segmented depth map
│   └── depth_plant_values.npy  # Masked depth values
├── models/                     # Pre-trained model weights
│   └── sam_vit_b_01ec64.pth    # SAM ViT-B checkpoint (for segmentation)
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- A CUDA-capable GPU is **not** required (all demos run on CPU)

### Installation

```bash
git clone <repo-url>
cd intrinsic-perception-demo

# Create and activate a python or conda environment, install dependencies
pip install -r requirements.txt
# or 
conda env create -f environment.yml 
# replace prefix with appropriate path
```

The core dependencies are:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, linear algebra |
| `scipy` | Rotation utilities (`spatial.transform.Rotation`) |
| `opencv-python` | Image I/O, resizing |
| `matplotlib` | 2D/3D plotting and visualisation |
| `open3d` | Point cloud & mesh processing (Demos 3–4) |
| `trimesh` | Mesh loading (Demo 4) |

### Running the Demos

All scripts are in `scripts/` and resolve data paths automatically relative to the project root, so they work from any working directory.

```bash
# Run any demo directly
python scripts/frames.py
python scripts/rgbd_geometry.py
python scripts/plant_pose_pipeline.py
python scripts/pose_cad_reprojection_diagnostics.py
python scripts/failure_analysis.py
```

The Jupyter notebooks in `notebooks/` provide the same content in notebook form (To be updated).

## Demo Details

### Demo 1: Coordinate Frame Transformations

Introduces SE(3) homogeneous transformation matrices. Defines camera and object poses in the world frame, computes the object-to-camera transform `T_{cam_obj} = T_{cam_world} · T_{world_obj}`, and visualises all three coordinate frames in 3D.

### Demo 2: RGB-D Geometry

Loads an RGB image, monocular depth map, and segmentation mask. Aligns them to a common resolution, back-projects masked depth into a 3D point cloud using pinhole camera intrinsics, computes the 3D centroid, and scales from relative to metric depth using a known object height (25 cm plant).

### Demo 3: Plant Pose Pipeline

End-to-end pipeline: loads RGB-D + mask → back-projects to metric 3D → builds an Open3D point cloud with colours → estimates normals → reconstructs a triangular surface mesh via Ball Pivoting → saves as OBJ → reprojects mesh vertices onto the original RGB image to verify alignment.

### Demo 4: Pose Representation & CAD Reprojection

Covers pose representations (4×4 SE(3) matrix, R+t dictionary, JSON serialisation). Loads the CAD mesh from Demo 3 and reprojects it onto the RGB image. Rebuilds the full depth → mesh → reprojection pipeline with detailed diagnostic outputs and sanity checks.

### Demo 5: Failure Analysis

Analyses five spatial failure modes common in pose estimation:

1. **Symmetry flips**: 180° rotational ambiguity produces near-identical 2D reprojections but large geodesic rotation errors
2. **Depth ambiguity**: monocular scale–translation coupling: an object at depth Z with extent S projects identically to one at 2Z with extent 2S
3. **Occlusion drift**: partial visibility shifts the visible centroid away from the true centroid, biasing both R and t
4. **Scale inconsistency**: wrong metric prior (assumed object height) causes proportional translation error
5. **Multi-view noise**: sensor noise in depth maps produces inconsistent per-view poses

Each failure mode is illustrated with distinct visualisation types (20+ figures saved to `output/failure_figs/`), and a summary figure maps each failure to its unique error signature in (R, t) space.

## Shared Utilities (`scripts/utils.py`)

Key helper functions used across all demos:

| Function | Description |
|----------|-------------|
| `make_T(R, t)` | Build a 4×4 SE(3) matrix from rotation and translation |
| `invert_T(T)` | Fast SE(3) inverse (exploits rigid-body structure) |
| `build_intrinsics(fx, fy, cx, cy)` | Construct a 3×3 camera intrinsic matrix K |
| `intrinsics_from_fov(W, H, fov)` | Derive K from horizontal field-of-view |
| `project_points(pts, K)` | Project 3D points → 2D pixel coordinates |
| `depth_mask_to_points(depth, mask, K)` | Back-project masked depth to 3D point cloud |
| `scale_to_metric(pts, height)` | Scale relative-depth points to metric units |
| `load_rgb / load_depth / load_mask` | Data loading helpers |
| `load_mesh_vertices / load_mesh_vertices_o3d` | Mesh loading (trimesh / Open3D) |
| `save_pose_json / load_pose_json` | Pose I/O in JSON format |
| `draw_frame(ax, R, t, label)` | Draw a 3D coordinate frame on matplotlib axes |

## Camera Convention

All demos follow the **OpenCV camera convention**:
- **+X** → right
- **+Y** → down
- **+Z** → forward (into the scene)

## License

This project is provided for educational and demonstration purposes.

