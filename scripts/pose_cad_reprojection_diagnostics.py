"""
Demo 4 — Pose Representation, CAD Reprojection 

Camera frame: OpenCV convention:
 +X right, +Y down, +Z forward

Demonstrates:
  1. Pose representation: 4×4 SE(3) matrix, R+t dict, JSON export/import.
  2. Loading a CAD mesh and transforming it into the camera frame.
  3. Projecting CAD vertices onto an RGB image.
  4. Constructing a look-at camera pointing at the object.
  5. Rebuilding the full pipeline (depth→mesh→reproject) with proper
     metric scaling and sanity checks.
"""

# %%
import os
import numpy as np
import cv2
import json
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from utils import (
    make_T,
    invert_T,
    draw_frame,
    build_intrinsics,
    intrinsics_from_fov,
    project_points,
    look_at_opencv,
    load_rgb,
    load_depth,
    load_mask,
    resize_to_rgb,
    depth_mask_to_points,
    scale_to_metric,
    load_mesh_vertices,
    save_pose_json,
    load_pose_json,
)

# ---------------------------------------------------------------------------
# Path constants (relative to project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
MESH_DIR = os.path.join(_PROJECT_ROOT, "meshes")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

# =====================================================================
# PART A — Frame transforms & pose representation
# =====================================================================

# %% [markdown]
# # Part A — Pose representation & JSON round-trip

# %%
# Define poses in world frame
t_world_cam = np.array([0.0, 0.0, 1.5])
R_world_cam = Rotation.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()

t_world_obj = np.array([0.3, 0.2, 0.0])
R_world_obj = Rotation.from_euler("z", 45, degrees=True).as_matrix()

# Build SE(3) and compute camera-frame pose
T_world_cam = make_T(R_world_cam, t_world_cam)
T_world_obj = make_T(R_world_obj, t_world_obj)

T_cam_obj = np.linalg.inv(T_world_cam) @ T_world_obj
R_cam_obj = T_cam_obj[:3, :3]
t_cam_obj = T_cam_obj[:3, 3]

print("Object pose in camera frame:")
print(f"  R:\n{R_cam_obj}")
print(f"  t: {t_cam_obj}")

# %% [markdown]
# ## Visualise world & camera frames

# %%
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

draw_frame(ax, np.eye(3), np.zeros(3), "World", scale=0.2)
draw_frame(ax, R_world_cam, t_world_cam, "Camera", scale=0.2)
draw_frame(ax, R_world_obj, t_world_obj, "Object", scale=0.2)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("All frames in world coordinates")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Export & re-import pose as JSON

# %%
save_pose_json(T_cam_obj, os.path.join(OUTPUT_DIR, "pose_cam_obj.json"))
T_cam_obj_loaded = load_pose_json(os.path.join(OUTPUT_DIR, "pose_cam_obj.json"))

assert T_cam_obj_loaded.shape == (4, 4)
assert np.allclose(T_cam_obj, T_cam_obj_loaded), "Round-trip mismatch!"
print("Pose JSON round-trip OK ✓")

# =====================================================================
# PART B — CAD model loading & reprojection
# =====================================================================

# %% [markdown]
# # Part B — Load CAD model & reproject onto RGB
#
# `plant_metric_mesh.obj` was created by `plant_pose_pipeline.py`, whose
# vertices live in the **metric camera frame** (back-projected with
# FOV-based intrinsics, then scaled to metres).  To reproject correctly
# we must use the **same K** and leave the vertices in camera frame
# (no centering, no arbitrary Z shift).

# %%
rgb = load_rgb(os.path.join(DATA_DIR, "plant.jpeg"))
H_img, W_img = rgb.shape[:2]

# Same FOV-based intrinsics used when the mesh was created
FOV_DEG = 60.0
K_b = intrinsics_from_fov(W_img, H_img, FOV_DEG)
print(f"Intrinsics: fx={K_b[0,0]:.1f}  fy={K_b[1,1]:.1f}  "
      f"cx={K_b[0,2]:.1f}  cy={K_b[1,2]:.1f}")

# Load CAD mesh — vertices are already in metric camera frame
OBJ_PATH = os.path.join(MESH_DIR, "plant_metric_mesh.obj")
cad_cam = load_mesh_vertices(OBJ_PATH)

print(f"CAD extents (m): {cad_cam.ptp(axis=0)}")
print(f"CAD centroid (m): {cad_cam.mean(axis=0)}")
print(f"CAD Z range in camera: [{cad_cam[:, 2].min():.3f}, {cad_cam[:, 2].max():.3f}]")

assert cad_cam[:, 2].min() > 0, "CAD mesh has points behind camera"

# %% [markdown]
# ## Project CAD vertices onto RGB

# %%
uv = project_points(cad_cam, K_b)

valid = (
    (uv[:, 0] >= 0) & (uv[:, 0] < W_img) &
    (uv[:, 1] >= 0) & (uv[:, 1] < H_img)
)
uv_valid = uv[valid]

plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.scatter(uv_valid[:, 0], uv_valid[:, 1], s=1, c="red", alpha=0.5)
plt.title("CAD reprojection onto RGB")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Pure geometric reprojection (black canvas)

# %%
canvas = np.zeros((H_img, W_img, 3), dtype=np.uint8)
uv_int = uv.astype(int)
valid_int = (
    (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W_img)
    & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H_img)
)
canvas[uv_int[valid_int, 1], uv_int[valid_int, 0]] = [255, 0, 0]

plt.figure(figsize=(8, 8))
plt.imshow(canvas)
plt.title("Pure geometric reprojection (no RGB)")
plt.axis("off")
plt.tight_layout()
plt.show()

# =====================================================================
# PART C — Full depth→mesh→reproject pipeline with metric scale
# =====================================================================

# %% [markdown]
# # Part C — Depth segmentation & mesh reconstruction

# %%
rgb = load_rgb(os.path.join(DATA_DIR, "plant.jpeg"))
depth = load_depth(os.path.join(DATA_DIR, "depth.npy"))
mask = load_mask(os.path.join(DATA_DIR, "mask.png"))

H, W = rgb.shape[:2]
depth_resized, mask_resized = resize_to_rgb(rgb, depth, mask)

# %%
# Masked depth map
depth_plant = depth_resized.copy()
depth_plant[~mask_resized] = 0.0
np.save(os.path.join(OUTPUT_DIR, "depth_plant.npy"), depth_plant)

depth_plant_values = depth_resized[mask_resized]
np.save(os.path.join(OUTPUT_DIR, "depth_plant_values.npy"), depth_plant_values)

print(f"RGB: {rgb.shape}  Depth: {depth_resized.shape}  Mask: {mask_resized.shape}")
assert np.count_nonzero(depth_plant) > 500, "Plant mask too small"

# %%
plt.figure(figsize=(6, 6))
plt.imshow(depth_plant, cmap="inferno")
plt.title("Segmented Plant Depth")
plt.colorbar()
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Back-project, scale, and reconstruct mesh
#
# Uses the same approach as plant_pose_pipeline.py:
#   1. FOV-based intrinsics  →  depth_mask_to_points  →  scale_to_metric
#   2. Same K is used for both back-projection **and** reprojection,
#      so the pixel ↔ 3-D round-trip is exact (metric scale cancels in X/Z, Y/Z).

# %%
# Camera intrinsics (FOV-based, matching plant_pose_pipeline.py)
FOV_DEG = 60.0
K = intrinsics_from_fov(W, H, FOV_DEG)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"Intrinsics: fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

# Back-project masked depth → 3D points (standard OpenCV convention)
points_cam = depth_mask_to_points(depth_resized, mask_resized, K)
colors_valid = rgb[mask_resized] / 255.0

print(f"Point cloud: {points_cam.shape[0]} points")
assert points_cam.shape[0] > 100, "Too few points — check mask"
assert np.all(points_cam[:, 2] > 0), "Points behind camera"

# Scale to metric (same as rgbd_geometry.py / plant_pose_pipeline.py)
REAL_HEIGHT_M = 0.25
pts_metric, metric_scale = scale_to_metric(points_cam, REAL_HEIGHT_M)
print(f"Scale factor: {metric_scale:.4f}")
print(f"Metric extents: {pts_metric.max(0) - pts_metric.min(0)}")

# %%
# Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_metric)
pcd.colors = o3d.utility.Vector3dVector(colors_valid)

# Estimate & orient normals (required for Ball Pivoting)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(k=30)

# Surface reconstruction (Ball Pivoting — same as plant_pose_pipeline.py)
# Unlike Poisson, Ball Pivoting does NOT create a closed watertight surface,
# so it won't inflate the mesh into a sphere around the data.
radii = [0.005, 0.01, 0.02]
mesh_recon = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)
mesh_recon.remove_unreferenced_vertices()

# --- mesh in camera frame ---
verts_cam = np.asarray(mesh_recon.vertices).copy()

print("Mesh Z range:", verts_cam[:, 2].min(), verts_cam[:, 2].max())

# save camera-frame mesh
o3d.io.write_triangle_mesh(os.path.join(MESH_DIR, "plant_cam.obj"), mesh_recon)

# --- object-centered mesh ---
center = verts_cam.mean(axis=0)
verts_obj = verts_cam - center

mesh_recon.vertices = o3d.utility.Vector3dVector(verts_obj)
o3d.io.write_triangle_mesh(os.path.join(MESH_DIR, "plant_centered.obj"), mesh_recon)

print("Saved meshes/plant_cam.obj and meshes/plant_centered.obj")

extent_m = np.ptp(np.asarray(mesh_recon.vertices), axis=0)
print(f"Plant extent (m): {extent_m}")

# %% [markdown]
# ## Reproject mesh onto RGB

# %%
# Mesh vertices are in metric camera frame (same coord system we back-projected into).
# Reprojecting with the SAME K recovers the original pixel locations because
# metric scaling cancels:  u = fx * X_metric / Z_metric + cx  =  fx * X/Z + cx  =  x_pixel.
mesh_cam = verts_cam

assert mesh_cam[:, 2].min() > 0, "Mesh behind camera!"

uv_mesh = project_points(mesh_cam, K)

u = uv_mesh[:, 0].astype(int)
v = uv_mesh[:, 1].astype(int)

valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

overlay = rgb.copy()
overlay[v[valid], u[valid]] = [255, 0, 0]

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title("Reprojected plant mesh overlay")
plt.axis("off")
plt.tight_layout()
plt.show()

# Diagnostics
print("Mesh extent (m):", mesh_cam.ptp(axis=0))
print("Mesh Z range:", mesh_cam[:, 2].min(), mesh_cam[:, 2].max())
print("UV range:", uv_mesh.min(axis=0), uv_mesh.max(axis=0))

centroid_uv = project_points(mesh_cam.mean(axis=0, keepdims=True), K)
print("Projected centroid:", centroid_uv)
