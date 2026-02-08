"""
Demo 3 — Plant Pose Pipeline: RGB → Metric Mesh → Reprojection

End-to-end pipeline that:
  1. Loads RGB, relative depth, and segmentation mask.
  2. Resizes depth & mask to match RGB resolution.
  3. Back-projects the masked depth to a 3D point cloud in camera frame.
  4. Scales points to metric units using a known plant height.
  5. Builds an Open3D point cloud with RGB colours.
  6. Estimates normals and creates a triangular mesh (Ball Pivoting).
  7. Saves the mesh as OBJ in metric scale.
  8. Reprojects mesh vertices onto the RGB image to verify alignment.
"""

# %%
import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from utils import (
    load_rgb,
    load_depth,
    load_mask,
    resize_to_rgb,
    intrinsics_from_fov,
    depth_mask_to_points,
    scale_to_metric,
    project_points,
)

# ---------------------------------------------------------------------------
# Path constants (relative to project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
MESH_DIR = os.path.join(_PROJECT_ROOT, "meshes")

# %% [markdown]
# # Step 1 — Load and align data

# %%
rgb = load_rgb(os.path.join(DATA_DIR, "plant.jpeg"))
depth = load_depth(os.path.join(DATA_DIR, "depth.npy"))
mask = load_mask(os.path.join(DATA_DIR, "mask.png"))

H, W = rgb.shape[:2]
depth_resized, mask_resized = resize_to_rgb(rgb, depth, mask)

print(f"RGB: {rgb.shape}  Depth: {depth_resized.shape}  Mask: {mask_resized.shape}")

# %% [markdown]
# # Step 2 — Camera intrinsics (estimated from FOV)

# %%
FOV_DEG = 60.0
K = intrinsics_from_fov(W, H, FOV_DEG)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print(f"fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

# %% [markdown]
# # Step 3 — Back-project depth → 3D & scale to metric

# %%
pts = depth_mask_to_points(depth_resized, mask_resized, K)

REAL_HEIGHT_M = 0.25  # known plant height in metres
pts_metric, scale = scale_to_metric(pts, REAL_HEIGHT_M)

print(f"Points: {pts_metric.shape[0]},  scale factor: {scale:.4f}")
print(f"Metric extents: {pts_metric.max(0) - pts_metric.min(0)}")

# %% [markdown]
# # Step 4 — Open3D point cloud with colours

# %%
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_metric)
colors = rgb[mask_resized] / 255.0
pcd.colors = o3d.utility.Vector3dVector(colors)

# Estimate normals (required for meshing)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(k=30)

# %% [markdown]
# # Step 5 — Surface mesh via Ball Pivoting

# %%
radii = [0.005, 0.01, 0.02]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)
mesh.remove_unreferenced_vertices()

MESH_OUT = os.path.join(MESH_DIR, "plant_metric_mesh.obj")
o3d.io.write_triangle_mesh(MESH_OUT, mesh)
print(f"Saved {MESH_OUT}")

# %% [markdown]
# # Step 6 — Reproject mesh onto RGB image

# %%
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

uv = project_points(vertices, K)
u_px, v_px = uv[:, 0], uv[:, 1]

valid = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

# Build valid triangle list for filled overlay
tri_valid = []
valid_indices = np.where(valid)[0]
valid_set = set(valid_indices.tolist())
index_map = {old: new for new, old in enumerate(valid_indices)}

for tri in triangles:
    if all(v in valid_set for v in tri):
        tri_valid.append([index_map[v] for v in tri])
tri_valid = np.array(tri_valid) if len(tri_valid) > 0 else np.empty((0, 3), dtype=int)

# %%
plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.axis("off")

if len(tri_valid) > 0:
    tri_obj = Triangulation(u_px[valid], v_px[valid], triangles=tri_valid)
    plt.tripcolor(
        tri_obj,
        facecolors=np.ones(len(tri_obj.triangles)),
        edgecolors="r",
        alpha=0.4,
    )

plt.title("Plant mesh reprojected onto RGB")
plt.tight_layout()
plt.show()
