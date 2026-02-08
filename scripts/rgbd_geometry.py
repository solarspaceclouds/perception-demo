"""
Demo 2 — RGB-D Geometry: Depth Back-Projection & Point Clouds

Demonstrates:
  1. Loading RGB, depth, and mask data.
  2. Aligning depth/mask to the RGB resolution.
  3. Back-projecting masked depth to a 3D point cloud in camera frame.
  4. Computing the 3D centroid.
  5. Scaling from relative depth to metric units using a known object height.
  6. Visualising the point cloud and centroid.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    build_intrinsics,
    load_rgb,
    load_depth,
    load_mask,
    resize_to_rgb,
    depth_mask_to_points,
    compute_centroid,
    scale_to_metric,
)

# ---------------------------------------------------------------------------
# Path constants (relative to project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# %% [markdown]
# # Step 1 — Load RGB, depth, and mask

# %%
rgb = load_rgb(os.path.join(DATA_DIR, "plant.jpeg"))
depth = load_depth(os.path.join(DATA_DIR, "depth.npy"))
mask = load_mask(os.path.join(DATA_DIR, "mask.png"))

print(f"RGB:   {rgb.shape}  depth: {depth.shape}  mask: {mask.shape}")

# %% [markdown]
# # Step 2 — Resize depth & mask to match RGB

# %%
depth_resized, mask_resized = resize_to_rgb(rgb, depth, mask)
H, W = rgb.shape[:2]

assert depth_resized.shape == (H, W)
assert mask_resized.shape == (H, W)

# %%
# Sanity check — edges of depth should roughly match the RGB
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_resized, cmap="inferno")
plt.title("Depth (aligned)")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Step 3 — Define camera intrinsics
#
# Using a reasonable default focal length.  For real sensors replace with
# calibrated values; for monocular depth this is approximate.

# %%
fx = fy = 500.0
cx, cy = W / 2.0, H / 2.0
K = build_intrinsics(fx, fy, cx, cy)

print("Intrinsic matrix K:\n", K)

# %% [markdown]
# # Step 4 — Back-project depth → 3D points (camera frame)
#
# Camera model (OpenCV convention):
# - +X → right,  +Y → down,  +Z → forward
#
# $$X = (u - c_x) \cdot Z / f_x, \quad Y = (v - c_y) \cdot Z / f_y$$

# %%
pts = depth_mask_to_points(depth_resized, mask_resized, K)

print(f"Point cloud: {pts.shape[0]} points")
assert pts.shape[1] == 3
assert np.all(np.isfinite(pts))
assert pts.shape[0] > 100, "Mask too small or misaligned"

# %%
# Quick 3D visualisation
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.3)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Raw point cloud (relative depth)")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Step 5 — Compute 3D centroid

# %%
centroid = compute_centroid(pts)
print("3D centroid (camera frame):", centroid)

assert centroid[2] > 0, "Object behind camera"
assert np.std(pts[:, 2]) > 1e-3, "Depth collapsed"

# %%
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.3, label="points")
ax.scatter(*centroid, c="r", s=100, label="centroid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("Point cloud with centroid")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Step 6 — Scale to metric units
#
# Monocular depth is *relative* (unitless).  We recover metric scale using a
# known real-world height of the plant.

# %%
REAL_HEIGHT_M = 0.25  # plant is ~25 cm tall

pts_metric, scale = scale_to_metric(pts, REAL_HEIGHT_M)
centroid_metric = centroid * scale

print(f"Scale factor: {scale:.4f}")
print(f"Metric extents: {pts_metric.max(0) - pts_metric.min(0)}")
print(f"Centroid (m):   {centroid_metric}")

# Validate metric depth range
z_mean = centroid_metric[2]
assert 0.2 < z_mean < 5.0, f"Depth scale likely wrong (z_mean={z_mean:.3f})"

# %%
print(f"Z min:  {pts_metric[:, 2].min():.4f} m")
print(f"Z mean: {pts_metric[:, 2].mean():.4f} m")
print(f"Z max:  {pts_metric[:, 2].max():.4f} m")

