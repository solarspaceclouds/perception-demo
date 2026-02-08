"""
Demo 5 — Spatial Failure Modes & Pose-Based Failure Diagnosis

Demonstrates five common failure modes in 6-DoF pose estimation from
RGB-D data, and shows how examining the SE(3) pose matrix exposes each
failure quantitatively.

Failure modes analysed (all spatial):
  1. Pose flips under symmetry    — 180° rotational ambiguity
  2. Depth ambiguity              — monocular scale–translation coupling
  3. Occlusion-induced translation drift — centroid shift from partial views
  4. Scale inconsistency          — wrong metric prior ⇒ wrong 3D pose
  5. Multi-view disagreement      — noisy depth ⇒ inconsistent poses

How pose exposes each failure:
  - Rotational flips    → large geodesic rotation error, near-zero 2D error
  - Translation drift   → centroid shift visible in SE(3) t-vector
  - Symmetry collapse   → multiple SE(3) solutions with similar reprojections
  - Occlusion hallucination → biased centroid, orientation jitter

Each section uses **different** visualisation types to make the specific
failure mode visually obvious:
  1. Dual-colour 2D overlay + 3D rotated point clouds
  2. XZ side-view depth cross-section + bar chart
  3. Coloured mask regions + centroid drift arrows
  4. Nested 3D bounding boxes + metric bar chart
  5. Error-vs-noise line plots + centroid scatter cloud

All figures are saved as PNG files in the `output/failure_figs/` directory.

Prerequisites:
  - data/plant.jpeg, data/depth.npy, data/mask.png (from the demo dataset)
  - utils.py (shared helpers)
"""

# %%
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from utils import (
    make_T,
    draw_frame,
    intrinsics_from_fov,
    project_points,
    load_rgb,
    load_depth,
    load_mask,
    resize_to_rgb,
    depth_mask_to_points,
    compute_centroid,
    scale_to_metric,
    set_equal_axes,
)

# ---------------------------------------------------------------------------
# Path constants (relative to project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

# Create output directory for figures
FIG_DIR = os.path.join(OUTPUT_DIR, "failure_figs")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name):
    """Save and show a figure."""
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  → Saved {FIG_DIR}/{name}")


# =====================================================================
# Helpers
# =====================================================================

def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic (angle-axis) rotation error between two rotation matrices, in degrees."""
    cos_ang = (np.trace(R1.T @ R2) - 1.0) / 2.0
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def translation_error_m(t1: np.ndarray, t2: np.ndarray) -> float:
    """Euclidean translation error in metres."""
    return float(np.linalg.norm(t1 - t2))


def pca_orientation(pts: np.ndarray) -> np.ndarray:
    """
    Return a right-handed 3×3 rotation whose columns are the PCA axes
    (descending eigenvalue order).
    """
    pts_c = pts - pts.mean(axis=0)
    cov = (pts_c.T @ pts_c) / len(pts_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    R = eigvecs[:, ::-1]  # descending eigenvalue order
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def draw_bbox_3d(ax, pts, color="red", linestyle="-", linewidth=1.5, label=None):
    """Draw a 3D axis-aligned bounding box around a point cloud."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    # 12 edges of a box
    edges = [
        ([lo[0], lo[0]], [lo[1], lo[1]], [lo[2], hi[2]]),
        ([lo[0], lo[0]], [hi[1], hi[1]], [lo[2], hi[2]]),
        ([hi[0], hi[0]], [lo[1], lo[1]], [lo[2], hi[2]]),
        ([hi[0], hi[0]], [hi[1], hi[1]], [lo[2], hi[2]]),
        ([lo[0], hi[0]], [lo[1], lo[1]], [lo[2], lo[2]]),
        ([lo[0], hi[0]], [hi[1], hi[1]], [lo[2], lo[2]]),
        ([lo[0], hi[0]], [lo[1], lo[1]], [hi[2], hi[2]]),
        ([lo[0], hi[0]], [hi[1], hi[1]], [hi[2], hi[2]]),
        ([lo[0], lo[0]], [lo[1], hi[1]], [lo[2], lo[2]]),
        ([hi[0], hi[0]], [lo[1], hi[1]], [lo[2], lo[2]]),
        ([lo[0], lo[0]], [lo[1], hi[1]], [hi[2], hi[2]]),
        ([hi[0], hi[0]], [lo[1], hi[1]], [hi[2], hi[2]]),
    ]
    for i, (xs, ys, zs) in enumerate(edges):
        ax.plot(xs, ys, zs, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label if i == 0 else None)


# =====================================================================
# Load shared data (used by all sections)
# =====================================================================

# %% [markdown]
# # Setup — Load data & establish ground-truth pose

# %%
rgb = load_rgb(os.path.join(DATA_DIR, "plant.jpeg"))
depth = load_depth(os.path.join(DATA_DIR, "depth.npy"))
mask = load_mask(os.path.join(DATA_DIR, "mask.png"))

H, W = rgb.shape[:2]
depth_resized, mask_resized = resize_to_rgb(rgb, depth, mask)

FOV_DEG = 60.0
K = intrinsics_from_fov(W, H, FOV_DEG)

REAL_HEIGHT_M = 0.25
pts_cam = depth_mask_to_points(depth_resized, mask_resized, K)
pts_metric, metric_scale = scale_to_metric(pts_cam, REAL_HEIGHT_M)

# Ground-truth pose: centroid + PCA orientation
centroid_gt = compute_centroid(pts_metric)
R_gt = pca_orientation(pts_metric)
T_gt = make_T(R_gt, centroid_gt)

print(f"Image: {W}x{H},  Points: {pts_metric.shape[0]}")
print(f"GT centroid (m): [{centroid_gt[0]:.4f}, {centroid_gt[1]:.4f}, {centroid_gt[2]:.4f}]")
euler_gt = Rotation.from_matrix(R_gt).as_euler("xyz", degrees=True)
print(f"GT orientation (Euler XYZ deg): [{euler_gt[0]:.1f}, {euler_gt[1]:.1f}, {euler_gt[2]:.1f}]")
print(f"Metric scale factor: {metric_scale:.6f}")
print()

pts_obj = pts_metric - centroid_gt  # object-centred coordinates


# =====================================================================
# FAILURE 1 — Pose flips under symmetry
# =====================================================================

# %% [markdown]
# # Failure 1 — Pose Flips Under Symmetry
#
# A near-symmetric object admits multiple orientations that produce
# **similar** 2D reprojections.  The silhouette barely changes under
# a 180° flip, but the SE(3) rotation is 180° wrong.

# %%
print("=" * 72)
print("  FAILURE 1 — Pose Flips Under Symmetry")
print("=" * 72)

# Generate flipped versions
flip_angles = [0, 45, 90, 135, 180]
flip_data = []  # (label, R_flipped, pts_cam_flipped, rot_err)

for angle in flip_angles:
    R_sym = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    R_f = R_gt @ R_sym
    pts_f = (R_f @ pts_obj.T).T + centroid_gt
    err = rotation_error_deg(R_gt, R_f)
    flip_data.append((f"{angle}°", R_f, pts_f, err))

# --- Figure 1A: Top-down XZ view showing actual rotation clearly ---
# This view makes the angular difference between rotations visually obvious,
# unlike 2D reprojection where all angles look nearly identical.
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
flip_colors = ["lime", "gold", "orange", "orangered", "red"]

for ax, (label, R_f, pts_f, err), color in zip(axes, flip_data, flip_colors):
    # Object-centred points rotated to camera frame (subtract centroid for vis)
    pts_vis = pts_f - centroid_gt
    sub = pts_vis[::3]
    ax.scatter(sub[:, 0], sub[:, 2], s=0.3, alpha=0.3, color=color)
    # Show GT as faint reference in all panels
    sub_gt_vis = pts_obj[::3]
    ax.scatter(sub_gt_vis[:, 0], sub_gt_vis[:, 2], s=0.3, alpha=0.1,
               color="gray")
    # Draw coordinate frame arrows at origin
    frame_scale = 0.08
    R_rel = R_f  # orientation
    # X-axis of frame
    ax.annotate("", xy=(R_rel[0, 0]*frame_scale, R_rel[2, 0]*frame_scale),
                xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=2))
    # Z-axis of frame
    ax.annotate("", xy=(R_rel[0, 2]*frame_scale, R_rel[2, 2]*frame_scale),
                xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Z (m)", fontsize=9)
    title_color = "green" if err < 0.1 else "red"
    ax.set_title(f"Y-flip: {label}\nRot err: {err:.0f}°",
                 fontsize=11, color=title_color, fontweight="bold")
    ax.grid(alpha=0.2)

fig.suptitle("Failure 1A — Top-down XZ view: rotation is clear in 3D\n"
             "(colored = rotated cloud, gray = GT reference, "
             "arrows = frame axes)",
             fontsize=13, y=1.05)
plt.tight_layout()
savefig(fig, "f1a_symmetry_topdown_rotation.png")

# --- Figure 1B: All rotations in one 3D plot from two viewing angles ---
fig = plt.figure(figsize=(18, 7))
sub_gt = pts_metric[::5]

# View 1: front-side view
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(sub_gt[:, 0], sub_gt[:, 1], sub_gt[:, 2],
            s=0.5, alpha=0.15, c="lime", label="GT (0°)")

for (label, R_f, pts_f, err), color in zip(flip_data[1:], flip_colors[1:]):
    sub_f = pts_f[::8]
    ax1.scatter(sub_f[:, 0], sub_f[:, 1], sub_f[:, 2],
                s=0.5, alpha=0.1, c=color, label=f"{label} flip")
    draw_frame(ax1, R_f, centroid_gt, label, scale=0.05)

draw_frame(ax1, R_gt, centroid_gt, "GT", scale=0.05)
ax1.set_title("Front-side view — all rotations overlaid", fontsize=11)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend(fontsize=7, loc="upper left", markerscale=5)
ax1.view_init(elev=15, azim=-60)

# View 2: top-down view (the best view to see Y-axis rotation)
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(sub_gt[:, 0], sub_gt[:, 1], sub_gt[:, 2],
            s=0.5, alpha=0.15, c="lime", label="GT (0°)")

for (label, R_f, pts_f, err), color in zip(flip_data[1:], flip_colors[1:]):
    sub_f = pts_f[::8]
    ax2.scatter(sub_f[:, 0], sub_f[:, 1], sub_f[:, 2],
                s=0.5, alpha=0.1, c=color, label=f"{label} flip")
    draw_frame(ax2, R_f, centroid_gt, label, scale=0.05)

draw_frame(ax2, R_gt, centroid_gt, "GT", scale=0.05)
ax2.set_title("Top-down view — Y-rotation visible as spread in XZ",
              fontsize=11)
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.legend(fontsize=7, loc="upper left", markerscale=5)
ax2.view_init(elev=85, azim=-60)

fig.suptitle("Failure 1B — 3D point clouds: each rotation angle is\n"
             "clearly distinct in 3D (same centroid, different orientation)",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f1b_symmetry_3d_all_rotations.png")

# --- Figure 1C: Bar chart of rotation error ---
fig, ax = plt.subplots(figsize=(8, 4))
angles = [d[0] for d in flip_data]
errors = [d[3] for d in flip_data]
bars = ax.bar(angles, errors, color=["lime", "gold", "orange", "orangered", "red"])
ax.set_ylabel("Geodesic Rotation Error (°)", fontsize=12)
ax.set_xlabel("Y-axis Flip Angle", fontsize=12)
ax.set_title("Failure 1C — Rotation error grows with flip angle\n"
             "(but 2D reprojection stays similar)", fontsize=12)
for bar, err in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{err:.0f}°", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 200)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
savefig(fig, "f1c_symmetry_error_bar.png")

for label, _, _, err in flip_data:
    print(f"  {label:>4} flip  →  rotation error: {err:.1f}°")
print("  ⇒ Rotation flips are invisible in 2D but obvious in the SE(3) matrix.\n")


# =====================================================================
# FAILURE 2 — Depth ambiguity (scale–translation coupling)
# =====================================================================

# %% [markdown]
# # Failure 2 — Depth Ambiguity
#
# Monocular depth is scale-ambiguous: an object at depth Z with extent S
# projects identically to one at depth 2Z with extent 2S.
# 2D reprojection is pixel-perfect in all cases, but the 3D translation
# and metric size are completely wrong.

# %%
print("=" * 72)
print("  FAILURE 2 — Depth Ambiguity (Scale–Translation Coupling)")
print("=" * 72)

scale_factors = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
centroids_depth = []
extents_depth = []
for sf in scale_factors:
    pts_scaled = pts_metric * sf
    centroids_depth.append(compute_centroid(pts_scaled))
    extents_depth.append(np.ptp(pts_scaled, axis=0))

# --- Figure 2A: XZ side-view cross-section ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: XZ side view (bird's eye) showing depth separation
ax = axes[0]
colors_sf = plt.cm.coolwarm(np.linspace(0, 1, len(scale_factors)))
for i, sf in enumerate(scale_factors):
    pts_s = pts_metric[::10] * sf  # subsample for speed
    lbl = f"×{sf}" + (" (GT)" if sf == 1.0 else "")
    ax.scatter(pts_s[:, 0], pts_s[:, 2], s=0.3, alpha=0.4,
               color=colors_sf[i], label=lbl)
    c = centroids_depth[i]
    ax.plot(c[0], c[2], "^", color=colors_sf[i], markersize=10,
            markeredgecolor="black", markeredgewidth=0.5)

ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Z — Depth (m)", fontsize=12)
ax.set_title("XZ Side View — depth ambiguity\n"
             "Each scale factor places the object at a different depth",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

# Right: YZ side view
ax = axes[1]
for i, sf in enumerate(scale_factors):
    pts_s = pts_metric[::10] * sf
    lbl = f"×{sf}" + (" (GT)" if sf == 1.0 else "")
    ax.scatter(pts_s[:, 1], pts_s[:, 2], s=0.3, alpha=0.4,
               color=colors_sf[i], label=lbl)
    c = centroids_depth[i]
    ax.plot(c[1], c[2], "^", color=colors_sf[i], markersize=10,
            markeredgecolor="black", markeredgewidth=0.5)

ax.set_xlabel("Y (m)", fontsize=12)
ax.set_ylabel("Z — Depth (m)", fontsize=12)
ax.set_title("YZ Side View — depth ambiguity\n"
             "All scales produce IDENTICAL 2D projections",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

fig.suptitle("Failure 2A — Cross-section view reveals depth ambiguity "
             "(invisible in 2D)", fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f2a_depth_ambiguity_sideview.png")

# --- Figure 2B: Bar chart of Z and extent ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Centroid Z
ax = axes[0]
z_vals = [c[2] for c in centroids_depth]
colors_bar = ["lime" if sf == 1.0 else "salmon" for sf in scale_factors]
bars = ax.bar([f"×{sf}" for sf in scale_factors], z_vals, color=colors_bar,
              edgecolor="gray")
for bar, z in zip(bars, z_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{z:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Centroid Z (m)", fontsize=12)
ax.set_xlabel("Depth Scale Factor", fontsize=12)
ax.set_title("Centroid depth varies linearly with scale\n"
             "(green = ground truth)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Physical extent
ax = axes[1]
ext_z = [e[2] for e in extents_depth]
ext_x = [e[0] for e in extents_depth]
x_pos = np.arange(len(scale_factors))
width = 0.35
ax.bar(x_pos - width/2, ext_z, width, label="Z extent", color="steelblue",
       edgecolor="gray")
ax.bar(x_pos + width/2, ext_x, width, label="X extent", color="coral",
       edgecolor="gray")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"×{sf}" for sf in scale_factors])
ax.set_ylabel("Physical Extent (m)", fontsize=12)
ax.set_xlabel("Depth Scale Factor", fontsize=12)
ax.set_title("Physical size scales proportionally\n"
             "(catastrophic for grasping/navigation)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Failure 2B — Depth ambiguity: Z and extent scale together",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f2b_depth_ambiguity_barchart.png")

c_ref = centroids_depth[scale_factors.index(1.0)]
print("  Scale | Centroid Z (m) |         Extent (m)          | t error (m)")
print("  ------|----------------|-----------------------------|----------")
for sf, c, ext in zip(scale_factors, centroids_depth, extents_depth):
    t_err = translation_error_m(c, c_ref)
    marker = " <-- GT" if sf == 1.0 else ""
    print(f"  x{sf:<4} | {c[2]:>13.4f}  | "
          f"[{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}] | {t_err:.4f}{marker}")
print("\n  ⇒ 2D reprojection CANNOT detect depth ambiguity — "
      "only the 3D pose reveals it.\n")


# =====================================================================
# FAILURE 3 — Occlusion-induced translation drift
# =====================================================================

# %% [markdown]
# # Failure 3 — Occlusion-Induced Translation Drift
#
# When part of the object is occluded, the visible centroid shifts away
# from the true centroid.  A pose estimator relying on the visible region
# will report a drifted pose.

# %%
print("=" * 72)
print("  FAILURE 3 — Occlusion-Induced Translation Drift")
print("=" * 72)

y_mid = H // 2
x_mid = W // 2

occlusion_configs = [
    ("Full view",        mask_resized),
    ("Top half only",    mask_resized & (np.arange(H)[:, None] < y_mid)),
    ("Bottom half only", mask_resized & (np.arange(H)[:, None] >= y_mid)),
    ("Left half only",   mask_resized & (np.arange(W)[None, :] < x_mid)),
    ("Right half only",  mask_resized & (np.arange(W)[None, :] >= x_mid)),
]

centroids_occ = []
orientations_occ = []
counts_occ = []
occ_pts_list = []

for label, occ_mask in occlusion_configs:
    n_pts = np.count_nonzero(occ_mask)
    if n_pts < 50:
        centroids_occ.append(centroid_gt)
        orientations_occ.append(R_gt)
        counts_occ.append(0)
        occ_pts_list.append(pts_metric)
        continue

    pts_occ = depth_mask_to_points(depth_resized, occ_mask, K)
    pts_occ_metric = pts_occ * metric_scale
    c = compute_centroid(pts_occ_metric)
    R_occ = pca_orientation(pts_occ_metric)
    centroids_occ.append(c)
    orientations_occ.append(R_occ)
    counts_occ.append(len(pts_occ))
    occ_pts_list.append(pts_occ_metric)

# --- Figure 3A: Visible mask regions highlighted on RGB ---
fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
mask_colors = ["#00ff00", "#3399ff", "#ff9900", "#cc33ff", "#ff3333"]

for ax, (label, occ_mask), color in zip(axes, occlusion_configs, mask_colors):
    # Show RGB with coloured mask overlay
    overlay = rgb.copy().astype(float)
    mask_rgb = np.zeros_like(rgb, dtype=float)
    c_rgb = matplotlib.colors.to_rgb(color)
    mask_rgb[occ_mask] = [c_rgb[0]*255, c_rgb[1]*255, c_rgb[2]*255]
    overlay = np.clip(overlay * 0.4 + mask_rgb * 0.6, 0, 255).astype(np.uint8)
    overlay[~occ_mask] = (rgb[~occ_mask] * 0.3).astype(np.uint8)

    n = np.count_nonzero(occ_mask)
    ax.imshow(overlay)
    ax.set_title(f"{label}\n({n} pixels)", fontsize=10, fontweight="bold")
    ax.axis("off")

fig.suptitle("Failure 3A — Occlusion masks: what the pose estimator sees",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f3a_occlusion_masks.png")

# --- Figure 3B: Top-down XY view with centroid drift arrows ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XY top-down view
ax = axes[0]
occ_colors = ["green", "royalblue", "orange", "purple", "red"]
for i, ((label, _), pts_o, color) in enumerate(
        zip(occlusion_configs, occ_pts_list, occ_colors)):
    sub = pts_o[::10]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.3, alpha=0.15, color=color)
    c = centroids_occ[i]
    ax.plot(c[0], c[1], "^", color=color, markersize=12,
            markeredgecolor="black", markeredgewidth=1, label=label, zorder=5)
    # Draw drift arrow from GT centroid
    if i > 0:
        ax.annotate("", xy=(c[0], c[1]), xytext=(centroid_gt[0], centroid_gt[1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2))

ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.set_title("XY Top-Down View\nCentroid drift direction under occlusion",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_aspect("equal")

# XZ side view
ax = axes[1]
for i, ((label, _), pts_o, color) in enumerate(
        zip(occlusion_configs, occ_pts_list, occ_colors)):
    sub = pts_o[::10]
    ax.scatter(sub[:, 0], sub[:, 2], s=0.3, alpha=0.15, color=color)
    c = centroids_occ[i]
    ax.plot(c[0], c[2], "^", color=color, markersize=12,
            markeredgecolor="black", markeredgewidth=1, label=label, zorder=5)
    if i > 0:
        ax.annotate("", xy=(c[0], c[2]), xytext=(centroid_gt[0], centroid_gt[2]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2))

ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Z — Depth (m)", fontsize=12)
ax.set_title("XZ Side View\nCentroid shift in depth under occlusion",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_aspect("equal")

fig.suptitle("Failure 3B — Occlusion shifts centroid: "
             "arrows show drift direction from GT",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f3b_occlusion_drift_arrows.png")

# --- Figure 3C: Grouped bar chart of translation drift + rotation error ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels_occ = [cfg[0] for cfg in occlusion_configs]
t_drifts = [translation_error_m(c, centroid_gt) for c in centroids_occ]
r_errors_occ = [rotation_error_deg(R_gt, R_o) if n > 0 else 0.0
                for R_o, n in zip(orientations_occ, counts_occ)]

ax = axes[0]
bars = ax.barh(labels_occ, t_drifts, color=occ_colors, edgecolor="gray")
for bar, val in zip(bars, t_drifts):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f} m", va="center", fontsize=10)
ax.set_xlabel("Translation Drift (m)", fontsize=12)
ax.set_title("Centroid Translation Drift", fontsize=11)
ax.grid(axis="x", alpha=0.3)

ax = axes[1]
bars = ax.barh(labels_occ, r_errors_occ, color=occ_colors, edgecolor="gray")
for bar, val in zip(bars, r_errors_occ):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}°", va="center", fontsize=10)
ax.set_xlabel("Rotation Error (°)", fontsize=12)
ax.set_title("Orientation Error (PCA)", fontsize=11)
ax.grid(axis="x", alpha=0.3)

fig.suptitle("Failure 3C — Occlusion: both translation and rotation "
             "degrade with partial visibility",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f3c_occlusion_error_bars.png")

print("  Occlusion          | Points |       Centroid (m)       "
      "     | t drift | R error")
print("  -------------------|--------|-------------------------"
      "------|---------|--------")
for (label, _), c, n, R_occ in zip(occlusion_configs, centroids_occ,
                                    counts_occ, orientations_occ):
    drift = translation_error_m(c, centroid_gt)
    r_err = rotation_error_deg(R_gt, R_occ) if n > 0 else 0.0
    print(f"  {label:<18} | {n:>6} | "
          f"[{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}] | "
          f"{drift:.4f}m | {r_err:.1f}°")
print("\n  ⇒ Occlusion shifts the centroid AND rotates the principal "
      "axes — both R and t drift.\n")


# =====================================================================
# FAILURE 4 — Scale inconsistency
# =====================================================================

# %% [markdown]
# # Failure 4 — Scale Inconsistency
#
# `scale_to_metric` relies on a known real-world height. If the assumed
# height is wrong, the metric pose is wrong even though 2D reprojection
# is unchanged.

# %%
print("=" * 72)
print("  FAILURE 4 — Scale Inconsistency")
print("=" * 72)

height_guesses = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
centroids_scale = []
extents_scale = []
pts_scale_list = []

for h in height_guesses:
    pts_h, _ = scale_to_metric(pts_cam, h)
    c = compute_centroid(pts_h)
    centroids_scale.append(c)
    extents_scale.append(np.ptp(pts_h, axis=0))
    pts_scale_list.append(pts_h)

# --- Figure 4A: 3D nested bounding boxes ---
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")

colors_h = plt.cm.RdYlGn_r(np.linspace(0, 1, len(height_guesses)))
gt_idx = height_guesses.index(REAL_HEIGHT_M)

for i, (h, pts_h) in enumerate(zip(height_guesses, pts_scale_list)):
    lw = 3.0 if h == REAL_HEIGHT_M else 1.0
    ls = "-" if h == REAL_HEIGHT_M else "--"
    lbl = f"h={h:.2f}m" + (" ← GT" if h == REAL_HEIGHT_M else "")
    draw_bbox_3d(ax, pts_h, color=colors_h[i], linestyle=ls,
                 linewidth=lw, label=lbl)
    c = centroids_scale[i]
    ax.scatter(*c, s=50, color=colors_h[i], marker="o",
               edgecolors="black", zorder=5)

# Show GT point cloud faintly
sub = pts_metric[::10]
ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.3, alpha=0.08, c="gray")

ax.set_xlabel("X (m)", fontsize=11)
ax.set_ylabel("Y (m)", fontsize=11)
ax.set_zlabel("Z (m)", fontsize=11)
ax.set_title("Scale inconsistency: nested bounding boxes\n"
             "Wrong height assumption → wrong 3D extent & position",
             fontsize=12)
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.view_init(elev=25, azim=-50)
plt.tight_layout()
savefig(fig, "f4a_scale_nested_bboxes.png")

# --- Figure 4B: Multi-metric bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

x_labels = [f"{h:.2f}" for h in height_guesses]
x_pos = np.arange(len(height_guesses))

# Centroid Z
ax = axes[0]
z_vals = [c[2] for c in centroids_scale]
colors_bar = ["limegreen" if h == REAL_HEIGHT_M else "salmon"
              for h in height_guesses]
bars = ax.bar(x_pos, z_vals, color=colors_bar, edgecolor="gray")
for bar, z in zip(bars, z_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{z:.2f}", ha="center", fontsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Centroid Z (m)", fontsize=11)
ax.set_xlabel("Assumed Height (m)", fontsize=11)
ax.set_title("Centroid depth", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Physical extent (Z and X)
ax = axes[1]
ext_z = [e[2] for e in extents_scale]
ext_x = [e[0] for e in extents_scale]
width = 0.35
ax.bar(x_pos - width/2, ext_z, width, label="Z extent",
       color="steelblue", edgecolor="gray")
ax.bar(x_pos + width/2, ext_x, width, label="X extent",
       color="coral", edgecolor="gray")
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Physical Extent (m)", fontsize=11)
ax.set_xlabel("Assumed Height (m)", fontsize=11)
ax.set_title("Metric extent", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Translation error from GT
ax = axes[2]
c_correct = centroids_scale[gt_idx]
t_errs = [translation_error_m(c, c_correct) for c in centroids_scale]
bars = ax.bar(x_pos, t_errs, color=colors_bar, edgecolor="gray")
for bar, te in zip(bars, t_errs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{te:.3f}", ha="center", fontsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Translation Error (m)", fontsize=11)
ax.set_xlabel("Assumed Height (m)", fontsize=11)
ax.set_title("3D translation error vs GT", fontsize=11)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Failure 4B — Scale inconsistency: wrong height assumption "
             "→ proportional 3D error\n(green = correct, red = wrong)",
             fontsize=13, y=1.05)
plt.tight_layout()
savefig(fig, "f4b_scale_barchart.png")

# --- Figure 4C: Continuous scale-error curve ---
fig, ax = plt.subplots(figsize=(10, 5))
h_range = np.linspace(0.05, 0.60, 50)
t_curve = []
for h in h_range:
    pts_h, _ = scale_to_metric(pts_cam, h)
    c = compute_centroid(pts_h)
    t_curve.append(translation_error_m(c, c_correct))

ax.plot(h_range, t_curve, "b-", linewidth=2)
ax.axvline(REAL_HEIGHT_M, color="green", linestyle="--", linewidth=2,
           label=f"GT height = {REAL_HEIGHT_M} m")
ax.set_xlabel("Assumed Object Height (m)", fontsize=12)
ax.set_ylabel("3D Translation Error (m)", fontsize=12)
ax.set_title("Failure 4C — Translation error is LINEAR in height error\n"
             "A 2× height mistake → 2× translation error",
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, "f4c_scale_error_curve.png")

print("  Assumed h |  Centroid Z (m) |        Extent (m)         | "
      "t error")
print("  ----------|-----------------|---------------------------|"
      "---------")
for h, c, ext in zip(height_guesses, centroids_scale, extents_scale):
    t_err = translation_error_m(c, c_correct)
    marker = " <-- correct" if h == REAL_HEIGHT_M else ""
    print(f"  {h:.2f} m    | {c[2]:>14.4f}  | "
          f"[{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}] | "
          f"{t_err:.4f} m{marker}")
print("\n  ⇒ A 2× height error causes a 2× translation error — "
      "invisible in 2D, catastrophic in 3D.\n")


# =====================================================================
# FAILURE 5 — Multi-view disagreement (noisy depth)
# =====================================================================

# %% [markdown]
# # Failure 5 — Multi-View Disagreement
#
# Different depth sensors (or repeated measurements) produce slightly
# different depth maps.  Each yields a different pose estimate.
# The spread of per-view poses reveals the uncertainty.

# %%
print("=" * 72)
print("  FAILURE 5 — Multi-View Disagreement (Noisy Depth)")
print("=" * 72)

np.random.seed(42)
noise_fractions = np.array([0.00, 0.02, 0.05, 0.10, 0.15, 0.20])
depth_range = depth_resized[mask_resized].ptp()

# Run multiple trials per noise level for statistics
N_TRIALS = 10
all_centroids_per_noise = {nf: [] for nf in noise_fractions}
all_rotations_per_noise = {nf: [] for nf in noise_fractions}

for nf in noise_fractions:
    for trial in range(N_TRIALS):
        noise_std = nf * depth_range
        depth_noisy = depth_resized + np.random.randn(*depth_resized.shape) * noise_std
        depth_noisy = np.clip(depth_noisy, 1e-6, None)

        pts_n = depth_mask_to_points(depth_noisy, mask_resized, K)
        pts_n_metric, _ = scale_to_metric(pts_n, REAL_HEIGHT_M)

        c = compute_centroid(pts_n_metric)
        R_n = pca_orientation(pts_n_metric)
        all_centroids_per_noise[nf].append(c)
        all_rotations_per_noise[nf].append(R_n)

# Aggregate statistics
mean_centroids = {nf: np.mean(cs, axis=0)
                  for nf, cs in all_centroids_per_noise.items()}
mean_t_err = {nf: np.mean([translation_error_m(c, centroid_gt)
                            for c in cs])
              for nf, cs in all_centroids_per_noise.items()}
std_t_err = {nf: np.std([translation_error_m(c, centroid_gt)
                          for c in cs])
             for nf, cs in all_centroids_per_noise.items()}
mean_r_err = {nf: np.mean([rotation_error_deg(R_gt, R)
                            for R in rs])
              for nf, rs in all_rotations_per_noise.items()}
std_r_err = {nf: np.std([rotation_error_deg(R_gt, R)
                          for R in rs])
             for nf, rs in all_rotations_per_noise.items()}

# --- Figure 5A: Error vs noise line plots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

nf_pct = noise_fractions * 100

# Translation error with error bars
ax = axes[0]
means_t = [mean_t_err[nf] for nf in noise_fractions]
stds_t = [std_t_err[nf] for nf in noise_fractions]
ax.errorbar(nf_pct, means_t, yerr=stds_t, fmt="o-", color="steelblue",
            linewidth=2, markersize=8, capsize=5, label="Mean ± 1σ")
ax.fill_between(nf_pct,
                [m - s for m, s in zip(means_t, stds_t)],
                [m + s for m, s in zip(means_t, stds_t)],
                alpha=0.2, color="steelblue")
ax.set_xlabel("Depth Noise Level (%)", fontsize=12)
ax.set_ylabel("Translation Error (m)", fontsize=12)
ax.set_title("Translation error grows with depth noise", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# Rotation error with error bars
ax = axes[1]
means_r = [mean_r_err[nf] for nf in noise_fractions]
stds_r = [std_r_err[nf] for nf in noise_fractions]
ax.errorbar(nf_pct, means_r, yerr=stds_r, fmt="s-", color="coral",
            linewidth=2, markersize=8, capsize=5, label="Mean ± 1σ")
ax.fill_between(nf_pct,
                [m - s for m, s in zip(means_r, stds_r)],
                [m + s for m, s in zip(means_r, stds_r)],
                alpha=0.2, color="coral")
ax.set_xlabel("Depth Noise Level (%)", fontsize=12)
ax.set_ylabel("Rotation Error (°)", fontsize=12)
ax.set_title("Rotation error grows with depth noise", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

fig.suptitle("Failure 5A — Multi-view disagreement: "
             "pose uncertainty grows with sensor noise\n"
             f"({N_TRIALS} trials per noise level)",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f5a_multiview_error_curves.png")

# --- Figure 5B: 3D centroid scatter cloud ---
fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121, projection="3d")
colors_noise = plt.cm.hot(np.linspace(0.1, 0.9, len(noise_fractions)))

for i, nf in enumerate(noise_fractions):
    cs = np.array(all_centroids_per_noise[nf])
    lbl = f"{nf*100:.0f}% noise" if nf > 0 else "0% (clean)"
    ax1.scatter(cs[:, 0], cs[:, 1], cs[:, 2], s=30, color=colors_noise[i],
                alpha=0.7, label=lbl, edgecolors="gray", linewidth=0.5)

ax1.scatter(*centroid_gt, s=100, c="lime", marker="*",
            edgecolors="black", linewidth=1, label="GT centroid", zorder=10)
ax1.set_xlabel("X (m)", fontsize=10)
ax1.set_ylabel("Y (m)", fontsize=10)
ax1.set_zlabel("Z (m)", fontsize=10)
ax1.set_title("3D centroid scatter\n(each dot = one trial)", fontsize=11)
ax1.legend(fontsize=7, loc="upper left")
ax1.view_init(elev=25, azim=-50)

# XZ projection (side view)
ax2 = fig.add_subplot(122)
for i, nf in enumerate(noise_fractions):
    cs = np.array(all_centroids_per_noise[nf])
    lbl = f"{nf*100:.0f}% noise" if nf > 0 else "0% (clean)"
    ax2.scatter(cs[:, 0], cs[:, 2], s=30, color=colors_noise[i],
                alpha=0.7, label=lbl, edgecolors="gray", linewidth=0.5)

ax2.scatter(centroid_gt[0], centroid_gt[2], s=100, c="lime", marker="*",
            edgecolors="black", linewidth=1, label="GT centroid", zorder=10)
ax2.set_xlabel("X (m)", fontsize=11)
ax2.set_ylabel("Z (m)", fontsize=11)
ax2.set_title("XZ projection — depth jitter\n"
              "Noisy depth → centroid spread", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("Failure 5B — Centroid scatter: noise spreads the estimated "
             "pose in 3D", fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f5b_multiview_centroid_scatter.png")

# --- Figure 5C: XZ side-view of noisy point clouds ---
# NOTE: 2D reprojection of back-projected points is always identical
# (back-proj then re-proj cancels out), so 2D plots can't show noise.
# Instead we show XZ cross-sections where the depth spread is visible.
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
np.random.seed(99)

sub_gt_xz = pts_metric[::5]

for ax, nf in zip(axes.flat, noise_fractions):
    depth_noisy = depth_resized + np.random.randn(*depth_resized.shape) * (nf * depth_range)
    depth_noisy = np.clip(depth_noisy, 1e-6, None)

    pts_n = depth_mask_to_points(depth_noisy, mask_resized, K)
    pts_n_metric, _ = scale_to_metric(pts_n, REAL_HEIGHT_M)

    sub_n = pts_n_metric[::5]

    # GT in green (faint)
    ax.scatter(sub_gt_xz[:, 0], sub_gt_xz[:, 2], s=0.3, alpha=0.15,
               c="lime")
    # Noisy in red
    ax.scatter(sub_n[:, 0], sub_n[:, 2], s=0.3, alpha=0.15, c="red")

    # Mark centroids
    c_n = compute_centroid(pts_n_metric)
    ax.plot(centroid_gt[0], centroid_gt[2], "^", color="lime",
            markersize=10, markeredgecolor="black", label="GT centroid")
    ax.plot(c_n[0], c_n[2], "v", color="red", markersize=10,
            markeredgecolor="black", label="Noisy centroid")

    t_err = translation_error_m(c_n, centroid_gt)
    color = "lime" if nf == 0 else "red"
    ax.set_title(f"Noise: {nf*100:.0f}%\nt err: {t_err:.3f} m",
                 fontsize=10, color=color, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Z — Depth (m)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    if nf == 0:
        ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Failure 5C — XZ side-view: depth noise spreads the "
             "point cloud in 3D\n(green = clean, red = noisy — "
             "note the growing depth scatter)",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f5c_multiview_xz_sideview.png")

# --- Figure 5D: Depth-map noise heatmaps ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
np.random.seed(99)

for ax, nf in zip(axes.flat, noise_fractions):
    depth_noisy = depth_resized + np.random.randn(*depth_resized.shape) * (nf * depth_range)
    # Show absolute depth difference as heatmap
    diff = np.abs(depth_noisy - depth_resized)
    # Mask to plant region only
    diff_vis = np.zeros_like(diff)
    diff_vis[mask_resized] = diff[mask_resized]
    im = ax.imshow(diff_vis, cmap="hot", vmin=0,
                   vmax=0.2 * depth_range)
    ax.set_title(f"Noise: {nf*100:.0f}%\n|Δdepth| map",
                 fontsize=10, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Failure 5D — Depth-noise magnitude heatmaps\n"
             "Each panel shows |noisy − clean| depth on the plant region",
             fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, "f5d_depth_noise_heatmaps.png")

print("  Noise % | Centroid Z (m) | t error (m)  | R error (°)")
print("  --------|----------------|--------------|------------")
for nf in noise_fractions:
    c_mean = mean_centroids[nf]
    te = mean_t_err[nf]
    re = mean_r_err[nf]
    print(f"  {nf*100:>5.0f}%  | {c_mean[2]:>13.4f}  | "
          f"{te:>11.4f}  | {re:>9.1f}")

print(f"\n  Max mean translation error:  {max(mean_t_err.values()):.4f} m")
print(f"  Max mean rotation error:     {max(mean_r_err.values()):.1f}°")
print("\n  ⇒ Noisy depth causes both R and t to jitter — "
      "multi-view consistency checks catch this.\n")


# =====================================================================
# Summary figure — comparison across all failure modes
# =====================================================================

# %%
print("=" * 72)
print("  SUMMARY — Pose-Based Failure Diagnosis")
print("=" * 72)

# Collect summary data
summary_labels = [
    "Symmetry flip\n(180° Y)",
    "Depth ambiguity\n(×2.0 scale)",
    "Occlusion\n(top half)",
    "Scale error\n(h=0.10 vs 0.25)",
    "Multi-view\n(20% noise)",
]

summary_rot_err = [
    flip_data[4][3],                    # 180° flip
    0.0,                                 # depth ambiguity
    rotation_error_deg(R_gt, orientations_occ[1]),  # occlusion top half
    0.0,                                 # scale inconsistency
    mean_r_err[0.20],                    # multi-view 20% noise
]

summary_trans_err = [
    0.0,                                 # symmetry flip
    translation_error_m(centroids_depth[scale_factors.index(2.0)],
                        centroids_depth[scale_factors.index(1.0)]),
    translation_error_m(centroids_occ[1], centroid_gt),
    translation_error_m(centroids_scale[height_guesses.index(0.10)],
                        centroids_scale[height_guesses.index(REAL_HEIGHT_M)]),
    mean_t_err[0.20],
]

# --- Summary figure: side-by-side bars ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(summary_labels))
colors_summary = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#2ecc71"]

# Rotation error
ax = axes[0]
bars = ax.bar(x, summary_rot_err, color=colors_summary, edgecolor="gray",
              width=0.6)
for bar, val in zip(bars, summary_rot_err):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}°", ha="center", fontsize=10, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 3,
                "0.0°", ha="center", fontsize=10, color="gray")
ax.set_xticks(x)
ax.set_xticklabels(summary_labels, fontsize=9)
ax.set_ylabel("Rotation Error (°)", fontsize=12)
ax.set_title("Which failures cause ROTATION error?", fontsize=12,
             fontweight="bold")
ax.set_ylim(0, 200)
ax.grid(axis="y", alpha=0.3)

# Translation error
ax = axes[1]
bars = ax.bar(x, summary_trans_err, color=colors_summary, edgecolor="gray",
              width=0.6)
for bar, val in zip(bars, summary_trans_err):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}m", ha="center", fontsize=10, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.01,
                "0.000m", ha="center", fontsize=10, color="gray")
ax.set_xticks(x)
ax.set_xticklabels(summary_labels, fontsize=9)
ax.set_ylabel("Translation Error (m)", fontsize=12)
ax.set_title("Which failures cause TRANSLATION error?", fontsize=12,
             fontweight="bold")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Summary — Each failure mode has a distinct error signature\n"
             "Pose inspection reveals failures that 2D reprojection cannot",
             fontsize=14, fontweight="bold", y=1.05)
plt.tight_layout()
savefig(fig, "f_summary_error_comparison.png")

# --- Summary: 2D-detectable vs 3D-only failures ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(summary_trans_err, summary_rot_err, s=200, c=colors_summary,
           edgecolors="black", linewidth=1.5, zorder=5)
for i, lbl in enumerate(summary_labels):
    lbl_clean = lbl.replace("\n", " ")
    ax.annotate(lbl_clean,
                (summary_trans_err[i], summary_rot_err[i]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

# Quadrant labels
ax.axhline(10, color="gray", linestyle=":", alpha=0.5)
ax.axvline(0.05, color="gray", linestyle=":", alpha=0.5)
ax.text(0.01, 150, "Rotation-only\n(symmetry flips)",
        fontsize=9, color="gray", style="italic")
ax.text(0.3, 3, "Translation-only\n(depth/scale errors)",
        fontsize=9, color="gray", style="italic")
ax.text(0.3, 150, "Both R+t\n(multi-view noise)",
        fontsize=9, color="gray", style="italic")
ax.text(0.01, 3, "No error\n(ideal)",
        fontsize=9, color="gray", style="italic")

ax.set_xlabel("Translation Error (m)", fontsize=12)
ax.set_ylabel("Rotation Error (°)", fontsize=12)
ax.set_title("Failure mode error signatures in R–t space\n"
             "Each failure occupies a different region",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_xlim(-0.03, max(summary_trans_err) * 1.3)
ax.set_ylim(-5, max(summary_rot_err) * 1.15)
plt.tight_layout()
savefig(fig, "f_summary_rt_scatter.png")

print()
print("  Failure                   | Rotation Error | Translation Error")
print("  --------------------------|----------------|------------------")
for lbl, re, te in zip(summary_labels, summary_rot_err, summary_trans_err):
    lbl_clean = lbl.replace("\n", " ")
    print(f"  {lbl_clean:<25} | {re:>13.1f}° | {te:>15.4f} m")

print()
print("  Key insight:")
print("    • Symmetry flips → pure rotation error (2D looks fine)")
print("    • Depth ambiguity & scale → pure translation error (2D identical)")
print("    • Occlusion → mainly translation drift + some rotation")
print("    • Multi-view noise → both R and t degrade")
print("    ⇒ Each failure has a DISTINCT signature in (R, t) space.")
print("    ⇒ You MUST inspect the full SE(3) pose to diagnose failures.")
print()
print(f"All figures saved to {FIG_DIR}/")
