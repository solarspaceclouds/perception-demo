"""
Shared utility functions for the Intrinsic Perception Demo.

Provides reusable helpers for:
  - SE(3) transformations
  - Camera intrinsics & projection
  - Depth back-projection to 3D point clouds
  - Metric scaling from monocular (relative) depth
  - Mesh loading (trimesh / Open3D)
  - 3D coordinate frame visualisation
  - Look-at camera construction (OpenCV convention)
"""

import json
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transformation (SE(3)) from R (3×3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) matrix (faster than np.linalg.inv for rigid transforms)."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def build_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Return a 3×3 camera intrinsic matrix K."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def intrinsics_from_fov(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Derive K from a horizontal field-of-view angle."""
    fx = fy = (width / 2) / np.tan(np.deg2rad(fov_deg / 2))
    cx = width / 2.0
    cy = height / 2.0
    return build_intrinsics(fx, fy, cx, cy)


def project_points(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D points in camera frame → 2D pixel coordinates.

    Args:
        pts_cam: (N, 3) points in camera frame.
        K: 3×3 intrinsic matrix.

    Returns:
        uv: (N, 2) pixel coordinates [u, v].
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Look-at camera (OpenCV convention: +X right, +Y down, +Z forward)
# ---------------------------------------------------------------------------
def look_at_opencv(cam_pos, target, up=np.array([0, 0, 1])):
    z = target - cam_pos
    z = z / np.linalg.norm(z)          # forward
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)          # right
    y = np.cross(z, x)                 # down (OpenCV)

    R = np.stack([x, y, z], axis=1)    # world → cam
    return R

# # Graphics-style look-at matrix
# def look_at_cv(
#     cam_pos: np.ndarray,
#     target: np.ndarray,
#     up: np.ndarray = np.array([0.0, 0.0, 1.0]),
# ) -> np.ndarray:
#     """
#     Compute a rotation matrix that makes the camera look from *cam_pos*
#     towards *target*, following the OpenCV convention (+Z forward, +Y down).
#     """
#     z = target - cam_pos
#     z = z / np.linalg.norm(z)

#     x = np.cross(up, z)
#     x = x / np.linalg.norm(x)

#     y = np.cross(z, x)

#     # OpenCV: Y points down, so negate y
#     return np.stack([x, -y, z], axis=1)


# ---------------------------------------------------------------------------
# Depth / point-cloud helpers
# ---------------------------------------------------------------------------

def depth_mask_to_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Back-project masked depth to 3D points in camera frame.

    Args:
        depth: (H, W) depth map.
        mask: (H, W) boolean mask selecting valid pixels.
        K: 3×3 intrinsic matrix.

    Returns:
        pts: (N, 3) array of 3D points.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    return pts[mask]  # (N, 3)


def compute_centroid(pts: np.ndarray) -> np.ndarray:
    """Return the mean 3D point (centroid)."""
    return np.mean(pts, axis=0)


def scale_to_metric(
    pts: np.ndarray,
    real_height_m: float,
    percentile: Tuple[float, float] = (5.0, 95.0),
) -> Tuple[np.ndarray, float]:
    """
    Scale relative-depth points to metric units using a known real-world height.

    Uses percentile-based robust height estimation to avoid outliers.

    Args:
        pts: (N, 3) points in relative-depth units.
        real_height_m: known height of the object in metres.
        percentile: (low, high) percentiles for robust height estimation.

    Returns:
        pts_metric: (N, 3) points in metres.
        scale: the scalar multiplier applied.
    """
    z_low, z_high = np.percentile(pts[:, 2], list(percentile))
    rel_height = z_high - z_low
    scale = real_height_m / rel_height
    return pts * scale, scale


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_rgb(path: str) -> np.ndarray:
    """Load an RGB image (returns H×W×3 uint8 in RGB order)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path: str) -> np.ndarray:
    """Load a depth map from .npy."""
    return np.load(str(path))


def load_mask(path: str) -> np.ndarray:
    """Load a binary mask from .png or .npy (returns bool H×W)."""
    p = Path(path)
    if p.suffix == ".npy":
        m = np.load(str(p))
    else:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
    return m > 0


def resize_to_rgb(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize depth and mask to match RGB resolution. Returns (depth, mask)."""
    H, W = rgb.shape[:2]
    depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(
        mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
    )
    return depth_resized, mask_resized > 0


# ---------------------------------------------------------------------------
# Mesh loading helpers
# ---------------------------------------------------------------------------

def load_mesh_vertices(mesh_path: str, scale: float = 1.0) -> np.ndarray:
    """
    Load mesh vertices using trimesh.

    Args:
        mesh_path: path to .obj / .ply / .stl file.
        scale: unit conversion factor (e.g. 0.001 for mm→m).

    Returns:
        vertices: (N, 3) float32 array.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices *= scale
    return vertices


def load_mesh_vertices_o3d(mesh_path: str, scale: float = 1.0) -> np.ndarray:
    """
    Load mesh vertices using Open3D.

    Args:
        mesh_path: path to .obj / .ply / .stl file.
        scale: unit conversion factor.

    Returns:
        vertices: (N, 3) float32 array.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices *= scale
    return vertices


# ---------------------------------------------------------------------------
# Pose I/O
# ---------------------------------------------------------------------------

def save_pose_json(T: np.ndarray, path: str) -> None:
    """Save a 4×4 SE(3) pose to JSON."""
    with open(str(path), "w") as f:
        json.dump(T.tolist(), f, indent=2)


def load_pose_json(path: str) -> np.ndarray:
    """Load a 4×4 SE(3) pose from JSON."""
    with open(str(path), "r") as f:
        return np.array(json.load(f))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_frame(ax, R: np.ndarray, t: np.ndarray, label: str, scale: float = 0.1):
    """
    Draw a 3D coordinate frame on a matplotlib 3D axis.

    Each column of R is drawn as a coloured arrow (R=x, G=y, B=z).
    """
    origin = t
    colors = ["r", "g", "b"]
    for i in range(3):
        axis = R[:, i] * scale
        ax.plot(
            [origin[0], origin[0] + axis[0]],
            [origin[1], origin[1] + axis[1]],
            [origin[2], origin[2] + axis[2]],
            color=colors[i],
        )
    ax.text(*origin, f"  {label}")


def set_equal_axes(ax, pts: Optional[np.ndarray] = None):
    """Set equal aspect ratio on a 3D matplotlib axis."""
    if pts is not None:
        mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
        span = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    else:
        mid = np.zeros(3)
        span = 1.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

