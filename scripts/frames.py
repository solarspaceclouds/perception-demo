"""
Demo 1 — Coordinate Frame Transformations (SE(3))

Demonstrates:
  1. Defining camera and object poses in the world frame.
  2. Building SE(3) homogeneous transformation matrices.
  3. Transforming the object pose into the camera frame.
  4. Visualising coordinate frames in 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from utils import make_T, draw_frame

# Step 1: Define poses in the world frame

# Camera: 1.5m above ground, tilted 20 deg downward about X
t_world_cam = np.array([0.0, 0.0, 1.5])
R_world_cam = Rotation.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()

# Object: sitting on the ground plane, rotated 45° about Z
t_world_obj = np.array([0.3, 0.2, 0.0])
R_world_obj = Rotation.from_euler("z", 45, degrees=True).as_matrix()

# # Step 2: Build SE(3) transformation matrices
T_world_cam = make_T(R_world_cam, t_world_cam)
T_world_obj = make_T(R_world_obj, t_world_obj)

print("T_world_cam:\n", T_world_cam)
print("T_world_obj:\n", T_world_obj)

# # Step 3: Transform object → camera frame
T_cam_world = np.linalg.inv(T_world_cam)
T_cam_obj = T_cam_world @ T_world_obj

R_cam_obj = T_cam_obj[:3, :3]
t_cam_obj = T_cam_obj[:3, 3]

print("Object position in camera frame:", t_cam_obj)

# # Step 4: Visualise all frames in the world

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
ax.set_title("World, Camera, and Object frames")
plt.tight_layout()
plt.show()

# # Step 5: Visualise object in camera frame

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

draw_frame(ax, np.eye(3), np.zeros(3), "Camera (origin)", scale=0.2)
draw_frame(ax, R_cam_obj, t_cam_obj, "Object", scale=0.2)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-2, 0.5])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Object pose in the camera frame")
plt.tight_layout()
plt.show()
