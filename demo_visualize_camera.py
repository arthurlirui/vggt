import open3d as o3d
import numpy as np

# Create a coordinate frame to represent world origin
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Create a dummy point cloud for context (optional)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# Define camera intrinsic parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=525.0,
    fy=525.0,
    cx=320.0,
    cy=240.0
)

# Create camera poses (4x4 transformation matrices)
# This is a list of example poses - replace with your actual camera poses
camera_poses = [
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 5],
        [0, 0, 0, 1]
    ]),
    np.array([
        [0, -1, 0, 2],
        [1, 0, 0, 1],
        [0, 0, 1, 4],
        [0, 0, 0, 1]
    ])
]

# Create camera frustums for visualization
camera_frustums = []
for pose in camera_poses:
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        intrinsic=intrinsic,
        extrinsic=np.linalg.inv(pose),  # Open3D expects camera-to-world
        scale=0.5
    )
    camera_frustums.append(frustum)

# Visualize everything together
o3d.visualization.draw_geometries([coordinate_frame, point_cloud] + camera_frustums)