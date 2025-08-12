import open3d as o3d
import numpy as np

# Helper function for look-at matrix
def look_at(eye, target, up=None):
    if up is None:
        up = np.array([0, -1, 0])
    z = (eye - target)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = x
    view_matrix[:3, 1] = y
    view_matrix[:3, 2] = z
    view_matrix[:3, 3] = eye
    return view_matrix

def visualize_camera_trajectory(poses, point_cloud=None):
    """
    Visualize camera trajectory with frustums and connecting lines

    Args:
        poses: List of 4x4 camera-to-world transformation matrices
        point_cloud: Optional point cloud to visualize with cameras
    """
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Create camera intrinsic (modify as needed)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=320.0,
        cy=240.0
    )

    # Create camera frustums
    frustums = []
    for pose in poses:
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=intrinsic,
            extrinsic=np.linalg.inv(pose),
            scale=0.3
        )
        frustums.append(frustum)

    # Create trajectory lines
    camera_centers = [pose[:3, 3] for pose in poses]
    lines = []
    for i in range(len(camera_centers) - 1):
        lines.append([i, i + 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(camera_centers),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([1, 0, 0])  # Red trajectory

    # Prepare visualization
    geometries = [coordinate_frame, line_set] + frustums
    if point_cloud is not None:
        geometries.insert(1, point_cloud)

    # Customize visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Set camera to look at the first camera pose
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(poses[0][:3, 3])
    ctr.set_zoom(0.5)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # Visualize the trajectory
    # Example usage:
    # Generate some example poses in a circular trajectory
    num_poses = 10
    radius = 5
    poses = []
    for i in range(num_poses):
        angle = 2 * np.pi * i / num_poses
        pose = np.eye(4)
        pose[0, 3] = radius * np.cos(angle)
        pose[1, 3] = radius * np.sin(angle)
        pose[2, 3] = 2.0
        # Make cameras point towards center
        #pose[:3, :3] = look_at(pose[:3, 3], np.zeros(3)).T
        tmp = look_at(pose[:3, 3], np.zeros(3)).T
        pose[:3, :3] = tmp[:3, :3]
        poses.append(pose)

    visualize_camera_trajectory(poses)