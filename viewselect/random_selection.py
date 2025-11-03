import numpy as np
import random


class RandomCameraSelection:
    def __init__(self, scene_bounds, num_candidates=1000):
        """
        Initialize random camera selector

        Args:
            scene_bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
            num_candidates: Number of candidate poses to generate
        """
        self.scene_bounds = scene_bounds
        self.num_candidates = num_candidates
        self.candidate_poses = []

    def generate_candidate_poses(self):
        """Generate random camera poses in the feasible space"""
        poses = []

        for i in range(self.num_candidates):
            # Random position within scene bounds
            x = random.uniform(self.scene_bounds[0], self.scene_bounds[1])
            y = random.uniform(self.scene_bounds[2], self.scene_bounds[3])
            z = random.uniform(self.scene_bounds[4], self.scene_bounds[5])
            position = [x, y, z]

            # Random orientation (yaw, pitch, roll)
            yaw = random.uniform(0, 2 * np.pi)  # Horizontal rotation
            pitch = random.uniform(-np.pi / 4, np.pi / 4)  # Limited vertical rotation
            roll = random.uniform(-np.pi / 12, np.pi / 12)  # Small roll variation

            orientation = [yaw, pitch, roll]

            poses.append({
                'position': position,
                'orientation': orientation,
                'id': i
            })

        self.candidate_poses = poses
        return poses

    def select_cameras(self, k, candidate_poses=None):
        """
        Randomly select k cameras from candidate poses

        Args:
            k: Number of cameras to select
            candidate_poses: Pre-computed candidate poses (optional)

        Returns:
            selected_poses: List of selected camera poses
            selection_indices: Indices of selected candidates
        """
        if candidate_poses is None:
            candidate_poses = self.candidate_poses

        if len(candidate_poses) < k:
            raise ValueError(f"Cannot select {k} cameras from {len(candidate_poses)} candidates")

        # Random sampling without replacement
        selected_indices = random.sample(range(len(candidate_poses)), k)
        selected_poses = [candidate_poses[i] for i in selected_indices]

        return selected_poses, selected_indices

    def evaluate_coverage(self, selected_poses, scene_point_cloud):
        """
        Evaluate coverage of randomly selected cameras

        Args:
            selected_poses: List of selected camera poses
            scene_point_cloud: [N, 3] array of 3D points

        Returns:
            coverage_ratio: Percentage of scene points visible
            visible_points: Mask of visible points
        """
        total_points = scene_point_cloud.shape[0]
        visibility_mask = np.zeros(total_points, dtype=bool)

        for pose in selected_poses:
            # Simple visibility check (simplified)
            pose_visibility = self.check_visibility(pose, scene_point_cloud)
            visibility_mask = visibility_mask | pose_visibility

        coverage_ratio = np.sum(visibility_mask) / total_points
        return coverage_ratio, visibility_mask

    def check_visibility(self, camera_pose, points):
        """
        Simplified visibility check for a camera pose
        """
        # Transform points to camera coordinates
        points_cam = self.transform_to_camera_frame(points, camera_pose)

        # Project to image plane (simplified)
        u, v, depth = self.project_to_image(points_cam)

        # Check if points are within field of view and in front of camera
        image_width, image_height = 1920, 1080  # Example resolution
        within_fov = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
        in_front = depth > 0

        visible = within_fov & in_front
        return visible