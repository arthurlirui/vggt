import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class BundleAdjustment:
    def __init__(self, camera_intrinsics):
        """
        Initialize Bundle Adjustment optimizer

        Parameters:
        - camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
        """
        self.intrinsics = camera_intrinsics
        self.fx = camera_intrinsics['fx']
        self.fy = camera_intrinsics['fy']
        self.cx = camera_intrinsics['cx']
        self.cy = camera_intrinsics['cy']

    def project_point(self, rotation_matrix, translation, point_3d):
        """
        Project 3D point to 2D image coordinates

        Parameters:
        - rotation_matrix: 3x3 rotation matrix
        - translation: 3x1 translation vector
        - point_3d: 3D point in world coordinates

        Returns:
        - projected 2D point [u, v]
        """
        # Transform point to camera coordinates
        point_cam = rotation_matrix @ point_3d + translation

        # Avoid division by zero
        if point_cam[2] <= 0:
            return None

        # Perspective projection
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]

        # Apply intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        return np.array([u, v])

    def compute_reprojection_error(self, params, observations, num_cameras, num_points):
        """
        Compute reprojection errors for all observations

        Parameters:
        - params: flattened array of all parameters [cameras..., points...]
        - observations: list of (camera_idx, point_idx, observed_uv)
        - num_cameras: number of cameras
        - num_points: number of 3D points

        Returns:
        - residuals: array of reprojection errors
        """
        residuals = []

        # Extract camera poses and 3D points from parameter vector
        camera_params = params[:num_cameras * 6].reshape(num_cameras, 6)
        points_3d = params[num_cameras * 6:].reshape(num_points, 3)

        for camera_idx, point_idx, observed_uv in observations:
            # Get camera pose (angle-axis + translation)
            cam_params = camera_params[camera_idx]
            angle_axis = cam_params[:3]
            translation = cam_params[3:]

            # Convert angle-axis to rotation matrix
            if np.linalg.norm(angle_axis) < 1e-12:
                rotation_matrix = np.eye(3)
            else:
                rotation_matrix = R.from_rotvec(angle_axis).as_matrix()

            # Get 3D point
            point = points_3d[point_idx]

            # Project point
            projected_uv = self.project_point(rotation_matrix, translation, point)

            if projected_uv is None:
                # Point behind camera, add large residual
                residuals.extend([100.0, 100.0])
            else:
                # Compute reprojection error
                error = projected_uv - observed_uv
                residuals.extend(error.tolist())

        return np.array(residuals)

    def bundle_adjustment(self, initial_points, initial_poses, observations):
        """
        Run bundle adjustment optimization

        Parameters:
        - initial_points: initial 3D points [N, 3]
        - initial_poses: initial camera poses [M, 6] (angle-axis + translation)
        - observations: list of (camera_idx, point_idx, observed_uv)

        Returns:
        - optimized_points: optimized 3D points
        - optimized_poses: optimized camera poses
        - optimization_result: scipy optimization result
        """
        num_cameras = len(initial_poses)
        num_points = len(initial_points)

        # Flatten parameters for optimization
        initial_params = np.concatenate([
            initial_poses.flatten(),
            initial_points.flatten()
        ])

        # Define cost function
        def cost_function(params):
            return self.compute_reprojection_error(params, observations, num_cameras, num_points)

        # Run optimization
        print("Starting bundle adjustment optimization...")
        result = opt.least_squares(
            cost_function,
            initial_params,
            method='lm',  # Levenberg-Marquardt
            verbose=2,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=200
        )

        # Extract optimized parameters
        optimized_params = result.x
        optimized_poses = optimized_params[:num_cameras * 6].reshape(num_cameras, 6)
        optimized_points = optimized_params[num_cameras * 6:].reshape(num_points, 3)

        return optimized_points, optimized_poses, result

    def evaluate_results(self, initial_points, initial_poses, optimized_points, optimized_poses, observations):
        """
        Evaluate optimization results
        """
        num_cameras = len(initial_poses)
        num_points = len(initial_points)

        # Initial errors
        initial_params = np.concatenate([initial_poses.flatten(), initial_points.flatten()])
        initial_errors = self.compute_reprojection_error(initial_params, observations, num_cameras, num_points)

        # Optimized errors
        optimized_params = np.concatenate([optimized_poses.flatten(), optimized_points.flatten()])
        optimized_errors = self.compute_reprojection_error(optimized_params, observations, num_cameras, num_points)

        print(f"\n=== Bundle Adjustment Results ===")
        print(f"Initial RMS error: {np.sqrt(np.mean(initial_errors ** 2)):.6f} pixels")
        print(f"Optimized RMS error: {np.sqrt(np.mean(optimized_errors ** 2)):.6f} pixels")
        err_perc = 100*(1 - np.sqrt(np.mean(optimized_errors ** 2) / np.sqrt(np.mean(initial_errors ** 2))))
        print(f"Error reduction: {err_perc:.2f} % ")

        return initial_errors, optimized_errors


# Utility functions
def create_synthetic_data(num_points=50, num_cameras=5, image_width=640, image_height=480):
    """
    Create synthetic test data for bundle adjustment
    """
    # Camera intrinsics
    intrinsics = {
        'fx': 800.0, 'fy': 800.0,
        'cx': image_width / 2, 'cy': image_height / 2
    }

    # Generate random 3D points
    points_3d = np.random.randn(num_points, 3) * 10.0

    # Generate camera poses in a circle looking at origin
    camera_poses = []
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        radius = 15.0

        # Camera position
        cam_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 5.0])

        # Look at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(np.array([0, 0, 1]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        rotation_matrix = np.column_stack([right, up, forward])
        translation = -rotation_matrix.T @ cam_pos

        # Convert to angle-axis + translation
        angle_axis = R.from_matrix(rotation_matrix).as_rotvec()
        pose = np.concatenate([angle_axis, translation])
        camera_poses.append(pose)

    camera_poses = np.array(camera_poses)

    # Generate observations
    ba = BundleAdjustment(intrinsics)
    observations = []

    for cam_idx in range(num_cameras):
        angle_axis = camera_poses[cam_idx, :3]
        translation = camera_poses[cam_idx, 3:]
        rotation_matrix = R.from_rotvec(angle_axis).as_matrix()

        for point_idx in range(num_points):
            # Project point
            projected = ba.project_point(rotation_matrix, translation, points_3d[point_idx])

            if projected is not None and (0 <= projected[0] < image_width and 0 <= projected[1] < image_height):
                # Add noise
                noisy_uv = projected + np.random.randn(2) * 2.0  # 2 pixel noise
                observations.append((cam_idx, point_idx, noisy_uv))

    return points_3d, camera_poses, observations, intrinsics


def pose_to_transform_matrix(angle_axis, translation):
    """Convert angle-axis + translation to 4x4 transformation matrix"""
    rotation_matrix = R.from_rotvec(angle_axis).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


# Example usage
def main():
    # Create synthetic data
    print("Generating synthetic data...")
    true_points, true_poses, observations, intrinsics = create_synthetic_data(
        num_points=100, num_cameras=4
    )

    # Add noise to initial estimates
    np.random.seed(42)
    initial_points = true_points + np.random.randn(*true_points.shape) * 0.5
    initial_poses = true_poses + np.random.randn(*true_poses.shape) * 0.1

    # Create bundle adjustment optimizer
    ba = BundleAdjustment(intrinsics)

    # Run bundle adjustment
    optimized_points, optimized_poses, result = ba.bundle_adjustment(initial_points, initial_poses, observations)

    # Evaluate results
    initial_errors, optimized_errors = ba.evaluate_results(initial_points, initial_poses, optimized_points, optimized_poses, observations)

    # Visualization
    plt.figure(figsize=(15, 5))

    # Plot 1: 3D points comparison
    plt.subplot(1, 3, 1)
    ax = plt.axes(projection='3d')
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2],
               c='g', marker='o', label='True Points', s=50)
    ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2],
               c='r', marker='^', label='Initial Points', s=30)
    ax.scatter(optimized_points[:, 0], optimized_points[:, 1], optimized_points[:, 2],
               c='b', marker='x', label='Optimized Points', s=40)

    # Plot camera positions
    for i, pose in enumerate(optimized_poses):
        cam_pos = -R.from_rotvec(pose[:3]).as_matrix().T @ pose[3:]
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='m', marker='s', s=100)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'Cam{i}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Reconstruction Comparison')

    # Plot 2: Error distribution before optimization
    plt.subplot(1, 3, 2)
    plt.hist(initial_errors, bins=50, alpha=0.7, color='red', label='Initial Errors')
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution - Before BA')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Error distribution after optimization
    plt.subplot(1, 3, 3)
    plt.hist(optimized_errors, bins=50, alpha=0.7, color='blue', label='Optimized Errors')
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution - After BA')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return optimized_points, optimized_poses, result


if __name__ == "__main__":
    optimized_points, optimized_poses, result = main()