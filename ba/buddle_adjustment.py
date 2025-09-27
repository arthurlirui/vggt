import numpy as np
import open3d as o3d
import cv2
import os
import sys
from typing import List, Tuple, Dict, Optional
import pycolmap
try:
    import pyceres
except ImportError:
    print("PyCeres 2.4 not installed. Please install it for bundle adjustment.")
    sys.exit(1)


class Camera:
    """Camera class storing intrinsic and extrinsic parameters"""

    def __init__(self, camera_id: int, intrinsic: np.ndarray, extrinsic: np.ndarray):
        self.camera_id = camera_id
        self.intrinsic = intrinsic.copy()  # [fx, fy, cx, cy]
        self.extrinsic = extrinsic.copy()  # 4x4 transformation matrix

        # Convert to parameter vector: [fx, fy, cx, cy, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z]
        rvec = cv2.Rodrigues(extrinsic[:3, :3])[0].flatten()
        tvec = extrinsic[:3, 3].flatten()
        self.param_vector = np.concatenate([intrinsic, rvec, tvec])

    def update_from_params(self, params: np.ndarray):
        """Update camera parameters from optimized vector"""
        self.param_vector = params.copy()
        self.intrinsic = params[0:4]

        # Update extrinsic matrix
        rvec = params[4:7]
        tvec = params[7:10]
        R = cv2.Rodrigues(rvec)[0]
        self.extrinsic = np.eye(4)
        self.extrinsic[:3, :3] = R
        self.extrinsic[:3, 3] = tvec


class Point3D:
    """3D point class"""

    def __init__(self, point_id: int, position: np.ndarray):
        self.point_id = point_id
        self.position = position.copy()  # [x, y, z]
        self.observations = []  # List of (camera_id, observation_2d)

    def add_observation(self, camera_id: int, observation: np.ndarray):
        """Add a 2D observation of this point"""
        self.observations.append((camera_id, observation.copy()))


class ReprojectionCostFunction:
    """Cost function for reprojection error"""

    def __init__(self, observed_x: float, observed_y: float):
        self.observed_x = observed_x
        self.observed_y = observed_y

    def __call__(self, camera_params: List[float], point_3d: List[float], residuals: List[float]) -> bool:
        """
        Compute reprojection error

        Args:
            camera_params: [fx, fy, cx, cy, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z]
            point_3d: [x, y, z]
            residuals: Output residuals [u_residual, v_residual]
        """
        try:
            # Convert to numpy arrays
            cam_params_np = np.array(camera_params, dtype=np.float64)
            point_np = np.array(point_3d, dtype=np.float64)

            # Extract parameters
            fx, fy, cx, cy = cam_params_np[0:4]
            rvec = cam_params_np[4:7]
            tvec = cam_params_np[7:10]

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Transform point to camera coordinates
            point_cam = R @ point_np + tvec

            # Avoid division by zero
            if point_cam[2] <= 1e-8:
                residuals[0] = 0.0
                residuals[1] = 0.0
                return True

            # Project to 2D
            x = point_cam[0] / point_cam[2]
            y = point_cam[1] / point_cam[2]

            # Apply intrinsic parameters
            u = fx * x + cx
            v = fy * y + cy

            # Calculate residuals
            residuals[0] = u - self.observed_x
            residuals[1] = v - self.observed_y

            return True

        except Exception as e:
            print(f"Error in reprojection: {e}")
            residuals[0] = 0.0
            residuals[1] = 0.0
            return True


class BundleAdjustment:
    """Main bundle adjustment class"""

    def __init__(self):
        self.cameras: Dict[int, Camera] = {}
        self.points: Dict[int, Point3D] = {}
        self.problem = None

    def add_camera(self, camera: Camera):
        """Add a camera to the optimization"""
        self.cameras[camera.camera_id] = camera

    def add_point(self, point: Point3D):
        """Add a 3D point to the optimization"""
        self.points[point.point_id] = point

    def setup_problem(self):
        """Set up the Ceres optimization problem"""
        self.problem = pyceres.Problem()

        # Add all camera parameters
        for camera in self.cameras.values():
            # Camera parameters: [fx, fy, cx, cy, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z]
            self.problem.add_parameter_block(np.array(camera.param_vector.tolist()), 10)

            # Add parameter bounds for positive focal length and principal point
            self.problem.set_parameter_lower_bound(np.array(camera.param_vector.tolist()), 0, 1.0)  # fx > 0
            self.problem.set_parameter_lower_bound(np.array(camera.param_vector.tolist()), 1, 1.0)  # fy > 0

        # Add all 3D points
        for point in self.points.values():
            self.problem.add_parameter_block(np.array(point.position.tolist()), 3)

        # Add residual blocks for all observations
        for point in self.points.values():
            for camera_id, observation in point.observations:
                if camera_id not in self.cameras:
                    continue

                camera = self.cameras[camera_id]

                # Create cost function
                cost_function = pyceres.CreateCostFunction(
                    ReprojectionCostFunction(observation[0], observation[1]),
                    2,  # Number of residuals
                    10,  # Size of camera parameter block
                    3  # Size of point parameter block
                )



                # Add robust loss function (Huber)
                loss_dict = {"type": "huber", "threshold": 1.0}

                # Add residual block
                self.problem.AddResidualBlock(
                    cost_function,
                    loss_dict,
                    camera.param_vector.tolist(),
                    point.position.tolist()
                )

    def optimize(self, max_iterations: int = 50, verbose: bool = True):
        """Run the optimization"""
        if self.problem is None:
            self.setup_problem()

        # Set solver options
        options = pyceres.SolverOptions()
        options.max_num_iterations = max_iterations
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
        options.minimizer_progress_to_stdout = verbose
        options.function_tolerance = 1e-6
        options.gradient_tolerance = 1e-10
        options.parameter_tolerance = 1e-8

        # Solve the problem
        summary = pyceres.SolverSummary()
        pyceres.Solve(options, self.problem, summary)

        if verbose:
            print(summary.BriefReport())

        # Update camera and point parameters from optimized values
        self._update_parameters()

        return summary.IsSolutionUsable()

    def _update_parameters(self):
        """Update camera and point objects from optimized parameters"""
        for camera in self.cameras.values():
            camera.update_from_params(np.array(camera.param_vector))

        # Points are updated automatically since we used their position arrays directly

    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get the optimized point cloud as Open3D object"""
        points = [point.position for point in self.points.values()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def save_point_cloud(self, filename: str):
        """Save the point cloud to file"""
        pcd = self.get_point_cloud()
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud saved to {filename}")

    def get_reprojection_errors(self) -> List[float]:
        """Calculate reprojection errors for all observations"""
        errors = []
        for point in self.points.values():
            for camera_id, observation in point.observations:
                if camera_id not in self.cameras:
                    continue

                camera = self.cameras[camera_id]
                projected = self.project_point(camera.param_vector, point.position)

                if projected is not None:
                    error = np.linalg.norm(projected - observation)
                    errors.append(error)

        return errors

    @staticmethod
    def project_point(camera_params: np.ndarray, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """Project a 3D point to 2D using camera parameters"""
        fx, fy, cx, cy = camera_params[0:4]
        rvec = camera_params[4:7]
        tvec = camera_params[7:10]

        R, _ = cv2.Rodrigues(rvec)
        point_cam = R @ point_3d + tvec

        if point_cam[2] <= 1e-8:
            return None

        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]

        u = fx * x + cx
        v = fy * y + cy

        return np.array([u, v])


def create_sample_data() -> BundleAdjustment:
    """Create sample data for testing"""
    ba = BundleAdjustment()

    # Create cameras
    intrinsic = np.array([800.0, 800.0, 320.0, 240.0])

    # Camera 1 at origin
    extrinsic1 = np.eye(4)
    extrinsic1[:3, 3] = [0, 0, 0]
    cam1 = Camera(0, intrinsic, extrinsic1)
    ba.add_camera(cam1)

    # Camera 2 moved along x-axis
    extrinsic2 = np.eye(4)
    extrinsic2[:3, 3] = [1.0, 0.5, 0.2]
    extrinsic2[:3, :3] = cv2.Rodrigues(np.array([0.1, 0.05, 0.02]))[0]
    cam2 = Camera(1, intrinsic, extrinsic2)
    ba.add_camera(cam2)

    # Camera 3 moved along y-axis
    extrinsic3 = np.eye(4)
    extrinsic3[:3, 3] = [-0.5, 1.0, 0.3]
    extrinsic3[:3, :3] = cv2.Rodrigues(np.array([-0.05, 0.1, 0.01]))[0]
    cam3 = Camera(2, intrinsic, extrinsic3)
    ba.add_camera(cam3)

    # Create 3D points
    points_3d = [
        np.array([0, 0, 5], np.float64),
        np.array([1, 0, 5], np.float64),
        np.array([0, 1, 5], np.float64),
        np.array([1, 1, 5], np.float64),
        np.array([0.5, 0.5, 6], np.float64),
        np.array([-0.5, 0.5, 4], np.float64),
        np.array([0.3, -0.2, 7], np.float64),
        np.array([-0.3, -0.4, 5], np.float64),
    ]

    for i, pos in enumerate(points_3d):
        point = Point3D(i, pos)
        ba.add_point(point)

    # Create observations with noise
    np.random.seed(42)  # For reproducible results
    for point in ba.points.values():
        for camera in ba.cameras.values():
            projected = BundleAdjustment.project_point(camera.param_vector, point.position)
            if projected is not None:
                # Add Gaussian noise
                noisy_obs = projected + np.random.normal(0, 0.5, 2)
                point.add_observation(camera.camera_id, noisy_obs)

    return ba


def main():
    """Main function demonstrating bundle adjustment"""
    print("Creating sample data...")
    ba = create_sample_data()

    # Print initial statistics
    print(f"Number of cameras: {len(ba.cameras)}")
    print(f"Number of points: {len(ba.points)}")

    total_observations = sum(len(point.observations) for point in ba.points.values())
    print(f"Total observations: {total_observations}")

    # Calculate initial reprojection errors
    initial_errors = ba.get_reprojection_errors()
    print(f"Initial mean reprojection error: {np.mean(initial_errors):.4f} pixels")
    print(f"Initial max reprojection error: {np.max(initial_errors):.4f} pixels")

    # Run bundle adjustment
    print("\nRunning bundle adjustment...")
    success = ba.optimize(max_iterations=50, verbose=True)

    if success:
        print("Optimization successful!")

        # Calculate final reprojection errors
        final_errors = ba.get_reprojection_errors()
        print(f"Final mean reprojection error: {np.mean(final_errors):.4f} pixels")
        print(f"Final max reprojection error: {np.max(final_errors):.4f} pixels")

        # Save results
        ba.save_point_cloud("optimized_point_cloud.ply")

        # Print some camera parameter changes
        print("\nCamera parameter changes:")
        for camera_id, camera in ba.cameras.items():
            print(f"Camera {camera_id}:")
            print(f"  Focal length: {camera.intrinsic[0]:.2f}, {camera.intrinsic[1]:.2f}")
            print(f"  Principal point: {camera.intrinsic[2]:.2f}, {camera.intrinsic[3]:.2f}")
            print(f"  Position: {camera.extrinsic[:3, 3]}")

    else:
        print("Optimization failed!")


if __name__ == "__main__":
    main()