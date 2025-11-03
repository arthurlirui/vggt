import open3d as o3d
import numpy as np
import cupy as cp
from numba import cuda
import time


class Open3DPointCloudRenderer:
    def __init__(self, num_points=1000000):
        self.num_points = num_points
        self.point_cloud = None

    def generate_sphere_point_cloud_gpu(self):
        """Generate points on a sphere surface using CUDA"""
        # Generate random points on CPU first (simpler approach)
        phi = np.random.uniform(0, 2 * np.pi, self.num_points)
        theta = np.random.uniform(0, np.pi, self.num_points)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        points = np.column_stack([x, y, z]) * 5.0  # Sphere radius 5

        # Generate colors based on position
        colors = (points + 5.0) / 10.0  # Normalize to 0-1

        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

    def generate_random_point_cloud_gpu(self):
        """Generate random points using CuPy for GPU computation"""
        # Use CuPy for GPU-accelerated random number generation
        points_gpu = cp.random.uniform(-10, 10, (self.num_points, 3), dtype=cp.float32)
        colors_gpu = cp.random.uniform(0, 1, (self.num_points, 3), dtype=cp.float32)

        # Convert back to numpy for Open3D
        points = cp.asnumpy(points_gpu)
        colors = cp.asnumpy(colors_gpu)

        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

    def visualize(self):
        """Visualize the point cloud with Open3D's GPU-accelerated renderer"""
        if self.point_cloud is None:
            self.generate_sphere_point_cloud_gpu()

        print(f"Rendering {len(self.point_cloud.points)} points...")

        # Set up visualization with better rendering options
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Point Cloud - {self.num_points} points")
        vis.add_geometry(self.point_cloud)

        # Set rendering options for better performance
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.light_on = True

        vis.run()
        vis.destroy_window()


# Usage example
if __name__ == "__main__":
    renderer = Open3DPointCloudRenderer(num_points=500000)
    renderer.visualize()