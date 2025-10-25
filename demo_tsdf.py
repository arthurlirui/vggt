import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import open3d as o3d
from tqdm import tqdm
import cv2


class TSDFFusion:
    def __init__(self, volume_size, voxel_size, trunc_margin=5.0):
        """
        Initialize TSDF volume

        Args:
            volume_size: Physical size of the volume in meters (x, y, z)
            voxel_size: Size of each voxel in meters
            trunc_margin: Truncation margin in voxel units
        """
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin * voxel_size

        # Calculate volume dimensions
        self.dims = (np.array(volume_size) / voxel_size).astype(int)
        print(f"TSDF volume dimensions: {self.dims}")

        # Initialize volumes
        self.tsdf_volume = np.ones(self.dims)  # Start with +1 (empty space)
        self.weight_volume = np.zeros(self.dims)

        # Compute world coordinates
        self._setup_world_coordinates(volume_size)

    def _setup_world_coordinates(self, volume_size):
        """Setup world coordinates for each voxel"""
        # Create voxel grid in volume coordinates
        x_range = np.linspace(-volume_size[0] / 2, volume_size[0] / 2, self.dims[0])
        y_range = np.linspace(-volume_size[1] / 2, volume_size[1] / 2, self.dims[1])
        z_range = np.linspace(-volume_size[2] / 2, volume_size[2] / 2, self.dims[2])

        self.x_coords, self.y_coords, self.z_coords = np.meshgrid(
            x_range, y_range, z_range, indexing='ij'
        )

        # Stack into a 4D array (i, j, k, 3) for efficient transformation
        self.voxel_points = np.stack([self.x_coords, self.y_coords, self.z_coords], axis=-1)

    def integrate(self, depth_map, intrinsic, camera_pose):
        """
        Integrate a single depth map into the TSDF volume

        Args:
            depth_map: 2D depth image (meters)
            intrinsic: Camera intrinsic matrix (3x3)
            camera_pose: Camera to world transformation (4x4)
        """
        height, width = depth_map.shape

        # Transform voxels to camera coordinates
        voxel_points_flat = self.voxel_points.reshape(-1, 3)
        voxel_points_homo = np.column_stack([voxel_points_flat, np.ones(len(voxel_points_flat))])

        # World to camera transform
        camera_to_world = camera_pose
        world_to_camera = np.linalg.inv(camera_to_world)

        # Transform to camera coordinates
        points_cam_homo = (world_to_camera @ voxel_points_homo.T).T
        points_cam = points_cam_homo[:, :3] / points_cam_homo[:, 3:4]

        # Project to image plane
        points_image_homo = (intrinsic @ points_cam.T).T
        points_image = points_image_homo[:, :2] / points_image_homo[:, 2:3]

        u = points_image[:, 0].astype(int)
        v = points_image[:, 1].astype(int)

        # Reshape back to volume shape
        u = u.reshape(self.dims)
        v = v.reshape(self.dims)
        z_cam = points_cam[:, 2].reshape(self.dims)

        # Create valid mask
        valid_mask = (
                (u >= 0) & (u < width) &
                (v >= 0) & (v < height) &
                (z_cam > 0)  # In front of camera
        )

        # Get measured depth values
        D_measured = np.full(self.dims, np.inf)
        D_measured[valid_mask] = depth_map[v[valid_mask], u[valid_mask]]

        # Additional validity check - remove points where depth is 0 or invalid
        valid_depth_mask = (D_measured > 0) & (D_measured < np.inf) & valid_mask

        # Calculate SDF
        sdf = D_measured - z_cam

        # Truncate SDF
        tsdf_new = np.clip(sdf / self.trunc_margin, -1, 1)

        # Update only valid voxels
        update_mask = valid_depth_mask & (np.abs(sdf) < self.trunc_margin)

        # Simple weighting (you can make this more sophisticated)
        w_new = 1.0

        # Running average update
        W_old = self.weight_volume[update_mask]
        TSDF_old = self.tsdf_volume[update_mask]

        W_new = W_old + w_new
        TSDF_new = (TSDF_old * W_old + tsdf_new[update_mask] * w_new) / W_new

        self.tsdf_volume[update_mask] = TSDF_new
        self.weight_volume[update_mask] = W_new

    def extract_mesh(self, level=0):
        """Extract mesh using marching cubes"""
        # Use marching cubes to extract the surface
        verts, faces, normals, _ = measure.marching_cubes(
            self.tsdf_volume, level=level, spacing=(self.voxel_size,) * 3
        )

        # Transform vertices to world coordinates
        verts_world = verts + self.voxel_points[0, 0, 0]  # Add volume offset

        return verts_world, faces, normals

    def render_depth_from_pose(self, intrinsic, camera_pose, image_size):
        """
        Render depth map from a novel viewpoint by raycasting
        """
        height, width = image_size

        # Generate ray directions in camera space
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        uv_homo = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)

        # Convert to camera rays (inverse projection)
        intrinsic_inv = np.linalg.inv(intrinsic)
        camera_rays = (intrinsic_inv @ uv_homo.T).T
        camera_rays = np.column_stack([camera_rays, np.ones(len(camera_rays))])

        # Transform rays to world coordinates
        world_to_camera = np.linalg.inv(camera_pose)

        # Raycast for each pixel
        depth_map = np.full((height, width), np.inf)

        # Simple raycasting implementation
        for i in tqdm(range(height), desc="Raycasting"):
            for j in range(width):
                idx = i * width + j
                ray_dir_cam = camera_rays[idx, :3]
                ray_dir_world = (camera_pose @ np.append(ray_dir_cam, 0))[:3]
                ray_origin_world = camera_pose[:3, 3]

                # Find intersection with TSDF volume
                depth = self.raycast(ray_origin_world, ray_dir_world)
                if depth is not None:
                    depth_map[i, j] = depth

        return depth_map

    def raycast(self, ray_origin, ray_dir, max_distance=10.0, step_size=None):
        """Simple raycasting through TSDF volume"""
        if step_size is None:
            step_size = self.voxel_size * 0.5

        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        current_pos = ray_origin.copy()
        distance = 0.0

        while distance < max_distance:
            # Check if current position is inside volume
            if not self.is_inside_volume(current_pos):
                break

            # Get TSDF value at current position
            tsdf_val = self.get_tsdf_at_point(current_pos)

            # If we found a surface (TSDF crosses zero)
            if tsdf_val <= 0:
                return distance

            # March forward
            current_pos += ray_dir * step_size
            distance += step_size

        return None

    def is_inside_volume(self, point):
        """Check if a point is inside the TSDF volume"""
        volume_min = self.voxel_points[0, 0, 0]
        volume_max = self.voxel_points[-1, -1, -1]

        return (point >= volume_min).all() and (point <= volume_max).all()

    def get_tsdf_at_point(self, point):
        """Get TSDF value at arbitrary point using trilinear interpolation"""
        # Convert world point to volume indices
        volume_min = self.voxel_points[0, 0, 0]
        indices = (point - volume_min) / self.voxel_size

        # Trilinear interpolation
        if (indices < 0).any() or (indices >= np.array(self.dims) - 1).any():
            return 1.0  # Outside volume

        x0, y0, z0 = np.floor(indices).astype(int)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # Get the 8 surrounding voxel values
        c000 = self.tsdf_volume[x0, y0, z0]
        c001 = self.tsdf_volume[x0, y0, z1]
        c010 = self.tsdf_volume[x0, y1, z0]
        c011 = self.tsdf_volume[x0, y1, z1]
        c100 = self.tsdf_volume[x1, y0, z0]
        c101 = self.tsdf_volume[x1, y0, z1]
        c110 = self.tsdf_volume[x1, y1, z0]
        c111 = self.tsdf_volume[x1, y1, z1]

        # Interpolation weights
        xd, yd, zd = indices - np.floor(indices)

        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        result = c0 * (1 - zd) + c1 * zd
        return result


def create_sample_data():
    """Create synthetic depth maps and camera poses for testing"""
    num_views = 8
    depth_maps = []
    camera_poses = []

    # Camera intrinsic matrix (assuming 640x480 resolution)
    intrinsic = np.array([
        [525.0, 0, 319.5],
        [0, 525.0, 239.5],
        [0, 0, 1]
    ])

    # Create a simple sphere-like object
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views

        # Camera pose - orbiting around the object
        camera_distance = 2.0
        camera_pose = np.eye(4)
        camera_pose[0, 3] = camera_distance * np.sin(angle)  # X
        camera_pose[1, 3] = 0.0  # Y
        camera_pose[2, 3] = camera_distance * np.cos(angle)  # Z

        # Look at origin
        forward = -camera_pose[:3, 3]
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward

        camera_poses.append(camera_pose)

        # Create synthetic depth map (sphere)
        depth_map = create_sphere_depth_map(intrinsic, camera_pose)
        depth_maps.append(depth_map)

    return depth_maps, camera_poses, intrinsic


def create_sphere_depth_map(intrinsic, camera_pose, sphere_radius=0.5):
    """Create a synthetic depth map of a sphere"""
    height, width = 480, 640
    depth_map = np.full((height, width), np.inf)

    # Sphere center at origin
    sphere_center = np.array([0, 0, 0])

    # Camera parameters
    camera_pos = camera_pose[:3, 3]
    R = camera_pose[:3, :3]

    for v in range(height):
        for u in range(width):
            # Generate ray
            pixel_pos = np.array([u, v, 1.0])
            ray_dir_camera = np.linalg.inv(intrinsic) @ pixel_pos
            ray_dir_world = R @ ray_dir_camera
            ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

            # Ray-sphere intersection
            oc = camera_pos - sphere_center
            a = np.dot(ray_dir_world, ray_dir_world)
            b = 2.0 * np.dot(oc, ray_dir_world)
            c = np.dot(oc, oc) - sphere_radius ** 2
            discriminant = b ** 2 - 4 * a * c

            if discriminant > 0:
                t = (-b - np.sqrt(discriminant)) / (2.0 * a)
                if t > 0:
                    depth_map[v, u] = t

    return depth_map


def visualize_results(tsdf_fusion, depth_maps, rendered_views=None):
    """Visualize the TSDF fusion results"""

    # Extract and visualize mesh
    print("Extracting mesh...")
    verts, faces, normals = tsdf_fusion.extract_mesh()

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_vertex_normals()

    # Visualize mesh
    print("Visualizing reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh])

    # Visualize TSDF slice
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    mid_z = tsdf_fusion.dims[2] // 2
    plt.imshow(tsdf_fusion.tsdf_volume[:, :, mid_z].T, cmap='coolwarm', origin='lower')
    plt.colorbar(label='TSDF Value')
    plt.title('TSDF Slice (XY plane)')

    plt.subplot(132)
    plt.imshow(tsdf_fusion.weight_volume[:, :, mid_z].T, cmap='hot', origin='lower')
    plt.colorbar(label='Weight')
    plt.title('Weight Slice (XY plane)')

    plt.subplot(133)
    if rendered_views:
        plt.imshow(rendered_views[0], cmap='plasma')
        plt.colorbar(label='Depth')
        plt.title('Rendered Depth View')

    plt.tight_layout()
    plt.show()

    return mesh


def main():
    """Main demonstration function"""
    print("TSDF Fusion System")
    print("=" * 50)

    # Create sample data
    print("Generating sample data...")
    depth_maps, camera_poses, intrinsic = create_sample_data()

    # Initialize TSDF volume
    volume_size = [2.0, 2.0, 2.0]  # 2m x 2m x 2m volume
    voxel_size = 0.02  # 2cm voxels

    tsdf_fusion = TSDFFusion(volume_size, voxel_size, trunc_margin=3.0)

    # Integrate all depth maps
    print("Integrating depth maps...")
    for i, (depth_map, camera_pose) in enumerate(zip(depth_maps, camera_poses)):
        print(f"Integrating view {i + 1}/{len(depth_maps)}")
        tsdf_fusion.integrate(depth_map, intrinsic, camera_pose)

    # Render novel views
    print("Rendering novel views...")
    rendered_views = []
    for i in range(2):  # Render 2 novel views
        # Create novel camera pose
        novel_pose = np.eye(4)
        angle = np.pi / 4 * i
        novel_pose[0, 3] = 1.5 * np.sin(angle)
        novel_pose[2, 3] = 1.5 * np.cos(angle)

        rendered_depth = tsdf_fusion.render_depth_from_pose(
            intrinsic, novel_pose, (480, 640)
        )
        rendered_views.append(rendered_depth)

    # Visualize results
    mesh = visualize_results(tsdf_fusion, depth_maps, rendered_views)

    # Save mesh
    o3d.io.write_triangle_mesh("reconstructed_mesh.ply", mesh)
    print("Mesh saved as 'reconstructed_mesh.ply'")


if __name__ == "__main__":
    main()