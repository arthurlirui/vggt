import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
import time


class CameraViewPlanner:
    def __init__(self, mesh_path: str):
        """
        Initialize camera view planner with 3D mesh

        Args:
            mesh_path: Path to 3D mesh file (.ply, .obj, .stl, etc.)
        """
        # Load and prepare mesh
        print(f"Loading mesh from {mesh_path}...")
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Compute normals for better rendering
        self.mesh.compute_vertex_normals()
        if not self.mesh.has_vertex_colors():
            self.mesh.paint_uniform_color([0.7, 0.7, 0.7])

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Camera View Planner",
                               width=1280, height=720)

        # Add mesh to visualizer
        self.vis.add_geometry(self.mesh)

        # Camera views storage
        self.camera_views = []  # List of camera parameters
        self.camera_spheres = []  # Visual markers for cameras
        self.view_images = []  # Rendered images

        # Camera intrinsic parameters
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640,  # Image width
            height=480,  # Image height
            fx=525.0,  # Focal length x
            fy=525.0,  # Focal length y
            cx=320.0,  # Principal point x
            cy=240.0  # Principal point y
        )

        # Renderer for image rendering
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)

        # Setup interactive controls
        self.setup_interactive_controls()

        # Get mesh bounds for camera placement
        self.mesh_bounds = self.mesh.get_axis_aligned_bounding_box()
        self.mesh_center = self.mesh_bounds.get_center()
        self.mesh_extent = self.mesh_bounds.get_extent()

        print(f"Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")
        print(f"Mesh bounds: {self.mesh_bounds}")
        print(f"Mesh center: {self.mesh_center}")

    def setup_interactive_controls(self):
        """Setup mouse and keyboard callbacks"""
        # Register mouse click callback
        self.vis.register_key_callback(ord("A"), self.add_camera_at_current_view)
        self.vis.register_key_callback(ord("S"), self.save_camera_views)
        self.vis.register_key_callback(ord("L"), self.load_camera_views)
        self.vis.register_key_callback(ord("R"), self.render_all_views)
        self.vis.register_key_callback(ord("D"), self.delete_last_camera)
        self.vis.register_key_callback(ord("C"), self.clear_all_cameras)
        self.vis.register_key_callback(ord(" "), self.take_screenshot)
        self.vis.register_key_callback(ord("1"), lambda vis: self.move_camera_to_view(0))
        self.vis.register_key_callback(ord("2"), lambda vis: self.move_camera_to_view(1))

        # Mouse click callback setup
        self.vis.register_animation_callback(self.mouse_click_callback)

        print("\n=== Interactive Controls ===")
        print("A: Add camera at current viewpoint")
        print("S: Save camera views to file")
        print("L: Load camera views from file")
        print("R: Render all camera views")
        print("D: Delete last camera")
        print("C: Clear all cameras")
        print("Space: Take screenshot")
        print("1/2: Jump to camera view 1/2")
        print("Mouse Click: Add camera at clicked position")
        print("============================\n")

    def mouse_click_callback(self, vis):
        """Handle mouse clicks to add cameras"""
        # This is a simplified mouse interaction
        # For more precise mouse picking, you'd need to implement ray casting
        pass

    def add_camera_at_current_view(self, vis):
        """Add a camera at the current viewpoint position"""
        # Get current camera parameters from visualizer
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()

        # Extract camera pose (4x4 transformation matrix)
        camera_pose = camera_params.extrinsic

        # Create a unique ID for this camera
        camera_id = len(self.camera_views)

        # Store camera parameters
        camera_data = {
            'id': camera_id,
            'pose': camera_pose.tolist(),
            'intrinsic': {
                'width': self.intrinsic.width,
                'height': self.intrinsic.height,
                'fx': self.intrinsic.intrinsic_matrix[0, 0],
                'fy': self.intrinsic.intrinsic_matrix[1, 1],
                'cx': self.intrinsic.intrinsic_matrix[0, 2],
                'cy': self.intrinsic.intrinsic_matrix[1, 2]
            },
            'position': camera_pose[:3, 3].tolist(),
            'rotation': camera_pose[:3, :3].tolist(),
            'timestamp': time.time()
        }

        self.camera_views.append(camera_data)

        # Add visual marker for the camera
        self.add_camera_marker(camera_pose[:3, 3], camera_id)

        # Render image from this viewpoint
        rendered_image = self.render_from_viewpoint(camera_pose)
        self.view_images.append(rendered_image)

        print(f"Camera {camera_id} added at position {camera_data['position']}")
        print(f"Total cameras: {len(self.camera_views)}")

        # Update visualization
        vis.update_geometry(self.mesh)
        for sphere in self.camera_spheres:
            vis.update_geometry(sphere)

        # Display the rendered image
        self.display_rendered_image(rendered_image, camera_id)

        return True

    def add_camera_marker(self, position: np.ndarray, camera_id: int):
        """Add a visual marker for the camera position"""
        # Create a sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(position)

        # Color code by camera ID
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [1, 0.5, 0],  # Orange
            [0.5, 0, 1],  # Purple
        ]

        color = colors[camera_id % len(colors)]
        sphere.paint_uniform_color(color)

        # Add coordinate axes to show camera orientation
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axis.translate(position)

        # Add to visualizer
        self.vis.add_geometry(sphere)
        self.vis.add_geometry(axis)

        # Store references
        self.camera_spheres.append(sphere)
        self.camera_spheres.append(axis)

    def render_from_viewpoint(self, camera_pose: np.ndarray) -> np.ndarray:
        """Render image from specified camera viewpoint"""
        # Create scene for rendering
        scene = o3d.visualization.rendering.Scene()

        # Add mesh to scene
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        scene.add_geometry("mesh", self.mesh, material)

        # Set camera
        scene.camera.set_projection(
            self.intrinsic.intrinsic_matrix,
            self.intrinsic.width,
            self.intrinsic.height,
            0.1,  # near plane
            100.0  # far plane
        )

        # Set camera pose
        scene.camera.look_at(
            [0, 0, 0],  # target
            camera_pose[:3, 3],  # eye (camera position)
            [0, 1, 0]  # up vector
        )

        # Render image
        image = self.renderer.render_to_image(scene)

        # Convert to numpy array
        img_np = np.asarray(image)

        return img_np

    def display_rendered_image(self, image: np.ndarray, camera_id: int):
        """Display rendered image in a separate window"""
        # Resize for display
        display_img = cv2.resize(image, (640, 480))

        # Add camera ID text
        cv2.putText(display_img, f"Camera {camera_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow(f"Camera View {camera_id}", display_img)
        cv2.waitKey(1)  # Non-blocking display

    def render_all_views(self, vis):
        """Render images from all camera viewpoints"""
        print(f"\nRendering all {len(self.camera_views)} camera views...")

        for i, camera_data in enumerate(self.camera_views):
            camera_pose = np.array(camera_data['pose'])
            rendered_image = self.render_from_viewpoint(camera_pose)

            # Display in grid
            plt.subplot(2, (len(self.camera_views) + 1) // 2, i + 1)
            plt.imshow(rendered_image)
            plt.title(f"Camera {i}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Save all rendered images
        self.save_rendered_images()

        return True

    def save_rendered_images(self):
        """Save all rendered images to disk"""
        import os
        os.makedirs("camera_views", exist_ok=True)

        for i, img in enumerate(self.view_images):
            img_path = f"camera_views/camera_{i:03d}.png"
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved {img_path}")

    def save_camera_views(self, vis, filename: str = "camera_views.json"):
        """Save camera views to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.camera_views, f, indent=2)
        print(f"Saved {len(self.camera_views)} camera views to {filename}")
        return True

    def load_camera_views(self, vis, filename: str = "camera_views.json"):
        """Load camera views from JSON file"""
        try:
            with open(filename, 'r') as f:
                loaded_views = json.load(f)

            # Clear existing cameras
            self.clear_all_cameras(vis)

            # Load new cameras
            for camera_data in loaded_views:
                camera_pose = np.array(camera_data['pose'])

                # Add visual marker
                self.add_camera_marker(camera_pose[:3, 3], camera_data['id'])

                # Render and store image
                rendered_image = self.render_from_viewpoint(camera_pose)
                self.view_images.append(rendered_image)

                # Store camera data
                self.camera_views.append(camera_data)

            print(f"Loaded {len(loaded_views)} camera views from {filename}")

            # Update visualization
            vis.update_geometry(self.mesh)
            for sphere in self.camera_spheres:
                vis.update_geometry(sphere)

        except FileNotFoundError:
            print(f"File {filename} not found")

        return True

    def delete_last_camera(self, vis):
        """Delete the last added camera"""
        if not self.camera_views:
            print("No cameras to delete")
            return False

        # Remove last camera
        camera_data = self.camera_views.pop()
        print(f"Deleted camera {camera_data['id']}")

        # Remove visual markers (last two geometries: sphere + axes)
        if self.camera_spheres:
            sphere = self.camera_spheres.pop()
            vis.remove_geometry(sphere, False)
            axis = self.camera_spheres.pop()
            vis.remove_geometry(axis, False)

        # Remove rendered image
        if self.view_images:
            self.view_images.pop()

        # Update visualization
        vis.update_geometry(self.mesh)

        return True

    def clear_all_cameras(self, vis):
        """Clear all camera views"""
        # Remove all camera spheres
        for sphere in self.camera_spheres:
            vis.remove_geometry(sphere, False)

        # Clear lists
        self.camera_views.clear()
        self.camera_spheres.clear()
        self.view_images.clear()

        print("Cleared all cameras")

        # Update visualization
        vis.update_geometry(self.mesh)

        return True

    def take_screenshot(self, vis):
        """Take a screenshot of the current view"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"

        # Capture screen
        image = vis.capture_screen_float_buffer(True)
        image_np = np.asarray(image) * 255
        image_np = image_np.astype(np.uint8)

        # Save
        cv2.imwrite(filename, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"Screenshot saved to {filename}")

        return True

    def move_camera_to_view(self, camera_idx: int):
        """Move the viewer camera to a specific camera view"""

        def callback(vis):
            if 0 <= camera_idx < len(self.camera_views):
                camera_data = self.camera_views[camera_idx]
                camera_pose = np.array(camera_data['pose'])

                # Set camera parameters
                ctr = vis.get_view_control()
                params = o3d.camera.PinholeCameraParameters()
                params.extrinsic = camera_pose
                params.intrinsic = self.intrinsic

                ctr.convert_from_pinhole_camera_parameters(params)
                print(f"Moved to camera view {camera_idx}")
            return True

        return callback

    def run_interactive(self):
        """Run the interactive visualization"""
        print("\nStarting interactive camera view planning...")
        print("Navigate the 3D view using:")
        print("  - Left mouse button: Rotate")
        print("  - Right mouse button: Pan")
        print("  - Mouse wheel: Zoom")
        print("  - Press 'H' for help in Open3D window")

        # Set initial view
        self.vis.get_view_control().set_zoom(0.8)
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_lookat(self.mesh_center)
        self.vis.get_view_control().set_up([0, 1, 0])

        # Run main loop
        self.vis.run()
        self.vis.destroy_window()

        # Close OpenCV windows
        cv2.destroyAllWindows()

        print("Interactive session ended")

        # Return collected camera views
        return self.camera_views


# Advanced version with more interactive features
class AdvancedCameraPlanner(CameraViewPlanner):
    def __init__(self, mesh_path: str):
        super().__init__(mesh_path)

        # Additional data structures
        self.view_coverage = {}  # Track coverage for each view
        self.view_quality = {}  # Track quality metrics
        self.view_graph = {}  # Graph of view relationships

    def add_camera_at_clicked_position(self, x: int, y: int):
        """Add camera at mouse-clicked position on mesh"""
        # This requires ray casting implementation
        # Placeholder for advanced feature
        print(f"Click at screen coordinates: ({x}, {y})")

        # In practice, you'd implement:
        # 1. Ray casting from screen coordinates
        # 2. Find intersection point with mesh
        # 3. Place camera at that position
        # 4. Orient camera to look at mesh center

    def calculate_view_coverage(self, camera_id: int) -> float:
        """Calculate coverage percentage for a camera view"""
        # This is a simplified calculation
        # In practice, you'd project the mesh and calculate visible area

        rendered_image = self.view_images[camera_id]

        # Simple heuristic: count non-background pixels
        gray = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

        return coverage

    def export_camera_data(self, filename: str = "camera_data.json"):
        """Export comprehensive camera data"""
        camera_data = {
            'mesh_info': {
                'vertices': len(self.mesh.vertices),
                'triangles': len(self.mesh.triangles),
                'bounds': self.mesh_bounds.get_box_points(),
                'center': self.mesh_center.tolist()
            },
            'cameras': self.camera_views,
            'statistics': {
                'num_cameras': len(self.camera_views),
                'total_coverage': self.calculate_total_coverage(),
                'average_quality': self.calculate_average_quality()
            }
        }

        with open(filename, 'w') as f:
            json.dump(camera_data, f, indent=2)

        print(f"Exported camera data to {filename}")

    def calculate_total_coverage(self) -> float:
        """Calculate total coverage of all cameras"""
        # This would require more sophisticated calculation
        # For now, return placeholder
        return min(1.0, len(self.camera_views) * 0.2)


# Example usage
if __name__ == "__main__":
    # Example mesh paths (replace with your own)
    EXAMPLE_MESHES = [
        "bunny.ply",  # Stanford Bunny
        "cube.obj",  # Simple cube
        "sphere.ply",  # Sphere
        # Add your own mesh file
    ]

    # Use the first available mesh
    mesh_path = None
    for mesh in EXAMPLE_MESHES:
        import os

        if os.path.exists(mesh):
            mesh_path = mesh
            break

    if mesh_path is None:
        print("No mesh file found. Please specify your own mesh path.")
        # Create a simple test mesh
        mesh = o3d.geometry.TriangleMesh.create_sphere()
        mesh_path = "test_sphere.ply"
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Created test mesh at {mesh_path}")

    # Create and run camera planner
    planner = CameraViewPlanner(mesh_path)

    # Alternative: Use advanced planner
    # planner = AdvancedCameraPlanner(mesh_path)

    # Run interactive session
    collected_views = planner.run_interactive()

    print(f"\nSession Summary:")
    print(f"Total cameras placed: {len(collected_views)}")
    print(f"Camera positions:")
    for i, view in enumerate(collected_views):
        print(f"  Camera {i}: {view['position']}")

    # Save final camera views
    if collected_views:
        planner.save_camera_views(planner.vis)
        planner.save_rendered_images()