"""
Multi-RGBD VGGT Pipeline Class

This class provides a unified interface for:
1. Reading multi-camera RGBD video streams
2. Running VGGT model inference for camera pose estimation
3. Generating point clouds from RGBD data
4. Visualizing results with Open3D with performance statistics
"""

import cv2
import numpy as np
import torch
import open3d as o3d
import threading
import time
from queue import Queue
from typing import List, Optional, Tuple
from pyorbbecsdk import (
    Pipeline, Config, Context, FrameSet, 
    OBFormat, OBSensorType, VideoStreamProfile, OBError
)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import utils


class MultiRGBDVGGTPipeline:
    """Multi-camera RGBD pipeline with VGGT model and Open3D visualization"""
    
    def __init__(
        self,
        max_devices: int = 5,
        max_queue_size: int = 6,
        conf_threshold: float = 1.1,
        use_gpu_render: bool = False,
        model_path: str = "model.pt",
        depth_scale: float = 0.001,
        depth_trunc: float = 3.0,
        depth_cutoff: float = 0.1
    ):
        """
        Initialize the pipeline.
        
        Args:
            max_devices: Maximum number of camera devices
            max_queue_size: Maximum queue size for frame buffering
            conf_threshold: Confidence threshold for point cloud filtering
            use_gpu_render: Whether to use GPU for rendering
            model_path: Path to the VGGT model checkpoint
            depth_scale: Scale factor for depth values
            depth_trunc: Maximum depth value for point cloud generation
            depth_cutoff: Minimum depth value for point cloud generation
        """
        # Device parameters
        self.max_devices = max_devices
        self.max_queue_size = max_queue_size
        self.curr_device_cnt = 0
        
        # Depth parameters for point cloud generation
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.depth_cutoff = depth_cutoff
        
        # Frame queues
        self.color_frames_queue: List[Queue] = [Queue() for _ in range(max_devices)]
        self.depth_frames_queue: List[Queue] = [Queue() for _ in range(max_devices)]
        self.has_color_sensor: List[bool] = [False for _ in range(max_devices)]
        
        # Camera pipelines and configs
        self.pipelines: List[Pipeline] = []
        self.configs: List[Config] = []
        
        # VGGT model parameters
        self.conf_threshold = conf_threshold
        self.use_gpu_render = use_gpu_render
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[VGGT] = None
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # Open3D visualization
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.is_initialized = False
        
        # Control flags
        self.stop_rendering = False
        self.start_record = False
        
        # ESC key code
        self.ESC_KEY = 27
        
        print(f"Initialized MultiRGBDVGGTPipeline on device: {self.device}")
    
    def init_cameras(self) -> bool:
        """
        Initialize camera devices and create pipelines.
        
        Returns:
            True if successful, False otherwise
        """
        print("Initializing cameras...")
        ctx = Context()
        device_list = ctx.query_devices()
        self.curr_device_cnt = device_list.get_count()
        
        if self.curr_device_cnt == 0:
            print("No camera devices connected")
            return False
        if self.curr_device_cnt > self.max_devices:
            print(f"Too many devices connected. Maximum: {self.max_devices}")
            return False
        
        print(f"Found {self.curr_device_cnt} camera(s)")
        
        for i in range(device_list.get_count()):
            camera = device_list.get_device_by_index(i)
            pipeline = Pipeline(camera)
            config = Config()
            
            # Try to enable color sensor
            try:
                profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                self.has_color_sensor[i] = True
                print(f"Device {i}: Color sensor enabled")
            except OBError as e:
                print(f"Device {i}: No color sensor - {e}")
                self.has_color_sensor[i] = False
            
            # Try to enable depth sensor
            try:
                profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(depth_profile)
                print(f"Device {i}: Depth sensor enabled")
            except OBError as e:
                print(f"Device {i}: No depth sensor - {e}")
            
            self.pipelines.append(pipeline)
            self.configs.append(config)
        
        return True
    
    def init_vggt_model(self) -> bool:
        """
        Initialize VGGT model.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Loading VGGT model from {self.model_path}...")
        try:
            self.model = VGGT()
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model = self.model.to(self.device)
            print("VGGT model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load VGGT model: {e}")
            return False
    
    def init_open3d_visualizer(
        self,
        window_name: str = '3D Scene - VGGT Point Cloud',
        width: int = 1080,
        height: int = 1080,
        top: int = 200,
        left: int = 200
    ) -> bool:
        """
        Initialize Open3D visualizer.
        
        Args:
            window_name: Window title
            width: Window width
            height: Window height
            top: Window top position
            left: Window left position
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=window_name,
                height=height,
                width=width,
                top=top,
                left=left
            )
            self.pcd = o3d.geometry.PointCloud()
            self.is_initialized = False
            print("Open3D visualizer initialized")
            return True
        except Exception as e:
            print(f"Failed to initialize Open3D visualizer: {e}")
            return False
    
    def _on_new_frame_callback(self, frames: FrameSet, index: int):
        """Callback for new camera frames"""
        if index >= self.max_devices:
            return
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if color_frame is not None:
            if self.color_frames_queue[index].qsize() >= self.max_queue_size:
                self.color_frames_queue[index].get()
            self.color_frames_queue[index].put(color_frame)
        
        if depth_frame is not None:
            if self.depth_frames_queue[index].qsize() >= self.max_queue_size:
                self.depth_frames_queue[index].get()
            self.depth_frames_queue[index].put(depth_frame)
    
    def start_cameras(self):
        """Start camera streams"""
        print("Starting camera streams...")
        for i, (pipeline, config) in enumerate(zip(self.pipelines, self.configs)):
            print(f"Starting device {i}")
            pipeline.start(
                config,
                lambda frame_set, curr_index=i: self._on_new_frame_callback(frame_set, curr_index)
            )
    
    def stop_cameras(self):
        """Stop camera streams"""
        print("Stopping camera streams...")
        for pipeline in self.pipelines:
            pipeline.stop()
    
    def capture_frames(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Capture RGBD frames from all cameras.
        
        Returns:
            Tuple of (color_images, depth_images)
        """
        color_images = [None for _ in range(self.curr_device_cnt)]
        depth_images = [None for _ in range(self.curr_device_cnt)]
        
        while not all(img is not None for img in color_images) or not all(depth is not None for depth in depth_images):
            for i in range(self.curr_device_cnt):
                color_frame = None
                depth_frame = None
                
                if not self.color_frames_queue[i].empty():
                    color_frame = self.color_frames_queue[i].get()
                if not self.depth_frames_queue[i].empty():
                    depth_frame = self.depth_frames_queue[i].get()
                
                if color_frame is None and depth_frame is None:
                    continue
                
                # Process color image
                if color_frame is not None:
                    color_images[i] = utils.frame_to_bgr_image(color_frame)
                
                # Process depth image
                if depth_frame is not None:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    height = depth_frame.get_height()
                    width = depth_frame.get_width()
                    depth_data = depth_data.reshape((height, width))
                    scale = depth_frame.get_depth_scale()
                    depth_float = depth_data.astype(np.float32) * scale
                    depth_images[i] = depth_float
        
        return color_images, depth_images
    
    def process_frames(self, images: List[np.ndarray]) -> dict:
        """
        Process captured frames with VGGT model.
        
        Args:
            images: List of captured color images
            
        Returns:
            Dictionary containing predictions
        """
        if self.model is None:
            raise RuntimeError("VGGT model not initialized")
        
        # Preprocess images
        images_tensor = load_and_preprocess_images(image_path_list=images, mode="crop")
        images_tensor = images_tensor.to(self.device)
        
        # Run inference
        print("Running VGGT inference...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                start_time = time.time()
                predictions = self.model(images_tensor)
                inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.3f}s")
        
        # Convert pose encoding to extrinsic and intrinsic
        print("Converting pose encodings...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            images_tensor.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # Convert tensors to numpy/cupy
        print("Processing model outputs...")
        start_time = time.time()
        if self.use_gpu_render:
            import cupy as cp
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = cp.asarray(predictions[key]).squeeze(0)
        else:
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        conversion_time = time.time() - start_time
        print(f"Conversion time: {conversion_time:.3f}s")
        
        return predictions
    
    def rgbd_to_pointcloud(
        self,
        color_images: List[np.ndarray],
        depth_images: List[np.ndarray],
        intrinsics: np.ndarray,
        extrinsics: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert RGBD images to point cloud using camera intrinsics and extrinsics.
        
        Args:
            color_images: List of color images [H, W, 3]
            depth_images: List of depth images [H, W]
            intrinsics: Camera intrinsics (S, 3, 3)
            extrinsics: Camera extrinsics (S, 3, 4)
            
        Returns:
            Tuple of (points, colors) as numpy arrays
        """
        all_points = []
        all_colors = []
        
        for i, (color_img, depth_img) in enumerate(zip(color_images, depth_images)):
            H, W = depth_img.shape
            
            # Get camera parameters
            fx = intrinsics[i, 0, 0]
            fy = intrinsics[i, 1, 1]
            cx = intrinsics[i, 0, 2]
            cy = intrinsics[i, 1, 2]
            
            # Create pixel coordinates
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u = u.reshape(-1)
            v = v.reshape(-1)
            
            # Get depth values
            depth = depth_img.reshape(-1) * self.depth_scale
            
            # Filter valid depth values
            valid_mask = (depth > self.depth_cutoff) & (depth < self.depth_trunc)
            
            if not np.any(valid_mask):
                continue
            
            # Project depth to 3D points in camera frame
            z_c = depth[valid_mask]
            x_c = (u[valid_mask] - cx) * z_c / fx
            y_c = (v[valid_mask] - cy) * z_c / fy
            
            # Transform to world frame
            extrinsic = extrinsics[i]  # (3, 4)
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            
            points_camera = np.stack([x_c, y_c, z_c], axis=1)  # (N, 3)
            points_world = (rotation @ points_camera.T).T + translation
            
            # Get colors
            colors = color_img.reshape(-1, 3)[valid_mask]
            
            all_points.append(points_world)
            all_colors.append(colors)
        
        # Concatenate all points
        points = np.vstack(all_points) if all_points else np.empty((0, 3))
        colors = np.vstack(all_colors) if all_colors else np.empty((0, 3))
        
        return points, colors

    def extract_point_cloud(self, predictions: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract point cloud from predictions.
        
        Args:
            predictions: Dictionary containing model predictions
            
        Returns:
            Tuple of (points, colors)
        """
        ptSHW = predictions['world_points']  # (S, H, W, 3)
        colorSHW = predictions['images']  # (S, 3, H, W)
        confSHW = predictions['world_points_conf']  # (S, H, W)
        
        # Convert color format
        colorSHW = colorSHW.transpose(0, 2, 3, 1)  # (S, H, W, 3)
        colorSHW = colorSHW[..., ::-1]  # RGB to BGR
        
        print(f"Point cloud shape: {ptSHW.shape}")
        print(f"Color shape: {colorSHW.shape}")
        
        # Flatten arrays
        ptN3 = ptSHW.reshape(-1, 3)
        colorN3 = colorSHW.reshape(-1, 3)
        confN = confSHW.reshape(-1, )
        
        # Apply confidence threshold
        mask = confN > self.conf_threshold
        
        if self.use_gpu_render:
            import cupy as cp
            points = cp.asarray(ptN3[mask, :])
            colors = cp.asarray(colorN3[mask, :])
            points = cp.asnumpy(points)
            colors = cp.asnumpy(colors)
        else:
            points = np.asarray(ptN3[mask, :], dtype=np.float64)
            colors = np.asarray(colorN3[mask, :], dtype=np.float64)
        
        return points, colors
    
    def visualize_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """
        Visualize point cloud with Open3D.
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
        """
        if self.vis is None:
            print("Open3D visualizer not initialized")
            return
        
        # Validate input shapes
        if points.shape[0] == 0:
            print("No valid points to visualize")
            return
        
        if points.shape[1] != 3:
            raise ValueError("Points must be an Nx3 array")
        if colors.shape[1] != 3:
            raise ValueError("Colors must be an Nx3 array")
        if points.shape[0] != colors.shape[0]:
            raise ValueError("Points and colors must have the same number of points")
        
        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Add or update geometry
        if not self.is_initialized:
            self.vis.add_geometry(self.pcd)
            self.is_initialized = True
        else:
            self.vis.update_geometry(self.pcd)
        
        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def display_images(self, predictions: dict):
        """
        Display color and depth images from predictions.
        
        Args:
            predictions: Dictionary containing model predictions
        """
        colorSHW = predictions['images']  # (S, 3, H, W)
        depthSHW = predictions['depth']  # (S, H, W, 1)
        
        colorSHW = colorSHW.transpose(0, 2, 3, 1)  # (S, H, W, 3)
        
        for i in range(min(self.curr_device_cnt, colorSHW.shape[0])):
            # Display color image
            img = colorSHW[i, :, :, 0:3] * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(f'Color-{i}', img)
            
            # Display depth colormap
            depth = depthSHW[i, :, :, 0]
            
            # Normalize depth for visualization
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                min_val = np.min(valid_depth)
                max_val = np.max(valid_depth)
                if max_val > min_val:
                    depth_normalized = cv2.convertScaleAbs(
                        depth, 
                        alpha=255.0 / (max_val - min_val), 
                        beta=-min_val * 255.0 / (max_val - min_val)
                    )
                else:
                    depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imshow(f'Depth-Colormap-{i}', depth_colormap)
    
    def run(self, use_rgbd_pointcloud: bool = True):
        """
        Main run loop.
        
        Args:
            use_rgbd_pointcloud: If True, use RGBD to generate point cloud directly.
                                 If False, use VGGT predicted point cloud.
        """
        mode_str = "RGBD" if use_rgbd_pointcloud else "VGGT"
        print(f"Starting main loop in {mode_str} mode...")
        print("Press 'q' to quit, 'v' to start recording")
        
        index = 0
        saved_data = []
        
        try:
            self.start_cameras()
            
            while not self.stop_rendering:
                frame_start_time = time.time()
                
                # Capture RGBD frames
                color_images, depth_images = self.capture_frames()
                
                if len(color_images) < self.curr_device_cnt or len(depth_images) < self.curr_device_cnt:
                    continue
                
                capture_time = time.time() - frame_start_time
                print(f"\nFrame {index}: Captured {len(color_images)} RGBD pairs")
                
                # Process frames (get camera parameters)
                predictions = self.process_frames(color_images)
                process_time = time.time() - frame_start_time - capture_time
                
                # Extract camera parameters
                #extrinsics = predictions['extrinsic'].cpu().numpy().squeeze(0)  # (S, 3, 4)
                #intrinsics = predictions['intrinsic'].cpu().numpy().squeeze(0)  # (S, 3, 3)
                extrinsics = predictions['extrinsic']  # (S, 3, 4)
                intrinsics = predictions['intrinsic']  # (S, 3, 3)

                # Generate point cloud
                pc_generation_start = time.time()
                if use_rgbd_pointcloud:
                    # Use RGBD to generate point cloud directly
                    points, colors = self.rgbd_to_pointcloud(
                        color_images, 
                        depth_images, 
                        intrinsics, 
                        extrinsics
                    )
                    source = "RGBD"
                else:
                    # Use VGGT predicted point cloud
                    points, colors = self.extract_point_cloud(predictions)
                    source = "VGGT"
                
                pc_generation_time = time.time() - pc_generation_start
                
                # Calculate statistics
                num_points = points.shape[0]
                points_memory_mb = (points.nbytes + colors.nbytes) / (1024 * 1024)
                
                # Visualize point cloud
                render_start = time.time()
                self.visualize_point_cloud(points, colors)
                render_time = time.time() - render_start
                
                # Display individual images
                self.display_images(predictions)
                
                # Print performance statistics
                total_time = time.time() - frame_start_time
                fps = 1.0 / total_time if total_time > 0 else 0
                
                print(f"{'='*60}")
                print(f"Point Cloud Source: {source}")
                print(f"{'='*60}")
                print(f"Capture time:        {capture_time*1000:.2f} ms")
                print(f"Process time:        {process_time*1000:.2f} ms")
                print(f"PC Generation time:  {pc_generation_time*1000:.2f} ms")
                print(f"Render time:         {render_time*1000:.2f} ms")
                print(f"{'='*60}")
                print(f"Total time:          {total_time*1000:.2f} ms")
                print(f"FPS:                 {fps:.2f}")
                print(f"{'='*60}")
                print(f"Number of points:    {num_points:,}")
                print(f"Point cloud size:     {points_memory_mb:.2f} MB")
                print(f"{'='*60}")
                
                # Save data if recording
                if self.start_record:
                    data_dict = utils.init_data_dict()
                    data_dict['colors'] = colors
                    data_dict['points'] = points
                    data_dict['color_images'] = color_images
                    data_dict['depth_images'] = depth_images
                    data_dict['time'] = time.time()
                    data_dict['index'] = index
                    data_dict['intrinsic'] = intrinsics
                    data_dict['extrinsic'] = extrinsics
                    data_dict['num_points'] = num_points
                    data_dict['memory_mb'] = points_memory_mb
                    data_dict['fps'] = fps
                    data_dict['source'] = source
                    saved_data.append(data_dict)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key == ord('v'):
                    print('Start Recording')
                    self.start_record = True
                elif key == ord('q') or key == self.ESC_KEY:
                    print("Quitting...")
                    self.stop_rendering = True
                    if saved_data:
                        save_path = input("Enter save path (press Enter for default): ")
                        if not save_path:
                            save_path = "saved/rgbd_data.pkl"
                        utils.save_data(save_path, saved_data)
                        print(f"Saved {len(saved_data)} frames to {save_path}")
                    break
                
                index += 1
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            self.stop_rendering = True
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
            self.stop_rendering = True
        finally:
            self.stop_cameras()
            cv2.destroyAllWindows()
            if self.vis is not None:
                self.vis.destroy_window()
            print("Pipeline stopped")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_rendering = True
        self.stop_cameras()
        cv2.destroyAllWindows()
        if self.vis is not None:
            self.vis.destroy_window()


if __name__ == "__main__":
    # Example usage
    pipeline = MultiRGBDVGGTPipeline(
        max_devices=5,
        max_queue_size=6,
        conf_threshold=1.1,
        use_gpu_render=False,
        model_path="model.pt",
        depth_scale=0.001,      # Depth scale factor
        depth_trunc=3.0,         # Maximum depth in meters
        depth_cutoff=0.1         # Minimum depth in meters
    )
    
    # Initialize components
    if not pipeline.init_cameras():
        print("Failed to initialize cameras")
        exit(1)
    
    if not pipeline.init_vggt_model():
        print("Failed to initialize VGGT model")
        exit(1)
    
    if not pipeline.init_open3d_visualizer():
        print("Failed to initialize Open3D visualizer")
        exit(1)
    
    # Run pipeline with RGBD point cloud generation (default)
    pipeline.run(use_rgbd_pointcloud=True)
    
    # Or run with VGGT predicted point cloud
    # pipeline.run(use_rgbd_pointcloud=False)
