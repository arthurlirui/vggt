from queue import Queue
from typing import List
import cv2
import numpy as np
from pyorbbecsdk import *
from pyorbbecsdk import FrameSet, OBFormat, Pipeline, Config, Context

import utils
from utils import frame_to_bgr_image

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import open3d as o3d
import cupy as cp
from numba import cuda
from pprint import pprint
import ba.buddle_adjustment as BA
from utils import init_data_dict, load_data

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

MAX_DEVICES = 5
curr_device_cnt = 0

MAX_QUEUE_SIZE = 6
ESC_KEY = 27

color_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
depth_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
has_color_sensor: List[bool] = [False for _ in range(MAX_DEVICES)]
stop_rendering = False
start_record = False


def on_new_frame_callback(frames: FrameSet, index: int):
    global color_frames_queue, depth_frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is not None:
        if color_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            color_frames_queue[index].get()
        color_frames_queue[index].put(color_frame)
    if depth_frame is not None:
        if depth_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            depth_frames_queue[index].get()
        depth_frames_queue[index].put(depth_frame)


def rendering_frames():
    global color_frames_queue, depth_frames_queue
    global curr_device_cnt
    global stop_rendering
    while not stop_rendering:
        for i in range(curr_device_cnt):
            color_frame = None
            depth_frame = None
            if not color_frames_queue[i].empty():
                color_frame = color_frames_queue[i].get()
            if not depth_frames_queue[i].empty():
                depth_frame = depth_frames_queue[i].get()
            if color_frame is None and depth_frame is None:
                continue
            color_image = None
            depth_image = None
            color_width, color_height = 0, 0
            if color_frame is not None:
                color_width, color_height = color_frame.get_width(), color_frame.get_height()
                color_image = frame_to_bgr_image(color_frame)
            if depth_frame is not None:
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_format = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue

                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_data = depth_data.reshape((height, width))
                except ValueError:
                    print("Failed to reshape depth data")
                    continue

                depth_data = depth_data.astype(np.float32) * scale

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            if color_image is not None and depth_image is not None:
                window_size = (color_width // 2, color_height // 2)
                color_image = cv2.resize(color_image, window_size)
                depth_image = cv2.resize(depth_image, window_size)
                image = np.hstack((color_image, depth_image))
            elif depth_image is not None and not has_color_sensor[i]:
                image = depth_image
            else:
                continue
            cv2.imshow("Device {}".format(i), image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                stop_rendering = True
                break
    cv2.destroyAllWindows()


def capture_frames():
    global color_frames_queue, depth_frames_queue
    global curr_device_cnt
    global stop_rendering
    images = [None for i in range(curr_device_cnt)]
    while not all(t is not None for t in images):
        for i in range(curr_device_cnt):
            color_frame = None
            depth_frame = None
            if not color_frames_queue[i].empty():
                color_frame = color_frames_queue[i].get()
                #print(color_frame.shape)
            if not depth_frames_queue[i].empty():
                depth_frame = depth_frames_queue[i].get()
            if color_frame is None and depth_frame is None:
                continue
            color_image = None
            depth_image = None
            color_width, color_height = 0, 0
            if color_frame is not None:
                color_width, color_height = color_frame.get_width(), color_frame.get_height()
                color_image = frame_to_bgr_image(color_frame)
            if depth_frame is not None:
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_format = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue
                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_data = depth_data.reshape((height, width))
                except ValueError:
                    print("Failed to reshape depth data")
                    continue

                depth_data = depth_data.astype(np.float32) * scale
                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            if color_image is not None and depth_image is not None:
                #window_size = (color_width // 2, color_height // 2)
                #color_image = cv2.resize(color_image, window_size)
                #depth_image = cv2.resize(depth_image, window_size)
                #image = np.hstack((color_image, depth_image))
                image = color_image
            elif color_image is not None and depth_image is None:
                image = color_image
            elif depth_image is not None and not has_color_sensor[i]:
                image = depth_image
            else:
                continue

            # capture images from each camera
            images[i] = image
    return images


def start_streams(pipelines: List[Pipeline], configs: List[Config]):
    index = 0
    for pipeline, config in zip(pipelines, configs):
        print("Starting device {}".format(index))
        pipeline.start(config, lambda frame_set, curr_index=index: on_new_frame_callback(frame_set, curr_index))
        index += 1


def stop_streams(pipelines: List[Pipeline]):
    for pipeline in pipelines:
        pipeline.stop()


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


def visualize_camera_pose(camera_poses, camera_intrinsic):
    # Create a coordinate frame to represent world origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Create a dummy point cloud for context (optional)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

    # Define camera intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=525.0, fy=525.0, cx=320.0, cy=240.0)

    # Create camera poses (4x4 transformation matrices)
    # This is a list of example poses - replace with your actual camera poses
    '''
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
    '''


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


def visualize_camera_trajectory():
    pass


def main():
    parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
    parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument("--conf_threshold", type=float, default=1.1, help="Initial percentage of low-confidence points to filter out")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--save_data", action="store_true", help="save images, point cloud and depth")
    parser.add_argument("--save_path", type=str, default="D:\\Code\\vggt\\saved", help="save path data")
    parser.add_argument("--use_GPU_render", action="store_true", help="Use GPU to render point clouds")

    """
        Main function for the VGGT demo with viser for 3D visualization.

        This function:
        1. Loads the VGGT model
        2. Processes input images from the specified folder
        3. Runs inference to generate 3D points and camera poses
        4. Optionally applies sky segmentation to filter out sky points
        5. Visualizes the results using viser

        Command-line arguments:
        --image_folder: Path to folder containing input images
        --use_point_map: Use point map instead of depth-based points
        --background_mode: Run the viser server in background mode
        --port: Port number for the viser server
        --conf_threshold: Initial percentage of low-confidence points to filter out
        --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    pprint(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    if False:
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    # init camera pipeline
    print("init camera pipeline")
    ctx = Context()
    device_list = ctx.query_devices()
    global curr_device_cnt
    curr_device_cnt = device_list.get_count()
    if curr_device_cnt == 0:
        print("No device connected")
        return
    if curr_device_cnt > MAX_DEVICES:
        print("Too many devices connected")
        return
    pipelines: List[Pipeline] = []
    configs: List[Config] = []
    images_buf = []
    pcs_buf = []
    depth_buf = []
    saved_data = []
    global has_color_sensor
    for i in range(device_list.get_count()):
        camera = device_list.get_device_by_index(i)
        pipeline = Pipeline(camera)
        config = Config()
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color_sensor[i] = True
        except OBError as e:
            print(e)
            has_color_sensor[i] = False

        # init depth sensor
        #profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        #depth_profile = profile_list.get_default_video_stream_profile()
        #config.enable_stream(depth_profile)

        #init IMU
        #config.enable_accel_stream()
        #config.enable_gyro_stream()

        pipelines.append(pipeline)
        configs.append(config)

    # start open3d windows
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Scene - RGB', height=1440, width=2800, top=0, left=0)
    pcd = o3d.geometry.PointCloud()
    is_initialized = False

    global stop_rendering, start_record
    start_streams(pipelines, configs)


    index = 0
    try:
        #rendering_frames()
        while not stop_rendering:
            images = capture_frames()
            print(len(images))
            data_dict = init_data_dict()
            if len(images) < MAX_DEVICES:
                continue
            images = load_and_preprocess_images(image_path_list=images, mode="crop")

            images = images.to(device)
            print(f"Preprocessed images shape: {images.shape}")
            print("Running inference...")
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    start_time = time.time()
                    predictions = model(images)
                    end_time = time.time()
            print(f"Computing Time:{end_time-start_time}")
            print("Converting pose encoding to extrinsic and intrinsic matrices...")
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            #pprint(extrinsic)
            #pprint(intrinsic)
            extrinsic_cpu = extrinsic.cpu().numpy().squeeze(0)
            intrinsic_cpu = intrinsic.cpu().numpy().squeeze(0)
            extrinsic_list = [extrinsic_cpu[i, :, :] for i in range(len(device_list))]
            intrinsic_list = [intrinsic_cpu[i, :, :] for i in range(len(device_list))]

            #pprint(extrinsic_list)
            #pprint(intrinsic_list)
            #continue

            print("Processing model outputs...")
            start_time = time.time()
            if args.use_GPU_render:
                for key in predictions.keys():
                    if isinstance(predictions[key], torch.Tensor):
                        # remove batch dimension and convert to pycuda numpy
                        predictions[key] = cp.asarray(predictions[key]).squeeze(0)
            else:
                for key in predictions.keys():
                    if isinstance(predictions[key], torch.Tensor):
                        # remove batch dimension and convert to numpy
                        predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            end_time = time.time()
            print(f"CPU-GPU-Communication Time:{end_time-start_time}")

            ptSHW = predictions['world_points']
            colorSHW = predictions['images']
            depthSHW = predictions['depth']
            confSHW = predictions['world_points_conf']
            colorSHW = colorSHW.transpose(0, 2, 3, 1)
            colorSHW = colorSHW[..., ::-1]
            print('Point cloud shape:', ptSHW.shape)
            print('Color shape:', colorSHW.shape)
            print('Depth shape:', depthSHW.shape)

            ptN3 = ptSHW.reshape(-1, 3)
            colorN3 = colorSHW.reshape(-1, 3)
            confN3 = confSHW.reshape(-1, )

            # Convert numpy arrays to correct types
            mask = confN3 > args.conf_threshold
            if args.use_GPU_render:
                points = cp.asnumpy(ptN3[mask, :])
                colors = cp.asnumpy(colorN3[mask, :])
            else:
                points = np.asarray(ptN3[mask, :], dtype=np.float64)
                colors = np.asarray(colorN3[mask, :], dtype=np.float64)

            if args.save_data and start_record:
                data_dict['colors'] = colors
                data_dict['points'] = points

            # Validate input shapes
            if points.shape[1] != 3:
                raise ValueError("Points must be an Nx3 array")
            if colors.shape[1] != 3:
                raise ValueError("Colors must be an Nx3 array")
            if points.shape[0] != colors.shape[0]:
                raise ValueError("Points and colors must have the same number of points")

            # Assign points and colors to the point cloud
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(pcd)
            start_time = time.time()
            if not is_initialized:
                vis.add_geometry(pcd)
                is_initialized = True
            else:
                vis.update_geometry(pcd)

            #vis.add_geometry(pcd, reset_bounding_box=False)
            vis.poll_events()
            vis.update_renderer()
            end_time = time.time()
            time.sleep(0.001)
            #vis.remove_geometry(pcd, reset_bounding_box=False)
            print(f"Rendering Time:{end_time-start_time}")

            if args.save_data and start_record:
                data_dict['images'] = colorSHW
                data_dict['depth'] = depthSHW
                data_dict['time'] = time.time()
                data_dict['index'] = index
                data_dict['intrinsic'] = intrinsic
                data_dict['extrinsic'] = extrinsic
                saved_data.append(data_dict)

            index += 1


            for i in range(MAX_DEVICES):
                if args.use_GPU_render:
                    continue
                img = colorSHW[..., i, :, :, 0:3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow(f'Color-{i}', img)
                depth = depthSHW[..., i, :, :, 0]

                # 为了显示，将深度值归一化到 0-255 范围
                # 首先，将无效深度值（例如0）屏蔽掉，以避免影响归一化
                # 假设无效的深度值为0
                clipped_depth = np.where(depth == 0, 0, depth)

                # 计算有效深度的最小值和最大值（忽略0）
                min_val = np.min(clipped_depth[np.nonzero(clipped_depth)])
                max_val = np.max(clipped_depth)

                # 对比度拉伸：将 [min_val, max_val] 映射到 [0, 255]
                depth_normalized = np.zeros_like(clipped_depth, dtype=np.uint8)
                if max_val > min_val:
                    depth_normalized = cv2.convertScaleAbs(depth, alpha=255.0 / (max_val - min_val), beta=-min_val * 255.0 / (max_val - min_val))
                else:
                    # 如果图像全黑或无效，避免除以零
                    depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)

                # 应用色彩映射（JET是常用的，但PARULA和VIRIDIS视觉效果更好）
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PARULA)

                # 显示原深度图（灰度）和伪彩色图
                # 原深度图需要先归一化才能用imshow正确显示
                #cv2.imshow('Original Depth (Normalized)', depth_normalized)
                #cv2.imshow('Depth Colormap', depth_colormap)

                #cv2.imshow(f'Depth-{i}', depth)
                cv2.imshow(f'Depth-Colormap-{i}', depth_colormap)
                end_time = time.time()
                #if end_time - start_time > 10:
                    #cv2.imwrite(os.path.join('tmp', f'rgb_{i}_{end_time}.jpg'), img*255)


            #cv2.imshow('Depth-1', depthSHW[..., 1, :, :, 0])
            #cv2.imshow('Depth-2', depthSHW[..., 2, :, :, 0])

            key = cv2.waitKey(1)
            if key == ord('v'):
                print('Start Recording')
                start_record = True
            if key == ord('q') or key == ESC_KEY:
                stop_rendering = True
                if args.save_data:
                    #utils.save_pointcloud(pcs_buf)
                    #utils.save_images(images_buf)
                    #utils.save_depth(depth_buf)
                    #for d in saved_data:
                    utils.save_data(os.path.join(args.save_path, f'frames_0922_%04d.pkl' % index), saved_data)
                break
        #cv2.destroyAllWindows()

    except KeyboardInterrupt:
        stop_rendering = True
    finally:
        stop_streams(pipelines)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if True:
        main()

    if False:
        from utils import load_data
        from pprint import pprint
        data_path = 'D:\\Code\\vggt\\saved\\frames_0922_0179.pkl'
        data = load_data(data_path=data_path)
        for d in data:
            print(d.keys())
            print(d['index'], d['time'], d['images'].shape, d['depth'].shape, d['points'].shape, d['colors'].shape)
        d = data[0]
        for i in range(MAX_DEVICES):
            img = d['images'][i, :, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow(f'Color-{i}', img)
            depth = d['depth'][i, :, :, 0]
            cv2.imshow(f'Depth-{i}', depth)
            cv2.waitKey(0)

