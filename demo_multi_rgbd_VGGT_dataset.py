from dataset import Dataset
from pprint import pprint
import torch
from queue import Queue
from typing import List
import cv2
import numpy as np
import os
import glob
import time
import threading
import argparse
from typing import List, Optional
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import open3d as o3d
try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import utils
from utils import init_data_dict,  load_data, frame_to_bgr_image


MAX_DEVICES = 5
curr_device_cnt = 0

MAX_QUEUE_SIZE = 6
ESC_KEY = 27

color_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
depth_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
has_color_sensor: List[bool] = [False for _ in range(MAX_DEVICES)]
stop_rendering = False
start_record = False


def main_dataset():
    parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
    parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/",
                        help="Path to folder containing images")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument("--conf_threshold", type=float, default=25.0,
                        help="Initial percentage of low-confidence points to filter out")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--save_data", action="store_true", help="save images, point cloud and depth")
    parser.add_argument("--save_path", type=str, default="D:\\Code\\vggt\\saved", help="save path data")

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
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device)

    # start open3d windows
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Scene - RGB', visible=True, width=640, height=640, top=100, left=100)
    pcd = o3d.geometry.PointCloud()
    is_initialized = False

    # init dataset
    #calib_path = 'Y:\\datasets\\1204\\people'
    #data_path = 'Y:\\datasets\\1204\\people'
    calib_path = 'D:\\Data\\ob\\1204\\rotated_data'
    data_path = 'D:\\Data\\ob\\1204\\rotated_data'
    #'D:\Data\ob\1219\rotated_data'
    #calib_path = 'D:\\Data\\ob\\1219\\rotated_data'
    #data_path = 'D:\\Data\\ob\\1219\\rotated_data'
    dataset = Dataset(calib_path=calib_path, data_path=data_path)

    device_list = list(dataset.data.keys())
    ind_list = sorted([int(fn.split('_')[-1].split('.')[0]) for fn in sorted(os.listdir(os.path.join(data_path, device_list[0], 'Color')))])
    print(sorted(ind_list))
    skeys = sorted(device_list)

    #index = ind_list[42]
    #i = index

    for ind in ind_list:
        index = ind
        rgbs, depths = {}, {}
        for ii in range(30):
            rgb0 = dataset.get_frames_by_index(ind + ii - 15, dict_name='rgb_dict')
            depth0 = dataset.get_frames_by_index(ind + ii - 15, dict_name='depth_dict')
            rgbs = {**rgbs, **rgb0}
            depths = {**depths, **depth0}
            #if len(rgbs) == len(device_list) and len(depths) == len(device_list):
            if len(rgbs) == len(device_list):
                break
        if len(rgbs) != len(device_list):
            print(f"No enough images - {ind}")
            continue

        print(ind, len(rgbs), len(depths), len(device_list))
        data_dict = init_data_dict()
        pprint(rgbs)
        images = [rgbs[k] for k in skeys]
        images = load_and_preprocess_images(image_path_list=images, mode="crop")
        images = images.to(device)
        print(f"Running inference: preprocessed images shape: {images.shape}")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                # remove batch dimension and convert to numpy
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        ptSHW = predictions['world_points']
        colorSHW = predictions['images']
        depthSHW = predictions['depth']
        colorSHW = colorSHW.transpose(0, 2, 3, 1)
        ConfSHW = predictions['world_points_conf']

        #colorSHW = colorSHW[..., ::-1]
        print('Point cloud shape:', ptSHW.shape)
        print('Color shape:', colorSHW.shape)
        print('Depth shape:', depthSHW.shape)
        print('Conf shape:', ConfSHW.shape)

        # Convert numpy arrays to correct types
        ptN3 = ptSHW.reshape(-1, 3)
        colorN3 = colorSHW.reshape(-1, 3)
        confN3 = ConfSHW.reshape(-1, )
        points = np.asarray(ptN3[confN3 > 1.05, :], dtype=np.float64)
        colors = np.asarray(colorN3[confN3 > 1.05, :], dtype=np.float64)
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

        if not is_initialized:
            vis.add_geometry(pcd)
            is_initialized = True
            # 获取渲染选项并设置点大小
            point_size = 1.0
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            render_option.background_color = np.array([1, 1, 1])  # 白色背景
        else:
            vis.update_geometry(pcd)

        # vis.add_geometry(pcd, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        #time.sleep(0.001)

        # vis.remove_geometry(pcd, reset_bounding_box=False)
        #cv2.waitKey()
        if False:
            data_dict['images'] = colorSHW
            data_dict['depth'] = depthSHW
            data_dict['time'] = time.time()
            data_dict['index'] = index
            data_dict['intrinsic'] = intrinsic
            data_dict['extrinsic'] = extrinsic

        for i in range(len(images[0:5])):
            img = colorSHW[..., i, :, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow(f'Color-{i}', img)

            # process depth
            depth = depthSHW[..., i, :, :, 0]
            clipped_depth = np.where(depth == 0, 0, depth)
            min_val = np.min(clipped_depth[np.nonzero(clipped_depth)])
            max_val = np.max(clipped_depth)
            depth_normalized = np.zeros_like(clipped_depth, dtype=np.uint8)
            if max_val > min_val:
                depth_normalized = cv2.convertScaleAbs(depth, alpha=255.0 / (max_val - min_val), beta=-min_val * 255.0 / (max_val - min_val))
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)

            # 应用色彩映射（JET是常用的，但PARULA和VIRIDIS视觉效果更好）
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            #cv2.imshow(f'Depth-{i}', depth)
            cv2.imshow(f'Depth-Colormap-{i}', depth_colormap)

            # process conf
            conf = ConfSHW[i, :, :]
            clipped_conf = np.where(conf == 0, 0, conf)
            min_val = np.min(clipped_conf[np.nonzero(clipped_conf)])
            max_val = np.max(clipped_conf)
            print(f"Min: {min_val} Max: {max_val}")
            conf_normalized = np.zeros_like(clipped_conf, dtype=np.uint8)
            if max_val > min_val:
                conf_normalized = cv2.convertScaleAbs(conf, alpha=255.0 / (max_val - min_val), beta=-min_val * 255.0 / (max_val - min_val))
            else:
                conf_normalized = np.zeros_like(conf, dtype=np.uint8)

            # 应用色彩映射（JET是常用的，但PARULA和VIRIDIS视觉效果更好）
            conf_colormap = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_JET)
            # cv2.imshow(f'Depth-{i}', depth)
            cv2.imshow(f'Conf-Colormap-{i}', conf_colormap)
            end_time = time.time()
        if args.save_data:
            ts = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            save_path = os.path.join(args.save_path, ts)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            vis.capture_screen_image(os.path.join(save_path, {ind}.png))
        vis.run()


if __name__ == '__main__':
    main_dataset()
    if False:
        calib_path = 'Y:\\datasets\\1204\\people'
        data_path = 'Y:\\datasets\\1204\\people'
        dataset = Dataset(calib_path=calib_path, data_path=data_path)
        # pprint(dataset.data)

        device_list = list(dataset.data.keys())
        ind_list = sorted([int(fn.split('_')[-1].split('.')[0]) for fn in
                           sorted(os.listdir(os.path.join(data_path, device_list[0], 'Color')))])
        print(sorted(ind_list))
        for i in ind_list:
            index = i
            rgbs, depths = {}, {}
            for ii in range(20):
                rgb0 = dataset.get_frames_by_index(i + ii - 10, dict_name='rgb_dict')
                depth0 = dataset.get_frames_by_index(i + ii - 10, dict_name='depth_dict')
                rgbs = {**rgbs, **rgb0}
                depths = {**depths, **depth0}
                if len(rgbs) == len(device_list) and len(depths) == len(device_list):
                    break
            if len(rgbs) != len(device_list) or len(depths) != len(device_list) or len(rgbs) != len(depths):
                continue

            print(i, len(rgbs), len(depths), len(device_list))

