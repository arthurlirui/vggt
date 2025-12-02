from queue import Queue
from typing import List
import cv2
import numpy as np
from pyorbbecsdk import *
from pyorbbecsdk import FrameSet, OBFormat, Pipeline, Config, Context, OBAlignMode

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
from models.depthanythingv2_DC import depthanythingv2_DC

img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ctx = Context()
    device_list = ctx.query_devices()
    curr_device_cnt = device_list.get_count()
    #for i in range(device_list.get_count()):
    camera_index = 1
    camera = device_list.get_device_by_index(camera_index)
    pipeline = Pipeline(camera)
    # 1.Create a pipeline with default device.
    #pipeline = Pipeline()
    # 2.Create config.
    config = Config()

    model = depthanythingv2_DC(encoder='vits').cuda()
    checkpoint = torch.load("models/best_rmse_model.pt")
    results = model.load_state_dict(checkpoint['net'], strict=True)
    print('Missing keys: {}'.format(results.missing_keys))
    print('Unexpected keys: {}'.format(results.unexpected_keys))

    # 3.Enable color profile
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 30)
    config.enable_stream(color_profile)

    # 4.Enable depth profile
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.Y16, 30)
    config.enable_stream(depth_profile)

    # 5.Set the frame aggregate output mode to ensure all types of frames are included in the output frameset
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    # 6.Start the pipeline with config.
    pipeline.start(config)

    # 7.Create a filter to align depth frame to color frame
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    while True:
        # 8.Wait for frames
        frames = pipeline.wait_for_frames(10)
        if frames is None:
            continue

        # 9.Filter the data
        frames = align_filter.process(frames)
        if not frames:
            continue
        frames = frames.as_frame_set()

        # Get the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        width, height = color_frame.get_width(), color_frame.get_height()
        print('Color:', width, height)
        color_data = np.asanyarray(color_frame.get_data())
        color_format = color_frame.get_format()
        # color_image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == OBFormat.RGB:
            color_image = np.resize(color_data, (height, width, 3))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        width, height = depth_frame.get_width(), depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        print('Depth:', width, height, scale)

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

        depth_image = depth_data.astype(np.float32) * scale
        # image_dict['depth'] = depth_data
        #color_image = cv2.resize(color_image, dsize=(504, 378))
        #depth_image = cv2.resize(depth_image, dsize=(504, 378))
        color_image = cv2.resize(color_image, dsize=(1280, 720))
        depth_image = cv2.resize(depth_image, dsize=(1280, 720))
        color_image = color_image[360 - 189:360 + 189, 640 - 252:640 + 252, :]
        depth_image = depth_image[360 - 189:360 + 189, 640 - 252:640 + 252]
        print(color_image.shape, depth_image.shape)

        color_image_temp = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image_temp = color_image_temp.astype(np.float32) / 255.0
        color_tensor = torch.from_numpy(color_image_temp).permute(2, 0, 1)
        color_tensor = (color_tensor - img_mean) / img_std

        depth_image_temp = depth_image.astype(np.float32)
        depth_tensor = torch.from_numpy(depth_image_temp).unsqueeze(0).unsqueeze(0)

        # print("image shape: ", color_tensor.shape)
        # print("depth shape: ", depth_tensor.shape)
        # print("depth_range:", depth_tensor.max(), depth_tensor.min())

        depth_tensor = depth_tensor / 1000 / 10 * 6

        color_tensor = color_tensor.cuda()
        depth_tensor = depth_tensor.cuda()

        output = model(color_tensor, depth_tensor)

        depth_output = output['depth_list'][-1]
        #depth_output = output['mona_depth']
        #
        #depth_output = (depth_output - depth_output.min()) / (depth_output.max() - depth_output.min())

        depth_output = depth_output * 10 * 1000 / 6

        depth_image = depth_output.squeeze().detach().cpu().numpy().astype(np.float32)


        min_val, max_val = 0, 10000

        # 对比度拉伸：将 [min_val, max_val] 映射到 [0, 255]
        if max_val > min_val:
            depth_normalized = cv2.convertScaleAbs(depth_image, alpha=255.0 / (max_val - min_val),
                                                   beta=-min_val * 255.0 / (max_val - min_val))
        else:
            # 如果图像全黑或无效，避免除以零
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
        depth_normalized = depth_normalized.astype('uint8')
        # 应用色彩映射（JET是常用的，但PARULA和VIRIDIS视觉效果更好）
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imshow(f'Depth', depth_colormap)
        cv2.imshow(f'Color', color_image)
        cv2.waitKey(1)

    # 10.Stop the pipeline
    pipeline.stop()


if __name__ == "__main__":
    from pyorbbecsdk import *
    main()