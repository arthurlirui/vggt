import os
import cv2
import numpy as np
import glob
import open3d as o3d
import json
from ultralytics import YOLO
from dataset import Dataset
import matplotlib.image as mpimg
import scipy
from scipy.ndimage import median_filter
import copy
from dataset import Dataset
from pprint import pprint
from utils.image_utils import depth_erode, depth_dilate


def calc_intrinsic_extrinsic(img_dict: dict, sz=26, psz=(8, 5), show_img=False, outpath=None):
    '''
    :param img_dict: {device_id: img_path}
    :param sz: chessboard size
    :param psz: chessboard pattern
    :param show_img:
    :param outpath:
    :return:
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((psz[0] * psz[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:psz[0], 0:psz[1]].T.reshape(-1, 2) * sz
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    img_shape = None

    exist_camera_info = False
    if os.path.exists(outpath):
        exist_camera_info = True

    for key in img_dict:
        fpath = img_dict[key]
        if exist_camera_info:
            continue
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape
        # Find the chess board corners
        # ret, corners = cv.findChessboardCorners(gray, psz, None)
        ret, corners = cv2.findChessboardCornersSB(gray, psz, None, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

        # If found, add object points, image points (after refining them)
        if ret:
            print('True ', key, fpath)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

            if show_img:
                cv2.drawChessboardCorners(img, psz, corners2, ret)
                cv2.imshow('img', img)
                if outpath is not None:
                    cv2.imwrite(os.path.join(outpath, os.path.basename(fpath)), img)
                cv2.waitKey(500)
        else:
            print('False ', key, fpath)
            print('Recapture images since there is a failure.')

    if show_img:
        cv2.destroyAllWindows()
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    res = {}
    for key in img_dict:
        res[key] = {'device_id': key, 'intrinsic': None, 'extrinsic': None}

    if exist_camera_info:
        with open(outpath, 'r', encoding='utf-8') as f:
            res = json.load(f)
            for key in res:
                res[key]['intrinsic'] = np.array(res[key]['intrinsic'])
                res[key]['extrinsic'] = np.array(res[key]['extrinsic'])
                res[key]['rvec'] = np.array(res[key]['rvec'])
                res[key]['tvec'] = np.array(res[key]['tvec'])
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)
        keys = img_dict.keys()
        for i, key in enumerate(keys):
            res[key]['device_id'] = key
            res[key]['intrinsic'] = mtx

            Rw, Tw = cv2.Rodrigues(rvecs[i])[0], tvecs[i]
            T = np.vstack([np.hstack([Rw, Tw]), np.array([0, 0, 0, 1])])
            res[key]['extrinsic'] = T
            res[key]['rvec'] = rvecs[i]
            res[key]['tvec'] = tvecs[i]
        if outpath is not None:
            tmp = {}
            for key in res:
                tmp[key] = {'device_id': key, 'intrinsic': res[key]['intrinsic'].tolist(),
                            'extrinsic': res[key]['extrinsic'].tolist(),
                            'rvec': res[key]['rvec'].tolist(), 'tvec': res[key]['tvec'].tolist()}
            with open(outpath, 'w+', encoding='utf-8') as f:
                json.dump(tmp, f, ensure_ascii=True, indent=4)
    return res


def merge_pointcloud_open3d(depths=[], rgbs=[], depthpath=None, colorpath=None, intrinsic=None, extrinsics=[],
                            save_subfile=False, visualize_pointcloud=False, use_color=False, output_path=None,
                            output_name='all_open3d.ply'):
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if intrinsic is not None:
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, fx=fx, fy=fy, cx=cx, cy=cy)

    #if len(depths) == 0:
    #    depth_list = glob.glob(os.path.join(depthpath, '*.png'))
    #    depth_list = sorted(depth_list)
    #    #depths = [cv.imread(fname, cv.IMREAD_UNCHANGED) for fname in depth_list]
    #    pprint(depth_list)

    #point_cloud_list = []
    if depthpath is None:
        depth_list = depths
    else:
        depth_list = sorted(glob.glob(os.path.join(depthpath, '*.png')))

    if colorpath is None:
        color_list = rgbs
    else:
        color_list = sorted(glob.glob(os.path.join(colorpath, '*.png')))

    point_cloud_list = []
    camera_extrinsic = []
    for i, depth in enumerate(depth_list):
        fname = os.path.basename(depth)
        extrinsic = extrinsics[i]
        Rw, Tw = extrinsic[3], extrinsic[2]
        #print(Rw.shape, Tw.shape)
        T = np.vstack([np.hstack([Rw, Tw]), np.array([0, 0, 0, 1])])
        depth_raw = o3d.io.read_image(depth_list[i])
        if use_color:
            #if colorpath is not None:
            #    colorfile_name = os.path.join(colorpath, fname)
            #    print(fname, colorfile_name)
            #    color_raw = o3d.io.read_image(colorfile_name)
            color_raw = o3d.io.read_image(color_list[i])
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                            depth_scale=1.0, depth_trunc=10000.0,
                                                                            convert_rgb_to_intensity=False)
            rgbd_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                                     intrinsic=camera_intrinsic,
                                                                     extrinsic=T)
            point_cloud_list.append(rgbd_pc)
            if save_subfile:
                o3d.io.write_point_cloud(os.path.join(output_path, f'{str(i)}_rgbd.ply'), rgbd_pc)
        else:
            depth_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_raw,
                                                                       intrinsic=camera_intrinsic,
                                                                       extrinsic=T,
                                                                       depth_scale=1.0,
                                                                       depth_trunc=10000.0)
            point_cloud_list.append(depth_pc)
            if save_subfile:
                o3d.io.write_point_cloud(os.path.join(output_path, f'{str(i)}_d.ply'), depth_pc)
    #allpc = sum(point_cloud_list)
    allpc = o3d.geometry.PointCloud()
    for t in point_cloud_list:
        allpc += t
    downply = allpc.voxel_down_sample(voxel_size=8.0)
    print(os.path.join(output_path, output_name))
    o3d.io.write_point_cloud(filename=os.path.join(output_path, output_name),
                             pointcloud=downply,
                             write_ascii=False,
                             compressed=True,
                             print_progress=True)


class Camera:
    def __init__(self, device_id, intrinsic: np.ndarray = None, extrinsic: np.ndarray = None, width=1920, height=1080):
        self.device_id = device_id
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.width = width
        self.height = height
        self.yolov_model = YOLO("yolov8x-seg.pt")  # load an official model

    def load_data(self, data_dict: dict):
        self.device_id = data_dict['device_id']
        self.intrinsic = data_dict['intrinsic']
        self.extrinsic = data_dict['extrinsic']
        self.rvec = data_dict['rvec']
        self.tvec = data_dict['tvec']
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.width, height=self.height, fx=fx, fy=fy, cx=cx,
                                                             cy=cy)
        self.intrinsic_pinhole = camera_intrinsic

    def depth_preprocess_bak(self, depth, img, d=35, sigmaColor=100.0, sigmaSpace=0.5):
        depth = np.asarray(depth)
        img = np.asarray(img)
        h, w = img.shape[0], img.shape[1]
        #depth = cv2.resize(depth, dsize=[w//2, h//2])
        #img = cv2.resize(img, dsize=[w//2, h//2])
        depth_mask = np.sum(depth, axis=2) <= 1
        depth_mask = depth_mask.astype(np.uint8) * 255
        depth_f = cv2.inpaint(depth[:, :, -1], depth_mask, 1, cv2.INPAINT_TELEA)
        depth_f = depth_f.astype(np.float32)
        depth_f = np.stack([depth_f, depth_f, depth_f], axis=2)
        depth_f = cv2.ximgproc.jointBilateralFilter(img.astype(np.float32), depth_f, d=d, sigmaColor=sigmaColor,
                                                    sigmaSpace=sigmaSpace)
        return depth_f, depth_mask

    def _depth2world(self, depth_path, depth_scale=1.0, depth_trunc=10000.0, detect_human=False, depth_proc=False):
        depth_raw = o3d.io.read_image(depth_path)
        h, w, d = depth_raw.shape
        depth_raw = cv2.resize(depth_raw, [w // 2, h // 2])
        #depth_raw = cv2.bilateralFilter(depth_raw, 15, 150, 150)
        depth_raw = cv2.resize(depth_raw, [w, h])

        depth_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_raw,
                                                                   intrinsic=self.intrinsic_pinhole,
                                                                   extrinsic=self.extrinsic,
                                                                   depth_scale=depth_scale,
                                                                   depth_trunc=depth_trunc)
        pc_o3d = depth_pc
        return pc_o3d

    def _rgbd2world(self, depth_path=None, rgb_path=None, depth_scale=1.0, depth_trunc=10000.0, use_morph=True,
                    depth_proc=False, detect_human=False):
        color_raw = o3d.io.read_image(rgb_path)
        depth_raw = o3d.io.read_image(depth_path)
        h, w, d = np.asarray(depth_raw).shape
        depth_raw2 = np.asarray(depth_raw)
        #color_raw2 = np.asarray(color_raw)

        if use_morph:
            #depth_nd = depth_dilate(depth_in=depth_raw2, ks=9, ops_iter=3)
            #es = 17
            #element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * es + 1, 2 * es + 1), (es, es))

            depth1c = depth_raw2[:, :, 0]
            mask = depth1c > 0
            emp_mask = depth1c == 0
            depth_dilate = np.zeros(depth1c.shape, dtype=np.float32)
            es = 7
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*es+1, 2*es+1), (es, es))
            mask_dilate = cv2.dilate(np.uint8(mask), element, iterations=1)
            cv2.dilate(src=np.float32(depth1c), kernel=element, dst=depth_dilate, iterations=1)
            depth1c[mask_dilate & emp_mask] = depth_dilate[mask_dilate & emp_mask]

            es = 9
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * es + 1, 2 * es + 1), (es, es))
            mask_erode = cv2.erode(np.uint8(depth1c > 0), element, iterations=2)
            depth1c[mask_erode == 0] = 0
            depth1c = np.uint16(depth1c)
            depth_raw2 = np.stack([depth1c, depth1c, depth1c], axis=2)

            #depth[erode_mask == 0] = 0
            #depth_raw2 = np.stack([depth, depth, depth], axis=2)

        depth_raw = o3d.geometry.Image(depth_raw2.astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,
                                                                        depth_raw,
                                                                        depth_scale=depth_scale,
                                                                        depth_trunc=depth_trunc,
                                                                        convert_rgb_to_intensity=False)

        rgbd_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                                 intrinsic=self.intrinsic_pinhole,
                                                                 extrinsic=self.extrinsic)
        pc_o3d = rgbd_pc
        return pc_o3d


    def rgbd2world(self, depth_path=None, rgb_path=None, depth_scale=1.0, depth_trunc=10000.0, detect_human=False, depth_proc=False):
        pc_o3d = None
        is_rgbd = rgb_path is not None and depth_path is not None
        is_depth = depth_path is not None and rgb_path is None
        if is_depth:
            return self._depth2world(depth_path, depth_scale=depth_scale, depth_trunc=depth_trunc)

        if is_rgbd:
            return self._rgbd2world(depth_path, rgb_path, depth_scale=depth_scale, depth_trunc=depth_trunc)

        '''
        if depth_path is not None and rgb_path is None:
            depth_raw = o3d.io.read_image(depth_path)
            h, w, d = depth_raw.shape
            depth_raw = cv2.resize(depth_raw, [w//2, h//2])
            #cv2.medianBlur(depth_raw, 15, depth_raw)
            depth_raw = cv2.bilateralFilter(depth_raw, 15, 150, 150)
            depth_raw = cv2.resize(depth_raw, [w, h])

            depth_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_raw,
                                                                       intrinsic=self.intrinsic_pinhole,
                                                                       extrinsic=self.extrinsic,
                                                                       depth_scale=depth_scale,
                                                                       depth_trunc=depth_trunc)
            pc_o3d = depth_pc
        '''

        '''
        
        '''
        if False and rgb_path is not None and depth_path is not None:
            color_raw = o3d.io.read_image(rgb_path)
            depth_raw = o3d.io.read_image(depth_path)

            #depth_raw, depth_mask = self.depth_preprocess(depth_raw, color_raw, d=35, sigmaColor=100.0, sigmaSpace=0.5)
            #depth_raw = o3d.geometry.Image(depth_raw)

            #if False:
                # cv2.medianBlur(depth_raw, 15, depth_raw)
                #depth_raw = np.asarray(depth_raw)
                #h, w, d = depth_raw.shape
                #depth_raw = cv2.resize(depth_raw, [w // 2, h // 2])
                #depth_raw = cv2.bilateralFilter(depth_raw.astype(np.float32), 15, 150, 50)
                #depth_raw = cv2.resize(depth_raw, [w, h])
                #depth_raw = depth_raw.astype(np.uint16)
                #depth_raw = o3d.geometry.Image(depth_raw)
            h, w, d = np.asarray(depth_raw).shape
            depth_raw2 = np.asarray(depth_raw)
            color_raw2 = np.asarray(color_raw)

            #depth_erode = erode_depth(depth_raw2[:, :, 0])
            #depth_raw3 = np.stack([depth_erode, depth_erode, depth_erode], axis=2)
            #depth_raw = o3d.geometry.Image(depth_raw3.astype(np.uint16))
            use_morph = True
            if use_morph:
                depth_nd = depth_dilate(depth_in=depth_raw2, ks=9, ops_iter=3)
                depth = depth_raw2[:, :, 0]
                depth_mask = depth > 0
                # cv2.imshow('mask', depth_mask)
                # erosion_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))
                es = 17
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*es+1, 2*es+1), (es, es))

                erode_mask = cv2.erode(np.uint8(depth_mask), element)
                depth[erode_mask == 0] = 0
                depth_raw2 = np.stack([depth, depth, depth], axis=2)

            # 1. inpainting depth
            if depth_proc:
                depth_f, depth_mask = depth_preprocess(depth_raw2, color_raw2, d=50, sigmaColor=10.0, sigmaSpace=20.0)
            else:
                depth_f = depth_raw

            # 2. segment person depth
            if detect_human:
                mask_list = YOLO_segment_person(color_raw2, self.yolov_model)
                if len(mask_list) > 0:
                    mask0 = np.sum(np.stack(mask_list, axis=2), axis=2)
                else:
                    mask0 = np.zeros((w, h))
                mask1 = cv2.resize(mask0, dsize=(w, h)).astype(np.bool8)
                #depth_f = depth_raw2
                depth_f[~mask1] = 0
                depth_f[depth_f > 3000] = 0
                depth_raw = o3d.geometry.Image(depth_f.astype(np.uint16))
            depth_raw = o3d.geometry.Image(depth_raw2.astype(np.uint16))
            #depth_raw = cv2.resize(depth_raw, [w // 2, h // 2])
            #depth_raw, depth_mask = self.depth_preprocess(depth_raw2, color_raw2, d=10, sigmaColor=100.0,
            #                                              sigmaSpace=0.5)
            #depth_raw = cv2.resize(depth_raw, [w, h])

            #color_raw = o3d.geometry.Image(color_raw.astype(np.uint8))


            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,
                                                                            depth_raw,
                                                                            depth_scale=depth_scale,
                                                                            depth_trunc=depth_trunc,
                                                                            convert_rgb_to_intensity=False)


            rgbd_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                                     intrinsic=self.intrinsic_pinhole,
                                                                     extrinsic=self.extrinsic)
            pc_o3d = rgbd_pc
            #pts = np.asarray(pc_o3d.points).astype(np.float32)
            #pc_o3d.points = o3d.utility.Vector3dVector(pts, np.float32)
            #pc_o3d.points = pts.Vector3dVector(np.float32)
            return pc_o3d

    def __repr__(self):
        print(self.device_id)
        print(self.intrinsic)
        print(self.extrinsic)

    def __str__(self):
        print(self.device_id)
        print(self.intrinsic)
        print(self.extrinsic)
        return self.device_id

    def print_device(self):
        pass

    def calib_intrinsic(self):
        pass

    def calib_extrinsic(self):
        pass


class CameraArray:
    def __init__(self, calib_path=None, data_path=None):
        self.camera_dict = {}
        if calib_path is not None and data_path is not None:
            self.dataset = Dataset(calib_path=calib_path, data_path=data_path)

    @staticmethod
    def calc_intrinsic_extrinsic(img_dict: dict, sz=26, psz=(8, 5), show_img=False, outpath=None):
        return calc_intrinsic_extrinsic(img_dict=img_dict, sz=sz, psz=psz, show_img=show_img, outpath=outpath)

    def calib(self, sz=60, psz=(8, 11), outpath=None):
        crgb, cdepth = self.dataset.get_calib_dict()
        res = CameraArray.calc_intrinsic_extrinsic(img_dict=crgb, sz=sz, psz=psz, show_img=False, outpath=outpath)

        for key in res:
            c = Camera(device_id=key)
            c.load_data(res[key])
            self.camera_dict[key] = c

    def load_camera_info(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
            for key in res:
                res[key]['intrinsic'] = np.array(res[key]['intrinsic'])
                res[key]['extrinsic'] = np.array(res[key]['extrinsic'])
                res[key]['rvec'] = np.array(res[key]['rvec'])
                res[key]['tvec'] = np.array(res[key]['tvec'])
        for key in res:
            c = Camera(device_id=key)
            c.load_data(res[key])
            self.camera_dict[key] = c

    def camera2world(self, rgb_path, depth_path):
        pass

    def get_camera_param_by_device(self, device_id):
        if device_id in self.camera_dict:
            return self.camera_dict[device_id]
        else:
            return None


    def get_frames_by_index(self, index):
        return self.dataset.get_frames_by_index(index=index)


def YOLO_segment_person(img, model):
    results = model(img.astype(np.uint8), imgsz=640)  # predict on an image
    masks = []
    for result in results:
        if result.masks is not None and len(result.masks) > 0:
            masks_data = result.masks.data
            cls = result.boxes.cls.cpu().numpy()
            for index, mask in enumerate(masks_data):
                if result.names[cls[index]] == 'person':
                    mask = mask.cpu().numpy() * 255
                    masks.append(mask)
    return masks


def run_dataset():
    if False:
        datapath = 'D:\\Data\\ob\\0607'
        rootpath = 'D:\\Data\\ob\\0607\\data'
        calib_path = os.path.join(datapath, 'calib')
        data_path = os.path.join(datapath, 'data')

    if True:
        #datapath = 'Z:\\datasets\\0813\\pointcloud_rgb_datasets'
        #datapath = 'D:\\Data\\ob\\0607\\test\\rgb3'
        datapath = 'Z:\\datasets\\1012\\calibrate'
        calib_path = datapath
        data_path = datapath
    output_path = os.path.join('output', 'tmp_1012')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ca = CameraArray(calib_path=calib_path, data_path=data_path)
    ca.calib(sz=60, psz=(8, 11), outpath=os.path.join(output_path, 'camera_info.json'))
    device_list = os.listdir(data_path)

    ind_list = sorted([int(fn.split('_')[-1].split('.')[0]) for fn in sorted(os.listdir(os.path.join(data_path, device_list[0], 'Color')))])
    print(sorted(ind_list))
    for i in ind_list:
        index = i
        rgbs, depths = {}, {}
        for ii in range(20):
            rgb0, depth0 = ca.dataset.get_frames_by_index(i + ii - 10)
            rgbs = {**rgbs, **rgb0}
            depths = {**depths, **depth0}
            if len(rgbs) == len(device_list) and len(depths) == len(device_list):
                break
        if len(rgbs) != len(device_list) or len(depths) != len(device_list) or len(rgbs) != len(depths):
            #print(i, len(rgbs), len(depths), len(device_list))
            continue

        #pprint(rgbs)
        #pprint(depths)

        print(i, len(rgbs), len(depths), len(device_list))
        # pprint(rgbs)
        pcs = o3d.geometry.PointCloud()
        for device_id in rgbs:
            if device_id not in depths:
                continue
            rgb_path = rgbs[device_id]
            depth_path = depths[device_id]
            c = ca.camera_dict[device_id]
            pc = c.rgbd2world(depth_path=depth_path, rgb_path=rgb_path)
            pcs += pc

        vsz = 1.0
        pcs = pcs.voxel_down_sample(voxel_size=vsz)
        # estimate surface normal
        pcs.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
        o3d.io.write_point_cloud(os.path.join(output_path, f'all_{index}.ply'), pcs, write_ascii=False, compressed=True)


def run_test_frames():
    rootpath = 'D:\\Data\\ob\\0510'
    calib_path = os.path.join(rootpath, '9-cameras-calibration')
    data_path = os.path.join(rootpath, '9-cameras-people')
    # output_path = 'D:\\Code\\RTGaussianPC\\camera_0509.json'
    #output_path = 'D:\\Code\\RTGaussianPC\\tmp_0509-1\\camera_0509.json'

    ca = CameraArray(calib_path=calib_path, data_path=data_path)
    ds = ca.dataset
    pprint(ds.data)


def test_single_RGBD2pc():
    datapath = 'Z:\\datasets\\0813\\pointcloud_rgb_datasets'
    device_id = 'CL8MB3300H0'
    #datapath = 'Z:\\datasets\\0813\\pointcloud_rgb_datasets\\CL8MB3300H0'
    depths = [os.path.join(datapath, device_id, 'Depth', 'CL8MB3300H0_38_3484.png')]
    rgbs = [os.path.join(datapath, device_id, 'Color', 'CL8MB3300H0_72_3283.png')]
    calib_path = datapath
    data_path = datapath
    ca = CameraArray(calib_path=calib_path, data_path=data_path)
    output_path = 'D:\\Code\\RTGaussianPC\\output\\tmp_0813'
    ca.calib(sz=60, psz=(8, 11), outpath=os.path.join(output_path, 'camera_info.json'))
    c = ca.camera_dict[device_id]
    intrinsic = c.intrinsic
    extrinsic = c.extrinsic
    output_path = 'D:\\Code\\RTGaussianPC\\output\\tmp_single_pc'

    color_raw = o3d.io.read_image(rgbs[0])
    depth_raw = o3d.io.read_image(depths[0])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,
                                                                    depth_raw,
                                                                    depth_scale=1.0,
                                                                    depth_trunc=10000.0,
                                                                    convert_rgb_to_intensity=False)

    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, fx=fx, fy=fy, cx=cx, cy=cy)
    rgbd_pc0 = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image, intrinsic=camera_intrinsic, extrinsic=np.eye(4))
    rgbd_pc1 = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image, intrinsic=camera_intrinsic, extrinsic=extrinsic)
    o3d.io.write_point_cloud(os.path.join(output_path, f'{device_id}_c.ply'), rgbd_pc0, write_ascii=True, compressed=False)
    o3d.io.write_point_cloud(os.path.join(output_path, f'{device_id}_w.ply'), rgbd_pc1, write_ascii=True, compressed=False)


    pc_path = os.path.join(datapath, device_id, 'PointCloud', 'CL8MB3300H0_801_34290.ply')
    tmp = o3d.io.read_point_cloud(pc_path)


    #pc = copy.deepcopy(tmp).transform(c.intrinsic)
    T = np.eye(4,4)
    T[:3, :3] = np.transpose(c.extrinsic[:3, :3])
    T[:, 3] = c.extrinsic[:, 3]
    tmp2 = copy.deepcopy(tmp).transform(T)

    #pc = copy.deetmp.transform(c.extrinsic)
    o3d.io.write_point_cloud(os.path.join(output_path, f'{device_id}_pc_w.ply'), tmp2, write_ascii=True, compressed=False)


def depth_preprocess(depth, img, d=35, sigmaColor=100.0, sigmaSpace=0.5, minD=0, maxD=10000):
    h, w, dm = img.shape
    if minD is not None and maxD is not None:
        depth[depth < minD] = 0
        depth[depth > maxD] = 0

    depth = cv2.resize(depth, dsize=[w//3, h//3], interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, dsize=[w//3, h//3], interpolation=cv2.INTER_NEAREST)
    depth_mask = np.sum(depth, axis=2) <= 0
    depth_mask = depth_mask.astype(np.uint8) * 255

    depth_f = depth[:, :, -1]
    rescaling_ratio = 100.0

    if False:
        depth_f = cv2.inpaint(depth[:, :, -1], depth_mask, 1, cv2.INPAINT_TELEA)
        print(np.min(depth_f[:]), np.max(depth_f[:]))

    if False:
        depth_f = depth_f.astype(np.float32) / rescaling_ratio
        depth_f = np.stack([depth_f, depth_f, depth_f], axis=2)
        depth_f = cv2.ximgproc.jointBilateralFilter(img.astype(np.float32), depth_f, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        depth_f = depth_f * rescaling_ratio

    #depth_f = median_filter(depth_f, 5)

    if False:
        #depth_f = np.stack([depth_f, depth_f, depth_f], axis=2)
        depth_f = depth_f // 10
        depth_f = depth_f.astype(np.uint8)
        depth_f = cv2.fastNlMeansDenoisingColored(depth_f, None, 5, 10, 7, 21)

    if False:
        depth_f = cv2.medianBlur(depth_f, ksize=5)
        #depth_f = cv2.medianBlur(depth_f, ksize=5)
        #depth_f = cv2.medianBlur(depth_f, ksize=5)


    depth_f = depth_f.astype(np.uint16)
    depth_f = cv2.resize(depth_f, dsize=[w, h], interpolation=cv2.INTER_NEAREST)
    depth_mask = cv2.resize(depth_mask, dsize=[w, h], interpolation=cv2.INTER_NEAREST)

    depth_f = depth_f.astype(np.uint16)
    return depth_f, depth_mask


if __name__ == '__main__':


    if False:
        calib_path = 'D:\\Data\\ob\\cameras-0418-11'
        data_path = 'D:\\Data\\ob\\11-cameras-timestamp'
        alldata = Dataset(calib_path=calib_path, data_path=data_path)
        crgb, cdepth = alldata.get_calib_dict()
        pprint(crgb)
        print(alldata.data.keys())
        #alldata.data[device_id]['calib_rgb_path']
        sz = 60  # mm
        psz = (8, 11)
        res = calc_intrinsic_extrinsic(img_dict=crgb, sz=sz, psz=psz, show_img=False, outpath=None)
        pprint(res)
        camera_dict = {}
        for key in res:
            c = Camera(device_id=key)
            c.load_data(res[key])
            camera_dict[key] = c

    if False:
        calib_path = 'D:\\Data\\ob\\cameras-0418-11'
        data_path = 'D:\\Data\\ob\\11-cameras-timestamp'
        ca = CameraArray(calib_path=calib_path, data_path=data_path)
        ca.calib(sz=60, psz=(8, 11))
        #pprint(ca.camera_dict)
        for key in ca.camera_dict:
            print(ca.camera_dict[key])

    if False:
        calib_path = 'D:\\Data\\ob\\cameras-0418-11'
        data_path = 'D:\\Data\\ob\\11-cameras-timestamp'
        ca = CameraArray(calib_path=calib_path, data_path=data_path)
        ca.calib(sz=60, psz=(8, 11))

        device_id = 'CL8MB3300MJ'
        c = ca.camera_dict[device_id]
        rgb_path = f'D:\\Data\\ob\\11-cameras-people\\{device_id}\\Color\\1920x1080_10809.png'
        depth_path = f'D:\\Data\\ob\\11-cameras-people\\{device_id}\\Depth\\1920x1080_10810.png'
        pc = c.rgbd2world(depth_path=depth_path, rgb_path=rgb_path)

        output_path = 'D:\\Code\\RTGaussianPC\\tmp_test'
        o3d.io.write_point_cloud(os.path.join(output_path, f'{device_id}_10809.ply'), pc)


    if False:
        # D:\Data\ob\11-cameras-people
        calib_path = 'D:\\Data\\ob\\0507\\calibration'
        data_path = 'D:\\Data\\ob\\0507\\12cameras-people'
        output_path = 'D:\\Code\\RTGaussianPC\\camera_0507.json'
        ca = CameraArray(calib_path=calib_path, data_path=data_path)
        ca.calib(sz=60, psz=(8, 11), outpath=output_path)

    if True:
        #test_single_RGBD2pc()
        run_dataset()

    if False:
        depth_path = f'D:\\Data\\ob\\0607\\data\\CL8MB3300EE\\Depth\\CL8MB3300EE_38_3277.png'
        color_path = f'D:\\Data\\ob\\0607\\data\\CL8MB3300EE\\Color\\CL8MB3300EE_74_3276.png'
        #depth_raw = cv2.imread(depth_path)
        #depth_raw = cv2.bilateralFilter(depth_raw, 15, 150, 150)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        #depth_raw = depth_raw.astype(np.float32)
        color_raw = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        depth_f, depth_mask = depth_preprocess(depth_raw, color_raw, d=10, sigmaColor=10, sigmaSpace=0.5)
        depth_f = depth_f.astype(np.uint16)
        #cv2.imwrite('test_depth.png', depth_f)
        depth = cv2.resize(depth_f, [640, 360])

        depth = depth // 10
        depth = (depth-np.min(depth[:]))/np.max(depth[:])*256

        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        color = cv2.resize(color_raw, [640, 360])
        #depth_f3 = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imshow('Depth', depth)
        cv2.imshow('Color', color)
        cv2.waitKey()

    if False:
        root_path = 'D:\\Data\\ob\\0607\\data\\CL8PB31002C\\Color'
        image_path = os.path.join(root_path, 'CL8PB31002C_312_12800.png')
        depth_path = os.path.join(root_path, 'CL8PB31002C_257_12004.png')

        #image_np = np.array(image)

        img = cv2.imread(image_path)
        depth = cv2.imread(depth_path)
        model = YOLO("yolov8x-seg.pt")
        mask_list = YOLO_segment_person(img, model)
        mask = np.sum(np.stack(mask_list, axis=2), axis=2)
        h, w, d = img.shape
        mask = cv2.resize(mask, [w, h])
        mask = np.stack([mask, mask, mask], axis=2).astype(np.bool8)
        image_person_np = img.copy()
        image_person_np[~mask] = 0
        cv2.imshow('test', image_person_np)
        cv2.waitKey()

    if False:
        run_test_frames()

    if False:
        # D:\Data\ob\11-cameras-people
        calib_path = 'D:\\Data\\ob\\0509\\11-cameras-calibration'
        data_path = 'D:\\Data\\ob\\0509\\11-cameras-pople'
        ca = CameraArray(calib_path=calib_path, data_path=data_path)
        output_path = 'D:\\Code\\RTGaussianPC\\camera_0509.json'
        ca.calib(sz=60, psz=(8, 11), outpath=output_path)

        if False:
            #device_list = ['CL8MB3300GL', 'CL8MB3300H0', 'CL8MB3300MJ', 'CL8MB33005K']
            device_list = []
            #device_list.append('CL8MB3300GL')
            #device_list.append('CL8MB3300H0')
            device_list.append('CL8MB3300MJ')
            #device_list.append('CL8MB33005K')
            #device_list.append('CL8MB33006M')
            #device_list.append('CL8MB330075')
            device_list.append('CL8PB31002V')
            device_list.append('CL8PB31005E')
            device_list.append('CL8PB310019')
            #device_list.append('CL8PB310036')
            #device_list.append('CL8PB310048')
        else:
            device_list = os.listdir(data_path)

        if False:
            img_id = '10809'
            depth_id = '10810'
            rgb0, depth0 = ca.dataset.get_frames_by_index(10808)
            rgb1, depth1 = ca.dataset.get_frames_by_index(10809)
            rgb2, depth2 = ca.dataset.get_frames_by_index(10810)
            rgb3, depth3 = ca.dataset.get_frames_by_index(10811)
            rgb4, depth4 = ca.dataset.get_frames_by_index(10812)
            rgbs = {**rgb0, **rgb1, **rgb2, **rgb3, **rgb4}
            depths = {**depth0, **depth1, **depth2, **depth3, **depth4}

        for i in range(0, 6000, 8):
            index = i
            #rgb0, depth0 = ca.dataset.get_frames_by_name_order(index=index)
            #rgb1, depth1 = ca.dataset.get_frames_by_name_order(index=index)
            #rgb2, depth2 = ca.dataset.get_frames_by_name_order(index=index)
            #rgb3, depth3 = ca.dataset.get_frames_by_name_order(index=index)
            #rgb4, depth4 = ca.dataset.get_frames_by_name_order(index=index)
            rgb0, depth0 = ca.dataset.get_frames_by_index(i + 0)
            rgb1, depth1 = ca.dataset.get_frames_by_index(i + 1)
            rgb2, depth2 = ca.dataset.get_frames_by_index(i + 2)
            rgb3, depth3 = ca.dataset.get_frames_by_index(i + 3)
            rgb4, depth4 = ca.dataset.get_frames_by_index(i + 4)
            rgb5, depth5 = ca.dataset.get_frames_by_index(i + 5)
            rgb6, depth6 = ca.dataset.get_frames_by_index(i + 6)
            rgbs = {**rgb0, **rgb1, **rgb2, **rgb3, **rgb4, **rgb5, **rgb6}
            depths = {**depth0, **depth1, **depth2, **depth3, **depth4, **depth5, **depth6}
            #rgbs = {**rgb0, **rgb1, **rgb2}
            #depths = {**depth0, **depth1, **depth2}
            #rgbs = {**rgb0}
            #depths = {**depth0}

            if len(rgbs) < 8 or len(depths) < 8:
                continue
            if len(rgbs) != len(depths):
                continue

            print(i, len(rgbs), len(depths), len(device_list))
            #pprint(rgbs)
            pcs = o3d.geometry.PointCloud()
            for device_id in rgbs:
                if device_id not in depths:
                    continue
                rgb_path = rgbs[device_id]
                depth_path = depths[device_id]
                c = ca.camera_dict[device_id]
                pc = c.rgbd2world(depth_path=depth_path, rgb_path=rgb_path)
                pcs += pc

            output_path = 'D:\\Code\\RTGaussianPC\\tmp_0509-1'
            vsz = 5.0
            pcs = pcs.voxel_down_sample(voxel_size=vsz)
            o3d.io.write_point_cloud(os.path.join(output_path, f'all_{index}.ply'), pcs,
                                     write_ascii=False,
                                     compressed=True)

        if False:
            for device_id in device_list:
                #device_id = 'CL8MB3300MJ'
                c = ca.camera_dict[device_id]

                rgb_path = f'D:\\Data\\ob\\11-cameras-people\\{device_id}\\Color\\1920x1080_{img_id}.png'
                depth_path = f'D:\\Data\\ob\\11-cameras-people\\{device_id}\\Depth\\1920x1080_{depth_id}.png'
                pc = c.rgbd2world(depth_path=depth_path, rgb_path=rgb_path)
                pcs += pc


