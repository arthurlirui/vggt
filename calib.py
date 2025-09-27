import os.path
import numpy as np
import cv2 as cv
import glob
from camera2world_selfcalib import depth_to_3D_coords, show_point_cloud
import open3d as o3d
from pprint import pprint
# import collections
# import struct
#import pcl
# from matplotlib import pyplot as plt
# from utils.graphics_utils import fov2focal
# from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
#     read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
# import cv2
# from scene.dataset_readers import readColmapSceneInfo
from plyfile import PlyData, PlyElement
from pprint import pprint

def calc_rotate_matrix(imgpath, sz=26, psz=(8, 5), show_img=False, outpath=None):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # cardboard mesh size 8*5 26mm*26mm
    #sz = 26  # mm
    #patter_size = (8, 5)
    objp = np.zeros((psz[0] * psz[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:psz[0], 0:psz[1]].T.reshape(-1, 2) * sz
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    #imgpath = 'D:\\Data\\ob\\3-1\\images'
    if type(imgpath) is list:
        images = imgpath
    else:
        images = glob.glob(os.path.join(imgpath, '*.png'))
        if images is None or len(images) == 0:
            images = glob.glob(os.path.join(imgpath, '*.jpg'))
    images = sorted(images)
    for fname in images:
        img = cv.imread(fname)
        #print(img.shape)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #print(gray.shape)
        # Find the chess board corners
        #ret, corners = cv.findChessboardCorners(gray, psz, None)

        ret, corners = cv.findChessboardCornersSB(gray, psz, None, cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY);

        # If found, add object points, image points (after refining them)
        if ret is True:
            # imgpoints = []
            print('True ', fname)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

            if show_img:
                cv.drawChessboardCorners(img, psz, corners2, ret)
                cv.imshow('img', img)
                if outpath is not None:
                    cv.imwrite(os.path.join(outpath, os.path.basename(fname)), img)

                cv.waitKey(500)
                #print(fname)
        else:
            print('False', fname)
    if show_img:
        cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    rmats = [cv.Rodrigues(r)[0] for r in rvecs]
    extrinsic = []
    #for i, rr in enumerate(rmats):
    #    extrinsic.append((i, rvecs[i], tvecs[i], rr))

    res = {'intrinsic': mtx, 'extrinsic': [(i, rvecs[i], tvecs[i], rr) for i, rr in enumerate(rmats)]}
    res['mtx'] = mtx
    res['dist'] = dist
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

def merge_pointcloud(depths=[], depthpath=None, intrinsic=None, extrinsics=[], save_subfile=False, visualize_pointcloud=False):
    if intrinsic is not None:
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    if len(depths) == 0:
        depth_list = glob.glob(os.path.join(depthpath, '*.png'))
        depth_list = sorted(depth_list)
        depths = [cv.imread(fname, cv.IMREAD_UNCHANGED) for fname in depth_list]
        pprint(depth_list)

    point_cloud_list = []
    for i, depth in enumerate(depths):
        PCc = depth_to_3D_coords(depth, cx, cy, fx, fy, sample_step=1, to_point_cloud=True)
        extrinsic = extrinsics[i]
        pprint(extrinsic)
        Rw, Tw = extrinsic[3], extrinsic[2]
        Rc = np.linalg.inv(Rw)
        Tc = Rw @ Tw
        #transformed_points = PCc @ Rc.T + Tc.reshape(1, 3)
        PCw = PCc @ Rw.T + Tw.reshape(1, 3)
        #PCw = PCc+Tw.reshape(1, 3)
        #transformed_points = cloud
        # transformed_points = camera_2_canonical(cloud, R.T, T)
        if visualize_pointcloud:
            point_cloud_list.append(PCw)

        vertex = np.zeros(PCw.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex['x'] = PCw[:, 0]
        vertex['y'] = PCw[:, 1]
        vertex['z'] = PCw[:, 2]

        ply_data = PlyData([PlyElement.describe(vertex, 'vertex')])

        if save_subfile:
            # 保存为PLY格式文件
            savepath = os.path.join('./tmp', f'transformed_cloud_{str(i)}.ply')
            print(savepath)
            ply_data.write(savepath)


    if visualize_pointcloud:
        show_point_cloud(point_cloud_list, axis_size=0.1)

    # 加载第一个点云文件
    #ply_data_combined = o3d.io.read_point_cloud("a1transformed_cloud.ply")

    # 加载并合并其他点云文件
    cont = 0
    ply_data_combined = None
    #for camera in Scene.train_cameras:
    for i, depth in enumerate(depths):
        savepath = os.path.join('tmp', f'transformed_cloud_{str(i)}.ply')
        ply_data = o3d.io.read_point_cloud(savepath)
        if ply_data_combined is None:
            ply_data_combined = ply_data
        else:
            ply_data_combined += ply_data

    # 保存合并后的点云文件
    o3d.io.write_point_cloud(os.path.join('tmp', 'combined_cloud.ply'), ply_data_combined)


def test_o3d():
    depthpath = 'D:\\Data\\ob\\4-15\\test1\\depth'
    depth_list = ['000000.png', '000001.png', '000002.png']
    pclist = []
    '''
    array([[1.11712899e+03, 0.00000000e+00, 9.69174500e+02],
       [0.00000000e+00, 1.11637317e+03, 5.65273625e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
       
    array([[-0.186572  , -0.84942829,  0.4936218 ],
       [ 0.96025867, -0.2638369 , -0.091068  ],
       [ 0.20759138,  0.45701387,  0.86489545]])
       
    array([[-0.13755825, -0.91253828,  0.3851644 ],
       [ 0.97328472, -0.19670166, -0.11842853],
       [ 0.18383304,  0.3585838 ,  0.9152175 ]])
       
    array([[ 0.00432174, -0.99772138,  0.06733028],
       [ 0.98824685, -0.00602733, -0.15274763],
       [ 0.15280539,  0.06719908,  0.98596896]])   
    '''

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080,
                                                         fx=1117.129, fy=1116.373,
                                                         cx=969.175, cy=565.274)
    camera_extrinsic = []
    ce1 = np.array([[-0.186572, -0.84942829,  0.4936218,  198.01322147],
                    [0.96025867, -0.2638369, -0.091068, 50.01940241],
                    [0.20759138,  0.45701387,  0.86489545, 999.38145416],
                    [0,  0,  0, 1]])

    ce2 = np.array([[-0.13755825, -0.91253828,  0.3851644, 82.89466426],
                    [0.97328472, -0.19670166, -0.11842853, -41.00819719],
                    [0.18383304,  0.3585838,  0.9152175, 878.28202198],
                    [0, 0, 0, 1]])
    ce3 = np.array([[0.00432174, -0.99772138,  0.06733028, 217.57702274],
                    [0.98824685, -0.00602733, -0.15274763, -32.73438582],
                    [0.15280539,  0.06719908,  0.98596896, 857.74300791],
                    [0, 0, 0, 1]])
    camera_extrinsic.append(ce1)
    camera_extrinsic.append(ce2)
    camera_extrinsic.append(ce3)
    depthpc = []
    for i, fn in enumerate(depth_list):
        filepath = os.path.join(depthpath, fn)
        depth_raw = o3d.io.read_image(filepath)
        depth = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_raw, intrinsic=camera_intrinsic, extrinsic=camera_extrinsic[i],
                                                                depth_scale=1.0, depth_trunc=10000.0)
        depthpc.append(depth)
        o3d.io.write_point_cloud(os.path.join('tmp', f'{str(i)}.ply'), depth)
    #depthpath = 'D:\\Data\\ob\\4-15\\test1\\depth\\000000.png'
    #depth_raw = o3d.io.read_image(depthpath)
    #print(depth_raw)
    allpc = depthpc[0] + depthpc[1] + depthpc[2]
    o3d.io.write_point_cloud(os.path.join('tmp', 'all_open3d.ply'), allpc)


def process_camera_pose(inpath, visualize=False):
    sz = 60  # mm
    psz = (8, 11)
    rootpath = 'D:\\Data\\ob\\0418-calib2'
    imgpath = os.path.join(rootpath, 'images')
    depthpath = os.path.join(rootpath, 'depth')
    colorpath = os.path.join(rootpath, 'images')
    # imgpath = 'D:\\Data\\ob\\4-15\\lab4-15\\images'
    # depthpath = 'D:\\Data\\ob\\4-15\\lab4-15\\depth'
    # colorpath = 'D:\\Data\\ob\\4-15\\lab4-15\\images'
    res = calc_rotate_matrix(imgpath, sz=sz, psz=psz)
    if visualize:
        pprint(res['intrinsic'])
        pprint(res['dist'])
    return res


def visualize_ply_series(data_path):
    import os
    import numpy as np
    import open3d as o3d

    files = sorted(os.listdir(data_path))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pointcloud = o3d.geometry.PointCloud()
    to_reset = False

    for f in files:
        ply = o3d.io.read_point_cloud(os.path.join(data_path, f))  # 此处读取的pcd文件,也可读取其他格式的
        #ply = np.asarray(ply.points).reshape((-1, 3))
        print(ply)
        downply = ply.voxel_down_sample(voxel_size=10.0)
        #pointcloud.points = o3d.utility.Vector3dVector(ply)  # 如果使用numpy数组可省略上两行
        #pointcloud = downply
        vis.add_geometry(downply)
        #vis.update_geometry()
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()
        #vis.run()
        vis.clear_geometries()


if __name__ == '__main__':
    if False:
        print("Load a ply point cloud, print it, and render it")
        #sample_ply_data = o3d.data.PLYPointCloud()
        inpath = 'D:\\Code\\RTGaussianPC\\tmp\\20_rgbd.ply'
        pcd = o3d.io.read_point_cloud(inpath)
        o3d.visualization.draw_geometries([pcd])

    if False:
        test_o3d()

    if False:
        inpath = 'D:\\Code\\RTGaussianPC\\tmp\\all_open3d.ply'
        ply = o3d.io.read_point_cloud(inpath)
        #o3d.visualization.draw_geometries([ply])
        print(ply)
        voxel_size = 3.0
        downply = ply.voxel_down_sample(voxel_size)
        print(downply)
        #o3d.visualization.draw_geometries([downply])
        o3d.io.write_point_cloud('D:\\Code\\RTGaussianPC\\tmp\\all_open3d_d.ply', downply)

    if False:
        data_path = 'D:\\Code\\RTGaussianPC\\tmp_array'
        visualize_ply_series(data_path)

    if True:
        import dataset
        sz = 60  # mm
        psz = (8, 11)
        calib_path = 'D:\\Data\\ob\\0507\\calibration'
        data_path = 'D:\\Data\\ob\\0507\\12cameras-people'
        alldata = dataset.Dataset(calib_path=calib_path, data_path=data_path)
        calib_rgb_list, calib_depth_list = alldata.get_calib_data()
        res = calc_rotate_matrix(imgpath=calib_rgb_list, sz=sz, psz=psz)
        pprint(res['intrinsic'])
        pprint(res['dist'])

        output_path = 'D:\\Code\\RTGaussianPC\\tmp_array'
        for i in range(0, 3000):
            rgb_dict0, depth_dict0 = alldata.get_frames_by_index(i)
            rgb_dict1, depth_dict1 = alldata.get_frames_by_index(i + 1)
            rgb_dict2, depth_dict2 = alldata.get_frames_by_index(i + 2)
            rgb_dict3, depth_dict3 = alldata.get_frames_by_index(i + 3)
            rgb_dict = {**rgb_dict0, **rgb_dict1, **rgb_dict2, **rgb_dict3}
            depth_dict = {**depth_dict0, **depth_dict1, **depth_dict2, **depth_dict3}
            if len(rgb_dict.keys()) == 11 and len(depth_dict.keys()) == 11:
                depths = sorted(depth_dict.values())
                rgbs = sorted(rgb_dict.values())
                merge_pointcloud_open3d(depths=depths, rgbs=rgbs,
                                        intrinsic=res['intrinsic'], extrinsics=res['extrinsic'],
                                        save_subfile=False, visualize_pointcloud=True, use_color=True,
                                        output_path=output_path, output_name=f'{i}.ply')

    if False:
        sz = 60  # mm
        psz = (8, 11)
        rootpath = 'D:\\Data\\ob\\0418-calib2'
        imgpath = os.path.join(rootpath, 'images')
        depthpath = os.path.join(rootpath, 'depth')
        colorpath = os.path.join(rootpath, 'images')
        #imgpath = 'D:\\Data\\ob\\4-15\\lab4-15\\images'
        #depthpath = 'D:\\Data\\ob\\4-15\\lab4-15\\depth'
        #colorpath = 'D:\\Data\\ob\\4-15\\lab4-15\\images'
        res = calc_rotate_matrix(imgpath, sz=sz, psz=psz)
        pprint(res['intrinsic'])
        pprint(res['dist'])

        merge_pointcloud_open3d(depths=[], depthpath=depthpath, colorpath=colorpath, intrinsic=res['intrinsic'], extrinsics=res['extrinsic'],
                                save_subfile=True, visualize_pointcloud=True, use_color=True)

    if False:
        foldername = 'camera_array2'
        datapath = f'D:/Data/IRCamera/{foldername}/image'
        outpath = f'D:/Data/IRCamera/{foldername}/outpath'
        depthpath = f'D:/Data/IRCamera/{foldername}/depth'
        sz = 60  # mm
        psz = (8, 11)
        #folderlist = os.listdir(datapath)
        #filelist = [os.path.join(datapath, f, 'IR', '640x576_50.png') for f in folderlist]
        filelist = os.listdir(datapath)
        pprint(filelist)
        res = calc_rotate_matrix(imgpath=datapath, sz=sz, psz=psz, show_img=True, outpath=outpath)

        pprint(res['intrinsic'])
        pprint(res['extrinsic'])
        pprint(res['dist'])
        mtx = res['mtx']
        dist = res['dist']
        imglist = os.listdir(datapath)
        for fname in imglist:
            ff = fname.split('.')[0]
            img = cv.imread(os.path.join(datapath, fname))
            img_corner = cv.imread(os.path.join(outpath, fname))
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            pprint(newcameramtx)
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv.imwrite(os.path.join(outpath, ff + '-rectified.png'), dst)
            dst2 = cv.undistort(img_corner, mtx, dist, None, newcameramtx)
            cv.imwrite(os.path.join(outpath, ff + '-corner-rectified.png'), dst2)


        merge_pointcloud(depths=[], depthpath=depthpath, intrinsic=res['intrinsic'], extrinsics=res['extrinsic'],
                         save_subfile=True,
                         visualize_pointcloud=True)

    if False:
        datapath = 'D:/Data/cameras-0403/cameras/1'
        outpath = f'D:/Data/cameras-0403/outpath'
        #datapath = 'D:/Data/IRCamera/test1/image'
        sz = 60  # mm
        psz = (8, 11)
        folderlist = os.listdir(datapath)
        filelist = [os.path.join(datapath, f, 'IR', '640x576_50.png') for f in folderlist]
        pprint(filelist)
        res = calc_rotate_matrix(imgpath=filelist, sz=sz, psz=psz, show_img=True, outpath=outpath)

        pprint(res['intrinsic'])
        pprint(res['extrinsic'])
        pprint(res['dist'])

        mtx = res['mtx']
        dist = res['dist']
        imglist = os.listdir(datapath)
        for i, fname in enumerate(imglist):
            ff = fname.split('.')[0]
            img = cv.imread(os.path.join(outpath, ff + f'-{i}.png'))
            img_corner = cv.imread(os.path.join(outpath, fname))
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            pprint(newcameramtx)
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv.imwrite(os.path.join(outpath, ff + f'-{i}-rectified.png'), dst)
            dst2 = cv.undistort(img_corner, mtx, dist, None, newcameramtx)
            cv.imwrite(os.path.join(outpath, ff + f'-{i}-corner-rectified.png'), dst2)


    if False:
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # cardboard mesh size 8*5 26mm*26mm
        sz = 26  # mm
        patter_size = (8, 5)
        objp = np.zeros((8 * 5, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * sz
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        imgpath = 'D:\\Data\\ob\\3-1\\images'
        images = glob.glob(os.path.join(imgpath, '*.png'))
        for fname in sorted(images):
            print(fname)
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, patter_size, None)
            # If found, add object points, image points (after refining them)
            if ret is True:
                #imgpoints = []
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv.drawChessboardCorners(img, patter_size, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        cv.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        rmats = [cv.Rodrigues(r)[0] for r in rvecs]
        #print(rvecs)
        print(mtx)
        print(len(rvecs))
        for i, rr in enumerate(rmats):
            print(i)
            print(rvecs[i])
            print(tvecs[i])
            print(rr)


