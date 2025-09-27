import os
import shutil
import cv2
from pprint import pprint
import numpy as np


class Dataset:
    def __init__(self, calib_path=None, data_path=None):
        self.calib_path = calib_path
        self.data_path = data_path
        self.device = {}
        self.data = {}
        if self.calib_path is not None:
            self.read_calib_data()

        if self.data_path is not None:
            self.read_scene_data()

    def print(self):
        pprint(self.calib_path)
        pprint(self.data_path)

    def read_device_data(self):
        device_list = os.listdir(self.calib_path)
        for device_id in device_list:
            if device_id not in self.data:
                self.data[device_id] = {}
                self.data[device_id]['rgb_dict'] = {}
                self.data[device_id]['depth_dict'] = {}
                self.data[device_id]['pc_dict'] = {}

    def read_calib_data(self):
        self.read_device_data()
        device_list = os.listdir(self.calib_path)
        for device_id in device_list:
            if device_id not in self.data:
                self.data[device_id] = {}
            rgb_path = os.path.join(self.calib_path, device_id, 'Color')
            if os.path.exists(rgb_path):
                rgb_list = os.listdir(rgb_path)
                if len(rgb_list) != 0:
                    file_name = rgb_list[0]
                    self.data[device_id]['calib_rgb_path'] = os.path.join(self.calib_path, device_id, 'Color', file_name)

            depth_path = os.path.join(self.calib_path, device_id, 'Depth')
            if os.path.exists(depth_path):
                depth_list = os.listdir(depth_path)
                if len(depth_list) != 0:
                    file_name = depth_list[0]
                    self.data[device_id]['calib_depth_path'] = os.path.join(self.calib_path, device_id, 'Depth', file_name)

    def get_calib_data(self):
        rgb_list = [self.data[device_id]['calib_rgb_path'] for device_id in self.data]
        depth_list = [self.data[device_id]['calib_depth_path'] for device_id in self.data]
        return sorted(rgb_list), sorted(depth_list)

    def get_calib_dict(self):
        rgb_dict, depth_dict = {}, {}
        for key in self.data:
            if 'calib_rgb_path' in self.data[key]:
                rgb_dict[key] = self.data[key]['calib_rgb_path']
            if 'calib_depth_path' in self.data[key]:
                depth_dict[key] = self.data[key]['calib_depth_path']
        return rgb_dict, depth_dict

    @staticmethod
    def get_frame_ind(filename):
        return int(filename.split('_')[-1].split('.')[0])

    def read_scene_data(self):
        device_list = os.listdir(self.data_path)
        for device_id in device_list:
            if device_id not in self.data:
                self.data[device_id] = {}
                self.data[device_id]['rgb_dict'] = {}
                self.data[device_id]['depth_dict'] = {}
                self.data[device_id]['pc_dict'] = {}
            rgb_list = os.listdir(os.path.join(self.data_path, device_id, 'Color'))
            for name in rgb_list:
                ind = Dataset.get_frame_ind(name)
                rgb_dict = self.data[device_id]['rgb_dict']
                rgb_dict[ind] = os.path.join(self.data_path, device_id, 'Color', name)

            depth_list = os.listdir(os.path.join(self.data_path, device_id, 'Depth'))
            for name in depth_list:
                ind = Dataset.get_frame_ind(name)
                depth_dict = self.data[device_id]['depth_dict']
                depth_dict[ind] = os.path.join(self.data_path, device_id, 'Depth', name)

            if os.path.exists(os.path.join(self.data_path, device_id, 'PointCloud')):
                pc_list = os.listdir(os.path.join(self.data_path, device_id, 'PointCloud'))
                for name in pc_list:
                    ind = Dataset.get_frame_ind(name)
                    pc_dict = self.data[device_id]['pc_dict']
                    pc_dict[ind] = os.path.join(self.data_path, device_id, 'PointCloud', name)

    def get_frames_by_name_order(self, index):
        rgb_dict = {}
        depth_dict = {}
        f = lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        for device_id in self.data:
            #print(sorted(self.data[device_id]['rgb_dict'].values(), key=f))
            if index < len(self.data[device_id]['rgb_dict'].values()):
                rgb_dict[device_id] = sorted(self.data[device_id]['rgb_dict'].values(), key=f)[index]
            if index < len(self.data[device_id]['depth_dict'].values()):
                depth_dict[device_id] = sorted(self.data[device_id]['depth_dict'].values(), key=f)[index]
        return rgb_dict, depth_dict

    '''
    def get_frames_by_index(self, index):
        rgb_dict = {}
        depth_dict = {}
        for device_id in self.data:
            if index in self.data[device_id]['rgb_dict']:
                rgb_dict[device_id] = self.data[device_id]['rgb_dict'][index]
            if index in self.data[device_id]['depth_dict']:
                depth_dict[device_id] = self.data[device_id]['depth_dict'][index]
        return rgb_dict, depth_dict
    '''

    def get_frames_by_index(self, index, dict_name='rgb_dict'):
        ret_dict = {}
        for device_id in self.data:
            if dict_name in self.data[device_id]:
                if index in self.data[device_id][dict_name]:
                    ret_dict[device_id] = self.data[device_id][dict_name][index]
        return ret_dict

    def get_pointcloud_by_index(self, index):
        return self.get_frames_by_index(index, 'pc_dict')

    def get_frames_by_device(self, device_id):
        if device_id in self.data:
            depth_dict = self.data[device_id]['depth_dict']
            rgb_dict = self.data[device_id]['rgb_dict']
            return rgb_dict, depth_dict

    def get_pointcloud_by_device(self, device_id):
        if device_id in self.data:
            pc_dict = self.data[device_id]['pc_dict']
            return pc_dict

    def get_all_data_by_index(self, index, ms_win=20):
        rgbs, depths, pcs = {}, {}, {}
        for ii in range(ms_win):
            rgb_dict = self.get_frames_by_index(index=index + ii - ms_win // 2, dict_name='rgb_dict')
            depth_dict = self.get_frames_by_index(index=index + ii - ms_win // 2, dict_name='depth_dict')
            pc_dict = self.get_frames_by_index(index=index + ii - ms_win // 2, dict_name='pc_dict')
            rgbs = {**rgbs, **rgb_dict}
            depths = {**depths, **depth_dict}
            pcs = {**pcs, **pc_dict}
        return rgbs, depths, pcs


def img2video(img_path, output_name):
    img_list = os.listdir(img_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, fourcc, 10.0, (1920, 1080), True)
    for fn in img_list:
        img = cv2.imread(os.path.join(img_path, fn))
        out.write(img)
    out.release()


def convert_image(inpath, outpath):
    device_list = os.listdir(inpath)
    for device_id in device_list:
        image_list = sorted(os.listdir(os.path.join(inpath, device_id, 'Color')), key=lambda x: os.path.basename(x).split('_')[1].split('.')[0])
        for i, image_path in enumerate(image_list):
            if not os.path.exists(os.path.join(outpath, str(i))):
                os.mkdir(os.path.join(outpath, str(i)))
            shutil.copy(os.path.join(inpath, device_id, 'Color', image_path), os.path.join(outpath, str(i), device_id+'.png'))


def convert_calib_image(inpath, outpath):
    device_list = os.listdir(inpath)
    for device_id in device_list:
        image_list = sorted(os.listdir(os.path.join(inpath, device_id, 'Color')), key=lambda x: os.path.basename(x).split('_')[1].split('.')[0])
        for i, image_path in enumerate(image_list):
            if i == 0:
                shutil.copy(os.path.join(inpath, device_id, 'Color', image_path), os.path.join(outpath, 'calib', device_id+'.png'))


if __name__ == '__main__':
    if True:
        #calib_path = 'D:\\Data\\ob\\cameras-0418-11'
        #data_path = 'D:\\Data\\ob\\11-cameras-timestamp'
        calib_path = 'Y:\\datasets\\1204\\people'
        data_path = 'Y:\\datasets\\1204\\people'
        dataset = Dataset(calib_path=calib_path, data_path=data_path)
        #pprint(dataset.data)
        # for i in range(8000, 10800):
        for i in range(0, 5):
            out = dataset.get_frames_by_index(i, dict_name='rgb_dict')
            pprint(out)
            #rgbs, depths = out[0], out[1]
            #if len(out[0]) > 0 or len(out[1]) > 0:
                #pprint(out)
        pprint(dataset.get_frames_by_device(device_id='CL8MB33005K')[0])

    if False:
        folder_list = os.listdir('D:\\Data\\ob\\11-cameras-timestamp')
        path = 'D:\\Data\\ob\\tmp'
        for folder in folder_list:
            folder_name = folder
            img_path = f'D:\\Data\\ob\\11-cameras-data\\{folder_name}\\Depth'
            output_name = os.path.join(path, f'{folder_name}.mp4')
            img2video(img_path, output_name)

    if False:
        calib_path = 'Y:\\datasets\\1204\\people'
        data_path = 'Y:\\datasets\\1204\\people'
        dataset = Dataset(calib_path=calib_path, data_path=data_path)
        # pprint(dataset.data)
        for i in range(0, 5):
            out = dataset.get_frames_by_name_order(i)
            if len(out[0]) > 0 or len(out[1]) > 0:
                print(i)
                pprint(out)
        #pprint(dataset.get_frames_by_device(device_id='CL8MB33005K')[0])

    if False:
        inpath = 'D:\\Data\\ob\\6cameras-1'
        outpath = 'D:\\Data\\ob\\6camera-device-frame'
        #convert_image(inpath, outpath)
        convert_calib_image(inpath, outpath)

    if False:
        import json
        path = 'D:\\Code\\RTGaussianPC\\tmp_0510'
        with open(os.path.join(path, 'camera_info.json'), 'r') as f:
            camera_info = json.load(f)
        pprint(camera_info)