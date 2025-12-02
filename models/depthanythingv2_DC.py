import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d  
from torch import nn
from models.depthanythingv2.dpt import DepthAnythingV2
from models.utils import fit_scale_shift_batch, depth_normalization
from models.refine_head import RefineHead

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


model_checkpoints = {
    'vitg': "pretrained/depthanythingv2/depth_anything_v2_vitg.pth",
    'vitl': "pretrained/depthanythingv2/depth_anything_v2_vitl.pth",
    'vitb': "pretrained/depthanythingv2/depth_anything_v2_vitb.pth",
    'vits': "pretrained/depthanythingv2/depth_anything_v2_vits.pth",
}

class depthanythingv2_DC(nn.Module):

    def __init__(self, encoder='vitl', args=None):
        super().__init__()
        self.mona = DepthAnythingV2(**model_configs[encoder])
        self.ft_refiner = RefineHead(model_configs[encoder]['features'], 
                                     increase_dims=48,
                                     hidden_dim=48,
                                     kernel_size=3)

    def forward(self, images, tof_depths):
        depth, conf, feature_list = self.mona(images, tof_depths)
        depth = 1 / (depth + 1e-6)  # 反转回深度

        depth = depth_normalization(depth)

        course_depth, s, t = fit_scale_shift_batch(depth, tof_depths)

        course_depth_fuse = conf * course_depth + (1 - conf) * tof_depths
        course_depth_fuse = torch.clamp(course_depth_fuse, 0.0, 6.0)

        depth_list = self.ft_refiner(course_depth_fuse, feature_list)  # from low to high


        output = {}
        output['mona_depth'] = depth
        output['mona_conf'] = conf
        output['course_depth'] = course_depth
        output['course_depth_fuse'] = course_depth_fuse
        output['depth_list'] = depth_list  # from low to high

        return output