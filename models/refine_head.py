import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d, DeformConv2d
from torch import nn
import math


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=True):
        super(BasicConv2d, self).__init__()
        if norm:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
            self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Post_process_deconv(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()

        self.stride = 1
        self.padding = int((kernel_size - 1) / 2)
        self.dilation = 1
        self.register_buffer('w', torch.ones((1, 1, kernel_size, kernel_size)))
        self.register_buffer('b', torch.zeros(1))

    def forward(self, depth, weight, offset):

        output = deform_conv2d(
            depth, offset, self.w, self.b, self.stride, self.padding,
            self.dilation, mask=weight)

        return output


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, depth_feat, fuse_feat):
        hx = torch.cat([h, depth_feat, fuse_feat], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, depth_feat, fuse_feat], dim=1)))

        h = (1-z) * h + z * q
        return h
    

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(in_planes)
        self.norm2 = nn.BatchNorm2d(in_planes)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.norm2(self.conv2(y))
        return self.relu(x+y)


class RefineHead(nn.Module):
    def __init__(self, feature_dim, increase_dims=None, hidden_dim=None, kernel_size=3):
        super(RefineHead, self).__init__()

        increase_dims = increase_dims or feature_dim
        hidden_dim = hidden_dim or feature_dim
        self.num = kernel_size**2 - 1
        self.idx_ref = self.num // 2

        self.depth_increase = BasicConv2d(1, increase_dims, kernel_size=1, padding=0) # Increase depth channel to dims

        # h生成
        self.h_conv = nn.Sequential(
            BasicConv2d(1, hidden_dim // 2, kernel_size=1, padding=0),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # ConvGRU逻辑
        # self.fuse_conv = BasicConv2d(feature_dim, increase_dims, kernel_size=3, padding=1)
        self.convgru = ConvGRU(hidden_dim=hidden_dim, input_dim=feature_dim + increase_dims)

        # offset和weight生成
        self.offset = nn.Conv2d(hidden_dim, 2 * (kernel_size**2 - 1), kernel_size=3, padding=1)
        self.weight = nn.Conv2d(hidden_dim, kernel_size**2 - 1, kernel_size=3, padding=1)

        # DC模块
        self.post_process = Post_process_deconv(kernel_size=kernel_size)
        self.apply(self._init_weights)
        self.relu = nn.ReLU()

    
    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        

    def forward(self, depth, features_list):

        B, _, H, W = depth.shape

        h = self.h_conv(depth)

        depths_list = []
        depths_list.append(depth)

        for i in range(len(features_list)):
            # depth_feat
            x = self.depth_increase(depth)

            # fuse_feat
            features = features_list[i]
            features = F.interpolate(features, size=x.shape[2:], mode='bilinear', align_corners=False)
            # features = self.fuse_conv(features)

            # ConvGRU
            h = self.convgru(h, depth_feat=x, fuse_feat=features)
            offset = self.offset(h)
            weight = torch.tanh(self.weight(h))

            weight_abs_sum = torch.sum(torch.abs(weight), dim=1, keepdim=True) + 1e-6
            weight = weight / weight_abs_sum
            weight_center = 1 - torch.sum(weight, dim=1, keepdim=True)
            # Insert center weight
            weight = weight.view(B, self.num, H, W)
            list_weight = list(torch.chunk(weight, self.num, dim=1))
            list_weight.insert(self.idx_ref,
                            weight_center)
            weight = torch.cat(list_weight, dim=1).view(B, -1, H, W)

            # Add zero reference offset
            offset = offset.view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                            torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            depth = self.post_process(depth, weight, offset)
            depth = torch.clamp(depth, 0.0, 6.0)
            depths_list.append(depth)
        
        return depths_list