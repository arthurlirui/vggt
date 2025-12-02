import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from models.utils import depth_normalization, interpolate_depth_nearest_zero_invalid


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()

        hidden_channels = hidden_channels or out_channels

        self.conv1x1_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.conv3x3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3)

        self.bn3x3 = nn.BatchNorm2d(hidden_channels)
        self.bn5x5 = nn.BatchNorm2d(hidden_channels)
        self.bn7x7 = nn.BatchNorm2d(hidden_channels)

        # 相加后ReLU
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # 若 in/out 通道不等，给残差做投影
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        identity = x
        x = self.conv1x1_in(x)
        x3 = self.bn3x3(self.conv3x3(x))
        x5 = self.bn5x5(self.conv5x5(x))
        x7 = self.bn7x7(self.conv7x7(x))
        y = x + x3 + x5 + x7  # [B, hidden, H, W]
        y = self.relu(y)
        y = self.conv1x1_out(y)             # [B, out_channels, H, W]

        if self.proj is not None:
            identity = self.proj(identity)

        return identity + y                  # 残差输出


class DepthPrompt(nn.Module):
    def __init__(self, channels=256, hidden=None):
        super().__init__()

        hidden = hidden or channels // 2

        # 输入: [features(channels) || depth(1)] -> out: channels
        self.mscb = MultiScaleConvBlock(channels + 1, hidden_channels=hidden, out_channels=channels)

        # 产生“提示/残差”表征
        self.prompt = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        # 像素级置信度门（0~1）
        self.confidence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

        self.apply(self._init_weights)
        with torch.no_grad():
            nn.init.constant_(self.confidence[-1].bias, 0.0)
            nn.init.constant_(self.confidence[-1].weight, 0.0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, features, depth):
        """
        features: [B, C, H, W]
        depth   : [B, 1, H0, W0] (建议预先做深度归一化/掩码填补)
        """
        identity = features

        # 尺度对齐
        depth = interpolate_depth_nearest_zero_invalid(depth, size=features.shape[2:])  # [B, 1, H, W]

        # 融合与抽取
        fused = self.mscb(torch.cat([features, depth], dim=1))  # [B, C, H, W]
        prompt_feat = self.prompt(fused)                        # [B, C, H, W]
        conf = self.confidence(fused)           # [B, 1, H, W]

        # 门控残差注入
        out = identity + prompt_feat * conf  # [B, C, H, W]

        return out, conf


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        # 新增加的模块
        self.ft_depth_prompt1 = DepthPrompt(channels=features, hidden=features // 2)
        self.ft_depth_prompt2 = DepthPrompt(channels=features, hidden=features // 2)
        self.ft_depth_prompt3 = DepthPrompt(channels=features, hidden=features // 2)
        self.ft_depth_prompt4 = DepthPrompt(channels=features, hidden=features // 2)

        self.ft_depth_conf = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.ft_depth_conf.apply(self._init_weights)
        nn.init.constant_(self.ft_depth_conf[-2].bias, -4.0)
        nn.init.constant_(self.ft_depth_conf[-2].weight, 0.0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, out_features, patch_h, patch_w, depth):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        feature_list = []
        # DepthPrompt 内部会做 features||depth 的提示与门控残差
        # depth = depth_normalization(depth)  # 归一化深度输入
        
        layer_4_rn, conf4 = self.ft_depth_prompt4(layer_4_rn, depth)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        feature_list.append(path_4)
        path_4, conf3 = self.ft_depth_prompt3(path_4, depth)        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # feature_list.append(path_3)
        path_3, conf2 = self.ft_depth_prompt2(path_3, depth)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # feature_list.append(path_2)
        path_2, conf1 = self.ft_depth_prompt1(path_2, depth)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        feature_list.append(path_1)

        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        temp = out
        out = self.scratch.output_conv2(out)

        conf = self.ft_depth_conf(temp)
        
        return out, conf, feature_list


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x, depth):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth, conf, features_list = self.depth_head(features, patch_h, patch_w, depth)
        depth = F.relu(depth)
        
        # 输出的是逆梯度
        return depth, conf, features_list
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
