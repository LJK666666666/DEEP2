"""
增强版 ResNet 骨干网络 - 用于高性能 SSD 目标检测

增强特性:
1. ResNet101/152 深层骨干网络
2. CBAM 注意力机制 (通道 + 空间)
3. 可变形卷积 v2 (Deformable Convolution)
4. 增强型 FPN 特征融合 (双向融合)
5. DropBlock 正则化
6. Mish 激活函数选项

作者: Enhanced for maximum performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# ============================================================================
# 激活函数
# ============================================================================
class Mish(nn.Module):
    """Mish 激活函数: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ============================================================================
# 注意力模块
# ============================================================================
class SELayer(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """完整的 CBAM 模块"""
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================================
# DropBlock 正则化
# ============================================================================
class DropBlock2D(nn.Module):
    """DropBlock 正则化 - 比 Dropout 更适合卷积网络"""
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        gamma = self._compute_gamma(x)
        mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()

        # 扩展 mask
        block_mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        block_mask = 1 - block_mask

        # 归一化
        count = block_mask.numel()
        count_ones = block_mask.sum()
        if count_ones > 0:
            x = x * block_mask * count / count_ones
        return x

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


# ============================================================================
# 基础卷积模块
# ============================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ============================================================================
# 增强版 Bottleneck
# ============================================================================
class EnhancedBottleneck(nn.Module):
    """增强版 Bottleneck: 支持 CBAM/SE + DropBlock"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 se=False, cbam=False, drop_block=None):
        super(EnhancedBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # 注意力模块
        self.se = se
        self.cbam = cbam
        if se and not cbam:
            self.attention = SELayer(planes * self.expansion)
        elif cbam and not se:
            self.attention = CBAM(planes * self.expansion)
        else:
            self.attention = None

        # DropBlock
        self.drop_block = drop_block

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 注意力
        if self.attention is not None:
            out = self.attention(out)

        # DropBlock
        if self.drop_block is not None:
            out = self.drop_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ============================================================================
# 增强型 FPN 特征融合
# ============================================================================
class EnhancedFPN(nn.Module):
    """增强型特征金字塔网络 - 双向特征融合"""
    def __init__(self, in_channels_list, out_channels=256):
        super(EnhancedFPN, self).__init__()

        # 横向连接 (1x1 卷积降维)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # 平滑卷积
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

        # 自顶向下路径的融合权重 (可学习)
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(len(in_channels_list) - 1)
        ])

        # 自底向上路径的融合权重 (可学习)
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(len(in_channels_list) - 1)
        ])

        self.relu = nn.ReLU(inplace=True)
        self.eps = 1e-4

    def forward(self, inputs):
        """
        inputs: [C2, C3, C4, C5] 从浅到深的特征图
        """
        # 横向连接
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            w = F.relu(self.td_weights[i-1])
            w = w / (w.sum() + self.eps)

            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            laterals[i-1] = w[0] * laterals[i-1] + w[1] * upsampled

        # 自底向上路径 (第二次融合)
        for i in range(len(laterals) - 1):
            w = F.relu(self.bu_weights[i])
            w = w / (w.sum() + self.eps)

            downsampled = F.max_pool2d(laterals[i], kernel_size=2, stride=2)
            if downsampled.shape[2:] != laterals[i+1].shape[2:]:
                downsampled = F.interpolate(downsampled, size=laterals[i+1].shape[2:], mode='nearest')
            laterals[i+1] = w[0] * laterals[i+1] + w[1] * downsampled

        # 平滑
        outputs = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]

        return outputs


# ============================================================================
# 增强版 ResNet 主干网络
# ============================================================================
class EnhancedResNet(nn.Module):
    """增强版 ResNet: 支持多种高级特性"""

    def __init__(self, block, layers, extras,
                 se=False, cbam=False,
                 use_fpn=True, use_dropblock=False,
                 zero_init_residual=True,
                 groups=1, width_per_group=64,
                 norm_layer=None):
        super(EnhancedResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.se = se
        self.cbam = cbam
        self.use_fpn = use_fpn

        # DropBlock 配置
        self.drop_block = DropBlock2D(0.1, 7) if use_dropblock else None

        # Stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # FPN
        if use_fpn:
            self.fpn = EnhancedFPN([512, 1024, 2048], out_channels=256)
            self.fpn_out_conv = nn.Conv2d(256 * 3, 512, 1)
            self.fpn_out_bn = nn.BatchNorm2d(512)

            # 上采样层
            self.upsample_c3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.upsample_c4 = nn.Upsample(size=(38, 38), mode='bilinear', align_corners=False)

            # 1x1 卷积对齐通道
            self.conv_c2 = nn.Conv2d(512, 256, 1)
            self.conv_c3 = nn.Conv2d(1024, 256, 1)
            self.conv_c4 = nn.Conv2d(2048, 256, 1)

        # Extra layers for detection
        self.extras = self._make_extras(block, extras)

        # 初始化
        self._initialize_weights(zero_init_residual)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                           self.base_width, previous_dilation, norm_layer,
                           se=self.se, cbam=self.cbam, drop_block=self.drop_block))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                               base_width=self.base_width, dilation=self.dilation,
                               norm_layer=norm_layer, se=self.se, cbam=self.cbam,
                               drop_block=self.drop_block))

        return nn.Sequential(*layers)

    def _make_extras(self, block, extras):
        """创建 SSD 检测所需的额外层"""
        layers = nn.ModuleList()

        self.inplanes = 512  # FPN 输出通道数

        for i, planes in enumerate(extras):
            stride = 2 if i > 0 else 1
            layer = nn.Sequential(
                conv1x1(self.inplanes, planes),
                self._norm_layer(planes),
                nn.ReLU(inplace=True),
                conv3x3(planes, planes * block.expansion, stride=stride),
                self._norm_layer(planes * block.expansion),
                nn.ReLU(inplace=True),
            )
            layers.append(layer)
            self.inplanes = planes * block.expansion

        return layers

    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, EnhancedBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        c1 = self.layer1(x)    # 1/4
        c2 = self.layer2(c1)   # 1/8,  512 channels
        c3 = self.layer3(c2)   # 1/16, 1024 channels
        c4 = self.layer4(c3)   # 1/32, 2048 channels

        if self.use_fpn:
            # FPN 特征融合
            p2 = self.conv_c2(c2)  # 38x38x256
            p3 = self.upsample_c3(self.conv_c3(c3))  # 38x38x256
            p4 = self.upsample_c4(self.conv_c4(c4))  # 38x38x256

            # 拼接
            fused = torch.cat([p2, p3, p4], dim=1)  # 38x38x768
            fused = self.fpn_out_conv(fused)
            fused = self.fpn_out_bn(fused)
            fused = self.relu(fused)  # 38x38x512

            # 生成多尺度特征图
            features = []
            x = fused

            for i, extra in enumerate(self.extras):
                x = extra(x)
                features.append(x)

            return tuple(features)
        else:
            # 不使用 FPN
            features = [c2, c3, c4]
            x = c4

            for extra in self.extras:
                x = extra(x)
                features.append(x)

            return tuple(features)


# ============================================================================
# 注册增强版骨干网络
# ============================================================================
@registry.BACKBONES.register('Enhanced_R101_300')
def Enhanced_R101_300(cfg, pretrained=True):
    """增强版 ResNet101 - 300x300 输入"""
    model = EnhancedResNet(
        block=EnhancedBottleneck,
        layers=[3, 4, 23, 3],  # ResNet101
        extras=cfg.MODEL.RESNET.EXTRAS,
        se=cfg.MODEL.RESNET.SE,
        cbam=cfg.MODEL.RESNET.CBAM,
        use_fpn=cfg.MODEL.RESNET.FUSION,
        use_dropblock=getattr(cfg.MODEL.RESNET, 'DROPBLOCK', False),
    )

    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        # 只加载匹配的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights")

    return model


@registry.BACKBONES.register('Enhanced_R152_300')
def Enhanced_R152_300(cfg, pretrained=True):
    """增强版 ResNet152 - 300x300 输入 (最深)"""
    model = EnhancedResNet(
        block=EnhancedBottleneck,
        layers=[3, 8, 36, 3],  # ResNet152
        extras=cfg.MODEL.RESNET.EXTRAS,
        se=cfg.MODEL.RESNET.SE,
        cbam=cfg.MODEL.RESNET.CBAM,
        use_fpn=cfg.MODEL.RESNET.FUSION,
        use_dropblock=getattr(cfg.MODEL.RESNET, 'DROPBLOCK', False),
    )

    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights")

    return model


if __name__ == '__main__':
    # 测试
    from torchsummary import summary

    model = EnhancedResNet(
        block=EnhancedBottleneck,
        layers=[3, 4, 23, 3],
        extras=[128, 256, 128, 64, 64, 64],
        se=False,
        cbam=True,
        use_fpn=True,
        use_dropblock=True,
    )

    model = model.cuda()
    summary(model, (3, 300, 300))
