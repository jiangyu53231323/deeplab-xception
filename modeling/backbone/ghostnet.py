import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _c(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class My_GhostNet(nn.Module):
    def __init__(self, w=1.0):
        super(My_GhostNet, self).__init__()

        # building first layer
        self.conv_stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)

        # building inverted residual blocks
        block = GhostBottleneck
        self.block1 = nn.Sequential(
            block(16, _c(16 * w), _c(16 * w), dw_kernel_size=3, stride=1, se_ratio=0),
        )
        self.block2 = nn.Sequential(
            block(_c(16 * w), _c(48 * w), _c(24 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(24 * w), _c(72 * w), _c(24 * w), dw_kernel_size=3, stride=1, se_ratio=0),
        )
        self.block3 = nn.Sequential(
            block(_c(24 * w), _c(72 * w), _c(40 * w), dw_kernel_size=5, stride=2, se_ratio=0.25),
            block(_c(40 * w), _c(120 * w), _c(40 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        )
        self.block4 = nn.Sequential(
            block(_c(40 * w), _c(240 * w), _c(80 * w), dw_kernel_size=3, stride=2, se_ratio=0),
            block(_c(80 * w), _c(200 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(184 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(184 * w), _c(80 * w), dw_kernel_size=3, stride=1, se_ratio=0),
            block(_c(80 * w), _c(480 * w), _c(112 * w), dw_kernel_size=3, stride=1, se_ratio=0.25),
            block(_c(112 * w), _c(672 * w), _c(112 * w), dw_kernel_size=3, stride=1, se_ratio=0.25),
        )
        self.block5 = nn.Sequential(
            block(_c(112 * w), _c(672 * w), _c(160 * w), dw_kernel_size=5, stride=2, se_ratio=0.25),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0),
            block(_c(160 * w), _c(960 * w), _c(160 * w), dw_kernel_size=5, stride=1, se_ratio=0.25),
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # x = self.blocks(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        low_level_feat = x3
        x4 = self.block4(x3)
        middle_level_feat = x4
        x5 = self.block5(x4)

        return x5, low_level_feat, middle_level_feat


def my_ghostnet(output_stride=16, BatchNorm=None):
    model = My_GhostNet(w=1.0)
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 640, 640)
    net = my_ghostnet()
    y, low_level_feat = net(input)
    print(y.size())
    print(low_level_feat.size())
