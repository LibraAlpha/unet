import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleConv, self).__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 输入
class InConv(nn.Module):
    conv: DoubleConv

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 先进行maxpool，再进行两层链接
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


# decoder中的向上传播：核心部分
class Up(nn.Module):
    """
    up path
    conv_transpose => double_conv
    """

    def __init__(self, in_ch, out_ch, Transpose=False):
        super(Up, self).__init__()

        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )

        self.conv = DoubleConv(in_ch, out_ch)
        self.up.apply(self.init_weights)

    def forward(self, x1, x2):  # x2是左侧的输出，x1是上一大层来的输出
        """
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]  # [N,C,H,W],diffY refers to height
        diffX = x2.size()[3] - x1.size()[3]  # [N,C,H,W],diffX refers to width

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)  # 在通道层将skip传递过来的数据与下层传递来的数据进行拼接
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
