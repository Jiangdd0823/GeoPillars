import math
import torch
from torch import nn

import torch.nn.functional as F


# 构建一个用于下采样的卷积池化模块
class ConvNormLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1):
        super(ConvNormLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


# ResNet基本的Bottleneck类(Resnet50/101/152)
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩增倍数(Resnet网络的结构)

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 初始的x
        out = self.bottleneck(x)
        # 残差融合前需保证out与identity的通道数以及图像尺寸均相同
        if self.downsample is not None:
            identity = self.downsample(x)  # 初始的x采取下采样
        out += identity
        out = self.relu(out)
        return out


class FPN(nn.Module):
    '''
    FPN需要初始化一个list，代表ResNet每一个阶段的Bottleneck的数量
    '''

    def __init__(self, layers, output_channel):
        super(FPN, self).__init__()
        # 构建C1
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 自下而上搭建C2、C3、C4、C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)  # c2->c3第一个bottleneck的stride=2
        self.layer3 = self._make_layer(256, layers[2], 2)  # c3->c4第一个bottleneck的stride=2
        self.layer4 = self._make_layer(512, layers[3], 2)  # c4->c5第一个bottleneck的stride=2

        # 对C5减少通道，得到P5
        self.toplayer = nn.Conv2d(2048, output_channel, 1, 1, 0)  # 1*1卷积

        # 横向连接，保证每一层通道数一致
        self.latlayer1 = nn.Conv2d(1024, output_channel, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, output_channel, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, output_channel, 1, 1, 0)

        # 平滑处理 3*3卷积
        self.smooth = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

    # 构建C2到C5
    def _make_layer(self, planes, blocks, stride=1, downsample=None):
        # 残差连接前，需保证尺寸及通道数相同
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))

        # 更新输入输出层
        self.inplanes = planes * Bottleneck.expansion

        # 根据block数量添加bottleneck的数量
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))  # 后面层stride=1
        return nn.Sequential(*layers)  # nn.Sequential接收orderdict或者一系列模型，列表需*转化

        # 自上而下的上采样

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape  # b c h w
        # 特征x 2倍上采样(上采样到y的尺寸)后与y相加
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.relu(self.bn1(self.conv1(x)))  # 1/2
        # c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))  此时为1/4
        c2 = self.layer1(self.maxpool(c1))  # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32

        # 自上而下，横向连接
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # 平滑处理
        p5 = p5  # p5直接输出
        p4 = self.smooth(p4)
        p3 = self.smooth(p3)
        p2 = self.smooth(p2)
        # return p2, p3, p4, p5
        return p5, p4, p3, p2


# 传入生成c2 c3 c4 c5特征层的bottleneck的堆叠数量，返回p2 p3 p4 p5
class PAFPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(PAFPN, self).__init__()
        self.fpn = FPN(in_channel_list, out_channel)
        self.bottom_up = ConvNormLayer(out_channel, out_channel, 3, 2)  # 2倍下采样模块
        self.inner_layer = []  # 1x1卷积，统一通道数，处理P3-P5的输出,这里要注意P2和P3-P5的输入通道是不同的，可以看2.1图3，
        self.out_layer = []  # 3x3卷积，对concat后的特征图进一步融合
        for i in range(len(in_channel_list)):
            if i == 0:
                self.inner_layer.append(nn.Conv2d(out_channel, out_channel, 1))  # 处理P2
            else:
                # self.inner_layer.append(nn.Conv2d(out_channel, out_channel, 1))  # 处理P3-P5
                self.inner_layer.append(nn.Conv2d(out_channel * 2, out_channel, 1))  # 处理P3-P5
            self.out_layer.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))

    def forward(self, x):
        head_output = []  # 存放最终输出特征图
        fpn_out = self.fpn(x)  # FPN操作
        print('------------FPN--PAFPN--------------')  # PAFPN操作分割线
        corent_inner = self.inner_layer[0](fpn_out[-1])  # 过1x1卷积，对P2统一通道数操作
        head_output.append(self.out_layer[0](corent_inner))  # 过3x3卷积，对统一通道后过的特征进一步融合，加入head_output列表
        print(self.out_layer[0](corent_inner).shape)
        for i in range(1, len(fpn_out), 1):
            pre_bottom_up = corent_inner
            pre_concat = self.bottom_up(pre_bottom_up)  # 下采样
            pre_inner = torch.concat([fpn_out[-1 - i], pre_concat], 1)  # concat
            corent_inner = self.inner_layer[i](pre_inner)  # 1x1卷积压缩通道
            head_output.append(self.out_layer[i](corent_inner))  # 3x3卷积进一步融合
            print(self.out_layer[i](corent_inner).shape)

        return head_output


if __name__ == '__main__':
    import os
    # device = torch.device('cuda:1')
    # fpn = FPN(layers=[5, 5, 5, 5])
    pafpn = PAFPN([5, 5, 5, 5], 256)
    img = torch.rand(1, 3, 448, 448)
    out = pafpn(img)
    print(out)

