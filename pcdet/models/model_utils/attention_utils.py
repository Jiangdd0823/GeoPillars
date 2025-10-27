import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F


# CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)


# SE
class SE(nn.Module):
    def __init__(self, c1, ratio=8):
        super().__init__()
        # c*1*1
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # _, b, c = x.size()
        y = self.l1(x)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        # y = y.view(b, c)
        return x * y.expand_as(x)


class SE_plus(nn.Module):
    def __init__(self, input_channel, ratio=8, weight=[0.5, 1.5]):
        super().__init__()

        self.low = weight[1]
        self.high = weight[0]
        self.l1_low = nn.Linear(input_channel, input_channel // ratio, bias=False)
        # revise 1123
        self.l1_high = nn.Linear(input_channel, input_channel * ratio, bias=False)
        # self.l1_high = nn.Linear(input_channel, input_channel * ratio // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        l2_input = int(input_channel * ratio + input_channel * 1/ratio)
        # l2_input = int(input_channel * ratio // 2 + input_channel * 1/ratio)
        self.l2 = nn.Linear(l2_input, input_channel, bias=False)
        self.sig = nn.Sigmoid()
        # revise 1127
        # self.norm = nn.BatchNorm1d(input_channel)

    def forward(self, x):
        y_low = self.l1_low(x)
        y_high = self.l1_high(x)
        y = torch.cat([y_low * self.low, y_high * self.high], dim=-1)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        # revise 1127
        # y = self.norm(y.permute(1, 2, 0)).permute(2, 0, 1)
        return x * y.expand_as(x)


# 效果不好
class SE_plus1110(nn.Module):
    def __init__(self, input_channel, ratio=8, weight=[0.5, 1.5]):
        super().__init__()

        self.low = weight[1]
        self.high = weight[0]
        self.l1_low = nn.Linear(input_channel, input_channel // ratio, bias=False)
        self.l1_high = nn.Linear(input_channel, input_channel * ratio // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.l2_low = nn.Linear(input_channel // ratio, input_channel, bias=True)
        self.l2_high = nn.Linear(input_channel * ratio // 2, input_channel, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y_low = self.l1_low(x)
        y_low = self.relu(y_low)
        y_high = self.l1_high(x)
        y_high = self.relu(y_high)

        y_low = self.l2_low(y_low)
        y_high = self.l2_high(y_high)
        y = y_high * self.high + y_low * self.low

        y = self.sig(y)

        return x * y.expand_as(x)


class SE_plus1120(nn.Module):
    def __init__(self, input_channel, ratio=8, weight=[0.5, 1.5]):
        super().__init__()

        self.channel_wise = nn.Sequential(
            nn.Linear(input_channel, input_channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(input_channel // ratio, input_channel, bias=False)
        )
        # self.global_wise = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(-1),
        # )
        self.sig = nn.Sigmoid()
        self.l2 = nn.Sequential(
            nn.Linear(input_channel * 2, input_channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_channel = self.channel_wise(x)
        y_global = torch.mean(x, dim=2).unsqueeze(-1)
        y_global = self.sig(y_global)
        y_global = x * y_global.expand_as(x)
        y = torch.cat([y_global, y_channel], dim=-1)
        y = self.l2(y)

        return x * y.expand_as(x)


class SplitAttention(nn.Module):
    def __init__(self, inp, oup, split_size=16):
        super(SplitAttention, self).__init__()
        reduce_ratio = inp // oup
        self.split_num = inp // split_size
        self.split_size = split_size

        self.l1 = nn.ModuleList()
        for i in range(self.split_num ):
            l1 = nn.Sequential(
                nn.Linear(inp//self.split_num, inp//self.split_num //reduce_ratio, bias=False),
                nn.ReLU(),
            )
            self.l1.append(l1)

        self.l2_low = nn.Sequential(
            nn.Linear(oup, oup//self.split_num , bias=False),
            nn.ReLU()
        )
        self.l2_high = nn.Sequential(
            nn.Linear(oup//self.split_num, oup, bias=False),
            nn.Sigmoid(),
        )
        # self.l2_norm = nn.BatchNorm1d(oup)

    def forward(self, feat, geo_feat):
        # 两特征交叠
        feature = torch.stack([feat, geo_feat], dim=3)
        _, n, d, c = feature.shape
        feature = feature.view(_, n, d*c)
        feature_split = torch.split(feature, self.split_size, dim=-1)
        feature = [self.l1[i](feature_split[i]) for i in range(self.split_num)]
        feature = torch.cat(feature, dim=-1)
        feature = self.l2_low(feature)
        feature = self.l2_high(feature)
        # feature = self.l2_norm(feature.permute(1, 2, 0)).permute(2, 0, 1)
        ouput = feat * feature
        return ouput


class SplitAttention1123(nn.Module):
    def __init__(self, inp, ratio=8):
        super().__init__()
        self.senet = SE(inp, ratio)
        self.l1 = nn.Sequential(
            nn.Linear(inp, inp, bias=False),
            # nn.BatchNorm1d(inp)
        )
        self.l2 = nn.Sequential(
            nn.Linear(inp, inp, bias=False),
            # nn.BatchNorm1d(inp)
        )

    def forward(self, feat, geo_feat):
        # 两特征交叠
        # feature = torch.cat([feat, geo_feat], dim=-1)
        feature = feat + geo_feat
        feature = self.senet(feature)
        feat1 = self.l1(feature)
        feat1_so = F.softmax(feat1, dim=-1)
        feat2 = self.l2(feature)
        feat2_so = F.softmax(feat2, dim=-1)
        ouput = feat1_so * feat + feat2_so * geo_feat

        return ouput


class SplitAttentionImage(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(inp, inp, bias=False),
        )
        # self.norm = nn.BatchNorm1d(inp)
        self.relu = nn.ReLU()
        self.l1 = nn.Sequential(
            nn.Linear(inp, inp, bias=False),
            # nn.BatchNorm1d(inp)
        )
        self.l2 = nn.Sequential(
            nn.Linear(inp, inp, bias=False),
            # nn.BatchNorm1d(inp)
        )
        self.l3 = nn.Sequential(
            nn.Linear(inp, inp, bias=False),
            # nn.BatchNorm1d(inp)
        )

    def forward(self, x):
        x1 = x[:, :128]
        x2 = x[:, 128:256]
        x3 = x[:, 256:]
        x = x1 + x2 + x3
        b, c, h, w = x.size()
        x_global = self.pool(x).view(b, c)
        x_global = self.mlp(x_global)
        x_global = self.relu(x_global)
        x1_channel = self.l1(x_global)
        x2_channel = self.l2(x_global)
        x3_channel = self.l3(x_global)
        x1_channel = F.softmax(x1_channel, dim=-1).view(b, c, 1, 1)
        x2_channel = F.softmax(x2_channel, dim=-1).view(b, c, 1, 1)
        x3_channel = F.softmax(x3_channel, dim=-1).view(b, c, 1, 1)
        x1 = x1 * x1_channel.expand_as(x1)
        x2 = x2 * x2_channel.expand_as(x1)
        x3 = x3 * x3_channel.expand_as(x1)

        return torch.cat([x1, x2, x3], dim=1)


# todo LLC SplitAttention注意力机制
class SplitAttentionLLC(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Linear(c, c, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(c, c * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        b, k, c, h, w = x.shape  # (b, k, c, h, w): (8, 3, 64, 246, 216 )
        x = x.reshape(b, k, c, -1)                  # b,k,c,n
        a = torch.sum(torch.sum(x, 1), -1)          # b,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # b,kc
        hat_a = hat_a.reshape(b, self.k, c)         # b,k,c
        bar_a = self.softmax(hat_a)                 # b,k,c
        attention = bar_a.unsqueeze(-1)             # b,k,c,1
        out = attention * x                         # b,k,c,n
        # revise 1204 相加试试
        # out = torch.sum(out, 1).reshape(b, c, h, w) # b,c,n -> b,c,h,w
        # revise 1203 llc 是将三个维度上相加，我改成cat
        out = out.reshape(b, -1, h, w)
        return out


class CoordAtt(nn.Module):
    """
    CA 注意力模块
    论文：Coordinate Attention for Efficient Mobile Network Design
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class SimAM(torch.nn.Module):
    """
    SimAM 注意力模块
    SimAM：A Simple，Parameter-Free Attention Module for Convolutional Neural Networks
    https://proceedings.mlr.press/v139/yang21o/yang21o.pdf
    https://github.com/ZjjConan/SimAM
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class SKAttention(nn.Module):
    """
    Selective Kernel Attention 注意力模块
    https://arxiv.org/pdf/1903.06586.pdf
    https://github.com/developer0hye/SKNet-PyTorch/tree/master
    """
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        # fuse
        U = sum(conv_outs)  # bs,c,h,w

        # reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class SKAttention_voxel(nn.Module):
    """
    Selective Kernel Attention 注意力模块
    https://arxiv.org/pdf/1903.06586.pdf
    https://github.com/developer0hye/SKNet-PyTorch/tree/master
    """
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

        self.conv_global = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(32, 64), padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, x):
        n, npoint, p_dim = x.size()
        x = x.unsqueeze(1).expand(n, 64, npoint, p_dim)
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        # fuse
        U = sum(conv_outs)  # bs,c,h,w

        # reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        out = self.conv_global(V)
        return out.squeeze()


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0")
    # attn = CoordAtt(256, 128)
    # attn = SimAM()
    attn = SpatialGroupEnhance(groups=8).to(device)
    # attn = SKAttention_voxel(64).to(device)
    non_voxel_feature = torch.rand(10000, 32, 64).to(device)
    image = torch.rand(2, 256, 100, 200).to(device)
    # ouput = attn(non_voxel_feature)
    ouput = attn(image)

    print(ouput)
