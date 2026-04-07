import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clsts_assign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        weight = torch.from_numpy(self.alpha * clsts_assign).unsqueeze(2).unsqueeze(3)
        self.conv.weight = nn.Parameter(weight)
        self.conv.bias = None

    def forward(self, x):
        n, c = x.shape[:2]
        x_flatten = x.view(n, c, -1)

        soft_assign = self.conv(x).view(n, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        vlad = torch.zeros([n, self.num_clusters, c], dtype=x.dtype, layout=x.layout, device=x.device)
        for cluster_idx in range(self.num_clusters):
            residual = (
                x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
                - self.centroids[cluster_idx : cluster_idx + 1, :]
                .expand(x_flatten.size(-1), -1, -1)
                .permute(1, 2, 0)
                .unsqueeze(0)
            )
            residual *= soft_assign[:, cluster_idx : cluster_idx + 1, :].unsqueeze(2)
            vlad[:, cluster_idx : cluster_idx + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _build_resnet34_encoder():
    encoder = ResNet(BasicBlock, [3, 4, 6, 3])
    layers = list(encoder.children())[:-4]
    return nn.Sequential(*layers)


class REM(nn.Module):
    def __init__(self, rotations=8):
        super().__init__()
        self.encoder = _build_resnet34_encoder()
        self.register_buffer(
            "angles",
            -torch.arange(0, 359.00001, 360.0 / rotations) / 180.0 * torch.pi,
        )

    def forward(self, x):
        equ_features = []
        batch_size = x.size(0)
        x_dtype = x.dtype
        x_device = x.device

        for i in range(len(self.angles)):
            aff = torch.zeros(batch_size, 2, 3, device=x_device, dtype=x_dtype)
            aff[:, 0, 0] = torch.cos(-self.angles[i])
            aff[:, 0, 1] = torch.sin(-self.angles[i])
            aff[:, 1, 0] = -torch.sin(-self.angles[i])
            aff[:, 1, 1] = torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True)
            warped_im = F.grid_sample(x, grid, align_corners=True, mode="bicubic")

            out = self.encoder(warped_im)
            if i == 0:
                init_size = out.size()

            aff = torch.zeros(batch_size, 2, 3, device=x_device, dtype=x_dtype)
            aff[:, 0, 0] = torch.cos(self.angles[i])
            aff[:, 0, 1] = torch.sin(self.angles[i])
            aff[:, 1, 0] = -torch.sin(self.angles[i])
            aff[:, 1, 1] = torch.cos(self.angles[i])
            grid = F.affine_grid(aff, torch.Size(init_size), align_corners=True)
            out = F.grid_sample(out, grid, align_corners=True, mode="bicubic")

            equ_features.append(out.unsqueeze(-1))

        equ_features = torch.cat(equ_features, axis=-1)
        equ_features = torch.max(equ_features, dim=-1, keepdim=False)[0]

        aff = torch.zeros(batch_size, 2, 3, device=x_device, dtype=x_dtype)
        aff[:, 0, 0] = 1.0
        aff[:, 1, 1] = 1.0

        b, c, h, w = x.size()
        grid = F.affine_grid(aff, torch.Size((b, c, h // 4, w // 4)), align_corners=True)
        out1 = F.grid_sample(equ_features, grid, align_corners=True, mode="bicubic")
        out1 = F.normalize(out1, dim=1)

        grid = F.affine_grid(aff, torch.Size((b, c, h, w)), align_corners=True)
        out2 = F.grid_sample(equ_features, grid, align_corners=True, mode="bicubic")
        out2 = F.normalize(out2, dim=1)
        return out1, out2


class REIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rem = REM()
        self.pooling = NetVLAD()
        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim * 64

    def forward(self, x):
        out1, local_feats = self.rem(x)
        global_desc = self.pooling(out1)
        return out1, local_feats, global_desc
