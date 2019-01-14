from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
import numpy as np
from torch.nn import init

# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(1, 1, 3, [8, 16, 32, 64]), (64, 64, 3, [4, 8, 16, 32]), (64, 64, 5, [8, 16]),
                 (128, 128, 5, [4, 8]), (256, 256, 5, []), (512, 512, 7, [])]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}

# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out


# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))

    def forward(self, x):
        return self.main(x)


# fusion features
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# extra part
def extra_layer(base, cfg):
    feat_layers, concat_layers, scale = [], [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return base, feat_layers, concat_layers



class BCN(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, connect, extract=[3, 8, 15, 22, 29], v2=True):
        super(BCN, self).__init__()
        self.extract = extract
        self.connect = connect
        self.base = base
        self.feat = nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.v2 = v2
        if v2: self.fuse = FusionLayer()

    def forward(self, x, label=None):
        prob, back, y, num = list(), list(), list(), 0
               
        # base 
        output_list = self.base(x)  #  x_out, d1, d2, d3, d4, e4
        for index, k in enumerate(output_list):  
            y.append(self.feat[index](k))        
        # side output
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))          
        # fusion map
        if self.v2:
            # version2: learning fusion
            back.append(self.fuse(back))
        else:
            # version1: mean fusion
            back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        # add sigmoid
        for i in back: prob.append(torch.sigmoid(i))
        return prob


# build the whole network
def build_model():
    return BCN(*extra_layer(ForwardConNet(1, 3, False), extra['dss']), connect['dss'])

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class ForwardConNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # torch.Size([1, 64, 64, 64])
        e1 = self.encoder1(x)  #  torch.Size([1, 64, 64, 64])
        e2 = self.encoder2(e1)  #  torch.Size([1, 128, 32, 32])
        e3 = self.encoder3(e2)  #  torch.Size([1, 256, 16, 16])
        e4 = self.encoder4(e3)  #  torch.Size([1, 512, 8, 8])

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3  #  torch.Size([1, 256, 16, 16])
        d3 = self.decoder3(d4) + e2  #  torch.Size([1, 128, 32, 32])
        d2 = self.decoder2(d3) + e1  #  torch.Size([1, 64, 64, 64])
        d1 = self.decoder1(d2)  #  torch.Size([1, 64, 128, 128])

        # Final Classification
        f1 = self.finaldeconv1(d1)  #  torch.Size([1, 32, 257, 257])
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)  #  torch.Size([1, 32, 255, 255])
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)  #  torch.Size([1, 1, 256, 256])

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = torch.sigmoid(f5)
        return x_out, d1, d2, d3, d4, e4
    
if __name__ == '__main__':
    net = BCN(*extra_layer(ForwardConNet(1, 3, True), extra['dss']), connect['dss'])
    img = torch.randn(1, 3, 256, 256)
    out = net(img)
    k = [out[x] for x in [1, 2, 3, 6]]
    print(len(k))
