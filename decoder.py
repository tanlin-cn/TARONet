import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import math
from segment_anything.modeling.common import LayerNorm2d
import ASConv

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(DecoderBlock, self).__init__()
        asconv_size = [5,9,13,17]
        self.asconv = ASConv.ASConv2(in_channels, asconv_size[1])
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1,
                                        output_padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels, filters, 1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU()
        self._init_weight()


    def forward(self, x):
        x = self.asconv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_topo(nn.Module):
    def __init__(self, config, filters = [96, 192, 384, 768]):
        super(Decoder_topo, self).__init__()
        self.config = config
        in_inplanes = filters[3]

        if self.config.PATCH_SIZE == 400:
            self.special_up = True
        else:
            self.special_up = False

        self.decoder4 = DecoderBlock(in_inplanes, filters[1])
        self.decoder3 = DecoderBlock(filters[2], filters[0])
        self.decoder2 = DecoderBlock(filters[1], int(filters[0] / 2))
        self.decoder1 = DecoderBlock(filters[0], int(filters[0] / 2))

        self.conv_e3 = nn.Sequential(nn.Conv2d(filters[2], filters[1], 1, bias=False),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(filters[1], filters[0], 1, bias=False),
                                     nn.BatchNorm2d(filters[0]),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(filters[0], int(filters[0]/2), 1, bias=False),
                                     nn.BatchNorm2d(int(filters[0]/2)),
                                     nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(int(filters[0]/2), int(filters[0]/2), 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(int(filters[0]/2)),
                                    nn.ReLU())

        self._init_weight()


    def forward(self, x):
        e1 = x[0]
        e2 = x[1]
        e3 = x[2]
        e4 = x[3]

        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        d1 = self.decoder1(d2)
        x = self.deconv(d1)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    import torch

    decoder = Decoder_topo(filters=[96, 192, 384, 768])
    # e1 = torch.rand(2, 256, 128, 128)
    # e2 = torch.rand(2, 512, 64, 64)
    # e3 = torch.rand(2, 1024, 32,32)
    # e4 = torch.rand(2, 2048, 16,16)
    e1 = torch.rand(1, 96, 128, 128)
    e2 = torch.rand(1, 192, 64, 64)
    e3 = torch.rand(1, 384, 32,32)
    e4 = torch.rand(1, 768, 16,16) # torch.Size([1, 12, 256, 256])
    feature = (e1, e2, e3, e4)
    x = decoder(feature)
    print(x.shape)
    # torch.Size([2, 64, 512, 512])

