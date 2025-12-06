import torch.nn as nn
import torch
torch.manual_seed(1)
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissStarAttention(nn.Module):
    """ Criss-Star Attention Module"""

    def __init__(self, in_dim):
        super(CrissStarAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.weight = nn.Parameter(torch.rand(4))

    def forward(self, x):
        batchsize, _, height, width = x.size()
        query = self.query_conv(x)
        # (b, w, c, h) -> (bw, c, h) -> (bw, h, c)
        query_H = query.permute(0, 3, 1, 2).contiguous().view(batchsize * width, -1, height).permute(0, 2, 1)
        dk_H = query_H.size(-1)
        # (b, h, c, w) -> (bh, c, w) -> (bh, w, c)
        query_W = query.permute(0, 2, 1, 3).contiguous().view(batchsize * height, -1, width).permute(0, 2, 1)
        dk_W = query_W.size(-1)
        # (b, c, h, 2w-1) -> (b, 2w-1, c, h) -> (b(2w-1), c, h) -> (b(2w-1), h, c)
        query_ldiagonal = self.h_transform(query).permute(0, 3, 1, 2).contiguous().view(batchsize * (2 * width - 1), -1, height).permute(0, 2, 1)
        dk_ldiagonal = query_ldiagonal.size(-1)
        # (b, c, 2h-1, w) -> (b, 2h-1, c, w) -> (b(2h-1), c, w) -> (b(2h-1), w, c)
        query_rdiagonal = self.v_transform(query).permute(0, 2, 1, 3).contiguous().view(batchsize * (2 * height - 1), -1, width).permute(0, 2, 1)
        dk_rdiagonal = query_rdiagonal.size(-1)

        key = self.key_conv(x)
        # (b, w, c, h) -> (bw, c, h)
        key_H = key.permute(0, 3, 1, 2).contiguous().view(batchsize * width, -1, height)
        # (b, h, c, w) -> (bh, c, w)
        key_W = key.permute(0, 2, 1, 3).contiguous().view(batchsize * height, -1, width)
        # (b, c, h, 2w-1) -> (b, 2w-1, c, h) -> (b(2w-1), c, h)
        key_ldiagonal = self.h_transform(key).permute(0, 3, 1, 2).contiguous().view(batchsize * (2 * width - 1), -1, height)
        # (b, c, 2h-1, w) -> (b, 2h-1, c, w) -> (b(2h-1), c, w)
        key_rdiagonal = self.v_transform(key).permute(0, 2, 1, 3).contiguous().view(batchsize * (2 * height - 1), -1, width)

        value = self.value_conv(x)
        # (b, w, c, h) -> (bw, c, h)
        value_H = value.permute(0, 3, 1, 2).contiguous().view(batchsize * width, -1, height)
        # (b, h, c, w) -> (bh, c, w)
        value_W = value.permute(0, 2, 1, 3).contiguous().view(batchsize * height, -1, width)
        # (b, c, h, 2w-1) -> (b, 2w-1, c, h) -> (b(2w-1), c, h)
        value_ldiagonal = self.h_transform(value).permute(0, 3, 1, 2).contiguous().view(batchsize * (2 * width - 1), -1, height)
        # (b, c, 2h-1, w) -> (b, 2h-1, c, w) -> (b(2h-1), c, w)
        value_rdiagonal = self.v_transform(value).permute(0, 2, 1, 3).contiguous().view(batchsize * (2 * height - 1), -1, width)

        # (bw, h, c) * (bw, c, h) -> (bw, h, h) -> (b, w, h, h) -> (b, h, w, h)
        # energy_H = (torch.bmm(query_H, key_H) + self.INF(batchsize, height, width)).view(batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_H = torch.bmm(query_H, key_H)
        energy_H = energy_H / torch.sqrt(torch.tensor(dk_H, dtype=torch.float32))
        energy_H = (energy_H + self.INF(batchsize, height, width)).view(batchsize, width, height, height).permute(0, 2, 1, 3)
        # (bh, w, c) * (bh, c, w) -> (bh, w, w) -> (b, h, w, w)
        # energy_W = (torch.bmm(query_W, key_W) + self.INF(batchsize, width, height)).view(batchsize, height, width, width)
        energy_W = torch.bmm(query_W, key_W)
        energy_W = energy_W / torch.sqrt(torch.tensor(dk_W, dtype=torch.float32))
        # energy_W = (energy_W + self.INF(batchsize, width, height)).view(batchsize, height, width, width)
        energy_W = energy_W.view(batchsize, height, width, width)
        # (b(2w-1), h, c) * (b(2w-1), c, h) -> (b(2w-1), h, h) -> (b, 2w-1, h, h) -> (b, h, h, 2w-1) -> (b, h, h, w) -> (b, h, w, h)
        # energy_ldiagonal = self.inv_h_transform(
        #     (torch.bmm(query_ldiagonal, key_ldiagonal) + self.INF(batchsize, height, 2 * width - 1)).view(batchsize, 2 * width - 1, height, height).permute(0, 3, 2, 1)
        # ).permute(0, 1, 3, 2)
        energy_ldiagonal = torch.bmm(query_ldiagonal, key_ldiagonal)
        energy_ldiagonal = energy_ldiagonal / torch.sqrt(torch.tensor(dk_ldiagonal, dtype=torch.float32))
        energy_ldiagonal = self.inv_h_transform(
            (energy_ldiagonal + self.INF(batchsize, height, 2 * width - 1)).view(batchsize, 2 * width - 1, height, height).permute(0, 3, 2, 1)
        ).permute(0, 1, 3, 2)
        # (b(2h-1), w, c) * (b(2h-1), c, w) -> (b(2h-1), w, w) -> (b, 2h-1, w, w) -> (b, w, 2h-1, w) -> (b, w, h, w) -> (b, h, w, w)
        # energy_rdiagonal = self.inv_v_transform(
        #     (torch.bmm(query_rdiagonal, key_rdiagonal) + self.INF(batchsize, width, 2 * height - 1)).view(batchsize, 2 * height - 1, width, width).permute(0, 2, 1, 3)
        # ).permute(0, 2, 1, 3)
        energy_rdiagonal = torch.bmm(query_rdiagonal, key_rdiagonal)
        energy_rdiagonal = energy_rdiagonal / torch.sqrt(torch.tensor(dk_rdiagonal, dtype=torch.float32))
        # energy_rdiagonal = self.inv_v_transform(
        #     (energy_rdiagonal + self.INF(batchsize, width, 2 * height - 1)).view(batchsize, 2 * height - 1, width, width).permute(0, 2, 1, 3)
        # ).permute(0, 2, 1, 3)
        energy_rdiagonal = self.inv_v_transform(
            energy_rdiagonal.view(batchsize, 2 * height - 1, width, width).permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)
        concate = self.softmax(torch.cat([energy_H, energy_W, energy_ldiagonal, energy_rdiagonal], 3))

        # (b, w, h, h) -> (bw, h, h)
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(batchsize * width, height, height)
        # (b, h, w, w) -> (bh, w, w)
        att_W = concate[:, :, :, height:height + width].contiguous().view(batchsize * height, width, width)
        # (b, h, w, h) -> (b, h, h, w) -> (b, h, h, 2w-1) -> (b, 2w-1, h, h) -> (b(2w-1), h, h)
        att_ldiagonal = self.h_transform(
            concate[:, :, :, height + width:height * 2 + width].permute(0, 1, 3, 2)
        ).permute(0, 3, 2, 1).contiguous().view(batchsize * (2 * width - 1), height, height)
        # (b, h, w, w) -> (b, w, h, w) -> (b, w, 2h-1, w) -> (b, 2h-1, w, w) -> (b(2h-1), w, w)
        att_rdiagonal = self.v_transform(
            concate[:, :, :, height * 2 + width:(height + width) * 2].permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3).contiguous().view(batchsize * (2 * height - 1), width, width)

        # (bw, c, h) * (bw, h, h) -> (bw, c, h) -> (b, w, c, h) -> (b, c, h, w)
        out_H = torch.bmm(value_H, att_H.permute(0, 2, 1)).view(batchsize, width, -1, height).permute(0, 2, 3, 1)
        # (bh, c, w) * (bh, w, w) -> (bh, c, w) -> (b, h, c, w) -> (b, c, h, w)
        out_W = torch.bmm(value_W, att_W.permute(0, 2, 1)).view(batchsize, height, -1, width).permute(0, 2, 1, 3)
        # (b(2w-1), c, h) * (b(2w-1), h, h) -> (b(2w-1), c, h) -> (b, 2w-1, c, h) -> (b, c, h, 2w-1) -> (b, c, h, w)
        out_ldiagonal = self.inv_h_transform(
            torch.bmm(value_ldiagonal, att_ldiagonal.permute(0, 2, 1)).view(batchsize, 2 * width - 1, -1, height).permute(0, 2, 3, 1)
        )
        # (b(2h-1), c, w) * (b(2h-1), w, w) -> (b(2h-1), c, w) -> (b, 2h-1, c, w) -> (b, c, 2h-1, w) -> (b, c, h, w)
        out_rdiagonal = self.inv_v_transform(
            torch.bmm(value_rdiagonal, att_rdiagonal.permute(0, 2, 1)).view(batchsize, 2 * height - 1, -1, width).permute(0, 2, 1, 3)
        )

        # return self.gamma * (1/4 * (out_H + out_W + out_ldiagonal + out_rdiagonal) + self.weight[0] * out_H + self.weight[1] * out_W + self.weight[2] * out_ldiagonal + self.weight[3] * out_rdiagonal) + x
        return self.gamma * (out_H + out_W + out_ldiagonal + out_rdiagonal) + x
    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

    def h_transform(self, x):
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x

    def inv_h_transform(self, x):
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1).contiguous()
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x

    def v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1)
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x.permute(0, 1, 3, 2)



class SAFF(nn.Module):
    def __init__(self, channels, r=4):
        super(SAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            CrissStarAttention(inter_channels),
            nn.Conv2d(inter_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.local_att(x)
        xg = self.global_att(x)
        wei = self.sigmoid(xg)
        xo = x * wei + x
        return xo
