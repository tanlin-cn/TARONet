import torch
import torch.nn as nn
import torch.nn.functional as F
def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref

def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid

def dcnv3_core(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    # for debug and test only,
    # need to use cuda version instead
    input = F.pad(
        input,
        [0, 0, pad_w, pad_w, pad_h, pad_h])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*kernel_h*kernel_w).to(input.device)
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_,
                                                   group*group_channels, H_out*W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()

# 将通道维度放在前面：(b,h,w,c)->(b,c,h,w)
class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x): # (b,h,w,c)
        return x.permute(0, 3, 1, 2) #(b,c,h,w)

# 将通道维度放在后面 (b,c,h,w)->(b,h,w,c)
class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x): # (b,c,h,w)
        return x.permute(0, 2, 3, 1) # (b,h,w,c)

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last()) # (b,c,h,w)->(b,h,w,c)
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

class DSConv(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 pad,
                 dilation,
                 group=4,
                 offset_scale=1.0,
                 norm_layer='LN'
    ):
        super(DSConv, self).__init__()
        # if channels % group != 0:  # 分组卷积必须保证通道数可以被组数整除
        #     raise ValueError(
        #         f'channels must be divisible by group, but got {channels} and {group}')
        emd_channels = channels // group * group

        self.offset_scale = offset_scale
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group

        self.input_proj = nn.Linear(channels, emd_channels)

        self.strip_conv = nn.Sequential(
            nn.Conv2d(
                emd_channels, emd_channels, kernel_size=kernel_size, padding=pad, dilation=dilation
            ),
            build_norm_layer(
                emd_channels,
                norm_layer,
                'channels_first',  # 如果是LN, (b,c,h,w)->(b,h,w,c)
                'channels_last'),
            nn.GELU()
        )

        self.offset = nn.Linear(emd_channels, group * kernel_size[0] * kernel_size[1] * 2)
        self.mask = nn.Linear(emd_channels, group * kernel_size[0] * kernel_size[1])

        self.output_proj = nn.Linear(emd_channels, channels)

    def forward(self, input):
        x = input.permute(0, 2, 3, 1)
        x = self.input_proj(x)
        batchsize, height, width, _ = x.size()

        x1 = self.strip_conv(x.permute(0, 3, 1, 2))

        offset = self.offset(x1)
        mask = self.mask(x1).reshape(batchsize, height, width, self.group, -1)

        mask = F.softmax(mask, -1).reshape(batchsize, height, width, -1)

        x = dcnv3_core(
            x, offset, mask,
            self.kernel_size[0], self.kernel_size[1],
            1, 1,
            self.pad[0], self.pad[1],
            self.dilation[0], self.dilation[1],
            self.group, self.group_channels,
            self.offset_scale
        )
        x = self.output_proj(x)
        return x.permute(0, 3, 1, 2)



class ASConv2(nn.Module):
    def __init__(self, in_channels, cov_size):
        super(ASConv2, self).__init__()
        if cov_size == 5:
            kernel = 3
            dilation = 2
        elif cov_size == 9:
            kernel = 5
            dilation = 2
        elif cov_size == 13:
            kernel = 7
            dilation = 2
        # elif cov_size == 13:
        #     kernel = 5
        #     dilation = 3
        elif cov_size == 17:
            kernel = 9
            dilation = 2

        self.conv_l= nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.deconv1 = DSConv(in_channels//4, (1, kernel), 1, (0, cov_size//2), (1, dilation))
        self.deconv2 = DSConv(in_channels//4, (kernel, 1), 1, (cov_size//2, 0), (dilation, 1))
        self.deconv3 = DSConv(in_channels//4, (kernel, 1), 1, (cov_size//2, 0), (dilation, 1))
        self.deconv4 = DSConv(in_channels//4, (1, kernel), 1, (0, cov_size//2), (1, dilation))
        self.conv_x1 = nn.Conv2d(in_channels//4 * 2, in_channels//4, 1)
        self.conv_x2 = nn.Conv2d(in_channels//4 * 2, in_channels//4, 1)
        self.conv_x3 = nn.Conv2d(in_channels//4 * 2, in_channels//4, 1)
        self.conv_x4 = nn.Conv2d(in_channels//4 * 2, in_channels//4, 1)

        self.bn2 = nn.BatchNorm2d(in_channels//4 * 4)
        self.relu2 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        local = x
        local = self.conv_l(local)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x1 = torch.cat((x1, local), dim=1)
        x1 = self.conv_x1(x1)

        x2 = self.deconv2(x)
        x2 = torch.cat((x2, local), dim=1)
        x2 = self.conv_x2(x2)

        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x3 = torch.cat((x3, local), dim=1)
        x3 = self.conv_x3(x3)

        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x4 = torch.cat((x4, local), dim=1)
        x4 = self.conv_x4(x4)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASConv(nn.Module):
    def __init__(self, in_channels, filters, cov_size):
        super(ASConv, self).__init__()
        if cov_size == 5:
            kernel = 3
            dilation = 2
        elif cov_size == 9:
            kernel = 5
            dilation = 2
        elif cov_size == 13:
            kernel = 7
            dilation = 2
        # elif cov_size == 13:
        #     kernel = 5
        #     dilation = 3
        elif cov_size == 17:
            kernel = 9
            dilation = 2

        self.conv_l= nn.Conv2d(in_channels, in_channels // 8, 3, 1, 1)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 8)
        self.relu1 = nn.ReLU()

        self.deconv1 = DSConv(in_channels//8, (1, kernel), 1, (0, cov_size//2), (1, dilation))
        self.deconv2 = DSConv(in_channels//8, (kernel, 1), 1, (cov_size//2, 0), (dilation, 1))
        self.deconv3 = DSConv(in_channels//8, (kernel, 1), 1, (cov_size//2, 0), (dilation, 1))
        self.deconv4 = DSConv(in_channels//8, (1, kernel), 1, (0, cov_size//2), (1, dilation))
        self.conv_x1 = nn.Conv2d(in_channels//8 * 2, filters//4, 1)
        self.conv_x2 = nn.Conv2d(in_channels//8 * 2, filters//4, 1)
        self.conv_x3 = nn.Conv2d(in_channels//8 * 2, filters//4, 1)
        self.conv_x4 = nn.Conv2d(in_channels//8 * 2, filters//4, 1)

        self.bn2 = nn.BatchNorm2d(filters//4 * 4)
        self.relu2 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        local = x
        local = self.conv_l(local)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x1 = torch.cat((x1, local), dim=1)
        x1 = self.conv_x1(x1)

        x2 = self.deconv2(x)
        x2 = torch.cat((x2, local), dim=1)
        x2 = self.conv_x2(x2)

        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x3 = torch.cat((x3, local), dim=1)
        x3 = self.conv_x3(x3)

        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x4 = torch.cat((x4, local), dim=1)
        x4 = self.conv_x4(x4)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
