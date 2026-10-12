import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class LocalPoolBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        self.proj = nn.Sequential(PointWiseTCN(in_channels, out_channels, stride=1), nn.LeakyReLU(0.1, inplace=True))
        self.local_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=((kernel_size - 1) // 2, 0), dilation=(1, 1), padding_mode="replicate", bias=False)
        self.local_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(kernel_size - 1, 0), dilation=(2, 1), padding_mode="replicate", bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.proj(x)
        return self.activation(self.fusion_bn(self.local_conv1(x) + self.local_conv2(x) + self.max_pool(x)))

class TemporalDWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, mode="replicate"):
        super(TemporalDWT, self).__init__()
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.mode = mode

    def forward(self, x):
        N, C, T, V = x.shape
        original_T = T

        if T % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode=self.mode)

        filt_len = self.dec_lo.numel()

        low_kernel = self.dec_lo.view(1, 1, filt_len, 1).repeat(C, 1, 1, 1)
        high_kernel = self.dec_hi.view(1, 1, filt_len, 1).repeat(C, 1, 1, 1)

        low = F.conv2d(x, low_kernel, stride=(2, 1), groups=C)
        high = F.conv2d(x, high_kernel, stride=(2, 1), groups=C)

        return low, high, original_T


class TemporalIDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi):
        super(TemporalIDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi

    def forward(self, low, high, output_length):
        N, C, T_half, V = low.shape
        filt_len = self.rec_lo.numel()

        low_kernel = self.rec_lo.view(1, 1, filt_len, 1).repeat(C, 1, 1, 1)
        high_kernel = self.rec_hi.view(1, 1, filt_len, 1).repeat(C, 1, 1, 1)

        low_rec = F.conv_transpose2d(low, low_kernel, stride=(2, 1), groups=C)
        high_rec = F.conv_transpose2d(high, high_kernel, stride=(2, 1), groups=C)

        return (low_rec + high_rec)[:, :, :output_length, :]


class ReducedTemporalAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ReducedTemporalAttention, self).__init__()

        hidden = max(channels // reduction, 8)

        self.query = nn.Conv1d(channels, hidden, kernel_size=1, bias=False)
        self.key = nn.Conv1d(channels, hidden, kernel_size=1, bias=False)
        self.value = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_attention=False):
        N, C, T, V = x.shape

        descriptor = x.mean(dim=-1)                                      # [N,C,T]

        q = self.query(descriptor).transpose(1, 2)                       # [N,T,D]
        k = self.key(descriptor)                                         # [N,D,T]
        v = self.value(descriptor).transpose(1, 2)                       # [N,T,C]

        attention = torch.softmax(torch.matmul(q, k) / math.sqrt(q.size(-1)), dim=-1)

        context = torch.matmul(attention, v).transpose(1, 2)             # [N,C,T]
        context = self.out_proj(context).unsqueeze(-1).expand(-1, -1, -1, V)

        out = x + torch.tanh(self.scale) * context

        return (out, attention) if return_attention else out


class WaveletGlobalTemporalBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=4):
        super(WaveletGlobalTemporalBranch, self).__init__()

        self.stride = stride

        self.proj = nn.Sequential(
            PointWiseTCN(in_channels, out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        s = 1.0 / math.sqrt(2.0)

        self.dec_lo = nn.Parameter(torch.tensor([s, s], dtype=torch.float32))
        self.dec_hi = nn.Parameter(torch.tensor([-s, s], dtype=torch.float32))

        self.rec_lo = nn.Parameter(torch.tensor([s, s], dtype=torch.float32))
        self.rec_hi = nn.Parameter(torch.tensor([-s, s], dtype=torch.float32))

        self.dwt = TemporalDWT(self.dec_lo, self.dec_hi)
        self.idwt = TemporalIDWT(self.rec_lo, self.rec_hi)

        self.low_attention = ReducedTemporalAttention(out_channels, reduction=reduction)

        self.high_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, return_attention=False):
        x = self.proj(x)                                                 # [N,C,T,V]

        low, high, original_T = self.dwt(x)                              # [N,C,T/2,V]

        if return_attention:
            low, attention = self.low_attention(low, return_attention=True)
        else:
            low = self.low_attention(low)

        high = self.high_refine(high)

        reconstructed = self.idwt(low, high, original_T)                  # [N,C,T,V]

        out = self.out_proj(reconstructed)

        if self.stride > 1:
            out = out[:, :, ::self.stride, :]

        return (out, attention) if return_attention else out

class MultiScale_TemporalModeling(nn.Module):
    """
    Four temporal branches:

    1. Wavelet-guided global temporal branch
    2. Local temporal convolution branch
    3. Local temporal convolution + max-pooling branch
    4. Pointwise projection branch
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, num_frame=64):
        super(MultiScale_TemporalModeling, self).__init__()

        if out_channels % 4 != 0:
            raise ValueError(f"out_channels must be divisible by 4, but got {out_channels}")

        branch_channels = out_channels // 4

        self.global_branch = WaveletGlobalTemporalBranch(
            in_channels,
            branch_channels,
            stride=stride,
            reduction=4
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            TemporalConv(
                branch_channels,
                branch_channels,
                kernel_size=kernel_size,
                stride=stride
            )
        )

        self.local_pool_branch = LocalPoolBranch(
            in_channels,
            branch_channels,
            kernel_size=kernel_size,
            stride=stride
        )

        self.conv1x1 = PointWiseTCN(
            in_channels,
            branch_channels,
            stride=stride
        )

    def forward(self, x):
        global_feature = self.global_branch(x)
        local_feature = self.local_branch(x)
        local_pool = self.local_pool_branch(x)
        identity = self.conv1x1(x)

        out = torch.cat([global_feature, local_feature, local_pool, identity], dim=1)

        return out