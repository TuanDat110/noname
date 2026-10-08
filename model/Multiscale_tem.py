import torch
import torch.nn as nn


class PatchMergingTconv(nn.Module):
    """Strided temporal convolution used to downsample between stages."""

    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(
            dim_in, dim_out,
            kernel_size=(kernel_size, 1), padding=(pad, 0),
            stride=(stride, 1), dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.reduction(x))


class MSTCN(nn.Module):
    """Multi-scale temporal convolution.

    Parallel temporal convolutions with different kernel sizes, each producing
    an equal share of the output channels, followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 7)):
        super().__init__()
        n_branches = len(kernel_sizes)
        assert out_channels % n_branches == 0, \
            f"out_channels={out_channels} must be divisible by {n_branches} branches"
        branch_ch = out_channels // n_branches

        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, branch_ch,
                      kernel_size=(k, 1),
                      padding=(k // 2, 0))
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C_in, T, V]
        outs = [b(x) for b in self.branches]
        y = torch.cat(outs, dim=1)   # [B, C_out, T, V]
        return self.act(self.bn(y))
