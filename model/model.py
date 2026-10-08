import math

import torch
import torch.nn as nn

try:
    from timm.layers import trunc_normal_, Mlp, DropPath
except ImportError:
    from timm.models.layers import trunc_normal_, Mlp, DropPath

from model.Multiscale_tem import MSTCN, PatchMergingTconv
from model.aux_group_proto import GroupHubProto, Class_Specific_Contrastive_Loss
from model.motion_family import InterAffinityContrastiveLoss
from model.hyper_edge import HyperEdgeGraph


class MultiPerspectiveDynamicGCN(nn.Module):
    """Graph convolution over K subsets with a multi-perspective adjacency:

        A[k] = A_0[k] + alpha[k] * A_inter[k] + beta[k] * A_intra[k] + gamma[k] * G[k]

    A_0      learned static adjacency
    A_inter  pairwise feature difference between joints (tanh)
    A_intra  pairwise feature correlation between joints (softmax)
    G        hypergraph adjacency from HyperEdgeGraph

    The virtual joints of HyperEdgeGraph are appended before the adjacencies
    are computed and sliced off again after the graph convolution.
    """

    def __init__(self, channels, num_subsets=8, num_points=27,
                 hyper_num=3, hyper_topk=9, hyper_reduction=4):
        super().__init__()
        assert channels % num_subsets == 0, (
            f"channels={channels} must be divisible by num_subsets={num_subsets}")
        self.num_subsets = num_subsets
        self.num_points = num_points
        self.mid_channels = channels // num_subsets
        Ve = num_points + hyper_num

        A_init = torch.zeros(num_subsets, Ve, Ve)
        trunc_normal_(A_init, std=0.02)
        self.A = nn.Parameter(A_init)

        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.post = nn.Conv2d(channels, channels, 1)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)

        self.hyper = HyperEdgeGraph(
            channels, num_subsets=num_subsets, num_points=num_points,
            hyper_num=hyper_num, topk=hyper_topk, reduction=hyper_reduction)

        # Zero init: every dynamic term starts switched off.
        self.gamma = nn.Parameter(torch.zeros(num_subsets))
        self.alpha = nn.Parameter(torch.zeros(num_subsets))
        self.beta = nn.Parameter(torch.zeros(num_subsets))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _scale(p):
        return p.view(1, -1, 1, 1, 1, 1)

    def forward(self, x):
        n, c, t, v = x.shape
        res = x

        x = self.hyper.cat_hyper(x)                     # [N, C, T, Ve]
        ve = x.shape[-1]

        A = self.A[None, :, None, None]                 # [1, K, 1, 1, Ve, Ve]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, ve)

        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, -1, ve)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, -1, ve)
        x1 = x1.mean(dim=-2, keepdim=True)              # [N, K, Cm, 1, Ve]
        x2 = x2.mean(dim=-2, keepdim=True)

        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)      # [N, K, Cm, 1, Ve, Ve]
        A = self.tanh(diff) * self._scale(self.alpha) + A

        aff = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        A = self.softmax(aff) * self._scale(self.beta) + A

        A = self._scale(self.gamma) * self.hyper(x)[:, :, None, None] + A

        A = A.squeeze(3)                                # [N, K, Cm, Ve, Ve]

        out = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
        out = self.post(out.reshape(n, -1, t, ve))
        out = out[..., :self.num_points]                # drop the virtual joints

        return self.act(self.bn(out) + res)


class HybridBlockPseudoGraph(nn.Module):
    """Two-branch block. The channels are split in half:

        C/2 -> MultiPerspectiveDynamicGCN   (spatial)
        C/2 -> chain of MSTCN               (temporal)

    The two halves are concatenated, projected and added back to the input,
    followed by a feed-forward network.
    """

    def __init__(self, in_channels, num_points=27, num_subsets=8,
                 drop_path=0., mlp_ratio=4.,
                 mstcn_kernel_sizes=(5, 7), mstcn_num_layers=3,
                 hyper_num=3, hyper_topk=9, hyper_reduction=4):
        super().__init__()
        assert in_channels % 2 == 0, f"in_channels={in_channels} must be even"
        half = in_channels // 2
        self.half = half

        self.norm_1 = nn.LayerNorm(in_channels)
        self.mapping = nn.Linear(in_channels, in_channels, bias=True)

        self.motion_gcn = MultiPerspectiveDynamicGCN(
            half, num_subsets=num_subsets, num_points=num_points,
            hyper_num=hyper_num, hyper_topk=hyper_topk,
            hyper_reduction=hyper_reduction)

        self.mstcn_chain = nn.Sequential(*[
            MSTCN(half, half, kernel_sizes=mstcn_kernel_sizes)
            for _ in range(mstcn_num_layers)
        ])

        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = nn.LayerNorm(in_channels)
        self.mlp = Mlp(in_features=in_channels,
                       hidden_features=int(mlp_ratio * in_channels),
                       act_layer=nn.GELU)

    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()      # [B, T, V, C]
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()
        f_gconv, f_mstcn = torch.split(f, [self.half, self.half], dim=1)

        y = [self.motion_gcn(f_gconv), self.mstcn_chain(f_mstcn)]

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous())
        output = skip + self.drop_path(output)
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        return output.permute(0, 3, 1, 2).contiguous()


class HybridBlockPseudoGraphDS(nn.Module):
    """HybridBlockPseudoGraph with an optional temporal downsampling in front."""

    def __init__(self, in_channels, out_channels, kernel_size=7, downscale=False,
                 **block_args):
        super().__init__()
        self.downsample = PatchMergingTconv(
            in_channels, out_channels, kernel_size=kernel_size) if downscale else None
        self.transformer = HybridBlockPseudoGraph(out_channels, **block_args)

    def forward(self, input):
        if self.downsample is not None:
            input = self.downsample(input)
        return self.transformer(input)


class HybridStagePseudoGraph(nn.Module):
    """A stage of `depth` blocks. Every stage but the first halves the temporal length."""

    def __init__(self, depth, in_channels, out_channels, first_depth=False,
                 kernel_size=7, drop_path=(), **block_args):
        super().__init__()
        self.blocks = nn.ModuleList([
            HybridBlockPseudoGraphDS(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                downscale=((index == 0) and (not first_depth)),
                drop_path=drop_path[index],
                **block_args)
            for index in range(depth)
        ])

    def forward(self, input):
        for block in self.blocks:
            input = block(input)
        return input


class HAPCNet(nn.Module):
    """Skeleton-based sign language recognition network.

    Input:  skeleton [B, C, T, V, M] and frame indices index_t [B, T]
    Output: class logits [B, num_classes]

    During training an auxiliary prototype branch reads the backbone feature;
    call `aux_losses` after `forward` to get its two contrastive losses, and
    `on_epoch_end` once per epoch. The branch is skipped at inference time.
    """

    def __init__(self, in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192),
                 num_classes=60, embed_dim=96, num_people=1, num_points=27,
                 kernel_size=7, num_subsets=8,
                 head_drop=0., drop_path=0., mlp_ratio=4.,
                 mstcn_kernel_sizes=(5, 7), mstcn_num_layers=3,
                 hyper_num=3, hyper_topk=9, hyper_reduction=4,
                 aux_attn_tau=1.0, aux_h_channel=256, aux_tmp=0.125, aux_mom=0.8,
                 aux_threshold=0.0,
                 mf_warmup=30, mf_k=10, mf_na=4, mf_center_mom=0.9):
        super().__init__()
        assert len(depths) == len(channels)

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=head_drop) if head_drop != 0 else None

        V_total = num_points * num_people

        self.stem = nn.ModuleList([
            nn.Conv2d(in_channels, 2 * in_channels, (1, 1)), nn.GELU(),
            nn.Conv2d(2 * in_channels, 3 * in_channels, (1, 1)), nn.GELU(),
            nn.Conv2d(3 * in_channels, embed_dim, (1, 1)),
        ])

        # Learned joint embedding, modulated by a sinusoidal encoding of index_t.
        self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, V_total))
        trunc_normal_(self.joint_person_embedding, std=.02)

        # Stochastic depth rate increases linearly over the blocks.
        dpr = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        self.stages = nn.ModuleList([
            HybridStagePseudoGraph(
                depth=depth,
                in_channels=embed_dim if idx == 0 else channels[idx - 1],
                out_channels=channel, first_depth=(idx == 0),
                kernel_size=kernel_size,
                drop_path=dpr[sum(depths[:idx]):sum(depths[:idx + 1])],
                num_points=V_total, num_subsets=num_subsets,
                mlp_ratio=mlp_ratio,
                mstcn_kernel_sizes=mstcn_kernel_sizes,
                mstcn_num_layers=mstcn_num_layers,
                hyper_num=hyper_num, hyper_topk=hyper_topk,
                hyper_reduction=hyper_reduction)
            for idx, (depth, channel) in enumerate(zip(depths, channels))
        ])
        self.head = nn.Linear(channels[-1], num_classes)

        # ---- auxiliary prototype branch (training only) ----
        self._z_aux = None
        self.aux_hub = GroupHubProto(channels[-1], attn_tau=aux_attn_tau)
        # The two losses have separate projection heads (cl_fc).
        self.aux_csc = Class_Specific_Contrastive_Loss(
            n_class=num_classes, n_channel=self.aux_hub.out_dim,
            h_channel=aux_h_channel, tmp=aux_tmp, mom=aux_mom,
            pred_threshold=aux_threshold)
        self.aux_inter = InterAffinityContrastiveLoss(
            n_class=num_classes, n_channel=self.aux_hub.out_dim,
            h_channel=aux_h_channel, tmp=aux_tmp, mom=mf_center_mom,
            pred_threshold=aux_threshold, start_epoch=mf_warmup,
            topk=mf_k, coincidence_count=mf_na)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {n for n, _ in self.named_parameters()
                if n.endswith('.alpha') or n.endswith('.beta')}

    def forward_features(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = x.mean(dim=(2, 3))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.head(x)

    def forward(self, input, index_t):
        B, C, T, V, M = input.shape
        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        for layer in self.stem:
            output = layer(output)

        te = torch.zeros(B, T, self.embed_dim, device=output.device)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=output.device)
            * -(math.log(10000.0) / self.embed_dim))
        te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
        te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
        output = output + torch.einsum('b t c, c v -> b c t v', te,
                                       self.joint_person_embedding)

        feat = self.forward_features(output)              # [B, C, T', V]
        self._z_aux = self.aux_hub(feat) if self.training else None
        return self.forward_head(feat)

    def aux_losses(self, label, logit):
        """Call after forward(). Both losses are zero outside training."""
        if self._z_aux is None:
            zero = torch.zeros((), device=logit.device)
            return dict(csc=zero, inter=zero)
        return dict(csc=self.aux_csc(self._z_aux, label, logit),
                    inter=self.aux_inter(self._z_aux, label, logit))

    def on_epoch_end(self, epoch):
        """Call at the end of every training epoch to rebuild the class families."""
        self.aux_inter.rebuild_family(epoch)
