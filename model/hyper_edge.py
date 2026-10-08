import torch
import torch.nn as nn


class HyperEdgeGraph(nn.Module):
    """Builds a sample-dependent hypergraph adjacency G of shape [N, K, V+P, V+P].

    P learnable virtual joints are appended to the V real joints. Each joint
    then forms a hyperedge with its top-k nearest neighbours in a learned
    embedding space, and the hyperedges are weighted by a learned score.
    """

    def __init__(self, in_channels, num_subsets=8, num_points=27,
                 hyper_num=3, topk=9, reduction=4):
        super().__init__()
        self.num_subsets = num_subsets
        self.topk = min(topk, num_points + hyper_num)
        self.hidden_channels = max(1, in_channels // num_subsets // reduction)
        hid = self.hidden_channels

        self.to_V = nn.Conv1d(in_channels, num_subsets * hid, 1, groups=num_subsets)
        self.to_W = nn.Sequential(
            nn.Conv1d(in_channels, num_subsets * hid, 1, groups=num_subsets),
            nn.LeakyReLU(),
            nn.Conv1d(num_subsets * hid, num_subsets, 1),
            nn.Tanh(),
        )
        # Each virtual joint is initialised independently so that they do not
        # start out as identical copies of one another.
        self.hyper_joint = nn.Parameter(torch.randn(hyper_num, in_channels))

        for c in (self.to_V, self.to_W[0], self.to_W[2]):
            nn.init.kaiming_normal_(c.weight, mode='fan_out')
            if c.bias is not None:
                nn.init.constant_(c.bias, 0)

    @staticmethod
    def hyper_norm(H, W):
        """Normalised hypergraph adjacency from incidence H and edge weights W."""
        w = torch.diag_embed(W)
        norm_w = torch.norm(H, 1, dim=2, keepdim=True) + 1e-8
        w_ = w / norm_w
        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=3, keepdim=True) + 1e-8
        h_ = H_w / norm_v
        return h_ @ w_ @ H.transpose(3, 2)

    def cat_hyper(self, x):
        """x [N, C, T, V] -> [N, C, T, V+P]"""
        n, c, t, v = x.shape
        h = self.hyper_joint.t().unsqueeze(1)              # [C, 1, P]
        return torch.cat([x, h.unsqueeze(0).expand(n, -1, t, -1)], dim=-1)

    def forward(self, x_ext):
        """x_ext [N, C, T, V+P] (virtual joints already appended) -> G [N, K, V+P, V+P]"""
        n, c, t, ve = x_ext.shape
        K, hid = self.num_subsets, self.hidden_channels

        t_x = x_ext.mean(2)                                # [N, C, V+P]
        v = self.to_V(t_x).view(n, K, hid, ve).permute(0, 1, 3, 2).contiguous()
        d = torch.cdist(v, v)                              # [N, K, V+P, V+P]

        topv, topi = torch.topk(d, self.topk, largest=False)
        topv = torch.softmax(-topv, dim=-1)
        H = torch.zeros_like(d).scatter_(3, topi, topv)

        W = self.to_W(t_x)                                 # [N, K, V+P]
        return self.hyper_norm(H, W)
