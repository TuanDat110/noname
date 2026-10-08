"""Auxiliary prototype branch (training only).

    H [B, C, T, V] --mean over T--> h [B, C, V]
                                      |
                       attention pooling over the V joints
                                      |
                                  z [B, C]
                                      |
               cl_fc -> L2 norm -> cosine vs. class memory bank -> L_CSC

The branch shares no parameters with the classifier and is skipped at
inference time, so it adds no test-time cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupHubProto(nn.Module):
    """Attention pooling of the backbone feature over all joints."""

    def __init__(self, in_channels, attn_tau=1.0):
        super().__init__()
        self.attn_tau = attn_tau
        self.out_dim = in_channels
        # Only output channel 0 is used. The layer keeps 3 channels so that the
        # released checkpoints load without remapping.
        self.score = nn.Conv1d(in_channels, 3, kernel_size=1)
        # Zero init -> uniform attention -> starts out as plain mean pooling.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, h_sta):
        """h_sta [B, C, T, V] -> z [B, C]"""
        h = h_sta.mean(dim=2)                              # [B, C, V]
        s = self.score(h)[:, 0]                            # [B, V]
        a = torch.softmax(s / self.attn_tau, dim=-1)
        return torch.einsum('bcv,bv->bc', h, a)


class Class_Specific_Contrastive_Loss(nn.Module):
    """Prototype contrastive loss against a momentum memory bank of class centres.

    Only correctly classified samples (with confidence above `pred_threshold`)
    update the bank.
    """

    def __init__(self, n_class, n_channel, h_channel=256, tmp=0.125,
                 mom=0.8, pred_threshold=0.0):
        super().__init__()
        self.n_class = n_class
        self.tmp = tmp
        self.mom = mom
        self.pred_threshold = pred_threshold
        self.register_buffer('avg_f', torch.randn(h_channel, n_class))
        # Marks the classes whose centre has been updated at least once.
        self.register_buffer('seen', torch.zeros(n_class))
        self.cl_fc = nn.Linear(n_channel, h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def onehot(self, label):
        ones = torch.eye(self.n_class, device=label.device)
        return ones.index_select(0, label.view(-1).long()).float()

    def local_average(self, f, mask):
        f = f.permute(1, 0)                                   # [h, B]
        avg_f = self.avg_f.detach().to(f.device)
        mask_sum = mask.sum(0, keepdim=True)                   # [1, n_class]
        f_mask = torch.matmul(f, mask) / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom                # class present in the batch
        has_object[has_object <= 0.1] = 1.0                    # class absent -> keep centre
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f.copy_(f_mem.detach())
            self.seen.clamp_(max=1.0).add_((mask_sum[0] > 1e-8).float()).clamp_(max=1.0)
        return f_mem

    def get_score(self, feature, f_mem):
        feature = F.normalize(feature, dim=1, eps=1e-12)
        f_mem = F.normalize(f_mem.permute(1, 0), dim=-1, eps=1e-12)
        return torch.matmul(f_mem, feature.permute(1, 0)) / self.tmp   # [n_class, B]

    def forward(self, feature, lbl, logit):
        feature = self.cl_fc(feature)
        pred_one = self.onehot(logit.max(1)[1])
        lbl_one = self.onehot(lbl)
        prob = torch.softmax(logit, 1)
        mask = lbl_one * pred_one * (prob > self.pred_threshold).float()
        f_mem = self.local_average(feature, mask)
        score = self.get_score(feature, f_mem).permute(1, 0).contiguous()
        return self.loss(score, lbl).mean()
