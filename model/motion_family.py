import torch
import torch.nn as nn
import torch.nn.functional as F


class InterAffinityContrastiveLoss(nn.Module):
    """Contrastive loss restricted to each class's "family" of confusable classes.

    A confusion matrix H is accumulated during training. At the end of every
    epoch `rebuild_family` turns it into a family mask: class j belongs to the
    family of class i if it is one of i's top-k most confused classes, or if the
    two share more than `coincidence_count` such neighbours. The loss then
    contrasts a sample against its own class centre and its family only.
    Classes without a family fall back to a contrast against all classes.
    The loss is exactly zero until `start_epoch` epochs have finished.
    """

    def __init__(self, n_class, n_channel, h_channel=256, tmp=0.125, mom=0.9,
                 pred_threshold=0.0, start_epoch=30, topk=10,
                 coincidence_count=4):
        super().__init__()
        self.n_class = n_class
        self.tmp = tmp                          # temperature of the fallback branch
        self.mom = mom                          # momentum of the class centres
        self.pred_threshold = pred_threshold
        self.start_epoch = start_epoch
        self.K = topk                           # number of direct neighbours
        self.coincidence_count = coincidence_count

        self.cl_fc = nn.Linear(n_channel, h_channel)

        self.register_buffer('avg_f', torch.randn(h_channel, n_class))
        self.register_buffer('H', torch.zeros(n_class, n_class))
        self.register_buffer('family', torch.zeros(n_class, n_class, dtype=torch.bool))
        self.register_buffer('fam_size', torch.zeros(n_class))
        self.register_buffer('active', torch.zeros(1))
        self.epoch = 0

    def onehot(self, label):
        return torch.eye(self.n_class, device=label.device).index_select(
            0, label.view(-1).long()).float()

    @torch.no_grad()
    def update_confusion(self, lbl, pred):
        """Accumulates confusions in both directions.

        1: class lbl[i] was predicted as pred[i]
        2: a sample of class lbl[j] was predicted as lbl[i] -> H[lbl[i], lbl[j]]
        """
        wrong = pred != lbl
        if wrong.any():
            idx = lbl[wrong] * self.n_class + pred[wrong]
            self.H.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=self.H.dtype))
        # pairs (i, j) in the batch with pred[j] == lbl[i] and lbl[j] != lbl[i]
        hit = (pred.unsqueeze(0) == lbl.unsqueeze(1)) & (lbl.unsqueeze(0) != lbl.unsqueeze(1))
        if hit.any():
            ii, jj = hit.nonzero(as_tuple=True)
            idx = lbl[ii] * self.n_class + lbl[jj]
            self.H.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=self.H.dtype))

    @torch.no_grad()
    def local_avg(self, f, mask):
        """EMA class centres, updated from correctly classified samples only."""
        f = f.permute(1, 0)
        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask) / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        self.avg_f.copy_(self.avg_f * has_object + (1 - has_object) * f_mask)
        return self.avg_f.permute(1, 0)          # [n_class, h]

    @torch.no_grad()
    def rebuild_family(self, epoch):
        """Rebuilds the family mask. Call once at the end of every epoch."""
        self.epoch = epoch + 1
        if self.epoch <= self.start_epoch:
            self.family.zero_()
            self.fam_size.zero_()
            self.active.zero_()
            return

        H = self.H.clone()
        H.fill_diagonal_(0)
        has_data = H.sum(1) > 0                              # classes with any confusion

        k = min(self.K, self.n_class - 1)
        val, idx = H.topk(k, dim=1)
        I = torch.zeros_like(H).scatter_(1, idx, 1.0)
        # Drop top-k entries with a zero count, so that classes with few
        # confusions are not assigned arbitrary neighbours.
        I = I * torch.zeros_like(H).scatter_(1, idx, (val > 0).to(H.dtype))
        I = I * has_data.unsqueeze(1).to(I.dtype)

        M = I @ I.t()                                        # shared neighbours
        M.fill_diagonal_(0)
        w = 0.5 * I + (M / max(k, 1)) * (M > self.coincidence_count).to(M.dtype)

        fam = w > 0
        fam.fill_diagonal_(False)                            # a family excludes the class itself
        fam[~has_data] = False
        self.family.copy_(fam)
        self.fam_size.copy_(fam.sum(1).float())
        self.active.fill_(1.0)

    def _adaptive_tau(self, n):
        """Temperature 0.1 / 0.5 / 1.0 depending on the family size."""
        tau = torch.full_like(n, 1.0)
        tau[n <= self.K] = 0.1
        tau[(n > self.K) & (n <= 2 * self.K)] = 0.5
        return tau

    def forward(self, feature, lbl, logit):
        f = self.cl_fc(feature)
        pred = logit.max(1)[1]

        # Only correctly classified samples update the class centres.
        prob = torch.softmax(logit, 1)
        mask = self.onehot(lbl) * self.onehot(pred) * (prob > self.pred_threshold).float()
        with torch.no_grad():
            if self.epoch >= self.start_epoch:
                self.update_confusion(lbl, pred)
        f_mem = self.local_avg(f.detach(), mask)             # [n_class, h]

        if self.active.item() < 0.5:
            return torch.zeros((), device=f.device)

        fn = F.normalize(f, dim=1)
        mn = F.normalize(f_mem, dim=-1).detach()
        cos = fn @ mn.t()                                    # [B, n_class]

        fam = self.family[lbl]                               # [B, n_class]
        n = self.fam_size[lbl]                               # [B]
        pos = self.onehot(lbl).bool()

        # Main branch: the denominator holds the class itself and its family.
        tau = self._adaptive_tau(n).unsqueeze(1)
        s = (cos / tau).masked_fill(~(fam | pos), float('-inf'))
        l_main = torch.logsumexp(s, 1) - s.masked_fill(~pos, float('-inf')).max(1).values

        # Fallback for classes without a family: contrast against all classes.
        sb = cos / self.tmp
        l_back = torch.logsumexp(sb, 1) - sb.masked_fill(~pos, float('-inf')).max(1).values

        return torch.where(n > 0, l_main, l_back).mean()
