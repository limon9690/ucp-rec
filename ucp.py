import torch


def coarse_percentile(feats, n_periods=12, n_bins=11):
    """Bin-weighted mean percentile over the coarse periods"""
    coarse = feats[..., : n_periods * n_bins].reshape(
        *feats.shape[:-1], n_periods, n_bins
    )
    centers = torch.arange(n_bins, device=feats.device, dtype=feats.dtype) * (
        100.0 / (n_bins - 1)
    )
    return (coarse * centers).sum(-1).mean(-1)


def fine_percentile(feats, offset=132, n_bins=6):
    fine = feats[..., offset : offset + n_bins]
    centers = torch.arange(n_bins, device=feats.device, dtype=feats.dtype) * (
        100.0 / (n_bins - 1)
    )
    return (fine * centers).sum(-1)


def _masked_mean_std(x, mask, n):
    mean = (x * mask).sum(1) / n
    var = (((x - mean.unsqueeze(1)) ** 2) * mask).sum(1) / n
    return mean, var.sqrt()


@torch.no_grad()
def user_profile(seq_feats, mask, maxlen):
    """7-dim user popularity profile"""
    mask = mask.to(seq_feats.dtype)
    n = mask.sum(1).clamp(min=1)
    pc = coarse_percentile(seq_feats)
    pf = fine_percentile(seq_feats)

    avg_c, std_c = _masked_mean_std(pc, mask, n)
    avg_f, std_f = _masked_mean_std(pf, mask, n)

    # position of each real item within the user's own history (0 .. n-1)
    order = torch.cumsum(mask, 1) - 1
    first = mask * (order < (n / 2).unsqueeze(1)).to(mask.dtype)
    second = mask - first
    n1, n2 = first.sum(1), second.sum(1)
    trend = torch.where(
        (n1 > 0) & (n2 > 0),
        (pc * second).sum(1) / n2.clamp(min=1) - (pc * first).sum(1) / n1.clamp(min=1),
        torch.zeros_like(n),
    )

    high = ((pc > 60).to(mask.dtype) * mask).sum(1) / n
    length = mask.sum(1) / maxlen

    profile = torch.stack(
        [
            avg_c / 100,
            std_c / 50,
            avg_f / 100,
            std_f / 50,
            (trend + 50) / 100,
            high,
            length,
        ],
        dim=1,
    )
    return profile.clamp(0, 1)


class APF(torch.nn.Module):
    """Adaptive popularity fusion"""

    def __init__(self, profile_dim=7, hidden=8):
        super().__init__()
        self.fc1 = torch.nn.Linear(profile_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)
        self.reset_output()

    def reset_output(self):
        # alpha = sigmoid(0) = 0.5 at the start of training
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def alpha(self, profile):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(profile))))

    def forward(self, deep, cand_feats, profile):
        """deep: (B, N) scores; cand_feats: (B, N, 138) candidate features; profile: (B, 7)."""
        a = self.alpha(profile)
        shallow = coarse_percentile(cand_feats) / 100
        return a * deep + (1 - a) * shallow
