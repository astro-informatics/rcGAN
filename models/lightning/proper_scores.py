"""Proper scoring-rule regularisers for conditional GANs (shared by all paper experiments).

Every loss takes
    gens: generated posterior samples, shape (B, P, C, H, W), P >= 2
    x:    ground truth,                shape (B, C, H, W)
and returns a scalar (mean over batch, channels and coefficients). Apply them to exactly the
tensor that the rcGAN regulariser is applied to in the corresponding model (same normalisation,
same channels), so that only the regulariser changes between runs.

Definitions (P = number of samples per measurement):
    fair CRPS  = mean_i |g_i - x| - 1/2 * mean_{i != j} |g_i - g_j|          (unbiased, Ferro 2014)
    plug-in    = mean_i |g_i - x| - 1/2 * mean_{i, j}   |g_i - g_j|          (biased, rewards under-dispersion)
    MS fair CRPS = mean over levels s = 0..S-1 of the fair CRPS of the orthonormal Haar scaling
                   (approximation) coefficients at level s, i.e. 2^s * avg_pool_{2^s}(field).
                   Level 0 is the per-pixel fair CRPS. Each level is a set of linear projections of
                   the field, so the sum is proper for the joint distribution and strictly proper for
                   the set of all pooled marginals; it is still unbiased (fair) for every P >= 2.
    MS-sigma   = the same with fixed per-level weights 1/sigma_s (sigma_s = std of the level-s coefficients of the
                   training truths), so every level contributes in standardised units. Added 2026-10-27 because the
                   unweighted MS is dominated by the coarse levels for red-spectrum images.
"""
import math

import torch
import torch.nn.functional as F


def _mean_abs_to_truth(gens, x):
    return (gens - x.unsqueeze(1)).abs().mean(dim=1)


def _mean_abs_pairs(gens, include_diagonal):
    """Mean of |g_i - g_j| over ordered pairs, via the sorted-sample identity (O(P log P) memory)."""
    P = gens.shape[1]
    s, _ = gens.sort(dim=1)
    shape = (1, P) + (1,) * (gens.dim() - 2)
    k = torch.arange(1, P + 1, device=gens.device, dtype=gens.dtype).view(shape)
    pair_sum = 2.0 * ((2 * k - P - 1) * s).sum(dim=1)  # sum over ordered pairs i != j
    return pair_sum / (P * P if include_diagonal else P * (P - 1))


def fair_crps(gens, x):
    return (_mean_abs_to_truth(gens, x) - 0.5 * _mean_abs_pairs(gens, include_diagonal=False)).mean()


def plugin_crps(gens, x):
    return (_mean_abs_to_truth(gens, x) - 0.5 * _mean_abs_pairs(gens, include_diagonal=True)).mean()


def default_levels(h, w, coarsest=4):
    """Number of Haar levels so that the coarsest level still has >= `coarsest` pixels per side."""
    return max(1, int(math.floor(math.log2(min(h, w) / coarsest))) + 1)


def haar_levels(field, n_levels):
    """Orthonormal Haar scaling coefficients of `field` (N, C, H, W) at levels 0..n_levels-1."""
    out = [field]
    for _ in range(1, n_levels):
        field = 2.0 * F.avg_pool2d(field, 2, ceil_mode=True)
        out.append(field)
    return out


def ms_fair_crps(gens, x, n_levels=None, level_weights=None):
    """Multiscale fair CRPS. level_weights=None gives the original unweighted average over levels ("MS").
    For "MS-sigma", pass level_weights = 1 / sigma_s, where sigma_s is the standard deviation of the level-s
    coefficients of the TRAINING truths (fixed constants, see `haar_level_std`), so each level's CRPS is in
    standardised units. Fixed positive weights keep the score proper and fair."""
    B, P, C, H, W = gens.shape
    if n_levels is None:
        n_levels = default_levels(H, W)
    if level_weights is None:
        level_weights = [1.0] * n_levels
    assert len(level_weights) == n_levels
    g = gens.reshape(B * P, C, H, W)
    t = x
    total = 0.0
    for s in range(n_levels):
        if s > 0:
            # orthonormal Haar scaling coefficients: block sum / block side = 2 * avg_pool each level
            g = 2.0 * F.avg_pool2d(g, 2, ceil_mode=True)
            t = 2.0 * F.avg_pool2d(t, 2, ceil_mode=True)
        total = total + level_weights[s] * fair_crps(g.view(B, P, C, g.shape[-2], g.shape[-1]), t)
    return total / n_levels


@torch.no_grad()
def haar_level_std(truth_batches, n_levels):
    """sigma_s for MS-sigma: std over the training truths of the level-s Haar scaling coefficients, pooled over
    coefficients and channels, computed in exactly the units the loss sees. `truth_batches` yields (N, C, H, W)
    tensors (use >= 1000 training items). Returns a list of n_levels floats; save it with the run config."""
    n = [0] * n_levels
    s1 = [0.0] * n_levels
    s2 = [0.0] * n_levels
    for xb in truth_batches:
        for s, c in enumerate(haar_levels(xb.double(), n_levels)):
            n[s] += c.numel()
            s1[s] += c.sum().item()
            s2[s] += (c * c).sum().item()
    return [math.sqrt(max(s2[s] / n[s] - (s1[s] / n[s]) ** 2, 1e-12)) for s in range(n_levels)]


def fair_energy_score(gens, x):
    """Fair energy score (q = 1, Euclidean norm over C*H*W) per measurement, averaged over the batch.
    Evaluation metric (and optional loss for the toy benchmark)."""
    B, P = gens.shape[:2]
    g = gens.reshape(B, P, -1)
    t = x.reshape(B, 1, -1)
    term1 = (g - t).norm(dim=-1).mean(dim=1)
    d = torch.cdist(g, g)  # (B, P, P), zero diagonal
    term2 = d.sum(dim=(1, 2)) / (P * (P - 1))
    return (term1 - 0.5 * term2).mean()


if __name__ == "__main__":
    # self-test: estimators against brute force, and propriety sanity check
    torch.manual_seed(0)
    g = torch.randn(3, 5, 2, 16, 16, dtype=torch.float64)
    x = torch.randn(3, 2, 16, 16, dtype=torch.float64)
    brute = (g.unsqueeze(1) - g.unsqueeze(2)).abs().sum(dim=(1, 2)) / (5 * 4)
    assert torch.allclose(_mean_abs_pairs(g, False), brute)
    # fair CRPS: N(0,1) truth; calibrated N(0,1) ensemble beats under-dispersed N(0,0.5^2) for P=2
    xs = torch.randn(200000, 1, 1, 1, dtype=torch.float64)
    cal = torch.randn(200000, 2, 1, 1, 1, dtype=torch.float64)
    print("fair  P=2: calibrated", fair_crps(cal, xs).item(), "| sd 0.5", fair_crps(0.5 * cal, xs).item())
    print("plugin P=2: calibrated", plugin_crps(cal, xs).item(), "| sd 0.5", plugin_crps(0.5 * cal, xs).item())
    print("levels for 300x300:", default_levels(300, 300), "| 384:", default_levels(384, 384))
    print("ms_fair_crps runs:", ms_fair_crps(g, x).item())
    sig = haar_level_std([x], 3)
    w = [1 / v for v in sig]
    print("haar_level_std:", [round(v, 3) for v in sig], "| MS-sigma runs:", ms_fair_crps(g, x, 3, w).item())
    assert abs(ms_fair_crps(g, x, 3, [1, 1, 1]).item() - ms_fair_crps(g, x, 3).item()) < 1e-12
