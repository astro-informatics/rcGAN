"""1-D theory: how much does the beta-tuned rcGAN minimiser lose to the CRPS minimiser (= p), as a function of the shape
of the per-pixel posterior p? Uses the toy agent's solver (~/crps_review/paper/toy/analytic_1d.py): 400 atoms, P = 2,
beta by bisection on the population PSNR-gain rule (the repo hook), exact expected loss.

For every target: CRPS excess of q_rc over p (relative to CRPS(p)), PIT calibration error, central-interval coverage
error at 50/90/95/99%, mass of q_rc outside the support of p, and the same for the fair-CRPS minimiser (solver control).
Writes candidates/shape_effect.json (+ a figure) under /share/gpu0/jjwhit/crps_review/candidates/.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path.home() / "crps_review/paper/toy"))
from analytic_1d import AbsMoment, Mix, fair_loss, optimise_atoms, psnr_gain_gap, rcgan_loss_P2  # noqa: E402

OUT = Path("/share/gpu0/jjwhit/crps_review/candidates")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)


class ZeroInflated:
    """mass pi0 at 0, else HalfNormal(scale) (sparse positive sky pixel)."""

    def __init__(self, pi0, scale=1.0):
        self.pi0, self.h = pi0, stats.halfnorm(scale=scale)

    def cdf(self, x):
        x = np.asarray(x, float)
        return np.where(x < 0, 0.0, self.pi0 + (1 - self.pi0) * self.h.cdf(np.maximum(x, 0)))

    def ppf(self, u):
        u = np.asarray(u, float)
        return np.where(u <= self.pi0, 0.0, self.h.ppf(np.clip((u - self.pi0) / (1 - self.pi0), 0, 1)))

    def mean(self):
        return (1 - self.pi0) * self.h.mean()

    def var(self):
        m2 = (1 - self.pi0) * (self.h.var() + self.h.mean() ** 2)
        return m2 - self.mean() ** 2


class Clipped:
    """N(mu, 1) clipped to [0, inf): atom at 0 of mass Phi(-mu) (bounded / saturated pixel, e.g. magnitude near 0)."""

    def __init__(self, mu):
        self.mu, self.n = mu, stats.norm(loc=mu)

    def cdf(self, x):
        x = np.asarray(x, float)
        return np.where(x < 0, 0.0, self.n.cdf(x))

    def ppf(self, u):
        return np.maximum(self.n.ppf(np.asarray(u, float)), 0.0)

    def mean(self):
        mu = self.mu
        return mu * stats.norm.cdf(mu) + stats.norm.pdf(mu)

    def var(self):
        mu = self.mu
        m2 = (mu ** 2 + 1) * stats.norm.cdf(mu) + mu * stats.norm.pdf(mu)
        return m2 - self.mean() ** 2


def targets():
    T = {
        "Gaussian": (stats.norm(), "reference"),
        "Gamma k=2 (skew 1.4)": (stats.gamma(2.0), "skewed"),
        "Gamma k=0.5 (skew 2.8)": (stats.gamma(0.5), "skewed"),
        "lognormal s=1 (skew 6.2)": (stats.lognorm(1.0), "skewed"),
        "Laplace": (stats.laplace(), "heavy-tailed"),
        "Student t3": (stats.t(3), "heavy-tailed"),
        "bimodal 0.5/0.5, sep 3 sd": (Mix([0.5, 0.5], [-1.5, 1.5], [1.0, 1.0]), "multimodal"),
        "bimodal 0.5/0.5, sep 5 sd": (Mix([0.5, 0.5], [-2.5, 2.5], [1.0, 1.0]), "multimodal"),
        "bimodal 0.8/0.2, sep 5 sd": (Mix([0.8, 0.2], [0.0, 5.0], [1.0, 1.0]), "multimodal"),
        "Beta(0.5,0.5) (bounded, U)": (stats.beta(0.5, 0.5), "bounded"),
        "Beta(2,5) (bounded)": (stats.beta(2, 5), "bounded"),
        "clipped N(1,1) (16% atom at 0)": (Clipped(1.0), "bounded"),
        "clipped N(0,1) (50% atom at 0)": (Clipped(0.0), "bounded"),
        "zero-inflated pi0=0.5": (ZeroInflated(0.5), "sparse"),
        "zero-inflated pi0=0.8": (ZeroInflated(0.8), "sparse"),
        "zero-inflated pi0=0.95": (ZeroInflated(0.95), "sparse"),
    }
    return T


def fair_crps_atoms(q, xs):
    """E|Q - X| - 0.5 E|Q - Q'| for atoms q (M,) and target draws xs (n,)."""
    qs = np.sort(q)
    M = len(qs)
    # E|Q-Q'| over distinct pairs (fair) via sorted formula
    i = np.arange(1, M + 1)
    eqq = 2 * ((2 * i - M - 1) * qs).sum() / (M * (M - 1))
    cs = np.concatenate([[0], np.cumsum(qs)])
    k = np.searchsorted(qs, xs)
    egx = (xs * (2 * k - M) - 2 * cs[k] + cs[-1]) / M
    return float(egx.mean() - 0.5 * eqq)


def crps_true(dist, xs, n=20000):
    u = (np.arange(n) + 0.5) / n
    return fair_crps_atoms(np.asarray(dist.ppf(u), float), xs)


def pit(q, xs, rng):
    qs = np.sort(q)
    below = np.searchsorted(qs, xs, side="left")
    ties = np.searchsorted(qs, xs, side="right") - below
    return (below + rng.uniform(size=len(xs)) * (ties + 1)) / (len(qs) + 1)


def summarise(q, dist, xs, rng, lo_support):
    u = pit(q, xs, rng)
    hist = np.histogram(u, bins=10, range=(0, 1))[0] / len(u) * 10
    cov = {str(a): float((np.abs(u - 0.5) <= a / 2).mean() - a) for a in (0.5, 0.9, 0.95, 0.99)}
    return dict(pit_hist10=hist.round(3).tolist(), pit_ce=float(np.sqrt(((hist - 1) ** 2).mean())), cov_err=cov,
                mass_below_support=float((q < lo_support - 1e-9).mean()) if lo_support is not None else None,
                sd_ratio=float(q.std() / math.sqrt(dist.var())))


def main(M=400):
    rng = np.random.default_rng(0)
    res = {}
    T = targets()
    fig, axs = plt.subplots(4, 4, figsize=(15, 12))
    for ax, (name, (dist, fam)) in zip(axs.flat, T.items()):
        h = AbsMoment(dist)
        u = (np.arange(M) + 0.5) / M
        init = torch.as_tensor(np.asarray(dist.ppf(u), float))
        init = init + 1e-3 * torch.randn(M)  # break atoms (zero-inflated targets)
        lo, hi, q = 0.3, 4.0, init
        for _ in range(22):
            beta = 0.5 * (lo + hi)
            q = optimise_atoms(rcgan_loss_P2(h, beta), q, iters=1500)
            if psnr_gain_gap(q, dist) > 0:
                hi = beta
            else:
                lo = beta
        q_rc = optimise_atoms(rcgan_loss_P2(h, beta), q, iters=6000, lr=5e-3).numpy()
        q_f = optimise_atoms(fair_loss(h), init + 0.05 * torch.randn(M) * math.sqrt(dist.var()), iters=6000).numpy()
        xs = np.asarray(dist.ppf(rng.uniform(size=200000)), float)
        c_p = crps_true(dist, xs)
        c_rc, c_f = fair_crps_atoms(q_rc, xs), fair_crps_atoms(q_f, xs)
        sup = 0.0 if fam in ("skewed", "bounded", "sparse") else None  # all bounded targets here start at 0
        res[name] = dict(family=fam, beta=beta, crps_true=c_p,
                         crps_excess_rc_pct=100 * (c_rc - c_p) / c_p, crps_excess_fair_pct=100 * (c_f - c_p) / c_p,
                         rc=summarise(q_rc, dist, xs, rng, sup), fair=summarise(q_f, dist, xs, rng, sup))
        r = res[name]
        print(f"{name:34s} beta={beta:.3f} CRPS excess rc {r['crps_excess_rc_pct']:6.2f}% fair {r['crps_excess_fair_pct']:5.2f}% "
              f"PIT-CE rc {r['rc']['pit_ce']:.3f} fair {r['fair']['pit_ce']:.3f} cov90 rc {r['rc']['cov_err']['0.9']:+.3f} "
              f"cov99 rc {r['rc']['cov_err']['0.99']:+.3f} below-support rc {r['rc']['mass_below_support']}", flush=True)
        lo_x = min(np.quantile(xs, 0.001), q_rc.min()) - 0.2
        hi_x = max(np.quantile(xs, 0.999), q_rc.max()) + 0.2
        bins = np.linspace(lo_x, hi_x, 60)
        ax.hist(xs, bins=bins, density=True, color="0.7", label="p")
        ax.hist(q_rc, bins=bins, density=True, histtype="step", color="C3", lw=1.5, label=f"rcGAN (β={beta:.2f})")
        ax.hist(q_f, bins=bins, density=True, histtype="step", color="C0", lw=1.2, label="fair CRPS")
        ax.set_title(f"{name}\nCRPS +{r['crps_excess_rc_pct']:.1f}%, PIT-CE {r['rc']['pit_ce']:.2f}", fontsize=8)
        ax.set_yscale("log")
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"shape_effect.{ext}", dpi=130)
    json.dump(res, open(OUT / "shape_effect.json", "w"), indent=1)


if __name__ == "__main__":
    main()
