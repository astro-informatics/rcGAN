"""Pilot: a trained rcGAN (R) vs fair CRPS (F) on a zero-inflated problem with an EXACT per-pixel posterior.

Question: does the 1-D theory (candidates/shape_effect.py: beta-tuned rcGAN minimiser loses 10-110% CRPS on zero-inflated
targets) survive a real network, a critic and ONE global beta serving pixels of different posterior shape?

Benchmark "ZI" (32x32, iid pixels; not image-like, by design, so the posterior is exact):
  x_i = 0 with prob. pi0 = 0.8, else |N(0, 1)|                (sparse positive "sky")
  fixed pattern (seed 1234): 40% unobserved (posterior = prior, zero-inflated),
                             30% y = x + N(0, 1.0^2)  (low SNR: atom weight depends on y),
                             30% y = x + N(0, 0.2^2)  (high SNR: near-Gaussian truncated posterior)
  posterior per observed pixel: w0 delta_0 + (1 - w0) N(m, s^2) truncated to x > 0, m = y/(1+sig^2), s^2 = sig^2/(1+sig^2),
  w0 = pi0 N(y;0,sig^2) / (pi0 N(y;0,sig^2) + (1-pi0) 2 N(y;0,1+sig^2) Phi(m/s)).
Training is the toy protocol (~/crps_review/paper/toy/train.py: same U-Net/critic, WGAN-GP, Adam(1e-3, (0, .99)), P = 2,
repo adversarial schedule, repo beta hook for R), imported unchanged. 80 epochs x 250 steps (= the toy's 20k steps, with
twice as many beta updates, because R's beta must climb further here).
Evaluation (512 test y, 128 model + 128 exact samples per y), per regime: fair CRPS vs truth (and the exact posterior's, the
floor), the per-pixel energy distance between model and exact samples (0 iff the marginals match), randomised-PIT coverage
error at 50/90%, sample mass below 0, sd ratio. -> /share/gpu0/jjwhit/crps_review/candidates/zi/<name>.json

Usage: python candidates/zi_pilot.py --reg {rcgan,fair} [--epochs 80 --steps-per-epoch 250 --seed 0]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOY = Path.home() / "crps_review/paper/toy"
sys.path.insert(0, str(TOY))
from losses import make_regulariser  # noqa: E402
from models import Critic, Generator  # noqa: E402
from train import ADV_UNIT, adv_weight_rc, gradient_penalty, psnr, sample  # noqa: E402

OUT = Path("/share/gpu0/jjwhit/crps_review/candidates/zi")
REGIMES = {0: "unobserved (prior, pi0=0.8)", 1: "low SNR (sigma=1)", 2: "high SNR (sigma=0.2)"}


class ZI:
    N, D, pi0 = 32, 1024, 0.8

    def __init__(self, dev):
        r = np.random.default_rng(1234).uniform(size=self.D)
        code = np.where(r < 0.4, 0, np.where(r < 0.7, 1, 2))
        self.code = torch.as_tensor(code, device=dev)
        self.sig = torch.as_tensor(np.where(code == 1, 1.0, 0.2), device=dev, dtype=torch.float64)
        self.obs = self.code > 0
        self.dev = dev

    def sample_prior(self, n, gen):
        z = torch.rand(n, self.D, device=self.dev, generator=gen, dtype=torch.float64) < self.pi0
        slab = torch.randn(n, self.D, device=self.dev, generator=gen, dtype=torch.float64).abs()
        return torch.where(z, torch.zeros_like(slab), slab)

    def observe(self, x, gen):
        y = x + self.sig * torch.randn(x.shape, device=self.dev, generator=gen, dtype=torch.float64)
        return torch.where(self.obs, y, torch.zeros_like(y))

    def y_image(self, y):
        n = y.shape[0]
        code = (self.code.float() / 2).expand(n, -1)
        return torch.stack([y.float(), code], 1).view(n, 2, self.N, self.N)

    def posterior(self, y):
        """(w0, m, s) per pixel; unobserved pixels: prior (w0 = pi0, m = 0, s = 1)."""
        sig2 = self.sig ** 2
        m = y / (1 + sig2)
        s = torch.sqrt(sig2 / (1 + sig2))
        lz = math.log(self.pi0) - 0.5 * y ** 2 / sig2 - 0.5 * torch.log(sig2)
        ls = (math.log(1 - self.pi0) + math.log(2) - 0.5 * y ** 2 / (1 + sig2) - 0.5 * torch.log(1 + sig2)
              + torch.log(torch.special.ndtr(m / s).clamp_min(1e-300)))
        w0 = torch.sigmoid(lz - ls)
        w0 = torch.where(self.obs, w0, torch.full_like(w0, self.pi0))
        m = torch.where(self.obs, m, torch.zeros_like(m))
        s = torch.where(self.obs, s, torch.ones_like(s))
        return w0, m, s

    def sample_posterior(self, y, n, gen):
        w0, m, s = self.posterior(y)  # (B, D)
        B = y.shape[0]
        zero = torch.rand(B, n, self.D, device=self.dev, generator=gen, dtype=torch.float64) < w0[:, None]
        lo = torch.special.ndtr(-m / s)[:, None]
        u = torch.rand(B, n, self.D, device=self.dev, generator=gen, dtype=torch.float64)
        slab = m[:, None] + s[:, None] * torch.special.ndtri((lo + u * (1 - lo)).clamp(1e-15, 1 - 1e-15))
        return torch.where(zero, torch.zeros_like(slab), slab.clamp_min(0))


def fair_crps_pix(S, t):
    """S (n, P) samples, t (P,) -> (P,)"""
    n = S.shape[0]
    s, _ = S.sort(0)
    i = torch.arange(1, n + 1, device=S.device, dtype=S.dtype)[:, None]
    return (S - t).abs().mean(0) - ((2 * i - n - 1) * s).sum(0) / (n * (n - 1))


def energy_dist_pix(A, B):
    """per-pixel unbiased energy distance between sample sets A (n, P), B (m, P)."""
    exy = (A[:, None] - B[None]).abs().mean((0, 1))
    n, m = A.shape[0], B.shape[0]
    sa, _ = A.sort(0)
    sb, _ = B.sort(0)
    ia = torch.arange(1, n + 1, device=A.device, dtype=A.dtype)[:, None]
    ib = torch.arange(1, m + 1, device=A.device, dtype=A.dtype)[:, None]
    eaa = 2 * ((2 * ia - n - 1) * sa).sum(0) / (n * (n - 1))
    ebb = 2 * ((2 * ib - m - 1) * sb).sum(0) / (m * (m - 1))
    return exy - 0.5 * eaa - 0.5 * ebb


def rpit(S, t, gen):
    below = (S < t).sum(0).double()
    eq = (S == t).sum(0).double()
    u = torch.rand(t.shape, device=t.device, generator=gen, dtype=torch.float64)
    return (below + u * (eq + 1)) / (S.shape[0] + 1)


@torch.no_grad()
def evaluate(G, bench, dev, n_test=512, n_samp=128, label="model"):
    g = torch.Generator(device=dev).manual_seed(2)
    x = bench.sample_prior(n_test, g)
    y = bench.observe(x, g)
    c = bench.y_image(y)
    ge = torch.Generator(device=dev).manual_seed(3)
    gz = torch.Generator(device=dev).manual_seed(4)
    gp = torch.Generator(device=dev).manual_seed(5)
    keys = ["crps", "crps_exact", "crps_clip", "edist", "edist_clip", "cov50", "cov90", "cov50_exact", "cov90_exact",
            "neg", "var", "var_exact"]
    acc = {k: torch.zeros(3, device=dev, dtype=torch.float64) for k in keys}
    cnt = torch.zeros(3, device=dev, dtype=torch.float64)
    for i in range(0, n_test, 16):
        sl = slice(i, i + 16)
        S = sample(G, c[sl], n_samp, gz).view(-1, n_samp, bench.D).double() if G is not None else None
        E = bench.sample_posterior(y[sl], n_samp, ge)
        E2 = bench.sample_posterior(y[sl], n_samp, ge)
        if S is None:  # null: a second independent exact set plays the model
            S = E2
        for b in range(S.shape[0]):
            t = x[sl][b]
            s, e = S[b], E[b]
            vals = dict(crps=fair_crps_pix(s, t), crps_exact=fair_crps_pix(e, t), crps_clip=fair_crps_pix(s.clamp_min(0), t),
                        edist=energy_dist_pix(s, e), edist_clip=energy_dist_pix(s.clamp_min(0), e),
                        neg=(s < 0).double().mean(0), var=s.var(0), var_exact=e.var(0))
            u, ue = rpit(s, t, gp), rpit(e, t, gp)
            vals["cov50"] = ((u - 0.5).abs() <= 0.25).double()
            vals["cov90"] = ((u - 0.5).abs() <= 0.45).double()
            vals["cov50_exact"] = ((ue - 0.5).abs() <= 0.25).double()
            vals["cov90_exact"] = ((ue - 0.5).abs() <= 0.45).double()
            for r in range(3):
                sel = bench.code == r
                for k in keys:
                    acc[k][r] += vals[k][sel].sum()
                cnt[r] += sel.sum()
    out = {}
    for r, lab in REGIMES.items():
        a = {k: float(acc[k][r] / cnt[r]) for k in keys}
        out[lab] = dict(crps=a["crps"], crps_exact=a["crps_exact"], crps_excess_pct=100 * (a["crps"] / a["crps_exact"] - 1),
                        crps_clip_excess_pct=100 * (a["crps_clip"] / a["crps_exact"] - 1),
                        edist_rel_pct=100 * a["edist"] / a["crps_exact"], edist_clip_rel_pct=100 * a["edist_clip"] / a["crps_exact"],
                        cov50_err=a["cov50"] - 0.5, cov90_err=a["cov90"] - 0.9,
                        cov50_err_exact=a["cov50_exact"] - 0.5, cov90_err_exact=a["cov90_exact"] - 0.9,
                        frac_neg=a["neg"], sd_ratio=math.sqrt(a["var"] / a["var_exact"]))
    tot = {k: float(acc[k].sum() / cnt.sum()) for k in keys}
    out["all"] = dict(crps_excess_pct=100 * (tot["crps"] / tot["crps_exact"] - 1),
                      crps_clip_excess_pct=100 * (tot["crps_clip"] / tot["crps_exact"] - 1),
                      edist_rel_pct=100 * tot["edist"] / tot["crps_exact"], frac_neg=tot["neg"],
                      sd_ratio=math.sqrt(tot["var"] / tot["var_exact"]))
    print(label, json.dumps(out, indent=1), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reg", default="fair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--steps-per-epoch", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-train", type=int, default=64000)
    p.add_argument("--P", type=int, default=2)
    p.add_argument("--adv", default="sched")
    p.add_argument("--null-only", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bench = ZI(dev)
    if a.null_only:
        json.dump(evaluate(None, bench, dev, label="EXACT-null"), open(OUT / "EXACT.json", "w"), indent=1)
        return
    name = f"{a.reg}_s{a.seed}"
    N = bench.N
    g0 = torch.Generator(device=dev).manual_seed(0)
    x_tr = bench.sample_prior(a.n_train, g0)
    c_tr = bench.y_image(bench.observe(x_tr, g0))
    x_tr = x_tr.float().view(-1, 1, N, N)
    g1 = torch.Generator(device=dev).manual_seed(1)
    x_val = bench.sample_prior(1000, g1)
    c_val = bench.y_image(bench.observe(x_val, g1))
    x_val = x_val.float().view(-1, 1, N, N)

    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
    G, D = Generator().to(dev), Critic().to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.0, 0.99))
    reg = make_regulariser(a.reg, N, dev, x_tr)
    beta, hist, t0 = 1.0, [], time.time()
    for epoch in range(a.epochs):
        lam = adv_weight_rc(a.adv, epoch, a.epochs) * ADV_UNIT
        perm = torch.randperm(a.n_train, device=dev, generator=gen)
        G.train()
        for it in range(a.steps_per_epoch):
            idx = perm[(it * a.batch) % a.n_train:][: a.batch]
            if idx.shape[0] < a.batch:
                idx = perm[: a.batch]
            x, c = x_tr[idx], c_tr[idx]
            B = x.shape[0]
            with torch.no_grad():
                fake = G(c, torch.randn(B, G.noise_chans, N, N, device=dev, generator=gen))
            gp = gradient_penalty(D, x, fake, c, gen)
            rp, fp = D(x, c), D(fake, c)
            d_loss = fp.mean() - rp.mean() + 10 * gp + 1e-3 * (rp ** 2).mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            gens = torch.stack([G(c, torch.randn(B, G.noise_chans, N, N, device=dev, generator=gen)) for _ in range(a.P)], 1)
            r = reg(gens, x, beta)
            g_loss = r - lam * D(gens.flatten(0, 1), c.repeat_interleave(a.P, 0)).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        if not math.isfinite(g_loss.item()):
            print("DIVERGED", flush=True)
            break
        G.eval()
        s = sample(G, c_val, 8, torch.Generator(device=dev).manual_seed(777))
        p1, p8 = psnr(s[:, 0], x_val).mean().item(), psnr(s.mean(1), x_val).mean().item()
        if a.reg == "rcgan":
            beta += 2e-2 * ((p1 + 2.5) - p8)
        hist.append(dict(epoch=epoch, reg=r.item(), d=d_loss.item(), psnr1=p1, psnr8=p8, beta=beta, wall=time.time() - t0))
        print(f"ep {epoch:3d} t {time.time() - t0:5.0f}s reg {r.item():.4f} d {d_loss.item():+.3f} psnr1 {p1:.2f} psnr8 {p8:.2f} "
              f"gain {p8 - p1:.2f} beta {beta:.3f}", flush=True)
    G.eval()
    res = dict(args=vars(a), beta=beta, hist=hist, eval=evaluate(G, bench, dev, label=name))
    torch.save(G.state_dict(), OUT / f"{name}_G.pt")
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
