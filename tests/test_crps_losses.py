"""CPU unit tests for the CRPS losses in models/lightning/mmGAN.py.

The loss methods are extracted from the source with `ast` and executed directly,
so the test checks the exact code in mmGAN.py without importing the whole
module (pytorch_lightning, lpips, ... are slow to import on the login node).

Run:  python tests/test_crps_losses.py
"""
import ast
import math
import os
import types
import unittest

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMGAN_PATH = os.path.join(REPO, "models", "lightning", "mmGAN.py")
LOSS_NAMES = ["_crps_core", "normal_crps_loss", "fair_crps_loss", "alpha_fair_crps_loss"]
MS = [2, 3, 4, 8]


def load_losses(path=MMGAN_PATH):
    """Return an object exposing the CRPS methods of mmGAN, taken verbatim from the source."""
    src = open(path).read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "mmGAN")
    funcs = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name in LOSS_NAMES]
    assert sorted(f.name for f in funcs) == sorted(LOSS_NAMES)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    obj = types.SimpleNamespace()
    for name in LOSS_NAMES:
        setattr(obj, name, types.MethodType(ns[name], obj))
    return obj


L = load_losses()


def committed_fair_crps_loss(gens, x):
    """fair_crps_loss as committed at ea08d39 (models/lightning/mmGAN.py:107)."""
    abs_diff_truth, abs_diff_samps = L._crps_core(gens, x)
    M = gens.shape[1]
    fair_crps = abs_diff_truth.mean(dim=1) - ((M - 1) / M) * 0.5 * abs_diff_samps.mean(dim=(1, 2))
    return fair_crps.mean()


# ---- textbook reference implementations (explicit loops) ----

def loop_crps(gens, x, spread):
    """mean_i |X_i - x| - spread(pair_sum, M), averaged over batch and pixels."""
    B, M = gens.shape[:2]
    g = gens.reshape(B, M, -1).double()
    t = x.reshape(B, -1).double()
    total, count = 0.0, 0
    for b in range(B):
        for p in range(g.shape[2]):
            skill = sum(abs(g[b, i, p] - t[b, p]) for i in range(M)) / M
            pair_sum = sum(abs(g[b, i, p] - g[b, j, p]) for i in range(M) for j in range(M) if i != j)
            total += float(skill - spread(pair_sum, M))
            count += 1
    return total / count


def loop_standard(gens, x):  # plug-in / energy form: 1/(2M^2) sum_{i,j}
    return loop_crps(gens, x, lambda s, M: s / (2 * M * M))


def loop_fair(gens, x):  # Ferro (2008) fair CRPS: 1/(2M(M-1)) sum_{i!=j}
    return loop_crps(gens, x, lambda s, M: s / (2 * M * (M - 1)))


def loop_alpha_fair(gens, x, alpha):  # Lang et al. (2024) afCRPS = alpha*fair + (1-alpha)*standard
    return alpha * loop_fair(gens, x) + (1 - alpha) * loop_standard(gens, x)


def gaussian_crps(mu, s, x):
    z = (x - mu) / s
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return s * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


class TestAgainstLoops(unittest.TestCase):
    """(i) each loss equals a direct loop over its textbook formula."""

    def setUp(self):
        torch.manual_seed(0)

    def _data(self, M):
        return torch.randn(3, M, 1, 4, 5, dtype=torch.float64), torch.randn(3, 1, 4, 5, dtype=torch.float64)

    def test_standard(self):
        for M in MS:
            gens, x = self._data(M)
            self.assertAlmostEqual(L.normal_crps_loss(gens, x).item(), loop_standard(gens, x), places=10, msg=f"M={M}")

    def test_fair(self):
        for M in MS:
            gens, x = self._data(M)
            self.assertAlmostEqual(L.fair_crps_loss(gens, x).item(), loop_fair(gens, x), places=10, msg=f"M={M}")

    def test_committed_fair_is_wrong(self):
        for M in MS:
            gens, x = self._data(M)
            self.assertGreater(abs(committed_fair_crps_loss(gens, x).item() - loop_fair(gens, x)), 1e-3, msg=f"M={M}")

    def test_alpha_fair(self):
        for M in MS:
            gens, x = self._data(M)
            for alpha in [0.0, 0.1, 0.5, 0.95, 1.0]:
                self.assertAlmostEqual(L.alpha_fair_crps_loss(gens, x, alpha).item(), loop_alpha_fair(gens, x, alpha),
                                       places=10, msg=f"M={M} alpha={alpha}")

    def test_alpha_endpoints(self):
        for M in MS:
            gens, x = self._data(M)
            self.assertAlmostEqual(L.alpha_fair_crps_loss(gens, x, 1.0).item(), L.fair_crps_loss(gens, x).item(), places=10)
            self.assertAlmostEqual(L.alpha_fair_crps_loss(gens, x, 0.0).item(), L.normal_crps_loss(gens, x).item(), places=10)


class TestMonteCarloUnbiased(unittest.TestCase):
    """(ii) E[fair CRPS] over Gaussian ensembles equals the closed-form CRPS of N(mu, s^2) at x."""

    N = 400_000  # independent ensembles (as pixels)

    def test_fair_unbiased_committed_biased(self):
        mu, s = 0.3, 1.7
        torch.manual_seed(1)
        for M in MS:
            for xv in [-1.0, 0.3, 2.5]:
                target = gaussian_crps(mu, s, xv)
                gens = (mu + s * torch.randn(1, M, 1, 1, self.N, dtype=torch.float64))
                x = torch.full((1, 1, 1, self.N), xv, dtype=torch.float64)
                # per-ensemble values, to get a standard error
                t, sp = L._crps_core(gens, x)
                fair_vals = t.mean(dim=1) - (M / (M - 1)) * 0.5 * sp.mean(dim=(1, 2))
                se = fair_vals.std().item() / math.sqrt(self.N)
                fair = L.fair_crps_loss(gens, x).item()
                committed = committed_fair_crps_loss(gens, x).item()
                print(f"  M={M} x={xv:+.1f}: closed={target:.4f} fair={fair:.4f} (se {se:.4f}) "
                      f"committed={committed:.4f} standard={L.normal_crps_loss(gens, x).item():.4f}")
                self.assertLess(abs(fair - target), 4 * se, msg=f"M={M} x={xv}")
                self.assertGreater(abs(committed - target), 10 * se, msg=f"M={M} x={xv}")


class TestMinimisingSpread(unittest.TestCase):
    """(iii) Truth y ~ N(0,1), forecast ensemble ~ N(0, sigma^2): argmin_sigma E[loss].

    Analytically E[loss] = sqrt(2/pi) sqrt(1+sigma^2) - 2k sigma/sqrt(pi), with k the
    effective weight on E|X-X'|, giving sigma* = sqrt(2k^2 / (1 - 2k^2)):
      fair k=1/2 -> 1;  standard k=(M-1)/(4M) -> 1/sqrt(7) at M=2;
      committed k=(M-1)^2/(2M^2) -> 1/sqrt(31) at M=2.
    Checked numerically on the code with common random numbers and a golden-section search.
    """

    N = 200_000

    @staticmethod
    def sigma_star(k):
        return math.sqrt(2 * k * k / (1 - 2 * k * k))

    def argmin(self, loss_fn, M):
        torch.manual_seed(2)
        z = torch.randn(1, M, 1, 1, self.N, dtype=torch.float64)
        y = torch.randn(1, 1, 1, self.N, dtype=torch.float64)
        f = lambda s: loss_fn(s * z, y).item()
        a, b = 0.01, 3.0
        g = (math.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(25):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        return (a + b) / 2

    def test_minimisers(self):
        for M in MS:
            cases = {
                "fair": (L.fair_crps_loss, 0.5),
                "standard": (L.normal_crps_loss, 0.5 * (M - 1) / M),
                "committed": (committed_fair_crps_loss, 0.5 * ((M - 1) / M) ** 2),
            }
            for name, (fn, k) in cases.items():
                expected = self.sigma_star(k)
                got = self.argmin(fn, M)
                print(f"  M={M} {name:9s}: sigma*={got:.4f} (analytic {expected:.4f})")
                self.assertAlmostEqual(got, expected, delta=0.03 * expected + 0.005, msg=f"M={M} {name}")
        self.assertAlmostEqual(self.sigma_star(0.25), 1 / math.sqrt(7), places=12)
        self.assertAlmostEqual(self.sigma_star(0.125), 1 / math.sqrt(31), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
