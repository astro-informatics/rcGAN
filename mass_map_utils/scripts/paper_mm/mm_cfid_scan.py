"""CFID vs RMSE checkpoint selection (reports/10_mm_cfid_vs_rmse.md). Run via mm_run.py (stageA mode).

For each checkpoint of a run, on fixed observations (sorted maps, noise seed offset + i, latent seed per batch, identical
for every checkpoint, so epochs are compared with common random numbers):
  * CFID as in the pre-10-Oct-2024 scripts/mass_map/validate.py (dc3e63b^): evaluation_scripts/mass_map_cfid CFIDMetric with
    VGG16Embedding for both image and condition, one sample per observation, get_cfid_torch_pinv. The old _get_embed_im reads
    channels 0 and 1 of the generator output, which no longer exist (out_chans = 1), so it is ported: the image is the
    sample's kappa map min-max normalised and repeated over 3 channels (identical to the old code for a real map, whose second
    channel was zero); the condition is the real part of unnormalize_complex(gamma1, gamma2), min-max normalised (unchanged).
  * RMSE / PSNR (mass_map_utils ks_utils, COSMOS mask) of the P-sample posterior mean, as the current validate.py (which
    uses P = num_z_test = 32; here P = 8 by default to save GPU time; the recorded RMSE pick is checkpoint_best itself).
Output: eval/cfid/<split>_<run>.json.
"""
import argparse
import glob
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import torch

from common import TestSet, load_cfg, load_model, to_kappa  # noqa: E402  (patched by mm_run.py)
from mass_map_utils.scripts.ks_utils import psnr, rmse
from mass_map_utils.scripts.transforms import unnormalize_complex
from evaluation_scripts.mass_map_cfid.cfid_metric import CFIDMetric
from utils.embeddings import VGG16Embedding

OLD = "/share/gpu0/jjwhit/mass_map/mm_models/"
NEW = "/share/gpu0/jjwhit/crps_review/paper/mm/ckpt/"
RUNS = {"R": OLD + "mmgan_training_real_output", "F100": OLD + "mmgan_crps_alpha_100", "S": OLD + "mmgan_crps_base",
        "FB": OLD + "mmgan_crps_fair2", "NL": OLD + "mmgan_no_loss", "F3": OLD + "mmgan_crps_fair3"}
OUT = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/cfid/"
VAL_DIR = "/share/gpu0/jjwhit/kappa_cosmos_simulations/cropped_dataset/kappa_val"


class ValSet(TestSet):
    """Sorted kappa_val files with fixed per-map observation noise (seed 2_000_000 + i, disjoint from the test seeds)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.files = sorted(pathlib.Path(VAL_DIR).iterdir())

    def get(self, i):
        kappa = np.load(self.files[i], allow_pickle=True).astype(np.float64)
        state = np.random.get_state()
        np.random.seed(2_000_000 + i)
        y, x, _, _ = self.transform(kappa)
        np.random.set_state(state)
        return y, x


def minmax3(im):  # (B, H, W) -> (B, 3, H, W) in [0, 1], per image, as CFIDMetric._get_embed_im
    lo = im.flatten(1).min(1)[0].view(-1, 1, 1)
    hi = im.flatten(1).max(1)[0].view(-1, 1, 1)
    return ((im - lo) / (hi - lo)).unsqueeze(1).repeat(1, 3, 1, 1)


def ckpt_list(path, which):
    if which == "best":
        return ["checkpoint_best.ckpt"]
    eps = sorted(int(re.search(r"epoch=(\d+)", f).group(1)) for f in glob.glob(path + "/checkpoint-epoch=*.ckpt"))
    return ["checkpoint_best.ckpt"] + [f"checkpoint-epoch={e}.ckpt" for e in eps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpts", default="all", help="all (checkpoint_best + every epoch file), best, or a comma list")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n-maps", type=int, default=0, help="0 = whole split")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--P", type=int, default=8)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--tag", default="", help="suffix for pilot outputs")
    a = p.parse_args()
    torch.set_float32_matmul_precision("medium")
    cfg = load_cfg()
    ds = ValSet(cfg) if a.split == "val" else TestSet(cfg)
    mask = ds.mask
    n = a.n_maps or len(ds)
    seeds = [int(s) for s in a.seeds.split(",")]
    path = RUNS.get(a.run) or (a.run if os.path.isabs(a.run) else None)
    if path is None:
        from mm_run import MODELS
        path = MODELS[a.run]
    ckpts = ckpt_list(path, a.ckpts) if a.ckpts in ("all", "best") else a.ckpts.split(",")
    os.makedirs(OUT, exist_ok=True)
    out_f = OUT + f"{a.split}_{os.path.basename(a.run)}{a.tag}.json"
    res = json.load(open(out_f)) if os.path.exists(out_f) else {"run": a.run, "path": path, "split": a.split, "n_maps": n,
                                                                  "P": a.P, "ckpts": {}}

    vgg = VGG16Embedding()
    cm = CFIDMetric(gan=None, loader=None, image_embedding=vgg, condition_embedding=vgg, cuda=True, args=cfg)
    emb = lambda ims: vgg(cm.transforms(ims)).double()  # noqa: E731

    # model-independent: observations, truths and their embeddings
    t0 = time.time()
    batches, cond_e, true_e, truths = [], [], [], []
    for b0 in range(0, n, a.batch):
        y, x = ds.batch(range(b0, min(n, b0 + a.batch)))
        batches.append(y)
        xk = to_kappa(x[:, 0].double(), cfg).float()
        truths.append(xk.numpy())
        yc = y[:, :2].permute(0, 2, 3, 1).cuda()  # (B, H, W, re/im) as the old reformatted tensor
        cond = unnormalize_complex(yc)[..., 0]
        with torch.no_grad():
            cond_e.append(emb(minmax3(cond.float())))
            true_e.append(emb(minmax3(xk.cuda())))
    cond_e, true_e, truths = torch.cat(cond_e), torch.cat(true_e), np.concatenate(truths)
    print(f"[{a.run}] {a.split}: {n} maps, embeddings {tuple(cond_e.shape)} in {time.time() - t0:.0f}s", flush=True)

    for ck in ckpts:
        if ck in res["ckpts"]:
            continue
        t0 = time.time()
        model = load_model(path, ckpt=ck)
        r = {"epoch": int(torch.load(os.path.join(path, ck), map_location="cpu")["epoch"])}
        with torch.no_grad():
            for s in seeds:
                img_e = []
                for bi, y in enumerate(batches):
                    torch.manual_seed(1000 * s + bi)
                    g = to_kappa(model(y.cuda())[:, 0].float(), model.args)
                    img_e.append(emb(minmax3(g)))
                cfid, m_d, c_d = cm.get_cfid_torch_pinv(y_predict=torch.cat(img_e), x_true=cond_e, y_true=true_e)
                r[f"cfid_s{s}"], r[f"cfid_m_s{s}"], r[f"cfid_c_s{s}"] = float(cfid), float(m_d), float(c_d)
            rm, ps = [], []
            for bi, y in enumerate(batches):
                torch.manual_seed(10_000 + bi)
                yy = y.cuda()
                mean = sum(to_kappa(model(yy)[:, 0].float(), model.args) for _ in range(a.P)) / a.P
                mean = mean.cpu().numpy()
                for j in range(len(y)):
                    xt = truths[bi * a.batch + j]
                    rm.append(rmse(xt.copy(), mean[j].copy(), mask))
                    ps.append(psnr(xt.copy(), mean[j].copy(), mask))
            r[f"rmse_P{a.P}"], r[f"psnr_P{a.P}"] = float(np.mean(rm)), float(np.mean(ps))
        res["ckpts"][ck] = r
        json.dump(res, open(out_f, "w"), indent=1)
        print(f"[{a.run}] {ck} (epoch {r['epoch']}): " + " ".join(f"cfid_s{s}={r[f'cfid_s{s}']:.4f}" for s in seeds) +
              f" rmse={r[f'rmse_P{a.P}']:.6f} psnr={r[f'psnr_P{a.P}']:.3f}  {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


main()
