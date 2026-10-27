"""Run an existing stage-A / joint-tests script unchanged, on the paper-mm models.

The stage-A (`crps-review` worktree) and joint-tests scripts are reused byte for byte, so the
test maps, observation noise (seed 1_000_000 + map), latent seeds (torch.manual_seed(map)),
N, geometry and statistics are identical to 02_stageA.md / 03_joint_and_collapse.md. Only
three things are patched before the script runs:
  * this worktree is put first on sys.path, so `models.lightning.mmGAN` is the paper-mm version
    (identical forward pass; it also knows the "ms" mode and adv_scale hyper-parameter);
  * MODELS / load_model resolve the new keys to the paper-mm checkpoint dir;
  * output roots point to /share/gpu0/jjwhit/crps_review/paper/mm/eval/.
R and F100 outputs are not recomputed: they are symlinked from the original runs.

Usage: python mm_run.py <stageA|joint> <script.py> [script args...]
"""
import os
import runpy
import sys

WT = "/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm"
REVIEW = "/home/jjwhit/.herdr/worktrees/rcGAN/crps-review/mass_map_utils/scripts/review"
JT = "/home/jjwhit/crps_review/joint_tests"
EVAL = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/"
OLD_CKPT = "/share/gpu0/jjwhit/mass_map/mm_models/"
NEW_CKPT = "/share/gpu0/jjwhit/crps_review/paper/mm/ckpt/"
MODELS = {
    "R": OLD_CKPT + "mmgan_training_real_output",
    "F100": OLD_CKPT + "mmgan_crps_alpha_100",
    "MS": NEW_CKPT + "mmgan_crps_ms",
    "MSS": NEW_CKPT + "mmgan_crps_ms_sigma",
    "FNA": NEW_CKPT + "mmgan_crps_fair_noadv",
    "MSNA": NEW_CKPT + "mmgan_crps_ms_noadv",
}

sys.path[:0] = [WT, REVIEW, JT]
import models.lightning.mmGAN  # noqa: E402,F401  (paper-mm version, cached before common.py adds crps-review)
import common  # noqa: E402


def load_model(exp, ckpt="checkpoint_best.ckpt", device="cuda"):
    from models.lightning.mmGAN import mmGAN

    path = exp if os.path.isabs(exp) else MODELS[exp]
    model = mmGAN.load_from_checkpoint(checkpoint_path=os.path.join(path, ckpt), map_location="cpu")
    print(f"loaded {path}/{ckpt}: crps_mode={model.crps_mode} adv_scale={getattr(model, 'adv_scale', 1.0)}", flush=True)
    return model.to(device).eval()


common.MODELS.clear()
common.MODELS.update(MODELS)
common.load_model = load_model
common.OUT_ROOT = EVAL + "stageA/"
assert models.lightning.mmGAN.__file__.startswith(WT), models.lightning.mmGAN.__file__

which, script = sys.argv[1], sys.argv[2]
if which == "joint":
    import jt_common
    import torch

    # torch 1.13's cuFFT fails on the L40S node (CUFFT_INTERNAL_ERROR). There, run jt_common's FFTs on the CPU
    # (same float32/complex64 transforms, results moved back to the GPU); nothing else changes.
    if torch.cuda.is_available() and "L40S" in torch.cuda.get_device_name():
        def _on_cpu(f):
            return lambda x, *a, **k: f(x.cpu(), *a, **k).to(x.device)

        torch.fft.fft2, torch.fft.ifft2 = _on_cpu(torch.fft.fft2), _on_cpu(torch.fft.ifft2)
        print("L40S: torch.fft.fft2/ifft2 run on the CPU", flush=True)

    jt_common.MODELS, jt_common.load_model, jt_common.OUT = common.MODELS, load_model, EVAL + "joint/"
    script = os.path.join(JT, script)
else:
    script = os.path.join(REVIEW, script)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
