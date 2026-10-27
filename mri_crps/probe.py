"""Memory/speed probe: a few D+G steps of rcGAN-MRI at several batch sizes (random data, same shapes)."""
import gc
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lightning.rcGAN import rcGAN  # noqa: E402
from utils.mri.get_mask import get_mask  # noqa: E402

cfg = types.SimpleNamespace(in_chans=16, out_chans=16, im_size=384, num_z_train=2, gp_weight=10, lr=1e-3,
                            beta_1=0, beta_2=0.99, adv_epochs=[4, 22])
for spec in sys.argv[1:]:
    bs, amp = int(spec.split(':')[0]), spec.endswith(':amp')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        m = rcGAN(cfg, 'probe', 1, reg='ms').cuda()
        m.current_epoch_ = 0
        opt_d, opt_g = m.configure_optimizers()[0]
        y = torch.randn(bs, 16, 384, 384, device='cuda')
        x = torch.randn(bs, 16, 384, 384, device='cuda')
        mask = get_mask(384)[None].repeat(bs, 1, 1, 1, 1).cuda()
        batch = (y, x, mask, None, None, None, None, None)
        times = []
        for it in range(6):
            torch.cuda.synchronize()
            t = time.time()
            for idx, opt in [(0, opt_d), (1, opt_g)]:
                # as Lightning's toggle_optimizer: only the stepped model's parameters require grad
                for q in m.discriminator.parameters():
                    q.requires_grad_(idx == 0)
                for q in m.generator.parameters():
                    q.requires_grad_(idx == 1)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', enabled=amp):
                    loss = m.training_step(batch, 0, idx)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        print(f'bs={bs} amp={amp}: {sum(times[2:]) / 4:.3f} s/step, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB',
              flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f'bs={bs} amp={amp}: OOM', flush=True)
    m = opt_d = opt_g = loss = y = x = mask = batch = None
    gc.collect()
    torch.cuda.empty_cache()
