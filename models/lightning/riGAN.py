#Mass Mapping

import torch

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
from matplotlib import cm

from PIL import Image
from torch.nn import functional as F
from utils.mri.fftc import ifft2c_new, fft2c_new #TODO: Unused imports.
from models.archs.radio.generator import UNetModel
from models.archs.radio.discriminator import DiscriminatorModel
from evaluation_scripts.metrics import psnr
from torchmetrics.functional import peak_signal_noise_ratio

class riGAN(pl.LightningModule):
    def __init__(self, args, exp_name, num_gpus):
        super().__init__()
        self.args = args # This is the cfg object 
        self.exp_name = exp_name
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans + 2  # Two extra dimensions of the added noise 
        self.out_chans = args.out_chans

        try:
            alt_upsample = self.args.alt_upsample
        except:
            alt_upsample = False
            
        self.generator = UNetModel(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            alt_upsample=alt_upsample
        )

        self.discriminator = DiscriminatorModel(
            in_chans=self.args.in_chans + self.args.out_chans, # Number of channels from x and y
            out_chans=self.out_chans,
            input_im_size=self.args.im_size
        )

        self.std_mult = 1
        self.is_good_model = 0
        self.resolution = self.args.im_size

        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors):
        z = torch.randn(num_vectors, 2, self.resolution, self.resolution, device=self.device)
        return z

    def reformat(self, samples):
        reformatted_tensor = torch.swapaxes(torch.clone(samples), 3, 1)
        return reformatted_tensor

    def readd_measures(self, samples, measures):
        return torch.clone(samples)

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, y=y)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(
            self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, y):
        num_vectors = y.size(0)
        noise = self.get_noise(num_vectors)
        samples = self.generator(torch.cat([y, noise], dim=1))
        samples = self.readd_measures(samples, y)     
        return samples

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, y, gens):
        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train), device=self.device)
        for k in range(y.shape[0]):
            cond = torch.zeros(
                1,
                self.args.in_chans,
                self.args.im_size,
                self.args.im_size,
                device=self.device
            )
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
            temp = self.discriminator(input=gens[k], y=cond)
            fake_pred[k] = temp[:, 0]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        adv_weight = 1e-5
        if self.current_epoch <= 4:
            adv_weight = 1e-2
        elif self.current_epoch <= 22:
            adv_weight = 1e-4

        return - adv_weight * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))
            ) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mean, std = batch

        # train generator
        if optimizer_idx == 1:
            gens = torch.zeros(
                size=(
                    y.size(0),
                    self.args.num_z_train,
                    self.args.out_chans,
                    self.args.im_size, 
                    self.args.im_size
                ),
                device=self.device)
            for z in range(self.args.num_z_train):
                gens[:, z, :, :, :] = self.forward(y)

            avg_recon = torch.mean(gens, dim=1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(y, gens)
            g_loss += self.l1_std_p(avg_recon, gens, x)

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 0:
            x_hat = self.forward(y)

            real_pred = self.discriminator(input=x, y=y)
            fake_pred = self.discriminator(input=x_hat, y=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        y, x, mean, std= batch

        fig_count = 0

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), num_code, self.args.out_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y) * std[:, None, None, None] + mean[:, None, None, None] 

        avg = torch.mean(gens, dim=1)
        avg_gen = self.reformat(avg)
        gt = self.reformat(x * std[:, None, None, None] + mean[:, None, None, None])

        mag_avg_list = []
        mag_single_list = []
        mag_gt_list = []
        psnr_8s = []
        psnr_1s = []

        for j in range(y.size(0)):
            psnr_8s.append(peak_signal_noise_ratio(avg_gen[j], gt[j]))
            psnr_1s.append(peak_signal_noise_ratio(self.reformat(gens[:, 0])[j], gt[j]))

            mag_avg_list.append(avg_gen[None, j, :, :, :])
            mag_single_list.append(self.reformat(gens[:, 0])[j])
            mag_gt_list.append(gt[None, j, :, :, :])

        psnr_8s = torch.stack(psnr_8s)
        psnr_1s = torch.stack(psnr_1s)
        mag_avg_gen = torch.cat(mag_avg_list, dim=0)
        mag_single_gen = torch.cat(mag_single_list, dim=0)
        mag_gt = torch.cat(mag_gt_list, dim=0)

        self.log('psnr_8_step', psnr_8s.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log('psnr_1_step', psnr_1s.mean(), on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 1 == 0 and fig_count == 0:
                fig_count += 1
                # Using single generation instead of avg generator (mag_avg_gen)
                avg_gen_np = mag_avg_gen[0, :, :, 0].cpu().numpy()
                gt_np = mag_gt[0, :, :, 0].cpu().numpy()

                plot_avg_np = (avg_gen_np - np.min(avg_gen_np)) / (np.max(avg_gen_np) - np.min(avg_gen_np))
                plot_gt_np = (gt_np - np.min(gt_np)) / (np.max(gt_np) - np.min(gt_np))

                np_psnr = psnr(gt_np, avg_gen_np)

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[
                        Image.fromarray(np.uint8(plot_gt_np*255), 'L'),
                        Image.fromarray(np.uint8(plot_avg_np*255), 'L'),
                        Image.fromarray(np.uint8(cm.jet(5*np.abs(plot_gt_np - plot_avg_np))*255))
                    ],
                    caption=["GT", f"Recon: PSNR (NP): {np_psnr:.2f}", "Error"]
                )

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8s.mean(), 'psnr_1': psnr_1s.mean()}

    def validation_epoch_end(self, validation_step_outputs):
        avg_psnr = self.all_gather(torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()).mean()
        avg_single_psnr = self.all_gather(torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()).mean()

        avg_psnr = avg_psnr.cpu().numpy()
        avg_single_psnr = avg_single_psnr.cpu().numpy()

        psnr_diff = (avg_single_psnr + 2.5) - avg_psnr

        mu_0 = 2e-2
        self.std_mult += mu_0 * psnr_diff

        if np.abs(psnr_diff) <= self.args.psnr_gain_tol:
            self.is_good_model = 1
        else:
            self.is_good_model = 0

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta_1, self.args.beta_2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta_1, self.args.beta_2)
        )
        return [opt_d, opt_g], []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
