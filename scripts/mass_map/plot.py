import torch
import yaml
import types
import json
import numpy as np
import matplotlib.patches as patches
import sys
sys.path.append('/home/jjwhit/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.MassMappingDataModule import MMDataTransform
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri.math import tensor_to_complex_np
from mass_map_utils.scripts.ks_utils import backward_model
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
import matplotlib.ticker as tkr

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MMDataModule(cfg)
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()


    with torch.no_grad():
        mmGAN_model = mmGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        mmGAN_model.cuda()

        mmGAN_model.eval()

        for i, data in enumerate(test_loader):
            y, x, mean, std = data
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens_mmGAN = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 2)).cuda()

            for z in range(cfg.num_z_test):
                gens_mmGAN[:, z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y))

            avg_mmGAN = torch.mean(gens_mmGAN, dim=1)

            gt = mmGAN_model.reformat(x)
            zfr = mmGAN_model.reformat(y)

            for j in range(y.size(0)):
                np_avgs = {
                    'mmGAN': None,
                }

                np_samps = {
                    'mmGAN': [],
                }

                np_stds = {
                    'mmGAN': None,
                }

                np_gt = None

                kappa_mean = cfg.kappa_mean
                kappa_std = cfg.kappa_std

                np_gt = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((gt[j] * kappa_std + kappa_mean).cpu())).numpy(), 180)
                np_zfr = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((zfr[j] * kappa_std + kappa_mean).cpu())).numpy(), 180)

                np_avgs['mmGAN'] = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((avg_mmGAN[j] * kappa_std + kappa_mean).cpu())).numpy(),
                    180)
                for z in range(cfg.num_z_test):
                    np_samps['mmGAN'].append(ndimage.rotate(torch.tensor(
                        tensor_to_complex_np((gens_mmGAN[j, z] * kappa_std + kappa_mean).cpu())).numpy(), 180))

                np_stds['mmGAN'] = np.std(np.stack(np_samps['mmGAN']), axis=0)

                method = 'mmGAN'
                zoom_length = 80  # Adjust this value based on your preference
                margin = 10  # Adjust this value to set the margin

                # Ensure the square is not touching the edge
                zoom_startx = np.random.randint(margin, cfg.im_size - zoom_length - margin)
                zoom_starty1 = np.random.randint(margin, int(cfg.im_size / 2) - zoom_length - margin)
                zoom_starty2 = np.random.randint(int(cfg.im_size / 2) + margin, cfg.im_size - zoom_length - margin)

                p = np.random.rand()
                zoom_starty = zoom_starty1 if p <= 0.5 else zoom_starty2

                x_coord = zoom_startx + zoom_length
                y_coords = [zoom_starty, zoom_starty + zoom_length]


                mask =  np.load(
                    cfg.cosmo_dir_path + 'cosmos_mask.npy', allow_pickle=True
                ).astype(bool)


                SMALL_SIZE = 8
                MEDIUM_SIZE = 10
                BIGGER_SIZE = 12

                plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
                plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
                plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
                plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


                #FIG 0.5: Gt, recon, error, std

                contours = find_contours(mask, 0.5)
                outer_contour = max(contours, key=lambda x: x.shape[0])


                fig, axes = plt.subplots(1,4)
                for axis in axes.flatten():
                    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
                vmin = np.min(np_gt.real)
                vmax = np.max(np_gt.real)
                
                im1 = axes[0].imshow(np_gt.real, cmap='inferno', vmin = vmin, vmax = vmax, origin='lower')
                axes[0].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[0].set_title('Ground Truth')
                
                im2 = axes[1].imshow(np_avgs[method].real, cmap='inferno', vmin = vmin, vmax = vmax, origin='lower')
                axes[1].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[1].set_title('Reconstruction')
                
                im3 = axes[2].imshow(np.abs(np_avgs[method]-np_gt),cmap='jet',vmin=np.min(np.abs(np_avgs[method]-np_gt)),
                                       vmax=np.max(np.abs(np_avgs['mmGAN']-np_gt)),origin='lower')
                axes[2].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[2].set_title('Absolute Error')
                
                im4 = axes[3].imshow(np_stds[method].real, cmap='viridis', vmin=np.min(np_stds[method].real), vmax = np.max(np_stds[method].real), origin='lower')
                axes[3].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[3].set_title('Standard Deviation')

                cbar1 = fig.colorbar(im1, ax=axes[0],format=tkr.FormatStrFormatter('%.2f'))
                cbar2 = fig.colorbar(im2, ax=axes[1],format=tkr.FormatStrFormatter('%.2f'))
                cbar3 = fig.colorbar(im3, ax=axes[2],format=tkr.FormatStrFormatter('%.2f'))
                cbar4 = fig.colorbar(im4, ax=axes[3],format=tkr.FormatStrFormatter('%.2f'))
                plt.subplots_adjust(wspace=0, hspace=.2)
                

                plt.savefig(f'/share/gpu0/jjwhit/plots/new/overview_long_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                #FIG 1: Gt, recon, error, std
                
                fig, axes = plt.subplots(2,2)
                for axis in axes.flatten():
                    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
                vmin = np.min(np_gt.real)
                vmax = np.max(np_gt.real)
                
                im1 = axes[0,0].imshow(np_gt.real, cmap='inferno', vmin = vmin, vmax = vmax, origin='lower')
                axes[0,0].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[0,0].set_title('Ground Truth')
                
                im2 = axes[0,1].imshow(np_avgs[method].real, cmap='inferno', vmin = vmin, vmax = vmax, origin='lower')
                axes[0,1].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[0,1].set_title('Reconstruction')
                
                im3 = axes[1,0].imshow(np.abs(np_avgs[method]-np_gt),cmap='jet',vmin=np.min(np.abs(np_avgs[method]-np_gt)),
                                       vmax=np.max(np.abs(np_avgs['mmGAN']-np_gt)),origin='lower')
                axes[1,0].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[1,0].set_title('Absolute Error')
                
                im4 = axes[1,1].imshow(np_stds[method].real, cmap='viridis', vmin=np.min(np_stds[method].real), vmax = np.max(np_stds[method].real), origin='lower')
                axes[1,1].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                axes[1,1].set_title('Standard Deviation')

                cbar1 = fig.colorbar(im1, ax=axes[0,0],format=tkr.FormatStrFormatter('%.2f'))
                cbar2 = fig.colorbar(im2, ax=axes[0,1],format=tkr.FormatStrFormatter('%.2f'))
                cbar3 = fig.colorbar(im3, ax=axes[1,0],format=tkr.FormatStrFormatter('%.2f'))
                cbar4 = fig.colorbar(im4, ax=axes[1,1],format=tkr.FormatStrFormatter('%.2f'))
                plt.subplots_adjust(wspace=0, hspace=.2)
                

                plt.savefig(f'/share/gpu0/jjwhit/plots/new/overview_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                # Plot 2: Truth; zoomed truth, sample, reconstruction, error and std dev.
                nrow = 4
                ncol = 2

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                    wspace=0.25, hspace=0.25,
                                    top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                    left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt.real, cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax
                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                        edgecolor='r',
                                        facecolor='none')

                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                        cmap='inferno',
                        vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Zoomed Truth')

                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                for samp in range(1):
                    ax = plt.subplot(gs[1, samp])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                              zoom_startx:zoom_startx + zoom_length].real, cmap='inferno', 
                              vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}')

                ax = plt.subplot(gs[1, 1])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Reconstruction')

                ax = plt.subplot(gs[2, 0])
                ax.imshow(np.abs(np_avgs[method][zoom_starty:zoom_starty + zoom_length,    
                          zoom_startx:zoom_startx + zoom_length] - np_gt[zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length]), cmap='jet', vmin=np.min(np.abs(np_avgs['mmGAN'] - np_gt)),
                               vmax=np.max(np.abs(np_avgs['mmGAN'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Absolute Error")

                ax = plt.subplot(gs[2, 1])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length].real, cmap='viridis', vmin=np.min(np_stds['mmGAN'].real),
                          vmax=np.max(np_stds['mmGAN'].real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.')

                plt.savefig(f'/share/gpu0/jjwhit/plots/new/zoomed_overview{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)




                #Plot 3: truth; zoomed truth, reconstruction, 8-, 4-, 2-avg, sample, std. dev.
                nrow = 4
                ncol = 2
                
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                
                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.3,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                
                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt.real, cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                plt.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')
                
                ax1 = ax
                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')
                
                ax.add_patch(rect)
                ax = plt.subplot(gs[0, 1])
                im = ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                          cmap='inferno',
                          vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')
                
                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)
                
                ax = plt.subplot(gs[1, 0])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Reconstruction')
                
                ax = plt.subplot(gs[1, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(8):
                    avg += np_samps[method][l]
                avg = avg / 8
                
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([]) 
                ax.set_title('8-Avg.')
                
                
                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(4):
                    avg += np_samps[method][l]
                
                avg = avg / 4
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.')
                
                
                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(2):
                    avg += np_samps[method][l]
                
                avg = avg / 2
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.')
                
                for samp in range(1):
                    ax = plt.subplot(gs[3, 0])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                                zoom_startx:zoom_startx + zoom_length].real, cmap='inferno', 
                                vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('Sample')
                
                ax = plt.subplot(gs[3, 1])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length].real, cmap='viridis', vmin=np.min(np_stds['mmGAN'].real),
                          vmax=np.max(np_stds['mmGAN'].real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.') 

                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Adjust the position and size as needed
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.8,format=tkr.FormatStrFormatter('%.2f'))

                plt.savefig(f'/share/gpu0/jjwhit/plots/new/zoomed_aes_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                #Plot 4: Zoomed diversity.
                nrow = 3
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                
                ax = plt.subplot(gs[0, 0])
                im = ax.imshow(np_gt.real, cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                plt.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax
                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')

                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                          cmap='inferno',
                          vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                for samp in range(2):
                    ax = plt.subplot(gs[1, samp])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                            zoom_startx:zoom_startx + zoom_length].real, cmap='inferno',
                             vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}')
                
                for samp in range(2):
                    ax = plt.subplot(gs[2, samp])
                    ax.imshow(np_samps[method][samp+2][zoom_starty:zoom_starty + zoom_length,
                            zoom_startx:zoom_startx + zoom_length].real, cmap='inferno',
                             vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 3}')
                
                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Adjust the position and size as needed
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.8,format=tkr.FormatStrFormatter('%.2f'))

                

                plt.savefig(f'/share/gpu0/jjwhit/plots/new/diversity_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                # #Plot 5: zoomed P-ascent.
                nrow = 3
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                ax = plt.subplot(gs[0, 0])
                im = ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                          cmap='inferno',
                          vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax
                ax = plt.subplot(gs[0, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(2):
                    avg += np_samps[method][l]

                avg = avg / 2
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.')


                ax = plt.subplot(gs[1, 0])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(4):
                    avg += np_samps[method][l]

                avg = avg / 4
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.')


                ax = plt.subplot(gs[1, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(8):
                    avg += np_samps[method][l]

                avg = avg / 8
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('8-Avg.')


                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(16):
                    avg += np_samps[method][l]

                avg = avg / 16
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('16-Avg.')

                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(32):
                    avg += np_samps[method][l]

                avg = avg / 32
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=np.min(np_gt.real), vmax=np.max(np_gt.real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('32-Avg.')

                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Adjust the position and size as needed
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.8,format=tkr.FormatStrFormatter('%.2f'))


                plt.savefig(f'/share/gpu0/jjwhit/plots/new/zoom_P_ascent_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                #Plot 6: Kaiser Squires comparison

                std1 = np.load(
                    cfg.cosmo_dir_path + 'cosmos_std1.npy', allow_pickle=True
                )
                std2 = np.load(
                    cfg.cosmo_dir_path + 'cosmos_std2.npy', allow_pickle=True
                )
                kernel = MMDataTransform.compute_fourier_kernel(cfg.im_size)
                gamma_sim = MMDataTransform.forward_model(np_gt, kernel) + (
                            std1 * np.random.randn(cfg.im_size, cfg.im_size) + 1.j * std2 * np.random.randn(cfg.im_size, cfg.im_size)
                        )

                nrow = 1
                ncol = 4
                fig, axes = plt.subplots(nrow, ncol, figsize=(12,3), constrained_layout=True)

                #TODO: Can loop for tick labels here.

                vmin = np.min(np_gt.real)
                vmax = np.max(np_gt.real)

                im1 = axes[0].imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
                axes[0].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=0.75)
                axes[0].set_title('Truth')
                axes[0].set_xticklabels([])
                axes[0].set_yticklabels([])
                axes[0].set_xticks([])
                axes[0].set_yticks([])

                im2 = axes[1].imshow(np_avgs[method].real, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
                axes[1].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=0.75)
                axes[1].set_title('cGAN')
                axes[1].set_xticklabels([])
                axes[1].set_yticklabels([])
                axes[1].set_xticks([])
                axes[1].set_yticks([])

                backward = backward_model(gamma_sim, kernel)
                im3 = axes[2].imshow(backward.real, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower') #Previously kappa_sim[0]
                axes[2].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=1)
                axes[2].set_title('Kaiser-Squires')
                axes[2].set_xticklabels([])
                axes[2].set_yticklabels([])
                axes[2].set_xticks([])
                axes[2].set_yticks([])

                ks = ndimage.gaussian_filter(backward, sigma=1/.29)

                im4 = axes[3].imshow(ks.real, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
                axes[3].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=1)
                axes[3].set_title('Smoothed Kaiser-Squires')
                axes[3].set_xticklabels([])
                axes[3].set_yticklabels([])
                axes[3].set_xticks([])
                axes[3].set_yticks([])

                cbar4 = fig.colorbar(im4, ax=axes[3], orientation='vertical', pad=0.02,format=tkr.FormatStrFormatter('%.2f'))
                cbar4.mappable.set_clim(vmin, vmax)

                plt.savefig(f'/share/gpu0/jjwhit/plots/new/ks_comp_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                #Plot 7: P-ascent.
                nrow = 3
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                ax = plt.subplot(gs[0, 0])
                im = ax.imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax

                ax = plt.subplot(gs[0, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(2):
                    avg += np_samps[method][l]

                avg = avg / 2
                ax.imshow(avg.real,cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.')


                ax = plt.subplot(gs[1, 0])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(4):
                    avg += np_samps[method][l]

                avg = avg / 4
                ax.imshow(avg.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.')


                ax = plt.subplot(gs[1, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(8):
                    avg += np_samps[method][l]

                avg = avg / 8
                ax.imshow(avg.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('8-Avg.')


                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(16):
                    avg += np_samps[method][l]

                avg = avg / 16
                ax.imshow(avg.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('16-Avg.')

                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size), dtype=np.complex128)
                for l in range(32):
                    avg += np_samps[method][l]

                avg = avg / 32
                ax.imshow(avg.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('32-Avg.')

                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Adjust the position and size as needed
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.8,format=tkr.FormatStrFormatter('%.2f'))


                plt.savefig(f'/share/gpu0/jjwhit/plots/new/P_ascent_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                # #Plot 8: Posterior samples.
                nrow = 2
                ncol = 3
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                ax = plt.subplot(gs[0, 0])
                im = ax.imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax)
                #ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax

                for samp in range(2):
                    ax = plt.subplot(gs[0, samp+1])
                    ax.imshow(np_samps[method][samp].real, cmap='inferno', 
                              vmin=vmin, vmax=vmax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}')

                for samp in range(3):
                    ax = plt.subplot(gs[1, samp])
                    ax.imshow(np_samps[method][samp+3].real, cmap='inferno', 
                              vmin=vmin, vmax=vmax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 3}')


                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Adjust the position and size as needed
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.8,format=tkr.FormatStrFormatter('%.2f'))


                plt.savefig(f'/share/gpu0/jjwhit/plots/new/samples_new_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                if fig_count == args.num_figs:
                    sys.exit()
                fig_count += 1
