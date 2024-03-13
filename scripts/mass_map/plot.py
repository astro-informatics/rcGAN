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

                # np_gt = ndimage.rotate(
                #     torch.tensor(tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).numpy(), 180)
                # np_zfr = ndimage.rotate(
                #     torch.tensor(tensor_to_complex_np((zfr[j] * std[j] + mean[j]).cpu())).numpy(), 180)
                # np_avgs['mmGAN'] = ndimage.rotate(
                #     torch.tensor(tensor_to_complex_np((avg_mmGAN[j] * std[j] + mean[j]).cpu())).numpy(),
                #     180)

                # for z in range(cfg.num_z_test):
                #     np_samps['mmGAN'].append(ndimage.rotate(torch.tensor(
                #         tensor_to_complex_np((gens_mmGAN[j, z] * std[j] + mean[j]).cpu())).numpy(), 180))

                np_gt = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((gt[j]).cpu())).numpy(), 180)
                np_zfr = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((zfr[j]).cpu())).numpy(), 180) #Removing std and mean contributions.
                
                np_avgs['mmGAN'] = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((avg_mmGAN[j]).cpu())).numpy(),
                    180)

                for z in range(cfg.num_z_test):
                    np_samps['mmGAN'].append(ndimage.rotate(torch.tensor(
                        tensor_to_complex_np((gens_mmGAN[j, z]).cpu())).numpy(), 180))


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


                #Fig 1: Global recon, error, std
                contours = find_contours(mask, 0.5)
                outer_contour = max(contours, key=lambda x: x.shape[0])
                

                fig, ax = plt.subplots(2,2, figsize=(8,8), constrained_layout=True)
                for axis in ax.flatten():
                    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                im1 = ax[0,0].imshow(np_gt.real, cmap='inferno', 
                                    vmin = np.min(np_gt.real), vmax = np.max(np_gt.real), origin='lower')
                ax[0,0].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                im2 = ax[0,1].imshow(np_avgs[method].real, cmap='inferno', 
                                    vmin = np.min(np_gt.real), vmax = np.max(np_gt.real), origin='lower')
                ax[0,1].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                im3 = ax[1,0].imshow((np_avgs[method]-np_gt).real,cmap='jet',
                                    vmin=np.min(np.abs(np_avgs[method]-np_gt)), vmax=np.max(np.abs(np_avgs[method]-np_gt)),origin='lower')
                ax[1,0].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                im4 = ax[1,1].imshow(np_stds[method].real, cmap='viridis', 
                                    vmin=np.min(np_stds[method].real), vmax = np.max(np_stds[method].real), origin='lower')
                ax[1,1].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.75)
                cbar1 = fig.colorbar(im1, ax=ax[0,0], shrink=0.8)
                cbar2 = fig.colorbar(im2, ax=ax[0,1], shrink=0.8)
                cbar3 = fig.colorbar(im3, ax=ax[1,0], shrink=0.8)
                cbar4 = fig.colorbar(im4, ax=ax[1,1], shrink=0.8)
                ax[0,0].set_title('Truth')
                ax[0,1].set_title('Reconstruction')
                ax[1,0].set_title('Absolute Error')
                ax[1,1].set_title('Std. Dev.')
                plt.subplots_adjust(wspace=0, hspace=.2)

                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/overview_new_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                # Plot 2: Truth; zoomed truth, sample, reconstruction, error and std dev.
                nrow = 4
                ncol = 2

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                    wspace=0.25, hspace=0.25,
                                    top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                    left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                
                vmin = np.min(np_gt.real)
                vmax=np.max(np_gt.real)

                ax = plt.subplot(gs[0, 0])
                im1 = ax.imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', labelsize=MEDIUM_SIZE)

                cbar_ax = fig.add_subplot(gs[2, :])
                fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

                ax1 = ax

                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                        edgecolor='r',
                                        facecolor='none')

                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                        cmap='inferno',
                        vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Zoomed Truth', labelsize=MEDIUM_SIZE)

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
                              vmin=vmin, vmax=vmax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}', labelsize=MEDIUM_SIZE)

                ax = plt.subplot(gs[1, 1])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Reconstruction', labelsize=MEDIUM_SIZE)

                ax = plt.subplot(gs[2, 0])
                ax.imshow((np_avgs[method][zoom_starty:zoom_starty + zoom_length,    
                          zoom_startx:zoom_startx + zoom_length] - np_gt[zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length]).real, cmap='jet', vmin=np.min(np.abs(np_avgs[method] - np_gt)),
                               vmax=np.max(np.abs(np_avgs[method] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Absolute Error", labelsize=MEDIUM_SIZE)

                ax = plt.subplot(gs[2, 1])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length].real, cmap='viridis', vmin=np.min(np_stds[method].real),
                          vmax=np.max(np_stds[method].real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.', labelsize=MEDIUM_SIZE)

                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/zoomed_overview_new_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)




                #Plot 3: truth; zoomed truth, reconstruction, 8-, 4-, 2-avg, sample, std. dev.
                nrow = 4
                ncol = 2
                
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                
                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                
                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax)
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
                          vmin=vmin, vmax=vmax)
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
                    cmap='inferno', vmin=vmin, vmax=vmax)
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
                    cmap='inferno', vmin=vmin, vmax=vmax)
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
                    cmap='inferno', vmin=vmin, vmax=vmax)
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
                    cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.')
                
                for samp in range(1):
                    ax = plt.subplot(gs[3, 0])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                                zoom_startx:zoom_startx + zoom_length].real, cmap='inferno', 
                                vmin=vmin, vmax=vmax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('Sample')
                
                ax = plt.subplot(gs[3, 1])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length].real, cmap='viridis', vmin=np.min(np_stds[method].real),
                          vmax=np.max(np_stds[method].real))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.')                
                
                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/zoomed_aes_new_{fig_count}.png', bbox_inches='tight', dpi=300)
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
                ax.imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax)
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
                          vmin=vmin, vmax=vmax)
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
                            zoom_startx:zoom_startx + zoom_length].real, cmap='inferno', vmin=vmin, vmax=vmax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}')
                
                for samp in range(2):
                    ax = plt.subplot(gs[2, samp])
                    ax.imshow(np_samps[method][samp+2][zoom_starty:zoom_starty + zoom_length,
                            zoom_startx:zoom_startx + zoom_length].real, cmap='inferno', vmin=vmin, vmax=vmax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 3}')
                

                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/diversity_new_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                # #Plot 5: zoomed P-ascent.
                nrow = 4
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                            cmap='inferno',vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax

                ax = plt.subplot(gs[0, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(2):
                    avg += np_samps[method][l]

                avg = avg / 2
                ax.imshow(avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                          cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.')


                ax = plt.subplot(gs[1, 0])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(4):
                    avg += np_samps[method][l]

                avg = avg / 4
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.')


                ax = plt.subplot(gs[1, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(8):
                    avg += np_samps[method][l]

                avg = avg / 8
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('8-Avg.')


                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(16):
                    avg += np_samps[method][l]

                avg = avg / 16
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('16-Avg.')

                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
                for l in range(32):
                    avg += np_samps[method][l]

                avg = avg / 32
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length].real,
                    cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('32-Avg.')

                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/zoom_P_ascent_new_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                #Plot 6: Kaiser Squires comparison


                std1 = np.load(
                    cfg.cosmo_dir_path + 'cosmos_std1.npy', allow_pickle=True
                )
                std2 = np.load(
                    cfg.cosmo_dir_path + 'cosmos_std2.npy', allow_pickle=True
                )
                D = MMDataTransform.compute_fourier_kernel(cfg.im_size)
                gamma_sim = MMDataTransform.forward_model(np_gt, D) + (
                            std1 * np.random.randn(cfg.im_size, cfg.im_size) + 1.j * std2 * np.random.randn(cfg.im_size, cfg.im_size)
                        )

                nrow = 1
                ncol = 4
                fig, axes = plt.subplots(nrow, ncol, figsize=(12,3), constrained_layout=True)

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

                kappa_sim = backward_model(gamma_sim, D)
                im3 = axes[2].imshow(kappa_sim.real, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower') #Previously kappa_sim[0]
                axes[2].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=1)
                axes[2].set_title('Kaiser-Squires')
                axes[2].set_xticklabels([])
                axes[2].set_yticklabels([])
                axes[2].set_xticks([])
                axes[2].set_yticks([])

                ks = ndimage.gaussian_filter(kappa_sim.real, sigma=1./29) #Previously kappa_sim[0]

                im4 = axes[3].imshow(ks, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
                axes[3].plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=1)
                axes[3].set_title('Smoothed Kaiser-Squires')
                axes[3].set_xticklabels([])
                axes[3].set_yticklabels([])
                axes[3].set_xticks([])
                axes[3].set_yticks([])

                # cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8, orientation='vertical', pad=0.02)
                # cbar1.mappable.set_clim(vmin, vmax)
                # cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8, orientation='vertical', pad=0.02)
                # cbar2.mappable.set_clim(vmin, vmax)
                # cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8, orientation='vertical', pad=0.02)
                # cbar3.mappable.set_clim(vmin, vmax)
                cbar4 = fig.colorbar(im4, ax=axes[3], orientation='vertical', pad=0.02)
                cbar4.mappable.set_clim(vmin, vmax)

                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/ks_comp_new_n_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                #Plot 7: P-ascent.
                nrow = 4
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt.real, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.plot(outer_contour[:, 1], outer_contour[:, 0], color='white', linewidth=.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax

                ax = plt.subplot(gs[0, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
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
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
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
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
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
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
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
                avg = np.zeros((cfg.im_size, cfg.im_size), dtype=np.complex128)
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

                plt.savefig(f'/share/gpu0/jjwhit/plots/no_norm/P_ascent_new_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)



                if fig_count == args.num_figs:
                    sys.exit()
                fig_count += 1
