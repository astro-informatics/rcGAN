
import numpy as np
import os
import matplotlib.pyplot as plt

'''This script Builds a COSMOS-like dataset of weak lensing maps from kappaTNG weak lensing suites.'''

# Configures redshift distribution.
redshift_distribution = np.load("/home/jjwhit/rcGAN/mass_map_utils/cosmos/hist_n_z.npy", allow_pickle=True)

redshift_vals_array = np.array((0, 
                                0.034, 0.070, 0.105, 0.142, 0.179,
                                0.216, 0.255, 0.294, 0.335, 0.376,
                                0.418, 0.462, 0.506, 0.552, 0.599,
                                0.648, 0.698, 0.749, 0.803, 0.858,
                                0.914, 0.973, 1.034, 1.097, 1.163,
                                1.231, 1.302, 1.375, 1.452, 1.532,
                                1.615, 1.703, 1.794, 1.889, 1.989,
                                2.094, 2.203, 2.319, 2.440, 2.568,
                                2.605, 2.642, 2.679,
                                2.716, 2.755, 2.794, 2.835, 2.876,
                                2.918, 2.962, 3.006, 3.052, 3.099,
                                3.148, 3.198, 3.249, 3.303, 3.358,
                                3.414, 3.473, 3.534, 3.597, 3.663,
                                3.731, 3.802, 3.875, 3.952, 4.032,
                                4.115, 4.203, 4.294, 4.389, 4.489,
                                4.594, 4.703, 4.819, 4.940, 5.068)
                                )

z = np.arange(0, 5.5, .01)

bin_edges = (np.concatenate([[0],
            [(redshift_vals_array[i] - redshift_vals_array[i-1])/2 + 
            redshift_vals_array[i-1] for i in range(1, len(redshift_vals_array))],
            [redshift_vals_array[-1] + (redshift_vals_array[-1] - redshift_vals_array[-2])/2]]))
# Create a histogram using your specified bin edges
hist, bin_edges, _ = (plt.hist(z, weights=redshift_distribution, bins=bin_edges, color="purple", 
                        alpha=0.5, ec="k", histtype='stepfilled', density=True, label='Normalized'))
plt.legend()
plt.savefig('Redshift Distribution Histogram')
pdf = hist


# Configures kappaTNG files. 

np.random.seed(0)
LPs = np.arange(1, 101)
runs = np.arange(1, 101)
mesh1, mesh2 = np.meshgrid(LPs, runs)
flat_mesh1 = mesh1.flatten()
flat_mesh2 = mesh2.flatten()
np.random.shuffle(flat_mesh1)
np.random.shuffle(flat_mesh2)


dst_dir = "/share/gpu0/jjwhit/kappa_cosmos_simulations/"
if not os.path.exists(dst_dir):
   # Create a new directory because it does not exist
   os.makedirs(dst_dir)
   print("The new directory has been made!")

dst_train_path = dst_dir + 'kappa_train/'
dst_test_path = dst_dir + 'kappa_test/'
dst_val_path = dst_dir + 'kappa_val/'

if not os.path.exists(dst_train_path):
    os.makedirs(dst_train_path)
if not os.path.exists(dst_test_path):
    os.makedirs(dst_test_path)
if not os.path.exists(dst_val_path):
    os.makedirs(dst_val_path)


n = 1
for it in range(len(flat_mesh1)):
    path_to_load = '/share/gpu0/jjwhit/kappaTNG_suites/LP{:03d}/run{:03d}/'.format(flat_mesh1[it], flat_mesh2[it])
    
    # Reset slice, kappa_tot, and omega for each different map.
    slice = 0
    kappa_tot = np.zeros((1024, 1024))
    omega = np.zeros(len(redshift_vals_array))
    for fname in range(len(redshift_vals_array)):
        if fname in range(0, 41):
            full_path = os.path.join(path_to_load, 'kappa' + str(fname).zfill(2) + '.dat')
        else:
            # Uses z=2.6 map (with appropriate weight) for redshifts beyond kappaTNG range.
            full_path = os.path.join(path_to_load, 'kappa40.dat')

        if not os.path.exists(full_path):
            print(f'The file at {full_path} does not exist.')
        else:
            with open(full_path, 'rb') as f:
                # Load file
                #print(f"loading z = {redshift_vals_array[fname]}...")
                dummy = np.fromfile(f, dtype="int32", count=1)
                kappa = np.fromfile(f, dtype="float", count=1024*1024)
                dummy = np.fromfile(f, dtype="int32", count=1)  
    
                kappa = kappa.reshape((1024, 1024))
            # Bin size halved for first and last bins.
            # delta_z is the range of each bin.
            if slice == 0:
                delta_z = (redshift_vals_array[slice + 1] - redshift_vals_array[slice])/2     

            elif slice == len(redshift_vals_array) - 1:
                delta_z = (redshift_vals_array[slice] - redshift_vals_array[slice - 1])/2

            else:
                delta_z = (
                    (redshift_vals_array[slice + 1] - redshift_vals_array[slice]) /2
                    - (redshift_vals_array[slice] - redshift_vals_array[slice - 1]) /2
                )

            omega[slice] = pdf[slice] * delta_z
            norm_factor = np.sum(omega)
            kappa_tot += (omega[slice]/norm_factor) * kappa
            slice +=1

    if (it/len(flat_mesh1)) <= 0.85:
        dst_dir = dst_train_path
    elif (it/len(flat_mesh1)) > 0.85 and (it/len(flat_mesh1)) <=0.95:
        dst_dir = dst_test_path
    else:
        dst_dir = dst_val_path

    save_path = '{:s}{:s}{:05d}{:s}'.format(dst_dir, "sim_", n, ".npy")
    np.save(save_path, kappa_tot, allow_pickle=True)
    n +=1