import matplotlib.pyplot as plt
import numpy as np
import math

# Wavelength (nm) of each filter
r_wl = np.array([447, 527, 805, 908, 937, 1013])
l_wl = np.array([445, 527, 676, 751, 867, 1012])

arr = '/Users/hannahrae/src/autoencoder/MastcamCAE/results/train_optimized_11/9153_sequence_id_17251_L1_sol1634_105/recon_mean_var_max.npy'
name = arr.split('/')[-2]
eye = arr.split('/')[-2].split('_')[4][:1]

data = np.load(arr)
mean = data[:6]
var = [math.sqrt(x) for x in data[6:12]]
max = data[12:]

x = []
if eye == 'R':
    x = r_wl
    # Right filters in wavelength order are 2 1 3 4 5 6
    mean = np.concatenate([[mean[1]], [mean[0]], mean[2:]])
    var = np.concatenate([[var[1]], [var[0]], var[2:]])
    max = np.concatenate([[max[1]], [max[0]], max[2:]])
elif eye == 'L':
    x = l_wl
    # Left filters in wavelength order are 2 1 4 3 5 6
    mean = np.concatenate([[mean[1]], [mean[0]], [mean[3]], [mean[2]], mean[4:]])
    var = np.concatenate([[var[1]], [var[0]], [var[3]], [var[2]], var[4:]])
    max = np.concatenate([[max[1]], [max[0]], [max[3]], [max[2]], max[4:]])
# TODO add else raise error

plt.errorbar(x, mean, var, linestyle='None', marker='.')
plt.plot(x, max, linestyle='None', marker='*', color='r')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean, Standard Deviation, and Max Pixel Difference in Each Filter')
plt.title(name)

plt.show()