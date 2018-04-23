import matplotlib.pyplot as plt
import numpy as np
import math

# Wavelength (nm) of each filter
r_wl = np.array([447, 527, 805, 908, 937, 1013])
l_wl = np.array([445, 527, 676, 751, 867, 1012])

x_stats = '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_optimized_12/2309_sequence_id_15465_R1_sol1505_0/input_mean_var_max.npy'
y_stats = '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_optimized_12/2309_sequence_id_15465_R1_sol1505_0/recon_mean_var_max.npy'

name = x_stats.split('/')[-2]
eye = x_stats.split('/')[-2].split('_')[4][:1]

data_x = np.load(x_stats)
mean_x = data_x[:6]
var_x = [math.sqrt(x) for x in data_x[6:12]]
max_x = data_x[12:]

data_y = np.load(y_stats)
mean_y = data_y[:6]
var_y = [math.sqrt(x) for x in data_y[6:12]]
max_y = data_y[12:]

x = []
if eye == 'R':
    x = r_wl
    # Right filters in wavelength order are 2 1 3 4 5 6
    mean_x = np.concatenate([[mean_x[1]], [mean_x[0]], mean_x[2:]])
    var_x = np.concatenate([[var_x[1]], [var_x[0]], var_x[2:]])
    max_x = np.concatenate([[max_x[1]], [max_x[0]], max_x[2:]])

    mean_y = np.concatenate([[mean_y[1]], [mean_y[0]], mean_y[2:]])
    var_y = np.concatenate([[var_y[1]], [var_y[0]], var_y[2:]])
    max_y = np.concatenate([[max_y[1]], [max_y[0]], max_y[2:]])
elif eye == 'L':
    x = l_wl
    # Left filters in wavelength order are 2 1 4 3 5 6
    mean_x = np.concatenate([[mean_x[1]], [mean_x[0]], [mean_x[3]], [mean_x[2]], mean_x[4:]])
    var_x = np.concatenate([[var_x[1]], [var_x[0]], [var_x[3]], [var_x[2]], var_x[4:]])
    max_x = np.concatenate([[max_x[1]], [max_x[0]], [max_x[3]], [max_x[2]], max_x[4:]])

    mean_y = np.concatenate([[mean_y[1]], [mean_y[0]], [mean_y[3]], [mean_y[2]], mean_y[4:]])
    var_y = np.concatenate([[var_y[1]], [var_y[0]], [var_y[3]], [var_y[2]], var_y[4:]])
    max_y = np.concatenate([[max_y[1]], [max_y[0]], [max_y[3]], [max_y[2]], max_y[4:]])
# TODO add else raise error

plt.errorbar(x, mean_x, var_x, linestyle='None', marker='.', color='r', label='Actual image mean/stddev')
plt.plot(x, max_x, linestyle='None', marker='*', color='r', label='Actual image max')
plt.errorbar(x, mean_y, var_y, linestyle='None', marker='.', color='b', label='Expected image mean/stddev')
plt.plot(x, max_y, linestyle='None', marker='*', color='b', label='Expected image max')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean, Standard Deviation, and Max Pixel Values in Each Filter')
plt.title('Egg rock: sequence mcam07642, sol 1505')
plt.legend(loc='upper left')

plt.show()