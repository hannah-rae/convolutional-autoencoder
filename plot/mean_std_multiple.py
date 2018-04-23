import matplotlib.pyplot as plt
import numpy as np
import math

# Wavelength (nm) of each filter
r_wl = np.array([447, 527, 805, 908, 937, 1013])
l_wl = np.array([445, 527, 676, 751, 867, 1012])

# arr_t = [
#          '/Users/hannahrae/src/autoencoder/MastcamCAE/results/reducedtrain_maxdiff/64517_sequence_id_9264_R1_sol1033_171/recon_mean_var_max.npy',
#          '/Users/hannahrae/src/autoencoder/MastcamCAE/results/reducedtrain_maxdiff/103286_sequence_id_3488_R1_sol0493_11/recon_mean_var_max.npy',
#          '/Users/hannahrae/src/autoencoder/MastcamCAE/results/reducedtrain_maxdiff/15612_sequence_id_17304_L1_sol1637_26/recon_mean_var_max.npy',
#          '/Users/hannahrae/src/autoencoder/MastcamCAE/results/reducedtrain_maxdiff/20721_sequence_id_10125_R1_sol1106_63/recon_mean_var_max.npy',
#          '/Users/hannahrae/src/autoencoder/MastcamCAE/results/reducedtrain_maxdiff/42035_sequence_id_9092_R1_sol0999_115/recon_mean_var_max.npy'
#         ]
arr_t = [
         '/Users/hannahrae/src/autoencoder/MastcamCAE/results/reducedtrain_maxdiff/15612_sequence_id_17304_L1_sol1637_26/recon_mean_var_max.npy'
        ]

# arr_a = [
#           '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_picks_maxdiff/212904_sequence_id_16106_R1_sol1552_0/recon_mean_var_max.npy',
#           '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_picks_maxdiff/124977_sequence_id_12960_R1_sol1326_0/recon_mean_var_max.npy',
#           '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_picks_maxdiff/112618_sequence_id_13423_R3_sol1359_0/recon_mean_var_max.npy',
#           '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_picks_maxdiff/128082_sequence_id_16788_R1_sol1608_0/recon_mean_var_max.npy',
#           '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_picks_maxdiff/67372_sequence_id_9241_R1_sol1032_0/recon_mean_var_max.npy'
#         ]
arr_a = [
          '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_picks_maxdiff/124977_sequence_id_12960_R1_sol1326_0/recon_mean_var_max.npy'
        ]

for arr in arr_t:
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

    plt.errorbar(x, mean, var, linestyle='None', marker='.', color='b')
    plt.plot(x, max, linestyle='None', marker='*', color='b', label='Typical')

for arr in arr_a:
    name = arr.split('/')[-2]
    eye = arr.split('/')[-2].split('_')[4][:1]

    data = np.load(arr)
    mean = data[:6]
    var = [math.sqrt(x) for x in data[6:12]]
    max = data[12:]

    x = []
    if eye == 'R':
        x = np.array([447+10, 527+10, 805+10, 908+10, 937+10, 1013+10])
        # Right filters in wavelength order are 2 1 3 4 5 6
        mean = np.concatenate([[mean[1]], [mean[0]], mean[2:]])
        var = np.concatenate([[var[1]], [var[0]], var[2:]])
        max = np.concatenate([[max[1]], [max[0]], max[2:]])
    elif eye == 'L':
        x = np.array([445+10, 527+10, 676+10, 751+10, 867+10, 1012+10])
        # Left filters in wavelength order are 2 1 4 3 5 6
        mean = np.concatenate([[mean[1]], [mean[0]], [mean[3]], [mean[2]], mean[4:]])
        var = np.concatenate([[var[1]], [var[0]], [var[3]], [var[2]], var[4:]])
        max = np.concatenate([[max[1]], [max[0]], [max[3]], [max[2]], max[4:]])
    # TODO add else raise error

    plt.errorbar(x, mean, var, linestyle='None', marker='.', color='r')
    plt.plot(x, max, linestyle='None', marker='*', color='r', label='Anomalous')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean, Standard Deviation, and Max Pixel Difference in Each Filter')
#plt.legend(loc='upper right')
plt.title('Typical (Blue) vs. Anomalous (Red) Image Reconstruction Error Statistics')

plt.show()