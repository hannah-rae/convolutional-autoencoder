# import matplotlib
# matplotlib.use('Agg')

from glob import glob
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np

# Get the list of images where filename contains reconstruction error
#results = glob('/Users/hannahrae/src/autoencoder/MastcamCAE/results/test_set_reducedtrain/*')

results_dw = glob('/scratch/hannah/MastcamCAE/results/DW_optimized_12/*')
results_train = glob('/scratch/hannah/MastcamCAE/results/train_optimized_12/*')
# results_dw = glob('/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_optimized_1/*')
# results_train = glob('/Users/hannahrae/src/autoencoder/MastcamCAE/results/train_optimized_1/*')

# Convert filenames to a list of integers and scale by mean and stddev
#results = [int(a.split('/')[-1].split('_')[0]) for a in results]
results_dw = [int(a.split('/')[-1].split('_')[0]) for a in results_dw]
#results_dw_drillholes = [int(a.split('/')[-1].split('_')[0]) for a in results_dw_drillholes]
# results_dw_mean = np.mean(results_dw)
# results_dw_std = np.std(results_dw)
# results_dw = [(x-results_dw_mean)/results_dw_std for x in results_dw]

results_train = [int(a.split('/')[-1].split('_')[0]) for a in results_train]
# results_train_mean = np.mean(results_train)
# results_train_std = np.std(results_train)
# results_train = [(x-results_train_mean)/results_train_std for x in results_train]

# Compute the KL-divergence between the two sequences (normalized automatically)
# kl = scipy.stats.entropy(results_dw, results_train)
# print 'KL Divergence between train and expert sets: %f' % kl

# Plot a histogram of reconstruction error for entire test dataset
plt.hist(results_train, bins=300, normed=True, alpha=0.7, label='Training set')
#plt.hist(results, bins=len(results), normed=True, alpha=0.7, label='Test set')
plt.hist(results_dw, bins=300, normed=True, alpha=0.7, label='Expert picks')
# plt.hist(results_dw_drillholes, bins=300, normed=True, alpha=0.7, label='Expert picks: veins only')

plt.xlabel('Reconstruction Error')
plt.ylabel('Error Probability')
plt.legend(loc='upper right')
plt.title('Histogram of Reconstruction Error in Test Set')

plt.show()