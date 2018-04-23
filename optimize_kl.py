import matplotlib.pyplot as plt
import numpy as np

import scipy.stats
from glob import glob

def weighted_mean(w, x):
    return np.average(x, weights=w)

def get_max_errors(path):
    all_max_errors = []
    for img_dir in glob(path+'/*'):
        max_errors = np.ndarray(6)
        for recon in glob(img_dir+'/*recon.png'):
            f, m, _ = recon.split('/')[-1].split('_')
            max_errors[int(f)-1] = m
        all_max_errors.append(max_errors)
    return all_max_errors

# def show_kl(wmean_typical, wmean_novel):


def kl(wmean_typical, wmean_novel):
    # Plot a histogram of reconstruction error for entire test dataset
    qk, bins_train, _ = plt.hist(wmean_typical, bins=300, normed=True, alpha=0.7, label='Training set')
    pk, bins_expert, _ = plt.hist(wmean_novel, bins=300, normed=True, alpha=0.7, label='Expert picks')

    # Change 0 bins to be very small values
    for i in range(len(qk)):
        if qk[i] == 0:
            qk[i] = 0.0000001
        if pk[i] == 0:
            pk[i] = 0.0000001

    # Compute the KL-divergence between the two distributions
    kl = scipy.stats.entropy(pk=qk, qk=pk)
    plt.clf()
    return kl

# Initialize variables
a = 0.01 # learning rate, alpha
h = 0.01 # step in derivative
c = 0.00000001 # epsilon, goal for dw
w = np.random.rand(6)
m_typical = get_max_errors(path='/scratch/hannah/MastcamCAE/results/train_optimization')
m_novel = get_max_errors(path='/scratch/hannah/MastcamCAE/results/DW_optimization')

# Gradient ascent
w_last = w
while True:
    print w
    # Compute the KL divergence given current values of w
    wmean_typical = [weighted_mean(w, x) for x in m_typical]
    wmean_novel = [weighted_mean(w, x) for x in m_novel]
    f = kl(wmean_typical, wmean_novel)
    print f
    
    # Compute the gradient of the KL divergence at this point
    wmean_typical_h = [weighted_mean(w+h, x) for x in m_typical]
    wmean_novel_h = [weighted_mean(w+h, x) for x in m_novel]
    df = (kl(wmean_typical_h, wmean_novel_h) - f) / h
    #print df
    # Update weights based on gradient ascent update rule
    w = w + a * df

    # print w_last
    # print w

    if np.all(np.abs(w_last - w) <= c):
        print "Weights:"
        print w
        print "Maximum KL divergence:"
        print f
        break
    else:
        w_last = w
