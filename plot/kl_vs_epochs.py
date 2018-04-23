import scipy
import matplotlib.pyplot as plt

import scipy.stats
from glob import glob

results = [['/scratch/hannah/MastcamCAE/results/DW_nodrop_brightflux_32x_15ep/*', # 15 epochs
              '/scratch/hannah/MastcamCAE/results/train_nodrop_brightflux_32x_15ep/*'],
           ['/scratch/hannah/MastcamCAE/results/DW_nodrop_brightflux_32x_30ep/*', # 30 epochs
              '/scratch/hannah/MastcamCAE/results/train_nodrop_brightflux_32x_30ep/*'],
           ['/scratch/hannah/MastcamCAE/results/DW_nodrop_brightflux_32x_60ep/*', # 60 epochs
              '/scratch/hannah/MastcamCAE/results/train_nodrop_brightflux_32x_60ep/*']
          ]

x = [15, 30, 60]
y_kl = []
# Iterate through pairs of expert and train results for different bottleneck sizes
for r in results:
    # Get the filenames
    expert = glob(r[0])
    train = glob(r[1])
    # Convert filenames to a list of integers representing reconstruction error
    expert = [int(a.split('/')[-1].split('_')[0]) for a in expert]
    train = [int(a.split('/')[-1].split('_')[0]) for a in train]
    
    # Plot a histogram of reconstruction error for entire test dataset
    qk, bins_train, _ = plt.hist(train, bins=300, normed=True, alpha=0.7, label='Training set')
    pk, bins_expert, _ = plt.hist(expert, bins=300, normed=True, alpha=0.7, label='Expert picks')

    # Change 0 bins to be very small values
    for i in range(len(qk)):
        if qk[i] == 0:
            qk[i] = 0.0000001
        if pk[i] == 0:
            pk[i] = 0.0000001

    # Compute the KL-divergence between the two distributions
    kl = scipy.stats.entropy(pk=pk, qk=qk)
    print kl
    y_kl.append(kl)

print x
print y_kl
plt.clf()
plt.plot(x, y_kl)
plt.xlabel('Number of Epochs Trained For')
plt.ylabel('KL Divergence from Typical to Expert Picks Distribution')
plt.title('Novelty Separation with Increasing Training Epochs')
plt.show()