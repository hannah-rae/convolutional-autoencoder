from glob import glob

import numpy as np

TYPICAL = 'typical'
NOVEL = 'novel'

typical_dir = '/Users/hannahrae/src/autoencoder/MastcamCAE/results/train_optimized_11'
novel_dir = '/Users/hannahrae/src/autoencoder/MastcamCAE/results/DW_optimized_11'
train_weka = '/Users/hannahrae/data/mcam_recon_spectra_train_9.arff'
test_weka = '/Users/hannahrae/data/mcam_recon_spectra_test_9.arff'

typical_files = glob(typical_dir+'/*')
np.random.shuffle(typical_files)
train_samples = [[t, TYPICAL] for t in typical_files[:250]]
test_samples = [[t, TYPICAL] for t in typical_files[250:632]]

novel_files = glob(novel_dir+'/*')
np.random.shuffle(novel_files)
train_samples = train_samples + [[t, NOVEL] for t in novel_files[:250]]
test_samples = test_samples + [[t, NOVEL] for t in novel_files[250:332]]

with open(train_weka, 'w+') as f:
    for file, lbl in train_samples:
        arr = np.load(file + '/recon_mean_var_max.npy')[:6]
        #arr = np.concatenate([np.load(file + '/recon_mean_var_max.npy')[:6], np.load(file + '/recon_mean_var_max.npy')[12:]])
        f.write('   ')
        for x in arr:
            f.write('%f,' % x)
        #f.write('%s\n' % lbl)
        f.write('%s %% %s\n' % (lbl, file.split('/')[-1]))

with open(test_weka, 'w+') as f:
    for file, lbl in test_samples:
        arr = np.load(file + '/recon_mean_var_max.npy')[:6]
        #arr = np.concatenate([np.load(file + '/recon_mean_var_max.npy')[:6], np.load(file + '/recon_mean_var_max.npy')[12:]])
        f.write('   ')
        for x in arr:
            f.write('%f,' % x)
        f.write('%s %% %s\n' % (lbl, file.split('/')[-1]))
