import cv2
import numpy as np
import csv
from glob import glob
from subprocess import call, check_call
from random import randint

data_dir = '/home/hannah/data/sammie/mcam_Lall_Rall_udrs'
new_data_dir = '/home/hannah/data/sammie/mcam_Lall_Rall_64x64_stride8'

#call(['mkdir', new_data_dir])

def slice_image(image, window_size=64, stride=8, margin=0):
    (rows, cols, _) = image.shape
    return [
                image[r:r+window_size, c:c+window_size]
                for r in range(margin, rows - margin - window_size + 1, stride)
                for c in range(margin, cols - margin - window_size + 1, stride)
            ]

# Get list of sols containing novel sequences
novel_sols = [fn.split('sol')[-1].split('_')[0] for fn in glob('/home/hannah/data/sammie/mcam_multispec_DW/*.npy')]

# Get a mapping between the database seqid and the mission seqid
seq_id_map = {}
with open('/home/hannah/data/sammie/seq_id_map.csv', 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        seq_id_map[row[0]] = row[1]

# Slice each thumbnail image into 64x64-pixel tiles with a stride size of 8.
# Iterate through sequence_id_XXXX directories
for seq_dir in glob(data_dir + '/*'):
    # Iterate through RX and LX directories
    seq_id = seq_dir.split('/')[-1].split('_')[-1]
    for obs in glob(seq_dir + '/*'):
        obs_id = obs.split('/')[-1]
        # Iterate through filters of each observation
        img_filters = glob(obs + '/*')
        # Find out the size of the images in this observation
        height, width, _ = cv2.imread(img_filters[0]).shape
        sol = img_filters[0].split('_')[-1]
        # Skip images that are too small
        if height < 64 or width < 64:
            continue
        # Create the multispectral image array
        img = np.ndarray([height, width, 6])
        for img_f in img_filters:
            # We are ignoring filter 0 (RGB) images for now
            if 'filter0' in img_f:
                continue
            if 'filter1' in img_f:
                img[:,:,0] = cv2.imread(img_f, 0)
            elif 'filter2' in img_f:
                img[:,:,1] = cv2.imread(img_f, 0)
            elif 'filter3' in img_f:
                img[:,:,2] = cv2.imread(img_f, 0)
            elif 'filter4' in img_f:
                img[:,:,3] = cv2.imread(img_f, 0)
            elif 'filter5' in img_f:
                img[:,:,4] = cv2.imread(img_f, 0)
            elif 'filter6' in img_f:
                img[:,:,5] = cv2.imread(img_f, 0)
        slices = slice_image(img)

        # Write the numpy array to the training example directory
        # Skip the image if it was taken on a sol containing a novel sequence
        if sol[3:] not in novel_sols:
            for idx, s in enumerate(slices):
                np.save(new_data_dir + '/%s_%s_%s_%d.npy' % (seq_id_map[seq_id], obs_id, sol, idx), img)