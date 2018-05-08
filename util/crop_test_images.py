import cv2
import numpy as np
from glob import glob
from subprocess import call, check_call

data_dir = '/home/hannah/data/mcam_Lall_Rall_udrs_sol1667to1925'
new_data_dir = '/home/hannah/data/mcam_Lall_Rall_udrs_sol1667to1925/cropped'

def slice_image(image, window_size=64, margin=0, stride=32):
    (rows, cols, chans) = image.shape
    return [
        image[r:r+window_size, c:c+window_size]
        for r in range(margin, rows - margin - window_size + 1, stride)
        for c in range(margin, cols - margin - window_size + 1, stride)
    ]

for seq_dir in glob(data_dir + '/sequence_id*'):
    # Iterate through RX and LX directories
    seq_id = seq_dir.split('/')[-1]
    for obs in glob(seq_dir + '/*'):
        obs_id = obs.split('/')[-1]
        # Iterate through filters of each observation
        img_filters = glob(obs + '/*')
        # Find out the size of the images in this observation
        height, width, _ = cv2.imread(img_filters[0]).shape
        # Skip images that are too small
        if height < 64 or width < 64:
            continue
        # Create a blank numpy array to hold the image
        img = np.ndarray([height, width, 6])
        for img_f in img_filters:
            sol = img_f.split('_')[-1][:-4]
            # We are ignoring filter 0 (RGB) images for now
            if 'filter0' in img_f:
                continue
            img_f_arr = cv2.imread(img_f, 0)
            if 'filter1' in img_f:
                img[:,:,0] = img_f_arr
            elif 'filter2' in img_f:
                img[:,:,1] = img_f_arr
            elif 'filter3' in img_f:
                img[:,:,2] = img_f_arr
            elif 'filter4' in img_f:
                img[:,:,3] = img_f_arr
            elif 'filter5' in img_f:
                img[:,:,4] = img_f_arr
            elif 'filter6' in img_f:
                img[:,:,5] = img_f_arr
        # Slice the image into 64x64x6 squares
        cropped_images = slice_image(img)
        print "Saving %d images for %s %s %s" % (len(cropped_images), seq_id, obs_id, sol)
        for i, c_i in enumerate(cropped_images):
            np.save(new_data_dir + '/%s_%s_%s_%d.npy' % (seq_id, obs_id, sol, i), c_i)