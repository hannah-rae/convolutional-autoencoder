import cv2
import numpy as np
from glob import glob
from subprocess import call, check_call
from random import randint

data_dir = '/home/hannah/data/mcamrdrs/train'
out_dir = '/home/hannah/data/mcamrdrs/train/cropped'

def get_random_crop_point(height, width):
    max_width = width - 64
    max_height = height - 64
    return (randint(0, max_width), randint(0, max_height))

def crop_img(crop_point, img):
    y, x = crop_point
    cropped = img[x:x+64, y:y+64, :]
    assert cropped.shape == (64,64,6)
    return cropped

# Go through all the data multiple times: we are doing random crops
# in each iteration to augment our dataset. There are 739 images in
# the source dataset that are > 64 x 64 pixels, thus we can set n 
# to get N = n * 617 images.
n = 160
for i in range(n):
    for img_name in glob(data_dir + '/*.npy'):
        # Extract info from file name
        info = img_name.split('/')[-1].split('_')
        seq = info[0]
        sol = int(info[1])
        ins = info[2]
        if len(info) > 4:
            identifier = info[4][:-4]
        else:
            identifier = None
        # Load the image
        img = np.load(img_name)
        height, width, _ = img.shape
        # Skip images that are too small
        if height < 64 or width < 64:
            continue
        # Crop random square from image
        crop_point = get_random_crop_point(height, width)
        img_crop = crop_img(crop_point, img)
        # Write the numpy array to the training example directory
        if identifier != None:
            np.save(out_dir + '/' + '%s_%d_%s_%s_%d.npy' % (seq, sol, ins, identifier, i), img_crop)
        else:
            np.save(out_dir + '/' + '%s_%d_%s_%d.npy' % (seq, sol, ins, i), img_crop)