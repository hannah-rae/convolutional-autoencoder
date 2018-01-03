import cv2
import numpy as np
from glob import glob
from subprocess import call, check_call
from random import randint

data_dir = '/home/hannah/data/mcam_Lall_Rall_udrs'
new_data_dir = '/home/hannah/data/mcam_Lall_Rall_64x64'

call(['mkdir', new_data_dir])

def get_random_crop_point(height, width):
	max_width = width - 64
	max_height = height - 64
	return (randint(0, max_width), randint(0, max_height))

def crop_img(crop_point, img):
	y, x = crop_point
	cropped = img[x:x+64, y:y+64]
	assert cropped.shape == (64,64)
	return cropped

# Go through all the data multiple times: we are doing random crops
# in each iteration to augment our dataset. There are 739 images in
# the source dataset that are > 64 x 64 pixels, thus we can set n 
# to get N = n * 739 images.
n = 6
for i in range(n):
	# Iterate through sequence_id_XXXX directories
	for seq_dir in glob(data_dir + '/*'):
		# Iterate through RX and LX directories
		seq_id = seq_dir.split('/')[-1]
		for obs in glob(seq_dir + '/*'):
			obs_id = obs.split('/')[-1]
			# Create a blank numpy array to fill with 64x64x6 image
			img = np.ndarray([64,64,6])
			# Iterate through filters of each observation
			img_filters = glob(obs + '/*')
			# Find out the size of the images in this observation
			height, width, _ = cv2.imread(img_filters[0]).shape
			# Skip images that are too small
			if height < 64 or width < 64:
				continue
			crop_point = get_random_crop_point(height, width)
			for img_f in img_filters:
				sol = img_f.split('_')[-1]
				# We are ignoring filter 0 (RGB) images for now
				if 'filter0' in img_f:
					continue
				img_f_arr = cv2.imread(img_f, 0)
				img_f_crop = crop_img(crop_point, img_f_arr)
				if 'filter1' in img_f:
					img[:,:,0] = img_f_crop
				elif 'filter2' in img_f:
					img[:,:,1] = img_f_crop
				elif 'filter3' in img_f:
					img[:,:,2] = img_f_crop
				elif 'filter4' in img_f:
					img[:,:,3] = img_f_crop
				elif 'filter5' in img_f:
					img[:,:,4] = img_f_crop
				elif 'filter6' in img_f:
					img[:,:,5] = img_f_crop
			# Write the numpy array to the training example directory
			np.save(new_data_dir + '/%s_%s_%s_%d.npy' % (seq_id, obs_id, sol, i), img)