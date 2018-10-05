import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import argparse
import os.path
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--test_image_dir', help='Directory of multispectral images stored as single-band images (should be .npy files)')
parser.add_argument('--cae_model_file', help='CAE model file to use')
parser.add_argument('--cnn_model_dir', help='CNN model file directory to use')
parser.add_argument('--stride_size', type=int, default=4, help='Stride size (in pixels) to use to create salience map')
parser.add_argument('--margin', type=int, default=1, help='Number of pixels on each side to ignore')
parser.add_argument('--save_dir', help='Directory to save the resulting salience map in')
parser.add_argument('--overlay', action='store_true', help='Overlay the salience maps on the input (RGB) image')
args = parser.parse_args()

from cae_multispec import autoencoder
# We have to build this exactly once
# If we do it repeatedly in get_cae_error_map(),
# tf error "Not found: Key Variable_10 not found in checkpoint"
ae = autoencoder(input_shape=[None, 64, 64, 6])
# We create a session to use the graph
sess = tf.Session()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
saver.restore(sess, args.cae_model_file)

from cnn_multispec import cnn_model_fn
# It's faster to not load the estimator with every function call
# Create the Estimator
multispec_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=args.cnn_model_dir)

def get_cae_error_map(tile):
    recon, mse = sess.run([ae['y'], ae['cost']], feed_dict={ae['x']: np.expand_dims(tile, axis=0), ae['keep_prob']: 1.0})
    # Compute the error map
    error_map = np.zeros((64, 64, 6), dtype=np.float32)
    for f in range(6):
        error_map[:,:,f] = np.square(np.subtract(tile[:,:,f], recon[0,:,:,f])).astype(np.float32)
    return error_map, recon[0]

# Note: if we don't specify the error map to be float32, we get DataLoss error

def get_cnn_prediction(error_map):
    # Classify this error tile
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.expand_dims(error_map.astype(np.float32), axis=0)},
        num_epochs=1,
        shuffle=False)
    predictions = list(multispec_classifier.predict(input_fn=predict_input_fn))
    # Convert output to single probability (of novelty)
    p_novel = [p["probabilities"] for p in predictions][0][1]
    return p_novel

# Load the image
rows, cols = cv2.imread(glob(os.path.join(args.test_image_dir, '*'))[0], 0).shape
img = np.ndarray([rows, cols, 6])
img_rgb = np.ndarray([rows, cols, 3])
for im in glob(os.path.join(args.test_image_dir, '*')):
    _im = cv2.imread(im, 0)
    # We don't want the RGB image in this cube,
    # but we want it for overlaying the maps
    if 'filter0' in im:
        img_rgb = cv2.imread(im)
    elif 'filter1' in im:
        img[:,:,0] = _im
    elif 'filter2' in im:
        img[:,:,1] = _im
    elif 'filter3' in im:
        img[:,:,2] = _im
    elif 'filter4' in im:
        img[:,:,3] = _im
    elif 'filter5' in im:
        img[:,:,4] = _im
    elif 'filter6' in im:
        img[:,:,5] = _im

# Create base for salience map
smap_probs = np.zeros([rows,cols])
smap_err = np.zeros(img.shape)
smap_recon = np.zeros(img.shape)
smap_freq = np.zeros([rows,cols])

# Make 64x64x6-pixel strided patches
for r in range(0, rows - 64 + 1, args.stride_size):
    for c in range(0, cols - 64 + 1, args.stride_size):
        tile = img[r:r+64, c:c+64]
        # Compute CAE error map
        tile_error, recon = get_cae_error_map(tile)
        # Compute CNN prediction based on error map
        p_novel = get_cnn_prediction(tile_error)
        # +1 for each probability computed in each pixel
        smap_freq[r:r+64, c:c+64] += 1.
        # Sum all the probabilities computed in each pixel
        smap_probs[r:r+64, c:c+64] += np.full(shape=(64,64), fill_value=p_novel)
        # Sum up the error values in each pixel for each band
        smap_err[r:r+64, c:c+64] += tile_error
        # Combine the reconstructions too
        smap_recon[r:r+64, c:c+64] += recon

# Get the mean probability in each pixel
smap_probs /= smap_freq
# Get the mean error in each pixel
for f in range(6):
    smap_err[:,:,f] /= smap_freq
    smap_recon[:,:,f] /= smap_freq
    # Save them each independently
    cv2.imwrite(os.path.join(args.save_dir, 'salience_error%d_map_test.png' % f), smap_err[:,:,f])
    cv2.imwrite(os.path.join(args.save_dir, 'salience_recon%d_map_test.png' % f), smap_recon[:,:,f])
# Blend the filter error maps into one
# May want to do some noise filter before this step
smap_err_mean = np.mean(smap_err, axis=2)
if args.overlay:
    fig, ax1 = plt.subplots(1)
    ax1.imshow(img_rgb)
    im1 = ax1.imshow(smap_probs, cmap='YlOrRd', alpha=0.5)
    ax1.tick_params(
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)
    fig.colorbar(im1, ax=ax1)
    fig.savefig(os.path.join(args.save_dir, 'salience_prob_map_overlay.png'))
else:
    # Rescale the float values from (0, 1) to (0, 255)
    smap_probs *= 255.0
    cv2.imwrite(os.path.join(args.save_dir, 'salience_prob_map_test.png'), smap_probs)

# Rescale the float values from (0, 1) to (0, 255)
smap_err_mean *= 255.0
cv2.imwrite(os.path.join(args.save_dir, 'salience_error_map_test.png'), smap_err_mean)

