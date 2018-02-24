"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math

import dataset
# from libs.activations import lrelu
# from libs.utils import corrupt

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# %%
def autoencoder(input_shape,
                n_filters=[1, 72, 48, 24],
                filter_sizes=[7, 5, 5, 6]):

    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')
    # Use this to randomly crop images on the fly, rather than holding everything in memory
    #x = tf.map_fn(lambda frame: tf.random_crop(value=frame, size=[64,64,6]), x)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Linearly scale image to have zero mean and unit norm
    current_input = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        W = tf.clip_by_norm(W, clip_norm=4.0)
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        output_drop = tf.nn.dropout(x=output, keep_prob=keep_prob, name='conv_dropout')
        current_input = output_drop

    # %%
    # store the latent representation
    z = current_input
    print z.shape
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        # W = tf.Variable(
        #     tf.random_uniform(encoder[layer_i].get_shape().as_list(),
        #         -1.0 / math.sqrt(encoder[layer_i].get_shape().as_list()[2]),
        #         1.0 / math.sqrt(encoder[layer_i].get_shape().as_list()[2])))
        W = encoder[layer_i] # should be clipped already
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output
        # If it's the last layer, don't add dropout
        # if (layer_i + 1) == len(shapes):
        #     current_input = output
        # else:
        #     output_drop = tf.nn.dropout(x=output, keep_prob=0.7, name='conv_dropout')
        #     current_input = output_drop

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.losses.mean_squared_error(x_tensor, y)
    # cost = tf.reduce_sum(tf.square(y - x_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'keep_prob': keep_prob}


def test_multispec():
    import tensorflow as tf
    from time import time
    from os import mkdir
    import cv2

    # load mastcam data
    mastcam = dataset.load_mcam_6f()
    print mastcam.shape
    # mean_img = np.mean(mastcam, axis=0)
    # std_img = np.std(mastcam, axis=0)
    ae = autoencoder(input_shape=[None, 64, 64, 6])

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    n_epochs = 15
    batch_size = 5
    num_batches = mastcam.shape[0] / batch_size
    print "num batches = %d", num_batches
    for epoch_i in range(n_epochs):
        for batch_i in range(num_batches):
            idx = batch_i*batch_size
            batch_xs = mastcam[idx:idx+batch_size]
            #train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.7})
            #print(batch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.7}))
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.7}))

    # %%
    # Plot example reconstructions and record reconstruction errors
    #n_examples = 10
    test_xs = dataset.load_mcam_6f_test()
    #test_xs_norm = np.array([img - mean_img for img in test_xs])
    #recon, err = sess.run([ae['y'], ae['cost']], feed_dict={ae['x']: test_xs, ae['keep_prob']: 1.0})
    #recon = np.array([img + mean_img for img in recon])
    t = str(int(time()))
    mkdir('./results/' + t)

    for i, example in enumerate(test_xs):
        # Run model on test example
        ex = np.zeros((1, 64, 64, 6))
        ex[0] = example
        recon, err = sess.run([ae['y'], ae['cost']], feed_dict={ae['x']: ex, ae['keep_prob']: 1.0})
        # Extract filters 3,5,6 for R,G,B
        test_vis = np.zeros((64, 64, 3))
        recon_vis = np.zeros((64, 64, 3))
        # We want to visualize filters 3,5,6 as R,G,B
        test_vis[:,:,0] = example[:,:,2]
        test_vis[:,:,1] = example[:,:,4]
        test_vis[:,:,2] = example[:,:,5]
        # Same for the reconstructed image
        recon_vis[:,:,0] = recon[0,:,:,2]
        recon_vis[:,:,1] = recon[0,:,:,4]
        recon_vis[:,:,2] = recon[0,:,:,5]

        cv2.imwrite('./results/' + t + '/' + str(i) + '_input.png', test_vis)
        cv2.imwrite('./results/' + t + '/' + str(i) + '_' + str(int(err)) + '_recon.png', recon_vis)



# %%
if __name__ == '__main__':
    test_multispec()
