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
                n_filters=[1, 36, 24, 12],
                filter_sizes=[7, 3, 3, 3],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')


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

    # %%
    # Optionally apply denoising autoencoder
    # if corruption:
    #     current_input = corrupt(current_input)

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
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        noise_shape = tf.stack([tf.shape(output)[0], 1, 1, tf.shape(output)[3]])
        output_drop = tf.nn.dropout(x=output, keep_prob=0.7, noise_shape=(noise_shape), name='conv_dropout')
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
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.losses.mean_squared_error(x_tensor, y)
    # cost = tf.reduce_sum(tf.square(y - x_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


def test_mastcam_slices():
    import tensorflow as tf
    from time import time
    from os import mkdir
    import cv2

    # load mastcam data
    mastcam = dataset.load_mcam_slices()
    print mastcam.shape
    mean_img = np.mean(mastcam, axis=0)
    ae = autoencoder(input_shape=[None, 36, 40, 3])

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    n_epochs = 3
    batch_size = 10
    num_batches = mastcam.shape[0] / batch_size
    print "num batches = %d", num_batches
    for epoch_i in range(n_epochs):
        for batch_i in range(num_batches):
            idx = batch_i*batch_size
            batch_xs = mastcam[idx:idx+batch_size]
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 10
    test_xs = dataset.test_mcam_slices()[:batch_size]
    #test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
    #recon = np.array([img + mean_img for img in recon])
    t = str(int(time()))
    mkdir('./results/' + t)
    for example_i in range(n_examples):
        print 'one test example shape'
        print test_xs[example_i].shape
        cv2.imwrite('./results/' + t + '/' + str(example_i) + '_input.png', test_xs[example_i])
        cv2.imwrite('./results/' + t + '/' + str(example_i) + '_recon.png', recon[example_i])

def test_mastcam_rgb():
    import tensorflow as tf
    from time import time
    from os import mkdir
    import cv2

    # load mastcam data
    mastcam = dataset.load_mcam_rgb()
    print mastcam.shape
    mean_img = np.mean(mastcam, axis=0)
    std_img = np.std(mastcam, axis=0)
    ae = autoencoder(input_shape=[None, 144, 160, 3])

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    n_epochs = 3
    batch_size = 10
    num_batches = mastcam.shape[0] / batch_size
    print "num batches = %d", num_batches
    for epoch_i in range(n_epochs):
        for batch_i in range(num_batches):
            idx = batch_i*batch_size
            batch_xs = mastcam[idx:idx+batch_size]
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 10
    test_xs = dataset.load_mcam_rgb()[:batch_size]
    #test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
    #recon = np.array([img + mean_img for img in recon])
    t = str(int(time()))
    mkdir('./results/' + t)
    for example_i in range(n_examples):
        print 'one test example shape'
        print test_xs[example_i].shape
        cv2.imwrite('./results/' + t + '/' + str(example_i) + '_input.png', test_xs[example_i])
        cv2.imwrite('./results/' + t + '/' + str(example_i) + '_recon.png', recon[example_i])


def test_mastcam_gray():
    import tensorflow as tf
    from time import time
    from os import mkdir
    from matplotlib.pyplot import imsave

    # load mastcam data
    mastcam = dataset.load_mcam_gray()
    mean_img = np.mean(mastcam, axis=0)
    ae = autoencoder(input_shape=[None, 144*144])

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 10
    n_epochs = 3
    num_batches = mastcam.shape[0] / batch_size
    print "num batches = %d", num_batches
    for epoch_i in range(n_epochs):
        for batch_i in range(num_batches):
            idx = batch_i*batch_size
            batch_xs = mastcam[idx:idx+batch_size]
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 10
    test_xs = dataset.load_test_gray()[0:batch_size]
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon, recon_err = sess.run([ae['y'], ae['cost']], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    t = str(int(time()))
    mkdir('./results/' + t)
    for example_i in range(n_examples):
        imsave('./results/' + t + '/' + str(example_i) + '_input.png', np.reshape(test_xs[example_i, :], (144, 144)), cmap='gray')
        imsave('./results/' + t + '/' + str(example_i) + '_' + str(int(recon_err)) + '_recon.png', np.reshape(np.reshape(recon[example_i, ...], (144*144,)) + mean_img, (144, 144)),  cmap='gray')


# %%
def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(input_shape=[None, 784])

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)), cmap='gray')
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)), cmap='gray')
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


# %%
if __name__ == '__main__':
    test_mastcam_rgb()
