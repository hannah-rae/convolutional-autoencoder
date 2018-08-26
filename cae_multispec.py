"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
import argparse
import dataset
# from libs.activations import lrelu
# from libs.utils import corrupt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--summaries_dir', default='/scratch/hannah/tf-summaries')
args = parser.parse_args()


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
                n_filters=[1, 12, 8, 3],
                filter_sizes=[7, 5, 3, 6]):

    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')
    # fraction of neurons to keep in dropout layer
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
    # Randomly fluctuate the brightness to help with overfitting
    # current_input = tf.map_fn(lambda frame: tf.image.random_brightness(frame, max_delta=32. / 255.), current_input)
    # Randomly flip horizontally to help with overfitting
    #current_input = tf.map_fn(lambda frame: tf.image.random_flip_left_right(frame), current_input)

    input_image = current_input

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        if layer_i == 0:
            stride=1
        else:
            stride=2
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
                current_input, W, strides=[1, stride, stride, 1], padding='SAME'), b))
        #output_bn = tf.layers.batch_normalization(inputs=output, axis=1, training=keep_prob != 1.0)
        #output_drop = tf.nn.dropout(x=output, keep_prob=keep_prob, name='conv_dropout')
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    print z.shape
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        if layer_i == 2:
            stride=1
        else:
            stride=2
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
                strides=[1, stride, stride, 1], padding='SAME'), b))
        current_input = output
        # If it's the last layer, don't do batch norm
        # if (layer_i + 1) == len(shapes):
        #     current_input = output
        # else:
        #     output_bn = tf.layers.batch_normalization(inputs=output, axis=1, training=keep_prob != 1.0)
        #     current_input = output_bn

    # %%
    # now have the reconstruction through the network
    y = current_input

    cost = tf.losses.mean_squared_error(x_tensor, y)
    tf.summary.scalar('train_loss', cost)

    # Merge all the summaries 
    merged = tf.summary.merge_all()

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'keep_prob': keep_prob, 'merged': merged}


def test_multispec():
    import tensorflow as tf
    from time import time
    from os import mkdir
    import cv2

    # load mastcam data
    # mastcam = dataset.load_mcam_6f()
    # print mastcam.shape
    # mean_img = np.mean(mastcam, axis=0)
    # std_img = np.std(mastcam, axis=0)
    ae = autoencoder(input_shape=[None, 64, 64, 6])

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(args.summaries_dir + '/train',
                                          sess.graph)
    # Fit all training data
    num_batches = dataset.num_train_ex / args.batch_size
    print "num batches = %d", num_batches
    for epoch_i in range(args.epochs):
        for batch_i in range(num_batches):
            # idx = batch_i*batch_size
            # batch_xs = mastcam[idx:idx+batch_size]
            batch_xs = dataset.next_batch_6f(batchsize=args.batch_size)
            #train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.6})
            summary = sess.run(ae['merged'], feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.6})
            train_writer.add_summary(summary, batch_i)
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs, ae['keep_prob']: 1.0}))
    
    # Save the model for future training or testing
    name = 'udr_12-8-3_7-5-3_nodrop_epochs%d_stride4data' % args.epochs
    save_path = saver.save(sess, "/scratch/hannah/saved_sessions/%s.ckpt" % name)
    print("Model saved in path: %s" % save_path)

    # # %%
    # # Plot example reconstructions and record reconstruction errors
    # #n_examples = 10
    # test_xs, x_names = dataset.load_mcam_DW_test(product='UDR')
    # #test_xs_norm = np.array([img - mean_img for img in test_xs])
    # #recon, err = sess.run([ae['y'], ae['cost']], feed_dict={ae['x']: test_xs, ae['keep_prob']: 1.0})
    # #recon = np.array([img + mean_img for img in recon])

    # mkdir('./results/DW_' + t)

    # for i, example in enumerate(test_xs):
    #     # Run model on test example
    #     ex = np.zeros((1, 64, 64, 6))
    #     ex[0] = example
    #     recon, err = sess.run([ae['y'], ae['cost']], feed_dict={ae['x']: ex, ae['keep_prob']: 1.0})
    #     # # Extract filters 3,5,6 for R,G,B
    #     # test_vis = np.zeros((64, 64, 3))
    #     # recon_vis = np.zeros((64, 64, 3))
    #     # # We want to visualize filters 3,5,6 as R,G,B
    #     # test_vis[:,:,0] = example[:,:,2]
    #     # test_vis[:,:,1] = example[:,:,4]
    #     # test_vis[:,:,2] = example[:,:,5]
    #     # # Same for the reconstructed image
    #     # recon_vis[:,:,0] = recon[0,:,:,2]
    #     # recon_vis[:,:,1] = recon[0,:,:,4]
    #     # recon_vis[:,:,2] = recon[0,:,:,5]

    #     # cv2.imwrite('./results/' + t + '/' + str(i) + '_input.png', test_vis)
    #     # cv2.imwrite('./results/' + t + '/' + str(i) + '_' + str(int(err)) + '_recon.png', recon_vis)

    #     input_per_filter_total_error = [np.sum(example[:,:,f]) for f in range(6)]
    #     input_per_filter_mean_error = [np.mean(example[:,:,f]) for f in range(6)]
    #     input_per_filter_var_error = [np.var(example[:,:,f]) for f in range(6)]
    #     input_per_filter_max_error = [np.max(example[:,:,f]) for f in range(6)]
    #     input_per_filter_mean_var_max = np.concatenate([input_per_filter_mean_error, input_per_filter_var_error, input_per_filter_max_error])

    #     recon_per_filter_total_error = [np.sum(recon[0,:,:,f]) for f in range(6)]
    #     recon_per_filter_mean_error = [np.mean(recon[0,:,:,f]) for f in range(6)]
    #     recon_per_filter_var_error = [np.var(recon[0,:,:,f]) for f in range(6)]
    #     recon_per_filter_max_error = [np.max(recon[0,:,:,f]) for f in range(6)]
    #     recon_per_filter_mean_var_max = np.concatenate([recon_per_filter_mean_error, recon_per_filter_var_error, recon_per_filter_max_error])

    #     # Find the reconstruction difference for each filter
    #     diff = np.zeros((64, 64, 6))
    #     for f in range(6):
    #         diff[:,:,f] = np.square(np.subtract(example[:,:,f], recon[0,:,:,f]))

    #     # Compute a per-pixel reconstruction error image through all filters
    #     r_im = np.ndarray([64,64])
    #     weights=[1./diff.shape[2]]*diff.shape[2]
    #     for j in range(diff.shape[0]):
    #         for k in range(diff.shape[1]):
    #             # compute a weighted sum in each pixel through all filters
    #             r_im[j,k] = np.dot(diff[j,k,:], weights) 

    #     # Computing per filter error as the sum of all error in the image 
    #     # because we don't want to dampen out the contributions of small 
    #     # anomalies to the error
    #     per_filter_total_error = [np.sum(diff[:,:,f]) for f in range(6)]
    #     per_filter_mean_error = [np.mean(diff[:,:,f]) for f in range(6)]
    #     per_filter_var_error = [np.var(diff[:,:,f]) for f in range(6)]
    #     per_filter_max_error = [np.max(diff[:,:,f]) for f in range(6)]
    #     diff_per_filter_mean_var_max = np.concatenate([per_filter_mean_error, per_filter_var_error, per_filter_max_error])

    #     #w = [0.026427, 1.903135, 0.022009, 0.105691, 0.194797, 0.130667]
    #     diff_mean = np.average(per_filter_max_error)

    #     # Make a new directory for each image
    #     #img_dir = './results/DW_' + t + '/' + str(int(diff_mean*100000000)) + '_' + x_names[i] # RDR
    #     img_dir = './results/DW_' + t + '/' + str(int(diff_mean)) + '_' + x_names[i] # UDR
    #     mkdir(img_dir)

    #     # Write the explanation product
    #     np.save(img_dir + '/diff_mean_var_max.npy', diff_per_filter_mean_var_max)
    #     np.save(img_dir + '/input_mean_var_max.npy', input_per_filter_mean_var_max)
    #     np.save(img_dir + '/recon_mean_var_max.npy', recon_per_filter_mean_var_max)
    #     #cv2.imwrite(img_dir + '/explanation.png', r_im)
    #     np.save(img_dir + '/explanation.npy', r_im)
    #     np.save(img_dir + '/diff_6f.npy', diff)

    #     # Six filters in each example
    #     for f in range(6):
    #         # UDRs
    #         cv2.imwrite(img_dir + '/' + str(f+1) + '_input.png', example[:,:,f])
    #         cv2.imwrite(img_dir + '/' + str(f+1) + '_' + str(int(np.max(diff[:,:,f]))) + '_recon.png', recon[0,:,:,f])

    #         # Write filter f as grayscale image - RDRs
    #         # scaled_ex = np.interp(example[:,:,f], (example[:,:,f].min(), example[:,:,f].max()), (0, 255))
    #         # cv2.imwrite(img_dir + '/' + str(f+1) + '_input.png', scaled_ex)
    #         # scaled_recon = np.interp(recon[0,:,:,f], (recon[0,:,:,f].min(), recon[0,:,:,f].max()), (0, 255))
    #         # cv2.imwrite(img_dir + '/' + str(f+1) + '_' + str(int(np.max(diff[:,:,f])*100000000)) + '_recon.png', scaled_recon)
    #         #cv2.imwrite(img_dir + '/' + str(f+1) + '_' + str(int(np.max(diff[:,:,f]))) + '_diff.png', diff[:,:,f])


# %%
if __name__ == '__main__':
    test_multispec()
