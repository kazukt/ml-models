import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

def generator(
    code, weight_decay=2.5e-5, is_training=True):
    normalizer_fn_args = {
        'is_training': is_training
    }

    with slim.arg_scope(
        [slim.batch_norm], **normalizer_fn_args):
        with slim.arg_scope(
            [slim.fully_connected, slim.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
            weights_regularizer=slim.l2_regularizer(weight_decay)):

            net = slim.fully_connected(code, 1024)
            net = slim.fully_connected(net, 7 * 7 * 128)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = slim.conv2d_transpose(net, 64, [4, 4], stride=2)
            net = slim.conv2d_transpose(net, 32, [4, 4], stride=2)

            net = slim.conv2d(
                net, 1, [4, 4], stride=1,
                normalizer_fn=None, activation_fn=tf.nn.tanh)

            return net

def discriminator(image, weight_decay=2.5e-5):
    with slim.arg_scope(
        [slim.fully_connected, slim.conv2d],
        activation_fn=functools.partial(tf.nn.leaky_relu, alpha=0.01),
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(weight_decay),
        biases_regularizer=slim.l2_regularizer(weight_decay)):
        net = slim.conv2d(image, 64, [4, 4], stride=2)
        net = slim.conv2d(net, 128, [4, 4], stride=2)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, normalizer_fn=slim.batch_norm)
        net = slim.linear(net, 1)

        return net

def encoder(image, latent_size, weight_decay=2.5e-5, is_training=True):
    normalizer_fn_args = {
        'is_training': is_training
    }

    with slim.arg_scope(
        [slim.batch_norm], **normalizer_fn_args):
        with slim.arg_scope(
            [slim.fully_connected, slim.conv2d],
            activation_fn=functools.partial(tf.nn.leaky_relu, alpha=0.01),
            normalizer_fn=slim.batch_norm,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_regularizer=slim.l2_regularizer(weight_decay)):
            net = slim.conv2d(image, 64, [4, 4], stride=2)
            net = slim.conv2d(net, 128, [4, 4], stride=2)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = slim.linear(net, latent_size, normalizer_fn=None)

            return net

def code_discriminator(code):
    net = slim.linear(code, 1)

    return net
