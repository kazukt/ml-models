import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

def generator(code, training=True):

    net = tf.layers.dense(code, 1024, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.dense(net, 7 * 7 * 128, activation=tf.nn.relu)
    net = tf.reshape(net, [-1, 7, 7, 128])
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d_transpose(net, 64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d_transpose(net, 32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d(net, 1, kernel_size=4, strides=1, padding='same', activation=tf.nn.tanh)
    
    return net

def discriminator(image):
    net = tf.layers.conv2d(image, 64, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
    net = tf.layers.conv2d(net, 128, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024, activation=tf.nn.leaky_relu)
    net = tf.layers.dense(net, 1)
    return net

def encoder(image, latent_size, weight_decay=2.5e-5, is_training=True):
    net = tf.layers.conv2d(image, 64, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.conv2d(net, 128, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024, activation=tf.nn.leaky_relu)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.dense(net, latent_size)
    return net

def code_discriminator(code):
    net = tf.layers.dense(code, 1)

    return net
