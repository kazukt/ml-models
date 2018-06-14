from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def add_model_summaries(model, scope=None):
    with tf.name_scope(scope):
        for var in model.variables:
            tf.summary.histogram(var)

def add_image_summaries(real_images, generated_images, grid_size=4):
    num_images = grid_size ** 2
    real_image_shape = real_images.shape.as_list()[1:3]
    generated_image_shape = generated_images.shape.as_list()[1:3]
    real_channels = real_images.shape.as_list()[3]
    generated_channels = generated_images.shape.as_list()[3]

    tf.summary.image(
        'real_images',
        image_grid(
            real_images[:num_images],
            grid_shape=(grid_size, grid_size),
            image_shape=real_image_shape,
            num_channels=real_channels),
        max_outputs=1)

    tf.summary.image(
        'generated_images',
        image_grid(
            generated_images[:num_images],
            grid_shape=(grid_size, grid_size),
            image_shape=generated_image_shape,
            num_channels=generated_channels),
        max_outputs=1)

def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
    """Arrange a minibatch of images into a grid to form a single image.
    Args:
    input_tensor: Tensor. Minibatch of images to format, either 4D
        ([batch size, height, width, num_channels]) or flattened
        ([batch size, height * width * num_channels]).
    grid_shape: Sequence of int. The shape of the image grid,
        formatted as [grid_height, grid_width].
    image_shape: Sequence of int. The shape of a single image,
        formatted as [image_height, image_width].
    num_channels: int. The number of channels in an image.
    Returns:
    Tensor representing a single image in which the input images have been
    arranged into a grid.
    Raises:
    ValueError: The grid shape and minibatch size don't match, or the image
        shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
        raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                        (grid_shape, int(input_tensor.shape[0])))
    if len(input_tensor.shape) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.shape[1]) != num_features:
            raise ValueError("Image shape and number of channels incompatible with "
                             "input tensor.")
    elif len(input_tensor.shape) == 4:
        if (int(input_tensor.shape[1]) != image_shape[0] or
            int(input_tensor.shape[2]) != image_shape[1] or
            int(input_tensor.shape[3]) != num_channels):
            raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
    else:
        raise ValueError("Unrecognized input tensor format.")
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(
      input_tensor, tuple(grid_shape) + tuple(image_sshape) + (num_channels,))
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
      input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = tf.reshape(
      input_tensor, [1, height, width, num_channels])
    return input_tensor
