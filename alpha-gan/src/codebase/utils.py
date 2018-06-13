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
            grid_shape=(grid_shape, grid_shape),
            image_shape=real_image_shape,
            num_channels=real_channels),
        max_outputs=1)

    tf.summary.image(
        'generated_images',
        image_grid(
            generated_images[:num_images],
            grid_shape=(grid_shape, grid_shape),
            image_shape=generated_image_shape,
            num_channels=generated_channels),
        max_outputs=1)
