import argparse
import os
import time

from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy
import tensorflow as tf

from codebase.alphagan import make_generator, make_discriminator
import codebase.losses as loss

from tensorflow.contrib.learn.python.learn.datasets import mnist

def generator_fn():
    """
    Args:
        z: A `float`-like `Tensor` [batch_size, 1, 1, latent_size]
    Returns:
        net:
    """
    net = z
    net = tf.layers.conv2d_transpose(
        net, 64, kernel_size=5, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(
        net, 32, kernel_size=5, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(
        net, 1, kernel_size=4, strides=2, activation=tf.nn.tanh)

    return net

def discriminator_fn(images):
    net = images
    net = tf.layers.conv2d(
        net, 32, kernel_size=4, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d(
        net, 64, kernel_size=5, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 1, kernel_size=5, strides=2)

    return net

def encoder_fn(images, latent_size):
    net = images
    net = tf.layers.conv2d(
        net, 32, kernel_size=4, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d(
        net, 64, kernel_size=5, strides=2, activation=tf.nn.relu)
    net = tf.layers.conv2d(net, latent_size, kernel_size=5, strides=2)

    return net

def code_discriminator(z):
    net = z
    net = tf.layers.dense(net, 1)

    return net


def save_imgs(x, fname):
    n = x.shapes[0]
    fig = figure.Figure(figsize=(n, 1), frameon=False)
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        ax.imshow(
            x[i].sqieeze(),
            interpolation='none',
            cmap=cm.get_cmap('binary'))
        ax.axis('off')
    canvas.print_figure(fname, format='png')
    print('saved %s'.format(fname))

def visualize_training(
    images, reconstructed_images, log_dir, prefix, viz_num=5):
    save_imgs(
        images[:viz_num],
        os.path.join(log_dir, '{}_inputs.png'.format(prefix)))
    save_imgs(
        reconstructed_images[:viz_num],
        os.path.joint(log_dir, '{}_reconstructions.png'.format(prefix)))

def build_input_pipline(mnist_data, batch_size, heldout_size):
    """Build an Iterator switching between train and heldout data."""
    # Build an iterator over training batches
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))
    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the heldout set with batch size=heldout_size
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images, np.int32(mnist_data.validation.labels)))
    heldout_frozen  = (heldout_dataset.take(heldout_size)).
                       repeat().batch(heldout_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator


def main():
    parser = argparse.ArgumentParser(description='Alpha GAN example: MNIST')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    args = parser.parse_args()

    if tf.gfile.Exists(args.model_dir):
        tf.logging.warn('Deleting old log directory at {}'.format(args.model_dir))
        tf.gfile.DeleteRecursively(args.model_dir)
    tf.gfile.MakeDirs(args.model_dir)

    mnist_data = mnist.read_data_sets(args.data_dir)

    with tf.Graph.as_default():
        (images, _, handle,
         training_iterator, heldout_iterator) = build_input_pipline(
             mnist_data, args.batch_size, mnist.validation.num_example)

        images = tf.reshape(images, shape=[-1] + IMAGE_SHAPE)
        noise  = tf.placeholder(tf.float32, [args.batch_size, args.latent_size])

        # model generator
        generator = make_generator(generator_fn, noise)
        discriminator = make_discriminator(
            discriminator_fn, generator.outputs, images)

        # model encoder
        encoder = make_generator(encoder_fn, images, scope='encoder')
        code_discriminator = make_discriminator(
            discriminator_fn, encoder.outputs,
            noise, scope='code_discriminator')

        with tf.variable_scope(generator.scope, reuse=True):
            reconstructed_data = generator.fn(encoder.outputs)

        with tf.variable_scope(discriminator.scope, reuse=True):
            d_rec_outputs = discriminator.fn(reconstructed_data)

        g_loss = loss.alphagan_generator_loss(
            discriminator.gen_outputs, d_rec_outputs,
            images, reconstructed_data, add_summaries=True)

        d_loss = loss.alphagan_discriminator_loss(
            discriminator.real_outputs, discriminator.gen_outputs,
            d_rec_outputs, add_summaries=True)

        e_loss = loss.alphagan_encoder_loss(
            code_discriminator.gen_outputs,
            images, reconstructed_data=, add_summaries=True)

        code_d_loss = loss.alphagan_code_discriminator_loss(
            code_discriminator.real_outputs,
            code_discriminator.gen_outputs,
            add_summaries=True)

        g_optimizer = tf.train.AdamOptimizer(args.learning_rate)
        g_train_op  = g_optimizer.minimize(
            g_loss, var_list=generator.variables)

        d_optimizer = tf.train.AdamOptimizer(args.learning_rate)
        d_train_op  = d_optimizer.minimize(
            d_loss, var_list=discriminator.variables)

        e_optimizer = tf.train.AdamOptimizer(args.learning_rate)
        e_train_op  = e_optimizer.minimize(e_loss, var_list=encoder.variables)

        code_d_optimizer = tf.train.AdamOptimizer(args.learning_rate)
        code_d_train_op  = code_d_optimizer.minimize(
            code_d_loss, var_list=code_discriminator.variables)

        # summary
        tf.summary.image('images', images, max_outputs=1)
        tf.summary.image('generated_images', generator.outputs, max_outputs=1)
        tf.summary.image('reconstructed_images', reconstructed_data, max_outputs=1)
        summary = tf.summary.merge_all()

        init    = tf.global_variables_initializer()
        saver   = tf.train.Saver()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
            sess.run(init)

            # Run the training loop
            train_handle = sess.run(training_iterator.string_handle())
            heldout_handle = sess.run(heldout_iterator.string_handle())
            for step in range(args.max_steps):
                feed_noise = tf.random_normal(
                    [args.batch_size, args.latent_size])
                start_time = time.time()

                _, e_loss_value = sess.run(
                    [e_train_op, e_loss],
                    feed_dict={
                        handle: train_handle, noise: feed_noise)})

                _, g_loss_value = sess.run(
                    [g_train_op, g_loss],
                    feed_dict={handle: train_handle, noise: feed_noise})

                _, d_loss_value = sess.run(
                    [d_train_op, d_loss],
                    feed_dict={handle: train_handle, noise: feed_noise})

                _, code_d_loss_value = sess.run(
                    [code_d_train_op, code_d_loss],
                    feed_dict={handle: train_handle, noise: feed_noise})

                duration = time.time() - start_time

                if step % 100 == 0:
                    print('Step: {:>3d} Discriminator Loss: {:.3f} '
                          'Code Discriminator Loss: {:.3f} ({:.3f} sec)'.format(
                              step, d_loss_value, code_d_loss_value, duration))

                    summary_str = sess.run(summary, feed_dict={handle: train_handle})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % args.viz_step == 0 or (step + 1) == args.max_step:
                    checkpoint_file = os.path.join(args.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                    images_value, reconstructions_value = sess.run(
                        [images, reconstructed_data],
                        feed_dict={handle_training_handle})
                    visualize_training(
                        images_value,
                        reconstructions_value,
                        log_dir=args.model_dir,
                        prefix='step{:0.5d}_train'.format(step))

                    heldout_images_value, heldout_reconstructions_value = sess.run(
                        [images, reconstructed_data],
                        feed_dict={handle: heldout_handle})
                    visualize_training(
                        heldout_images_value,
                        heldout_reconstructions_value,
                        log_dir=args.model_dir,
                        prefix='step{:5d}_validation'.format(step))






if __name__ == '__main__':
    main()
