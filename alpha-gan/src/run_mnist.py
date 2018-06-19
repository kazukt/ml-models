import argparse
import functools
import os
import time
from datetime import datetime

from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import numpy
import tensorflow as tf

from codebase.alphagan import make_gan
import codebase.losses as loss
import codebase.networks.mnist as networks
from codebase import mnist_dataset
from codebase.utils import add_model_summaries

IMAGE_SHAPE = [28, 28, 1]

def save_imgs(x, fname):
    n = x.shape[0]
    fig = figure.Figure(figsize=(n, 1), frameon=False)
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        ax.imshow(
            x[i].squeeze(),
            interpolation='none',
            cmap=cm.get_cmap('binary'))
        ax.axis('off')
    canvas.print_figure(fname, format='png')
    print('saved {:s}'.format(fname))

def visualize_training(
    images, reconstructed_images, log_dir, prefix, viz_num=5):
    save_imgs(
        images[:viz_num],
        os.path.join(log_dir, '{}_inputs.png'.format(prefix)))
    save_imgs(
        reconstructed_images[:viz_num],
        os.path.join(log_dir, '{}_reconstructions.png'.format(prefix)))


def build_input_pipline(training_dataset, batch_size, heldout_size):
    """Build an Iterator switching between train and heldout data."""
    # Build an iterator over training batches

    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the heldout set with batch size=heldout_size

    heldout_frozen = training_dataset.take(heldout_size).repeat().batch(heldout_size)
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator


def main():
    parser = argparse.ArgumentParser(description='Alpha GAN example: MNIST')
    parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'data/mnist'))
    parser.add_argument('--model_dir',
                        default='checkpoint/mnist',
                        help='Directory to put the model')

    # model parameters
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--heldout_size', type=int, default=10000)

    # train parameters
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--generator_learning_rate', type=float, default=1.0e-3)
    parser.add_argument('--discriminator_learning_rate', type=float, default=1.0e-5)
    parser.add_argument('--encoder_learning_rate', type=float, default=1.0e-3)
    parser.add_argument('--code_discriminator_learning_rate', type=float, default=1.0e-5)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--viz_steps', type=int, default=50)

    args = parser.parse_args()

    model_dir =  os.path.join(args.model_dir, datetime.now().strftime('%Y%m%d_%H%M'))

    if tf.gfile.Exists(model_dir):
        tf.logging.warn('Deleting old log directory at {}'.format(model_dir))
        tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MakeDirs(model_dir)

    mnist_data = mnist_dataset.train(args.data_dir)

    with tf.Graph().as_default():
        (images, _, handle,
         training_iterator, heldout_iterator) = build_input_pipline(
             mnist_data, args.batch_size, args.heldout_size)

        images = tf.reshape(images, shape=[-1] + IMAGE_SHAPE)
        noise  = tf.random_normal([args.batch_size, args.latent_size])

        is_training = tf.placeholder(bool)

        # model generator
        generator, discriminator = make_gan(
            functools.partial(
                networks.generator, training=is_training),
            networks.discriminator,
            images,
            noise)

        # model encoder
        encoder, code_discriminator = make_gan(
            functools.partial(
                networks.encoder, latent_size=args.latent_size, is_training=is_training),
            networks.code_discriminator,
            noise,
            images,
            generator_scope='encoder',
            discriminator_scope='code_discriminator')

        with tf.variable_scope(generator.scope, reuse=True):
            reconstructed_data = generator.fn(encoder.outputs)

        with tf.variable_scope(discriminator.scope, reuse=True):
            d_rec_outputs = discriminator.fn(reconstructed_data)
        
        g_loss = loss.alphagan_generator_loss(
            discriminator.gen_outputs, d_rec_outputs,
            images, reconstructed_data,
            adversarial_weights=1.0,
            likelihood_weights=5.0, add_summaries=True)
  
        d_loss = loss.alphagan_discriminator_loss(
            discriminator.real_outputs, discriminator.gen_outputs,
            d_rec_outputs, add_summaries=True)

        e_loss = loss.alphagan_encoder_loss(
            code_discriminator.gen_outputs,
            images, reconstructed_data,
            adversarial_weights=1.0,
            likelihood_weights=5.0, add_summaries=True)

        code_d_loss = loss.modified_discriminator_loss(
            code_discriminator.real_outputs,
            code_discriminator.gen_outputs,
            scope='alphagan_code_discriminator_loss',
            add_summaries=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_optimizer = tf.train.AdamOptimizer(args.generator_learning_rate)
            g_train_op  = g_optimizer.minimize(
                g_loss, var_list=generator.variables)

            d_optimizer = tf.train.AdamOptimizer(args.discriminator_learning_rate)
            d_train_op  = d_optimizer.minimize(
                d_loss, var_list=discriminator.variables)

            e_optimizer = tf.train.AdamOptimizer(args.encoder_learning_rate)
            e_train_op  = e_optimizer.minimize(e_loss, var_list=encoder.variables)

            code_d_optimizer = tf.train.AdamOptimizer(args.code_discriminator_learning_rate)
            code_d_train_op  = code_d_optimizer.minimize(
                code_d_loss, var_list=code_discriminator.variables)

        tf.summary.image('generated_images', generator.outputs)
        tf.summary.image('real_images', discriminator.real_data)
        tf.summary.image('reconstructed_data', reconstructed_data)
        add_model_summaries(generator)
        add_model_summaries(discriminator)
        add_model_summaries(encoder)
        add_model_summaries(code_discriminator)
        summary = tf.summary.merge_all()

        init    = tf.global_variables_initializer()
        saver   = tf.train.Saver()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
            sess.run(init)

            # Run the training loop
            train_handle = sess.run(training_iterator.string_handle())
            heldout_handle = sess.run(heldout_iterator.string_handle())
            for step in range(args.max_steps):
                start_time = time.time()

                _, e_loss_value = sess.run(
                    [e_train_op, e_loss],
                    feed_dict={handle: train_handle, is_training: True})

                _, g_loss_value = sess.run(
                    [g_train_op, g_loss],
                    feed_dict={handle: train_handle, is_training: True})

                _, d_loss_value = sess.run(
                    [d_train_op, d_loss],
                    feed_dict={handle: train_handle, is_training: True})

                _, code_d_loss_value = sess.run(
                    [code_d_train_op, code_d_loss],
                    feed_dict={handle: train_handle, is_training: True})

                duration = time.time() - start_time

                if step % 100 == 0:

                    print('Step: {:>3d} Discriminator Loss: {:.3f} '
                          'Code Discriminator Loss: {:.3f} ({:.3f} sec)'.format(
                              step, d_loss_value, code_d_loss_value, duration))

                    summary_str = sess.run(
                        summary, feed_dict={handle: train_handle, is_training: False})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % args.viz_steps == 0 or (step + 1) == args.max_steps:
                    checkpoint_file = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                    # generator
                    generated_value = sess.run(
                        [generator.outputs], {handle: train_handle, is_training: False})
                    save_imgs(
                        generated_value[0][:5],
                        os.path.join(model_dir, 'step{:05d}_generator.png'.format(step)))

                    # training data
                    images_value, reconstructions_value = sess.run(
                        [images, reconstructed_data],
                        feed_dict={handle: train_handle, is_training: False})
                    visualize_training(
                        images_value,
                        reconstructions_value,
                        log_dir=model_dir,
                        prefix='step{:05d}_train'.format(step))

                    # validation data
                    heldout_images_value, heldout_reconstructions_value = sess.run(
                        [images, reconstructed_data],
                        feed_dict={handle: heldout_handle, is_training: False})
                    visualize_training(
                        heldout_images_value,
                        heldout_reconstructions_value,
                        log_dir=model_dir,
                        prefix='step{:05d}_validation'.format(step))


if __name__ == '__main__':
    main()
