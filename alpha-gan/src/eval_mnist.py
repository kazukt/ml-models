import argparse
import errno
import functools
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import codebase.networks.mnist as networks
from codebase import mnist_dataset
from codebase.alphagan import make_gan
from codebase.utils import images_to_sprite

NUM_TRAINING = 60000
NUM_TEST = 10000

parser = argparse.ArgumentParser(description='Eval AlphaGAN example: MNIST')
parser.add_argument(
  '--data_dir', default=os.path.join(os.getcwd(), 'data/mnist'))
parser.add_argument(
  '--model_dir', default='checkpoint/mnist/', help='Directory to put the model')
parser.add_argument(
  '--date', type=str)

parser.add_argument('--latent_size', type=int, default=64)
parser.add_argument('--num_images_generated', type=int, default=1000)
parser.add_argument('--train_batch_size', type=int, default=60000)
parser.add_argument('--test_batch_size', type=int, default=10000)

args = parser.parse_args()

def build_input_pipeline(train_dataset, test_dataset, train_batch_size, test_batch_size):
  # Build a iterator.
  train_dataset = train_dataset.repeat().batch(train_batch_size)
  train_iterator = train_dataset.make_one_shot_iterator()

  # Build a iterator.
  test_dataset = test_dataset.repeat().batch(test_batch_size)
  test_iterator = test_dataset.make_one_shot_iterator()

  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, train_dataset.output_types, train_dataset.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, train_iterator, test_iterator

def main():
  if not tf.gfile.Exists(args.model_dir):
    raise FileNotFoundError(
      errno.ENOENT, os.strerror(errno.ENOENT), args.model_dir)
  
  with tf.Graph().as_default():
    with tf.name_scope('input'):
      (images, labels, handle, 
       train_iterator, test_iterator) = build_input_pipeline(
         mnist_dataset.train(args.data_dir), 
         mnist_dataset.test(args.data_dir), 
         train_batch_size=args.train_batch_size,
         test_batch_size=args.test_batch_size)
    
      images = tf.reshape(images, shape=[-1, 28, 28, 1])
      noise  = tf.random_normal([args.num_images_generated, args.latent_size])

    with tf.variable_scope('generator'):
      generated_images = networks.generator(noise, training=False)
    
    with tf.variable_scope('encoder'):
      codes = networks.encoder(images, args.latent_size, is_training=False)
    
    model_saver = tf.train.Saver()
    print('Embedding')
    with tf.Session() as sess:
      model_dir = os.path.join(args.model_dir, args.date)

      checkpoint = tf.train.get_checkpoint_state(model_dir)
      if checkpoint is None:
        raise ValueError('Invalid checkpoint state')
      trained_model = checkpoint.model_checkpoint_path
      model_saver.restore(sess, trained_model)
      
      output_dir = os.path.join(model_dir, 'embedding')
      embed_tensors = []
      summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
      config = projector.ProjectorConfig()
      for iterator, name in [(train_iterator, 'train'), (test_iterator, 'test')]:
        feedable_handle = sess.run(iterator.string_handle())
        imgs, ll, embed = sess.run(
          [images, labels, codes], feed_dict={handle: feedable_handle})
        embed_tensor = tf.Variable(embed, name='{}_embedding'.format(name))
        embed_tensors.append(embed_tensor)
        sess.run(embed_tensor.initializer)
        embedding = config.embeddings.add()
        embedding.tensor_name = embed_tensor.name
        embedding.metadata_path = os.path.join(
          output_dir, '{}_labels.tsv'.format(name))
        embedding.sprite.image_path = os.path.join(
          output_dir, '{}_sprite.png'.format(name))
        embedding.sprite.single_image_dim.extend([28, 28])
        projector.visualize_embeddings(summary_writer, config)
        
        sprite = images_to_sprite(imgs)
        sprite = np.reshape(sprite, [sprite.shape[0], sprite.shape[1]])
        scipy.misc.imsave(
          os.path.join(output_dir, '{}_sprite.png'.format(name)), sprite)
        with open(os.path.join(output_dir, '{}_labels.tsv'.format(name)), 'w') as f:
          f.write('Name\tClass\n')
          for i in range(len(ll)):
            f.write('{:6d}\t{:d}\n'.format(i, ll[i]))
        
      print(output_dir)
      result = sess.run(embed_tensors)
      embedding_saver = tf.train.Saver(embed_tensors)
      embedding_saver.save(sess, os.path.join(output_dir, 'model.ckpt'), 1)



if __name__ == '__main__':
  main()