import collections

import tensorflow as tf

class Generator(
<<<<<<< HEAD
    collections.namedtuple('Generator', (
        'fn',
        'inputs',
        'outputs',
        'variables',
        'scope',
    ))):
    """
    """
=======
    collections.namedtuple(
        'Generator',
        [fn, inputs, outputs, variables, scope])):
>>>>>>> 2573dc0d50f750ec48d42a577ce1094708c952c8

def make_generator(generator_fn, inputs, scope='generator'):
    with tf.variable_scope(scope) as gen_scope:
        inputs  = tf.convert_to_tensor(inputs)
        outputs = generator_fn(inputs)

<<<<<<< HEAD
    variables = tf.trainable_variables(gen_scope.name)
=======
    variables = tf.trainable_variables(gen_scope)
>>>>>>> 2573dc0d50f750ec48d42a577ce1094708c952c8

    return Generator(
        fn=generator_fn,
        inputs=inputs,
        outputs=outputs,
        variables=variables,
        scope=gen_scope)

class Discriminator(
<<<<<<< HEAD
    collections.namedtuple('Discriminator', (
        'fn',
        'generated_data',
        'real_data',
        'gen_outputs',
        'real_outputs',
        'variables',
        'scope',
    ))):
    """
    """
=======
    collections.namedtuple(
        'Discriminator',
        [fn, generated_data, real_data,
         gen_outputs, real_outputs, variables, scope])):
>>>>>>> 2573dc0d50f750ec48d42a577ce1094708c952c8

def make_discriminator(
    discriminator_fn,
    generated_data,
    real_data,
    scope='discriminator',
    check_shapes=True):
    with tf.variable_scope(scope) as dis_scope:
        generated_data = tf.convert_to_tensor(generated_data)
        gen_outputs = discriminator_fn(generated_data)

    with tf.variable_scope(dis_scope, reuse=True):
        real_data = tf.convert_to_tensor(real_data)
        real_outputs = discriminator_fn(real_data)

<<<<<<< HEAD
    variables = tf.trainable_variables(dis_scope.name)
=======
    variables = tf.trainable_variables(dis_scope)
>>>>>>> 2573dc0d50f750ec48d42a577ce1094708c952c8

    return Discriminator(
        fn=discriminator_fn,
        generated_data=generated_data,
        real_data=real_data,
        gen_outputs=gen_outputs,
        real_outputs=real_outputs,
        variables=variables,
        scope=dis_scope)
