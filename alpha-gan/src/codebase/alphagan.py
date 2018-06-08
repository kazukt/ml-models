import collections

import tensorflow as tf

class Generator(
    collections.namedtuple('Generator', (
        'fn',
        'inputs',
        'outputs',
        'variables',
        'scope',
    ))):
    """
    """

def make_generator(generator_fn, inputs, scope='generator'):
    with tf.variable_scope(scope) as gen_scope:
        inputs  = tf.convert_to_tensor(inputs)
        outputs = generator_fn(inputs)

    variables = tf.trainable_variables(gen_scope.name)

    return Generator(
        fn=generator_fn,
        inputs=inputs,
        outputs=outputs,
        variables=variables,
        scope=gen_scope)

class Discriminator(
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

    variables = tf.trainable_variables(dis_scope.name)

    return Discriminator(
        fn=discriminator_fn,
        generated_data=generated_data,
        real_data=real_data,
        gen_outputs=gen_outputs,
        real_outputs=real_outputs,
        variables=variables,
        scope=dis_scope)

def make_gan(
    generator_fn,
    discriminator_fn,
    real_data,
    generator_inputs,
    generator_scope='generator',
    discriminator_scope='discriminator',
    check_shapes=True):
    with tf.variable_scope(generator_scope) as gen_scope:
        generator_inputs = tf.convert_to_tensor(generator_inputs)
        generated_data = generator_fn(generator_inputs)
        
    if check_shapes:
        if not generated_data.shape.is_compatible_with(real_data.shape):
            raise ValueError(
                'Generator output shape ({:s}) must be the same shape as real data '
                '({:s}).'.format(generated_data.shape, real_data.shape))
    
    with tf.variable_scope(discriminator_scope) as dis_scope:
        discriminator_gen_outputs = discriminator_fn(generated_data)
    
    with tf.variable_scope(dis_scope, reuse=True):
        real_data = tf.convert_to_tensor(real_data)
        discriminator_real_outputs = discriminator_fn(real_data)
        
    if check_shapes:
        if not generated_data.shape.is_compatible_with(real_data.shape):
            raise ValueError(
                'Generator output shape ({:s}) must be the same shape as real data '
                '({:s}).'.format(generated_data.shape, real_data.shape))
    
    generator_variables = tf.trainable_variables(gen_scope.name)
    discriminator_variables = tf.trainable_variables(dis_scope.name)
    
    generator = Generator(
        fn=generator_fn,
        inputs=generator_inputs,
        outputs=generated_data,
        variables=generator_variables,
        scope=gen_scope)
    
    discriminator = Discriminator(
        fn=discriminator_fn,
        generated_data=generated_data,
        real_data=real_data,
        gen_outputs=discriminator_gen_outputs,
        real_outputs=discriminator_real_outputs,
        variables=discriminator_variables,
        scope=dis_scope)
    
    return generator, discriminator